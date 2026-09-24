"""Leak-free backtest: for every game in a past season, build features using ONLY data available
before that week, simulate with the closing spread/total, and score the predicted probabilities
against what actually happened.

Metrics
  Brier score      mean (p - outcome)^2           lower is better
  Brier skill      1 - Brier / Brier(base rate)   > 0 means better than always guessing the base rate
  Calibration      in each probability bucket, predicted vs actual hit rate (should match)
  Bias             mean projection - mean actual  (systematic over/under-projection)

Usage:  python backtest.py --season 2025 --weeks 4-18
        python backtest.py --season 2025 --grid     (tune funnel/script/TD parameters)
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import config as C
import data
import features as F
from game import Model
from sim import simulate_game

HIST = None   # historical-games pool for empirical margins (set to seasons before the backtest)

THRESHOLDS = {
    "receptions": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    "rec_yds": [19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 99.5],
    "rush_yds": [19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 99.5],
    "pass_yds": [174.5, 199.5, 224.5, 249.5, 274.5, 299.5],
    "anytime_td": [0.5],
}
ACTUAL_COL = {"receptions": "receptions", "rec_yds": "receiving_yards", "rush_yds": "rushing_yards",
              "pass_yds": "passing_yards", "anytime_td": "tds"}


def prepare(model, weeks, feats=None):
    """Build every game's inputs once (usage after inactives, QBs, actual outcomes)."""
    if True:
        ps = data.player_stats([model.season])
        ps["tds"] = ps["rushing_tds"].fillna(0) + ps["receiving_tds"].fillna(0)
        # who actually played = offensive snaps > 0; players with zero stats must count as zeros
        sn = model.snaps[(model.snaps["season"] == model.season) & (model.snaps["offense_snaps"] > 0)]
        played = sn[["week", "gsis_id"]].drop_duplicates().rename(columns={"gsis_id": "player_id"})
        ps = played.merge(ps, on=["week", "player_id"], how="left")
        for c in ACTUAL_COL.values():
            ps[c] = ps[c].fillna(0)
        rows = []
        for wk in weeks:
            tp, usage = feats[wk] if feats else model.features(wk)
            outs = F.injury_outs(model.inj, model.season, wk)
            games = model.sched[(model.sched["week"] == wk) & model.sched["spread_line"].notna()]
            actual = ps[ps["week"] == wk].set_index("player_id")
            for _, g in games.iterrows():
                h, a = g["home_team"], g["away_team"]
                if h not in tp.index or a not in tp.index:
                    continue
                us = {}
                # inactives: anyone who took no offensive snap is treated as ruled out (known at kickoff)
                played = set(model.snaps[(model.snaps["season"] == model.season) & (model.snaps["week"] == wk)
                                         & (model.snaps["offense_snaps"] > 0)]["gsis_id"])
                inactive = set(usage[usage["team"].isin([h, a]) & ~usage["pid"].isin(played)]["pid"])
                game_outs = outs | inactive
                for t in (h, a):
                    qb = F.primary_qb(model.pbp, t, model.season, wk, game_outs)
                    u = F.active_usage(usage, t, game_outs)
                    u = u[(u["pos"] != "QB") | (u["pid"] == qb)]
                    if qb is not None and qb not in set(u["pid"]):
                        u = pd.concat([u, usage[usage["pid"] == qb]])
                    us[t] = (u, qb)
                wx = data.weather_from_schedule(g, model._wx_text.get(g["game_id"]))
                # offensive line injuries, so the feature can be tested like any other
                ol = model.ol_injury_adjust(wk, game_outs) if hasattr(model, "ol_injury_adjust") else {}
                if ol:
                    tp = tp.copy(); tp.attrs = model.features(wk)[0].attrs
                    if "ol_missing" not in tp.columns:
                        tp["ol_missing"] = 0.0
                    for t_, lost in ol.items():
                        if t_ in tp.index:
                            tp.loc[t_, "ol_missing"] = lost
                rows.append(dict(wk=wk, h=h, a=a, spread=g["spread_line"], total=g["total_line"], tp=tp, wx=wx,
                                 uh=us[h][0], ua=us[a][0], qh=us[h][1], qa=us[a][1], actual=actual))
        return rows


def collect_prepared(prepared, n=3000, params=None):
    params = params or {}
    old = {k: getattr(C, k) for k in params}
    for k, v in params.items():
        setattr(C, k, v)
    try:
        rows = []
        for gm in prepared:
            wk, actual = gm["wk"], gm["actual"]
            if True:
                sim = simulate_game(gm["h"], gm["a"], gm["spread"], gm["total"], gm["tp"], gm["uh"], gm["ua"],
                                    gm["qh"], gm["qa"], n=n, seed=wk,
                                    weather=gm.get("wx") if getattr(C, "USE_WEATHER", True) else None, hist=HIST)
                for name, st in sim.players.items():
                    pid = sim.meta[name]["pid"]
                    if pid not in actual.index:          # didn't play -> in live use you'd see inactives
                        continue
                    act = actual.loc[pid]
                    if isinstance(act, pd.DataFrame):
                        act = act.iloc[0]
                    pos = sim.meta[name]["pos"]
                    for stat, ths in THRESHOLDS.items():
                        if stat == "pass_yds" and pos != "QB":
                            continue
                        if stat in ("receptions", "rec_yds") and pos == "QB":
                            continue
                        if stat == "rush_yds" and pos not in ("RB", "QB"):
                            continue
                        x = st[stat]
                        y = float(act[ACTUAL_COL[stat]]) if pd.notna(act[ACTUAL_COL[stat]]) else 0.0
                        mu = float(x.mean())
                        for th in ths:
                            rows.append((wk, name, pos, stat, th, float((x > th).mean()), float(y > th), mu, y))
        return pd.DataFrame(rows, columns=["week", "player", "pos", "stat", "line", "p", "hit", "proj", "actual"])
    finally:
        for k, v in old.items():
            setattr(C, k, v)


def collect(model, weeks, n=3000, params=None, feats=None):
    return collect_prepared(prepare(model, weeks, feats), n=n, params=params)


def score(df):
    out = []
    for stat, g in df.groupby("stat"):
        base = g.groupby("line")["hit"].transform("mean")
        brier = ((g["p"] - g["hit"]) ** 2).mean()
        brier0 = ((base - g["hit"]) ** 2).mean()
        u = g.drop_duplicates(["week", "player"])
        out.append({"stat": stat, "n": len(g), "brier": round(brier, 4),
                    "skill_vs_base_rate": round(1 - brier / brier0, 3),
                    "bias(proj-actual)": round((u["proj"] - u["actual"]).mean(), 2),
                    "mae": round((u["proj"] - u["actual"]).abs().mean(), 2)})
    return pd.DataFrame(out)


def calibration(df, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    df = df.copy()
    df["bucket"] = pd.cut(df["p"], bins, include_lowest=True)
    return df.groupby("bucket", observed=True).agg(n=("hit", "size"), predicted=("p", "mean"),
                                                   actual=("hit", "mean")).round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", default="4-18")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    lo, hi = map(int, args.weeks.split("-"))
    weeks = list(range(lo, hi + 1))
    m = Model(args.season)
    feats = {w: m.features(w) for w in weeks}
    if not args.grid:
        df = collect(m, weeks, n=args.n, feats=feats)
        print(score(df).to_string(index=False))
        print(calibration(df).to_string())
    else:
        grid = {"FUNNEL_BETA": [0.0, 0.015, 0.035], "SCRIPT_PASS_BETA": [0.04, 0.075, 0.11],
                "POINTS_PER_TD": [7.3, 8.0]}
        res = []
        prep = prepare(m, weeks, feats)
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid, vals))
            df = collect_prepared(prep, n=args.n, params=p)
            sc = score(df)
            res.append({**p, "mean_brier": sc["brier"].mean(),
                        **{f"brier_{r.stat}": r.brier for r in sc.itertuples()}})
            print(res[-1], flush=True)
        print(pd.DataFrame(res).sort_values("mean_brier").head(8).to_string(index=False))
