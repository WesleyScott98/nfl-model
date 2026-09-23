"""Bet tracking: the only real proof of an edge.

    import tracking as T
    bid = T.log_bet(season=2026, week=3, game="GB@CHI", price=-115, stake=1.0, p_final=0.56,
                    legs=[dict(player="D.J. Moore", stat="rec_yds", side="over", line=59.5)])
    T.set_close(bid, -135)        # FanDuel price right before kickoff (for closing-line value)
    T.grade()                     # after the games: settles from nflverse results
    T.report()                    # record, ROI with 95% CI, CLV, model calibration

Parlays: pass several legs and the parlay price. Team legs: dict(team="CHI", market="ml"),
dict(team="CHI", market="spread", line=-3.5), dict(market="total", side="over", line=44.5).

CLV (closing-line value): how much the market moved toward your bet after you placed it.
Positive average CLV over many bets is the strongest early evidence of a real edge; ROI takes
hundreds of bets to separate skill from luck.
"""
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import data
from odds import american_to_decimal, implied_prob

PATH = os.environ.get("NFL_EDGE_BETS", os.path.join(os.path.dirname(__file__), "bets.csv"))
COLS = ["bet_id", "placed_at", "season", "week", "game", "legs", "price", "stake", "p_model", "p_final",
        "close_price", "status", "pnl", "note"]
STAT_COL = {"receptions": "receptions", "rec_yds": "receiving_yards", "rush_yds": "rushing_yards",
            "pass_yds": "passing_yards", "pass_tds": "passing_tds", "targets": "targets", "carries": "carries",
            "completions": "completions", "pass_att": "attempts"}


def _load():
    if os.path.exists(PATH):
        return pd.read_csv(PATH, dtype={"bet_id": str})
    return pd.DataFrame(columns=COLS)


def _save(df):
    df.to_csv(PATH, index=False)


def log_bet(season, week, game, legs, price, stake=1.0, p_model=None, p_final=None, note=""):
    df = _load()
    bid = uuid.uuid4().hex[:8]
    row = {"bet_id": bid, "placed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "season": season, "week": week, "game": game, "legs": json.dumps(legs), "price": price,
           "stake": stake, "p_model": p_model, "p_final": p_final, "close_price": np.nan,
           "status": "open", "pnl": np.nan, "note": note}
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True))
    return bid


def set_close(bet_id, close_price):
    df = _load()
    df.loc[df["bet_id"] == bet_id, "close_price"] = close_price
    _save(df)


# ------------------------------------------------------------------ grading
def _norm(name):
    n = str(name).lower().replace(".", "").replace("'", "")
    for suf in (" jr", " sr", " iii", " ii", " iv"):
        n = n.replace(suf, "")
    return " ".join(n.split())


def _leg_result(leg, stats, played, sched_row):
    """-> 'win' | 'loss' | 'push' | 'void' | None (result not available yet)."""
    side = leg.get("side", "over")
    if "player" in leg:
        key = _norm(leg["player"])
        row = stats[stats["_n"] == key]
        if row.empty:
            if key not in played:
                return "void"                      # didn't take a snap -> book voids the leg
            r = pd.Series(dtype=float)             # played but recorded nothing -> zeros count
        else:
            r = row.iloc[0]
        if leg["stat"] == "anytime_td":
            val = r.get("rushing_tds", 0) + r.get("receiving_tds", 0)
            line = leg.get("line", 0.5)
        else:
            val, line = float(r.get(STAT_COL[leg["stat"]], 0) or 0), leg["line"]
        val = float(val or 0)
    else:
        if sched_row is None or pd.isna(sched_row["home_score"]):
            return None
        h, a = sched_row["home_score"], sched_row["away_score"]
        if leg["market"] == "total":
            val, line = h + a, leg["line"]
        else:
            own, opp = (h, a) if leg["team"] == sched_row["home_team"] else (a, h)
            if leg["market"] == "ml":
                return "win" if own > opp else ("push" if own == opp else "loss")
            val, line, side = own - opp + leg["line"], 0.0, "over"
    if val == line:
        return "push"
    return "win" if (val > line) == (side == "over") else "loss"


def grade():
    df = _load()
    open_ = df[df["status"] == "open"]
    if open_.empty:
        return df
    for (season, week), grp in open_.groupby(["season", "week"]):
        try:
            ps = data.player_stats([int(season)])
        except Exception:
            continue
        stats = ps[ps["week"] == week].copy()
        if stats.empty:
            continue
        stats["_n"] = stats["player_display_name"].map(_norm)
        info = data.player_info().set_index("gsis_id")["display_name"]
        sn = data.snaps([int(season)])
        played = {_norm(info.get(i, "")) for i in sn[(sn["week"] == week) & (sn["offense_snaps"] > 0)]["gsis_id"]}
        sched = data.schedules(int(season))
        for i, bet in grp.iterrows():
            legs = json.loads(bet["legs"])
            away, home = bet["game"].split("@") if "@" in str(bet["game"]) else (None, None)
            srow = sched[(sched["week"] == week) & (sched["home_team"] == home) & (sched["away_team"] == away)]
            srow = srow.iloc[0] if len(srow) else None
            res = [_leg_result(l, stats, played, srow) for l in legs]
            if any(r is None for r in res):
                continue
            live = [r for r in res if r not in ("void", "push")]
            if "loss" in live:
                df.loc[i, ["status", "pnl"]] = ["loss", -bet["stake"]]
            elif not live:
                df.loc[i, ["status", "pnl"]] = ["void", 0.0]
            elif len(live) < len(res):
                # books reprice parlays when a leg voids/pushes; record as won but flag for manual check
                df.loc[i, ["status", "pnl", "note"]] = ["win*", bet["stake"] * (american_to_decimal(bet["price"]) - 1),
                                                        f"{bet['note']} | leg voided/pushed: check FanDuel payout".strip(" |")]
            else:
                df.loc[i, ["status", "pnl"]] = ["win", bet["stake"] * (american_to_decimal(bet["price"]) - 1)]
    _save(df)
    return df


# ------------------------------------------------------------------ reporting
def report(df=None, n_boot=2000, seed=0):
    df = _load() if df is None else df
    s = df[df["status"].isin(["win", "win*", "loss"])].copy()
    out = {"bets_settled": len(s), "open": int((df["status"] == "open").sum())}
    if s.empty:
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
    s["won"] = s["status"].str.startswith("win").astype(float)
    staked = s["stake"].sum()
    out.update({"record": f"{int(s.won.sum())}-{int(len(s) - s.won.sum())}", "units_staked": round(staked, 2),
                "profit_units": round(s["pnl"].sum(), 2), "roi": round(s["pnl"].sum() / staked, 4)})
    rng = np.random.default_rng(seed)
    boots = [s.sample(len(s), replace=True, random_state=int(rng.integers(1e9))) for _ in range(n_boot)]
    rois = [b["pnl"].sum() / b["stake"].sum() for b in boots]
    out["roi_95ci"] = (round(float(np.percentile(rois, 2.5)), 4), round(float(np.percentile(rois, 97.5)), 4))
    if s["p_final"].notna().any():
        e = s[s["p_final"].notna()]
        exp_profit = (e["p_final"] * (e["price"].map(american_to_decimal) - 1) - (1 - e["p_final"])) * e["stake"]
        out["expected_profit_units"] = round(exp_profit.sum(), 2)
        out["model_avg_p_vs_hit_rate"] = (round(e["p_final"].mean(), 3), round(e["won"].mean(), 3))
    c = df[df["close_price"].notna() & df["price"].notna()]
    if len(c):
        clv = c["close_price"].map(implied_prob) - c["price"].map(implied_prob)
        out["avg_clv_pct_pts"] = round(100 * clv.mean(), 2)
        out["share_beat_close"] = round((clv > 0).mean(), 3)
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}


BOARD_PATH = os.environ.get("NFL_EDGE_BOARD", os.path.join(os.path.dirname(__file__), "boards.csv"))


def log_board(priced, season, week):
    """Save the whole priced board (every FanDuel line + the model's probability) for a week.

    Why: we have no history of sportsbook prop lines, so we can't yet test questions like
    "when the model disagrees with the market by 10+ points, who is right?" or tune how much to
    defer to the market (MODEL_WEIGHT). Logging every board builds that dataset week by week.
    After ~6-8 weeks, run the analysis in README ('Calibrating trust in the market')."""
    df = priced.drop(columns=[c for c in ["leg"] if c in priced.columns]).copy()
    df.insert(0, "season", season)
    df.insert(1, "week", week)
    df.insert(2, "logged_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    header = not os.path.exists(BOARD_PATH)
    df.to_csv(BOARD_PATH, mode="a", header=header, index=False)
    return len(df)


def board_review(season=None):
    """Once boards.csv has a few weeks and results exist, score the model against the market:
    on lines where they disagreed, who was closer? Returns hit rates by disagreement size."""
    if not os.path.exists(BOARD_PATH):
        return pd.DataFrame()
    b = pd.read_csv(BOARD_PATH)
    if season:
        b = b[b["season"] == season]
    rows = []
    for (season_, week), grp in b.groupby(["season", "week"]):
        try:
            ps = data.player_stats([int(season_)])
        except Exception:
            continue
        st = ps[ps["week"] == week].copy()
        if st.empty:
            continue
        st["_n"] = st["player_display_name"].map(_norm)
        sn = data.snaps([int(season_)])
        info = data.player_info().set_index("gsis_id")["display_name"]
        played = {_norm(info.get(i, "")) for i in sn[(sn["week"] == week) & (sn["offense_snaps"] > 0)]["gsis_id"]}
        for _, r in grp.iterrows():
            if "player" not in str(r.get("selection", "")) and pd.isna(r.get("player", np.nan)):
                continue
            leg = {"player": r.get("player"), "stat": r.get("stat"), "line": r.get("line"), "side": r.get("side", "over")}
            if not leg["player"] or leg["stat"] not in STAT_COL and leg["stat"] != "anytime_td":
                continue
            res = _leg_result(leg, st, played, None)
            if res in (None, "void", "push"):
                continue
            rows.append({"disagreement": round(r["p_model"] - r["p_market"], 3), "model_higher": r["p_model"] > r["p_market"],
                         "won": res == "win"})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["bucket"] = pd.cut(d["disagreement"].abs(), [0, .05, .10, .15, 1.0])
    return d.groupby(["bucket", "model_higher"], observed=True).agg(n=("won", "size"), hit_rate=("won", "mean")).round(3)
