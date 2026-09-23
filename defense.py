"""Defensive modelling beyond raw averages.

1. Opponent-adjusted ratings: EPA/play, yards/target, catch rate allowed and yards/carry are fit as
   value = league + offense effect + defense effect (ridge-shrunk, alternating least squares), so a
   defense that faced weak offenses isn't mistaken for an elite one.
2. Defense vs position: share of targets and yards/target allowed to WR / TE / RB relative to league.
3. Coverage shells (NGS participation): two-high (Cover 2/2-Man/4/6) vs single-high (Cover 0/1/3)
   rate for each defense, and each receiver's target share vs each shell.

All computed from data before (season, week), same weighting as features.py.
"""
import numpy as np
import pandas as pd

from features import _cutoff, _season_weight

TWO_HIGH = {"COVER_2", "2_MAN", "COVER_4", "COVER_6"}
ONE_HIGH = {"COVER_0", "COVER_1", "COVER_3"}
POS_GROUP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}


def _ridge(df, val, w, lam, iters=8):
    """val ~ mu + off[posteam] + def[defteam]; returns mu, off effects, def effects."""
    mu = float(np.average(df[val], weights=df[w]))
    d = pd.Series(0.0, index=pd.Index(df["defteam"].unique()))
    wo = df[w].groupby(df["posteam"]).sum()
    wd = df[w].groupby(df["defteam"]).sum()
    o = pd.Series(0.0, index=wo.index)
    for _ in range(iters):
        r = df[val] - mu - df["defteam"].map(d)
        o = (r * df[w]).groupby(df["posteam"]).sum() / (wo + lam)
        r = df[val] - mu - df["posteam"].map(o)
        d = (r * df[w]).groupby(df["defteam"]).sum() / (wd + lam)
    return mu, o, d


_COLS = ["season", "week", "game_id", "posteam", "defteam", "pass", "rush", "two_point_attempt", "pass_attempt",
         "sack", "receiver_player_id", "epa", "yards_gained", "complete_pass", "defense_coverage_type"]


def _plays(pbp, season, week):
    df = _cutoff(pbp[[c for c in _COLS if c in pbp.columns]], season, week)
    df = df[((df["pass"] == 1) | (df["rush"] == 1)) & (df["two_point_attempt"] != 1)].copy()
    for c in ("qb_kneel", "qb_spike"):
        if c in df.columns:
            df = df[df[c] != 1]
    df["w"] = _season_weight(df, season, week)
    df["pass_play"] = (df["pass_attempt"] == 1) | (df["sack"] == 1)
    df["is_tgt"] = (df["pass_attempt"] == 1) & (df["sack"] != 1) & df["receiver_player_id"].notna()
    return df


def defense_profiles(pbp: pd.DataFrame, info: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    df = _plays(pbp, season, week)
    out = {}
    lg = {}
    # ---- 1. opponent-adjusted ratings
    specs = [("pass_epa_adj", df[df["pass_play"]], "epa", 250),
             ("rush_epa_adj", df[~df["pass_play"]], "epa", 250),
             ("ypt_adj", df[df["is_tgt"]], "yards_gained", 250),
             ("cmp_adj", df[df["is_tgt"]], "complete_pass", 250),
             ("ypc_adj", df[~df["pass_play"]], "yards_gained", 300)]
    for name, sub, val, lam in specs:
        sub = sub[sub[val].notna()]
        mu, _, d = _ridge(sub, val, "w", lam)
        out[name] = mu + d
        lg[name] = mu
    prof = pd.DataFrame(out)
    zp = (prof["pass_epa_adj"] - prof["pass_epa_adj"].mean()) / prof["pass_epa_adj"].std()
    zr = (prof["rush_epa_adj"] - prof["rush_epa_adj"].mean()) / prof["rush_epa_adj"].std()
    prof["funnel_adj_z"] = zp - zr

    # ---- 2. defense vs position (targets and yards per target allowed by receiver position)
    t = df[df["is_tgt"]].copy()
    pos = info.set_index("gsis_id")["position"].map(POS_GROUP)
    t["pg"] = t["receiver_player_id"].map(pos)
    t = t[t["pg"].notna()]
    tot_w = t.groupby("defteam")["w"].sum()
    for p in ["WR", "TE", "RB"]:
        tp_ = t[t["pg"] == p]
        lg_share = tp_["w"].sum() / t["w"].sum()
        lg_ypt = np.average(tp_["yards_gained"], weights=tp_["w"])
        wp = tp_.groupby("defteam")["w"].sum().reindex(prof.index).fillna(0)
        yp = (tp_["yards_gained"] * tp_["w"]).groupby(tp_["defteam"]).sum().reindex(prof.index).fillna(0)
        share = (wp + 200 * lg_share) / (tot_w.reindex(prof.index).fillna(0) + 200)
        prof[f"tshare_{p}"] = share / lg_share
        prof[f"yptr_{p}"] = ((yp + 120 * lg_ypt) / (wp + 120)) / lg_ypt
        lg[f"ypt_{p}"] = lg_ypt

    # ---- 3. coverage shell mix
    if "defense_coverage_type" in df.columns:
        pp = df[df["pass_play"] & df["defense_coverage_type"].isin(TWO_HIGH | ONE_HIGH)]
        pp = pp.assign(two=pp["defense_coverage_type"].isin(TWO_HIGH).astype(float))
        lg_two = np.average(pp["two"], weights=pp["w"]) if len(pp) else 0.45
        n = pp.groupby("defteam")["w"].sum().reindex(prof.index).fillna(0)
        k2 = (pp["two"] * pp["w"]).groupby(pp["defteam"]).sum().reindex(prof.index).fillna(0)
        prof["two_high_rate"] = (k2 + 150 * lg_two) / (n + 150)
    else:
        lg_two = 0.45
        prof["two_high_rate"] = lg_two
    lg["two_high"] = lg_two
    prof.attrs["league_adj"] = lg
    return prof


def player_shell_splits(pbp: pd.DataFrame, snaps: pd.DataFrame, season: int, week: int, k=40) -> pd.DataFrame:
    """Each receiver's target share vs two-high and single-high shells, relative to his overall share."""
    df = _plays(pbp, season, week)
    if "defense_coverage_type" not in df.columns:
        return pd.DataFrame(columns=["pid", "s2r", "s1r"])
    t = df[df["is_tgt"] & df["defense_coverage_type"].isin(TWO_HIGH | ONE_HIGH)].copy()
    t["two"] = t["defense_coverage_type"].isin(TWO_HIGH)
    team = t.groupby(["game_id", "posteam", "two"]).size().unstack(fill_value=0).rename(columns={True: "t2", False: "t1"})
    ply = t.groupby(["game_id", "posteam", "receiver_player_id", "two"]).size().unstack(fill_value=0).rename(
        columns={True: "p2", False: "p1"})
    ply.index = ply.index.set_names(["game_id", "posteam", "pid"])
    sn = _cutoff(snaps, season, week)
    sn = sn[sn["offense_snaps"] > 0].rename(columns={"gsis_id": "pid", "team": "posteam"})[["game_id", "posteam", "pid", "season"]]
    g = sn.merge(team.reset_index(), on=["game_id", "posteam"]).merge(ply.reset_index(), on=["game_id", "posteam", "pid"], how="left")
    for c in ["t1", "t2", "p1", "p2"]:
        if c not in g:
            g[c] = 0
    g = g.fillna({"p1": 0, "p2": 0})
    g["w"] = _season_weight(g, season, week).values
    a = g.assign(p1=g.p1 * g.w, p2=g.p2 * g.w, t1=g.t1 * g.w, t2=g.t2 * g.w).groupby("pid")[["p1", "p2", "t1", "t2"]].sum()
    s = (a.p1 + a.p2) / (a.t1 + a.t2).replace(0, np.nan)
    s2 = (a.p2 + k * s) / (a.t2 + k)
    s1 = (a.p1 + k * s) / (a.t1 + k)
    res = pd.DataFrame({"s2r": (s2 / s).clip(0.6, 1.6), "s1r": (s1 / s).clip(0.6, 1.6)}).dropna()
    return res.reset_index()
