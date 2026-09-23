"""Feature engineering.

Team level  : pace, neutral pass rate, pass-rate-over-expected, offensive EPA,
              defensive EPA/yards allowed split pass vs rush -> "funnel",
              man/zone rate, blitz rate, pressure rate.
Player level: recency-weighted target / carry / red-zone shares (only games actually played,
              via snap counts), efficiency shrunk toward position means,
              yards-per-target vs man and vs zone.

Everything is computed as of a (season, week) cut-off so the same code drives live picks
and leak-free backtests."""
import numpy as np
import pandas as pd

import config as C

LEAGUE_POS = ["WR", "TE", "RB", "QB"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _cutoff(df, season, week):
    return df[(df["season"] < season) | ((df["season"] == season) & (df["week"] < week))]


def _season_weight(df, season, week):
    g = max(week - 1, 0)
    prior_w = C.PRIOR_SEASON_K / (C.PRIOR_SEASON_K + g)
    w = np.where(df["season"] == season, 1.0, prior_w * (0.5 ** (season - 1 - df["season"])))
    return pd.Series(w, index=df.index)


def _wmean(values, weights):
    m = weights > 0
    return float(np.average(values[m], weights=weights[m])) if m.any() else np.nan


def _shrink(value, n, prior, k):
    return (value * n + prior * k) / (n + k)


# ----------------------------------------------------------------------------
# team profiles
# ----------------------------------------------------------------------------
def team_profiles(pbp: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    df = _cutoff(pbp, season, week)
    df = df[(df["pass"] == 1) | (df["rush"] == 1)].copy()
    df = df[(df["two_point_attempt"] != 1) & (df.get("qb_kneel", 0) != 1) & (df.get("qb_spike", 0) != 1)]
    df["w"] = _season_weight(df, season, week)
    df["pass_play"] = ((df["pass_attempt"] == 1) | (df["sack"] == 1)).astype(float)  # scrambles count as runs
    df["neutral"] = (df["wp"].between(0.2, 0.8) & (df["down"] <= 3) &
                     (df["half_seconds_remaining"] > 120)).astype(float)
    df["tgt"] = ((df["pass_attempt"] == 1) & (df["sack"] != 1)).astype(float)
    df["neutral_pass_play"] = df["pass_play"] * df["neutral"]

    # per-game offensive rows
    g = df.groupby(["season", "game_id", "posteam"])
    off = pd.DataFrame({
        "w": g["w"].first(),
        "plays": g.size(),
        "pass_plays": g["pass_play"].sum(),
        "sacks": g["sack"].sum(),
        "neutral_n": g["neutral"].sum(),
        "neutral_pass": g["neutral_pass_play"].sum(),
        "pass_oe": g["pass_oe"].mean(),
        "pass_td": g["pass_touchdown"].sum(),
        "rush_td": g["rush_touchdown"].sum(),
    }).reset_index()

    # per-game defensive rows
    d = df.copy()
    d["is_man"] = (d.get("defense_man_zone_type") == "MAN_COVERAGE").astype(float)
    d["has_cov"] = d.get("defense_man_zone_type", pd.Series(index=d.index, dtype=object)).isin(
        ["MAN_COVERAGE", "ZONE_COVERAGE"]).astype(float)
    d["blitz"] = (d.get("n_blitzers", pd.Series(np.nan, index=d.index)) > 0).astype(float)
    d["has_blitz"] = d.get("n_blitzers", pd.Series(np.nan, index=d.index)).notna().astype(float)
    d["pressure"] = d.get("was_pressure", pd.Series(np.nan, index=d.index)).astype(float)
    gd = d.groupby(["season", "game_id", "defteam"])
    pas = d[d["pass_play"] == 1].groupby(["season", "game_id", "defteam"])
    tg = d[d["tgt"] == 1].groupby(["season", "game_id", "defteam"])
    ru = d[d["pass_play"] == 0].groupby(["season", "game_id", "defteam"])
    de = pd.DataFrame({
        "w": gd["w"].first(),
        "plays_allowed": gd.size(),
        "pass_n": pas.size(), "pass_epa_sum": pas["epa"].sum(),
        "tgt_n": tg.size(), "tgt_yds": tg["yards_gained"].sum(), "cmp": tg["complete_pass"].sum(),
        "rush_n": ru.size(), "rush_epa_sum": ru["epa"].sum(), "rush_yds": ru["yards_gained"].sum(),
        "man_n": gd["is_man"].sum(), "cov_n": gd["has_cov"].sum(),
        "blitz_n": gd["blitz"].sum(), "blitz_obs": gd["has_blitz"].sum(),
        "press_n": pas["pressure"].sum(), "press_obs": pas["pressure"].count(),
    }).fillna(0).reset_index()

    teams = sorted(set(off["posteam"]) | set(de["defteam"]))
    rows = []
    for t in teams:
        o = off[off["posteam"] == t]
        x = de[de["defteam"] == t]
        wo, wd = o["w"].values, x["w"].values
        rows.append({
            "team": t,
            "games_w": wo.sum(),
            "plays_pg": _wmean(o["plays"].values, wo),
            "plays_allowed_pg": _wmean(x["plays_allowed"].values, wd),
            "neutral_pass_rate": (o["neutral_pass"] * wo).sum() / max((o["neutral_n"] * wo).sum(), 1),
            "neutral_n": (o["neutral_n"] * wo).sum(),
            "pass_oe": _wmean(o["pass_oe"].fillna(0).values, wo),
            "sack_rate": (o["sacks"] * wo).sum() / max((o["pass_plays"] * wo).sum(), 1),
            "pass_td_share": (o["pass_td"] * wo).sum() / max(((o["pass_td"] + o["rush_td"]) * wo).sum(), 1),
            "td_n": ((o["pass_td"] + o["rush_td"]) * wo).sum(),
            "def_pass_n": (x["pass_n"] * wd).sum(),
            "def_pass_epa": (x["pass_epa_sum"] * wd).sum() / max((x["pass_n"] * wd).sum(), 1),
            "def_rush_n": (x["rush_n"] * wd).sum(),
            "def_rush_epa": (x["rush_epa_sum"] * wd).sum() / max((x["rush_n"] * wd).sum(), 1),
            "def_ypt": (x["tgt_yds"] * wd).sum() / max((x["tgt_n"] * wd).sum(), 1),
            "def_tgt_n": (x["tgt_n"] * wd).sum(),
            "def_cmp_rate": (x["cmp"] * wd).sum() / max((x["tgt_n"] * wd).sum(), 1),
            "def_ypc": (x["rush_yds"] * wd).sum() / max((x["rush_n"] * wd).sum(), 1),
            "man_rate": (x["man_n"] * wd).sum() / max((x["cov_n"] * wd).sum(), 1) if (x["cov_n"] * wd).sum() > 0 else np.nan,
            "cov_n": (x["cov_n"] * wd).sum(),
            "blitz_rate": (x["blitz_n"] * wd).sum() / (x["blitz_obs"] * wd).sum() if (x["blitz_obs"] * wd).sum() > 0 else np.nan,
            "pressure_rate": (x["press_n"] * wd).sum() / (x["press_obs"] * wd).sum() if (x["press_obs"] * wd).sum() > 0 else np.nan,
        })
    tp = pd.DataFrame(rows).set_index("team")

    # shrink noisy rates toward league means
    lg = {c: np.nanmean(tp[c]) for c in tp.columns}
    k = C.SHRINK_DEF_PLAYS
    tp["neutral_pass_rate"] = _shrink(tp["neutral_pass_rate"], tp["neutral_n"], lg["neutral_pass_rate"], 150)
    tp["pass_td_share"] = _shrink(tp["pass_td_share"], tp["td_n"], 0.6, 20)
    tp["def_pass_epa"] = _shrink(tp["def_pass_epa"], tp["def_pass_n"], lg["def_pass_epa"], k)
    tp["def_rush_epa"] = _shrink(tp["def_rush_epa"], tp["def_rush_n"], lg["def_rush_epa"], k)
    tp["def_ypt"] = _shrink(tp["def_ypt"], tp["def_tgt_n"], lg["def_ypt"], k)
    tp["def_cmp_rate"] = _shrink(tp["def_cmp_rate"], tp["def_tgt_n"], lg["def_cmp_rate"], k)
    tp["def_ypc"] = _shrink(tp["def_ypc"], tp["def_rush_n"], lg["def_ypc"], k)
    tp["man_rate"] = _shrink(tp["man_rate"].fillna(lg["man_rate"]), tp["cov_n"], lg["man_rate"], 150)

    # funnel: +ve = relatively weaker vs the pass -> offenses should throw more
    zp = (tp["def_pass_epa"] - tp["def_pass_epa"].mean()) / tp["def_pass_epa"].std()
    zr = (tp["def_rush_epa"] - tp["def_rush_epa"].mean()) / tp["def_rush_epa"].std()
    tp["funnel_z"] = zp - zr
    tp.attrs["league"] = {"ypt": lg["def_ypt"], "cmp": lg["def_cmp_rate"], "ypc": lg["def_ypc"],
                          "man_rate": lg["man_rate"]}
    return tp


# ----------------------------------------------------------------------------
# player usage
# ----------------------------------------------------------------------------
def player_usage(pbp: pd.DataFrame, snaps: pd.DataFrame, season: int, week: int,
                 info: pd.DataFrame = None, inj: pd.DataFrame = None) -> pd.DataFrame:
    df = _cutoff(pbp, season, week)
    df = df[((df["pass"] == 1) | (df["rush"] == 1)) & (df["two_point_attempt"] != 1)
            & (df.get("qb_kneel", 0) != 1) & (df.get("qb_spike", 0) != 1)].copy()
    df["is_tgt"] = ((df["pass_attempt"] == 1) & (df["sack"] != 1) & df["receiver_player_id"].notna())
    df["is_car"] = (df["rush_attempt"] == 1) & df["rusher_player_id"].notna()
    df["rz"] = df["yardline_100"] <= 20
    df["gl"] = df["yardline_100"] <= 10
    df["is_man"] = df.get("defense_man_zone_type") == "MAN_COVERAGE"
    df["is_zone"] = df.get("defense_man_zone_type") == "ZONE_COVERAGE"

    df["rz_tgt"] = df["is_tgt"] & df["rz"]
    df["gl_car"] = df["is_car"] & df["gl"]
    df["air_tgt"] = df["air_yards"].clip(lower=0).fillna(0) * df["is_tgt"]
    df["yds_man"] = df["yards_gained"] * df["is_man"]
    df["yds_zone"] = df["yards_gained"] * df["is_zone"]
    team_game = df.groupby(["game_id", "posteam"]).agg(
        team_tgt=("is_tgt", "sum"), team_car=("is_car", "sum"), team_rz_tgt=("rz_tgt", "sum"),
        team_gl_car=("gl_car", "sum"), team_air=("air_tgt", "sum"),
    ).reset_index()

    t = df[df["is_tgt"]].drop(columns=["name"], errors="ignore").rename(
        columns={"receiver_player_id": "pid", "receiver_player_name": "pname"})
    rec = t.groupby(["game_id", "posteam", "pid"]).agg(
        name=("pname", "first"), tgt=("is_tgt", "size"), rec=("complete_pass", "sum"),
        rec_yds=("yards_gained", "sum"), rec_td=("pass_touchdown", "sum"),
        rz_tgt=("rz", "sum"), air=("air_tgt", "sum"),
        tgt_man=("is_man", "sum"), yds_man=("yds_man", "sum"),
        tgt_zone=("is_zone", "sum"), yds_zone=("yds_zone", "sum"),
    ).reset_index()
    r = df[df["is_car"]].drop(columns=["name"], errors="ignore").rename(
        columns={"rusher_player_id": "pid", "rusher_player_name": "pname"})
    car = r.groupby(["game_id", "posteam", "pid"]).agg(
        name_r=("pname", "first"), car=("is_car", "size"), rush_yds=("yards_gained", "sum"),
        rush_td=("rush_touchdown", "sum"), gl_car=("gl", "sum"),
    ).reset_index()

    # appearance = any offensive snap (so games with zero targets still count against share)
    sn = _cutoff(snaps, season, week).rename(columns={"gsis_id": "pid", "team": "posteam"})
    sn = sn[sn["offense_snaps"] > 0][["game_id", "posteam", "pid", "season", "offense_pct"]]
    pg = sn.merge(rec, on=["game_id", "posteam", "pid"], how="left").merge(
        car, on=["game_id", "posteam", "pid"], how="left").merge(team_game, on=["game_id", "posteam"], how="left")
    pg["name"] = pg["name"].fillna(pg["name_r"])
    pg = pg.fillna({c: 0 for c in ["tgt", "rec", "rec_yds", "rec_td", "rz_tgt", "air", "tgt_man", "yds_man",
                                    "tgt_zone", "yds_zone", "car", "rush_yds", "rush_td", "gl_car"]})
    pg = pg[pg["team_tgt"].notna()]
    pg["week"] = pg["game_id"].str.slice(5, 7).astype(int)

    # injury-shortened games (a regular starter playing < 60% of his usual snaps) distort shares;
    # drop them. If a player is on a snap limit this week, use snap_override in active_usage().
    med = pg.groupby("pid")["offense_pct"].transform("median")
    short = (med >= 0.5) & (pg["offense_pct"] < 0.6 * med)
    if C.SHORT_GAME_NEEDS_INJURY and inj is not None and len(inj):
        hurt = set(zip(inj["season"], inj["week"], inj["gsis_id"]))
        on_report = pd.Series([(s, w, p) in hurt for s, w, p in zip(pg["season"], pg["week"], pg["pid"])], index=pg.index)
        short = short & on_report
    if C.SHORT_GAME_MODE == "drop":
        pg = pg[~short]
        pg["short_scale"], pg["short_w"] = 1.0, 1.0
    else:
        ratio = (med / pg["offense_pct"].clip(lower=0.05))
        pg["short_scale"] = np.where(short, np.clip(ratio ** C.SHORT_GAME_ALPHA, 1.0, 2.5), 1.0)
        pg["short_w"] = np.where(short, 1 / ratio, 1.0)

    # current team = team in the player's most recent game; only keep games with that team
    pg = pg.sort_values(["season", "week"])
    cur_team = pg.groupby("pid")["posteam"].last()
    team_w = np.where(pg["posteam"] == pg["pid"].map(cur_team), 1.0, C.PRIOR_TEAM_WEIGHT)
    pg["games_ago"] = pg.groupby("pid").cumcount(ascending=False)
    pg["w"] = (0.5 ** (pg["games_ago"] / C.USAGE_HALF_LIFE_GAMES)) * _season_weight(pg, season, week).values * team_w * pg["short_w"].values

    raw = {c: (pg[n] / pg[d].replace(0, np.nan)).fillna(0) * pg["short_scale"] for c, n, d in
           [("s_tgt", "tgt", "team_tgt"), ("s_car", "car", "team_car"), ("s_rz", "rz_tgt", "team_rz_tgt"),
            ("s_gl", "gl_car", "team_gl_car"), ("s_air", "air", "team_air")]}

    def base_shares(adj):
        tmp = pg[["pid", "w"]].copy()
        for c in raw:
            tmp[c] = raw[c] * adj[c] * pg["w"]
        g = tmp.groupby("pid")
        return g[list(raw)].sum().div(g["w"].sum(), axis=0)

    # with/without adjustment: in games where a CURRENT teammate was absent (injured, or not yet
    # on the team), everyone else's share was inflated. Deflate by the absent teammates' shares.
    # Iterate to a fixed point: shares used for deflation are themselves deflated, so one-game
    # samples of new arrivals can't wipe out everyone else. Only meaningful teammates count.
    team_of = pg.groupby("pid")["posteam"].last().rename("cur_team")
    # candidates for "absent": rostered on this team now AND on it that season (not later arrivals)
    member = pg[["pid", "posteam", "season"]].drop_duplicates()
    member = member[member["posteam"] == member["pid"].map(team_of)]
    games = pg[["game_id", "posteam", "season"]].drop_duplicates()
    cand = games.merge(member, on=["posteam", "season"])
    pres = sn[["game_id", "pid"]].drop_duplicates().assign(_p=1)
    absent = cand.merge(pres, on=["game_id", "pid"], how="left")
    absent = absent[absent["_p"].isna()][["game_id", "posteam", "pid"]]
    adj = {c: pd.Series(1.0, index=pg.index) for c in raw}
    key = pd.MultiIndex.from_frame(pg[["game_id", "posteam"]])
    for _ in range(4):
        base = base_shares(adj)
        b = absent.join(base, on="pid")
        for c in raw:
            vac = b[c].where(b[c] >= 0.05, 0).groupby([b["game_id"], b["posteam"]]).sum()
            adj[c] = pd.Series(np.clip(1 - vac.reindex(key).fillna(0).values, 0.55, 1.0), index=pg.index)

    def share(num, den, key):
        return raw[key] * adj[key] * pg["w"]

    pg["s_tgt"], pg["s_car"] = share("tgt", "team_tgt", "s_tgt"), share("car", "team_car", "s_car")
    pg["s_rz"], pg["s_gl"] = share("rz_tgt", "team_rz_tgt", "s_rz"), share("gl_car", "team_gl_car", "s_gl")
    pg["s_air"] = share("air", "team_air", "s_air")
    pg["w_snap"] = pg["offense_pct"] * pg["w"]
    pg["w_rush_yds"] = pg["rush_yds"] * pg["w"]      # recency-weighted rushing yards per game

    agg = pg.groupby("pid").agg(
        name=("name", "last"), team=("posteam", "last"), w=("w", "sum"), games=("w", "size"),
        w_max=("w", "max"),
        s_tgt=("s_tgt", "sum"), s_car=("s_car", "sum"), s_rz=("s_rz", "sum"), s_gl=("s_gl", "sum"),
        s_air=("s_air", "sum"), snap=("w_snap", "sum"), trail_rush=("w_rush_yds", "sum"),
        tgt=("tgt", "sum"), rec=("rec", "sum"), rec_yds=("rec_yds", "sum"), car=("car", "sum"),
        rush_yds=("rush_yds", "sum"), tgt_man=("tgt_man", "sum"), yds_man=("yds_man", "sum"),
        tgt_zone=("tgt_zone", "sum"), yds_zone=("yds_zone", "sum"),
    )
    for c in ["s_tgt", "s_car", "s_rz", "s_gl", "s_air", "snap", "trail_rush"]:
        agg[c] = agg[c] / agg["w"]
    agg["neff"] = agg["w"] / agg["w_max"].clip(lower=1e-9)
    agg = agg[(agg["tgt"] + agg["car"]) > 0]
    agg = agg[agg["name"].notna()]

    # real positions + full names (full names are needed to match sportsbook player names)
    if info is not None:
        inf = info.set_index("gsis_id")
        agg["full_name"] = pd.Series(agg.index.map(inf["display_name"]), index=agg.index).fillna(agg["name"])
        agg["pos"] = pd.Series(agg.index.map(inf["position"]), index=agg.index).fillna("WR")
    else:
        agg["full_name"] = agg["name"]
        agg["pos"] = np.where(agg["car"] > agg["tgt"] * 1.5, "RB", "WR")
    agg["pos"] = agg["pos"].where(agg["pos"].isin(LEAGUE_POS), "WR")

    # shrink shares toward a position prior (neff = effective number of games)
    k = agg["pos"].map(lambda p: getattr(C, "SHARE_PRIOR_GAMES_BY_POS", {}).get(p, C.SHARE_PRIOR_GAMES))
    pt = agg["pos"].map(lambda p: C.SHARE_PRIORS.get(p, (0.08, 0.0))[0])
    pc = agg["pos"].map(lambda p: C.SHARE_PRIORS.get(p, (0.08, 0.0))[1])
    for c, prior in [("s_tgt", pt), ("s_rz", pt), ("s_car", pc), ("s_gl", pc)]:
        agg[c] = (agg[c] * agg["neff"] + prior * k) / (agg["neff"] + k)

    # efficiency shrunk toward position-group means
    for pos, grp in agg.groupby("pos"):
        cr = grp["rec"].sum() / max(grp["tgt"].sum(), 1)
        ypr = grp["rec_yds"].sum() / max(grp["rec"].sum(), 1)
        ypc = grp["rush_yds"].sum() / max(grp["car"].sum(), 1)
        idx = grp.index
        agg.loc[idx, "catch_rate"] = _shrink(grp["rec"] / grp["tgt"].replace(0, np.nan), grp["tgt"], cr, C.SHRINK_TARGETS).fillna(cr)
        agg.loc[idx, "ypr"] = _shrink(grp["rec_yds"] / grp["rec"].replace(0, np.nan), grp["rec"], ypr, C.SHRINK_TARGETS).fillna(ypr)
        agg.loc[idx, "ypc"] = _shrink(grp["rush_yds"] / grp["car"].replace(0, np.nan), grp["car"], ypc, C.SHRINK_CARRIES).fillna(ypc)

    # man vs zone yards per target, shrunk toward the player's own overall YPT
    ypt = agg["rec_yds"] / agg["tgt"].replace(0, np.nan)
    agg["ypt_man"] = _shrink(agg["yds_man"] / agg["tgt_man"].replace(0, np.nan), agg["tgt_man"], ypt, C.SHRINK_COVERAGE_TGTS).fillna(ypt)
    agg["ypt_zone"] = _shrink(agg["yds_zone"] / agg["tgt_zone"].replace(0, np.nan), agg["tgt_zone"], ypt, C.SHRINK_COVERAGE_TGTS).fillna(ypt)
    agg["ypt"] = ypt
    return agg.reset_index()


def active_usage(usage: pd.DataFrame, team: str, out_ids=(), out_names=(), snap_override=None) -> pd.DataFrame:
    """Players on `team`, minus anyone ruled out, with shares renormalised so volume from
    missing players is redistributed (proportionally) to everyone still active.
    snap_override: {"Malik Nabers": 0.6} scales a player's usage for an expected snap share
    (pitch counts, returning from injury)."""
    u = usage[usage["team"] == team].copy()
    for nm, pct in (snap_override or {}).items():
        hit = (u["full_name"].str.lower() == nm.lower()) | (u["name"].str.lower() == nm.lower())
        for c in ["s_tgt", "s_car", "s_rz", "s_gl"]:
            u.loc[hit, c] = u.loc[hit, c] * pct / u.loc[hit, "snap"].clip(lower=0.2)
    out_names = {n.lower() for n in out_names}
    missing = (u["pid"].isin(out_ids) | u["name"].str.lower().isin(out_names)
               | u["full_name"].str.lower().isin(out_names))
    for c in ["s_tgt", "s_car", "s_rz", "s_gl"]:
        lost = u.loc[missing, c].sum()
        u.loc[~missing, c] = u.loc[~missing, c] / max(1 - lost, 0.2)
    u = u[~missing]
    # keep only meaningful contributors; the remainder becomes an 'other' bucket in the sim
    u = u[(u["s_tgt"] > 0.015) | (u["s_car"] > 0.015)].copy()
    # Shares estimated in different contexts rarely sum to 1. Remove any excess from the LEAST
    # reliable estimates first (weight share / effective games), so a star with a long track
    # record isn't scaled down as hard as a one-game sample.
    rel = u["neff"].clip(lower=0.5)
    for c, cap in [("s_tgt", 0.97), ("s_car", 0.98), ("s_rz", 0.97), ("s_gl", 0.98)]:
        for _ in range(3):
            excess = u[c].sum() - cap
            if excess <= 1e-6:
                break
            wgt = u[c] / rel
            cut = (excess * wgt / wgt.sum()).clip(upper=0.7 * u[c])
            u[c] = u[c] - cut
    return u


def injury_outs(inj: pd.DataFrame, season: int, week: int, include_doubtful=True):
    st = {"Out"} | ({"Doubtful"} if include_doubtful else set())
    x = inj[(inj["season"] == season) & (inj["week"] == week) & inj["report_status"].isin(st)]
    return set(x["gsis_id"].dropna())


def primary_qb(pbp: pd.DataFrame, team: str, season: int, week: int, out_ids=(), out_names=()):
    """Most recent primary passer for `team` who isn't ruled out (gsis id)."""
    df = _cutoff(pbp, season, week)
    df = df[(df["posteam"] == team) & df["passer_player_id"].notna()]
    out_names = {n.lower() for n in out_names}
    df = df[~df["passer_player_id"].isin(out_ids) & ~df["passer_player_name"].str.lower().isin(out_names)]
    if df.empty:
        return None
    df = df.sort_values(["season", "week"])
    last_games = df["game_id"].drop_duplicates().tail(3)
    return df[df["game_id"].isin(last_games)]["passer_player_id"].value_counts().idxmax()
