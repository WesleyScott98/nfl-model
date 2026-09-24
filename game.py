"""Build and simulate a game in one call.

    from game import Model
    m = Model(season=2026)
    sim = m.simulate("NYG", "LA", week=2, out_names=["Puka Nacua"],
                     questionable={"Davante Adams": 0.85},        # your read on who plays
                     weather={"outdoor": True, "wind": 18, "temp": 40, "precip": False},  # or "auto"
                     def_adjust={"NYG": {"def_ypt": 1.06}})      # e.g. two starting CBs out
    sim.prob("Malik Nabers", "rec_yds", 49.5)   # conditional on Nabers playing
"""
import numpy as np
import pandas as pd

import config as C
import data
import defense as D
import features as F
from sim import merge_results, simulate_game


class Model:
    def __init__(self, season: int, prior_seasons: int = 2):
        self.season = season
        seasons = list(range(season - prior_seasons, season + 1))
        self.pbp = data.pbp_with_scheme(seasons)
        self.snaps = data.snaps(seasons)
        self.info = data.player_info()
        self.sched = data.schedules(season)
        self.inj = data.injuries(season)
        self.inj_all = pd.concat([data.injuries(s) for s in seasons], ignore_index=True)
        try:
            self.rost = data.roster_status(season)
        except Exception:
            self.rost = None
        self._wx_text = self.pbp.groupby("game_id")["weather"].first().to_dict()
        self._cache = {}

    def features(self, week):
        if week not in self._cache:
            tp = F.team_profiles(self.pbp, self.season, week)
            dp = D.defense_profiles(self.pbp, self.info, self.season, week)
            attrs = {**tp.attrs, **dp.attrs}
            tp = tp.join(dp)
            tp.attrs = attrs
            u = F.player_usage(self.pbp, self.snaps, self.season, week, self.info)
            sh = D.player_shell_splits(self.pbp, self.snaps, self.season, week)
            u = u.merge(sh, on="pid", how="left").fillna({"s2r": 1.0, "s1r": 1.0})
            self._cache[week] = (tp, u)
        return self._cache[week]

    def game_row(self, away, home, week):
        g = self.sched[(self.sched["week"] == week) & (self.sched["home_team"] == home) & (self.sched["away_team"] == away)]
        return g.iloc[0] if len(g) else None

    UNAVAILABLE = {"RES", "EXE", "SUS", "CUT", "RET", "DEV"}

    def unavailable(self, week, usage):
        """Players the model would otherwise project but who can't play: latest official roster
        status (before this week) is IR / exempt / suspended / released / retired / practice squad,
        or they're now on a different team. Game-day inactives (INA) are NOT carried forward."""
        if self.rost is None or self.rost.empty:
            return set()
        r = self.rost[self.rost["week"] <= max(week - 1, 1)]
        if r.empty:
            return set()
        latest = r.sort_values("week").groupby("gsis_id").last()
        bad = set(latest.index[latest["status"].isin(self.UNAVAILABLE)])
        team_now = latest["team"]
        u = usage.set_index("pid")["team"]
        moved = {pid for pid, t in u.items() if pid in team_now.index and team_now[pid] != t}
        # Not on ANY roster this season = free agent, retired or unsigned. These have no status to
        # be "bad", so they used to slip through and get projected off an older season's usage.
        last_wk = r["week"].max()
        current = r[r["week"] == last_wk]
        unsigned = set()
        if len(current) > 1200:                     # only trust a fully populated roster week
            on_roster = set(current["gsis_id"].dropna())
            unsigned = {pid for pid in u.index if pid not in on_roster}
        return bad | moved | unsigned

    def def_injury_adjust(self, week, out_ids):
        """How much worse each defense should be, from the share of its normal defensive snaps that
        are unavailable. Returns {team: {"def_ypt": mult, "def_ypc": mult, "def_cmp_rate": mult}}."""
        if not getattr(C, "AUTO_DEF_ADJUST", False):
            return {}
        try:
            ds = self.dsnaps
        except AttributeError:
            try:
                self.dsnaps = ds = data.defense_snaps(list(range(self.season - 1, self.season + 1)))
            except Exception:
                self.dsnaps = ds = None
        if ds is None or ds.empty:
            return {}
        prior = ds[(ds["season"] == self.season) & (ds["week"] < week)]
        if prior.empty:
            prior = ds[ds["season"] == self.season - 1]
        if prior.empty:
            return {}
        # a player's normal role = his average defensive snap share so far
        usual = prior.groupby(["team", "gsis_id", "grp"])["defense_pct"].mean().reset_index()
        missing = usual[usual["gsis_id"].isin(out_ids)]
        if missing.empty:
            return {}
        adj = {}
        for (team, grp), grp_rows in missing.groupby(["team", "grp"]):
            lost = float(grp_rows["defense_pct"].sum())      # in starter-equivalents
            d = adj.setdefault(team, {})
            if grp == "DB":
                d["def_ypt"] = d.get("def_ypt", 1.0) + min(C.DEF_INJURY_DB_YPT * lost, C.DEF_INJURY_CAP)
                d["def_cmp_rate"] = d.get("def_cmp_rate", 1.0) + min(C.DEF_INJURY_DB_CMP * lost, C.DEF_INJURY_CAP)
            else:
                d["def_ypc"] = d.get("def_ypc", 1.0) + min(C.DEF_INJURY_FRONT_YPC * lost, C.DEF_INJURY_CAP)
                d["def_ypt"] = d.get("def_ypt", 1.0) + min(C.DEF_INJURY_FRONT_YPT * lost, C.DEF_INJURY_CAP)
        return adj

    def weather_for(self, g, home):
        if g is None:
            return None
        obs = data.weather_from_schedule(g, self._wx_text.get(g["game_id"]))
        if obs is not None:
            return obs
        return data.weather_forecast(home, g["gameday"], g["gametime"])

    def _team_inputs(self, usage, team, week, outs, out_names, snap_override):
        qb = F.primary_qb(self.pbp, team, self.season, week, outs, out_names)
        u = F.active_usage(usage, team, outs, out_names, snap_override)
        u = u[(u["pos"] != "QB") | (u["pid"] == qb)]
        if qb is not None and qb not in set(u["pid"]):
            row = usage[usage["pid"] == qb]
            if len(row):
                u = pd.concat([u, row])
        return u, qb

    def simulate(self, away, home, week, spread_home=None, total=None, out_names=(),
                 use_injury_report=True, snap_override=None, questionable=None,
                 weather="auto", def_adjust=None, qb=None, n=C.N_SIMS, seed=7):
        """spread_home: home team's margin as favourite (Rams -6.5 -> 6.5); defaults to schedule.
        questionable: {name: P(plays)} — merged with the injury report (report default
        C.P_PLAY_QUESTIONABLE). weather: "auto" (observed/forecast), None, or a dict.
        def_adjust: {team: {"def_ypt"|"def_ypc"|"def_cmp_rate": multiplier}} for defensive injuries."""
        tp, usage = self.features(week)
        auto_def = {}
        if def_adjust is None or getattr(C, "AUTO_DEF_ADJUST", False):
            pass  # filled in below once we know who's out
        if def_adjust:
            tp = tp.copy()
            tp.attrs = self.features(week)[0].attrs
            for t, mults in def_adjust.items():
                for col, mult in mults.items():
                    tp.loc[t, col] *= mult
        g = self.game_row(away, home, week)
        if spread_home is None:
            spread_home = float(g["spread_line"]) if g is not None else 0.0
        if total is None:
            total = float(g["total_line"]) if g is not None else 44.0
        outs = F.injury_outs(self.inj, self.season, week) if use_injury_report else set()
        outs = outs | self.unavailable(week, usage)
        # resolve any full names you passed (overrides, live feed) to player ids, so they also
        # apply where the data only has abbreviated names like "C.Rush" — e.g. picking the starting QB
        if out_names:
            low = {n.lower() for n in out_names}
            outs = outs | set(usage.loc[usage["full_name"].str.lower().isin(low), "pid"])
        auto_def = self.def_injury_adjust(week, outs)
        if auto_def:
            tp = tp.copy(); tp.attrs = self.features(week)[0].attrs
            for t, mults in auto_def.items():
                if t not in (home, away) or t not in tp.index:
                    continue
                for col, mult in mults.items():
                    tp.loc[t, col] *= mult
                shown = ", ".join(f"{k.replace('def_','')} x{v:.2f}" for k, v in mults.items())
                print(f"[defense out] {t} weakened: {shown}")
        wx = self.weather_for(g, home) if weather == "auto" else weather

        # questionable players -> probability of playing
        q = {}
        if use_injury_report:
            rep = self.inj[(self.inj["season"] == self.season) & (self.inj["week"] == week)
                           & (self.inj["report_status"] == "Questionable")]
            mine = usage[usage["team"].isin([home, away]) & usage["pid"].isin(set(rep["gsis_id"]))]
            q = {nm: C.P_PLAY_QUESTIONABLE for nm in mine["full_name"]}
        q.update(questionable or {})
        outs_lower = {o.lower() for o in out_names}
        q = {k: v for k, v in q.items() if k.lower() not in outs_lower and v < 1}

        qb_named = {k: v for k, v in (qb or {}).items()}

        def run(extra_out, n_, seed_):
            ins = {}
            for t in (home, away):
                u, q = self._team_inputs(usage, t, week, outs, list(out_names) + extra_out, snap_override)
                # an announced starter (overrides.json "qb") beats whoever took the snaps last week
                named = qb_named.get(t)
                if named:
                    row = usage[usage["full_name"].str.lower() == named.lower()]
                    if len(row):
                        q = row["pid"].iloc[0]
                        if q not in set(u["pid"]):
                            u = pd.concat([u, row])
                        u = u[(u["pos"] != "QB") | (u["pid"] == q)]
                    else:
                        print(f"[qb override] {named} not found in usage data for {t}; leaving the model's pick")
                ins[t] = (u, q)
            return simulate_game(home, away, spread_home, total, tp, ins[home][0], ins[away][0],
                                 ins[home][1], ins[away][1], n=n_, seed=seed_, weather=wx)

        if not q:
            sim = run([], n, seed)
        else:
            rng = np.random.default_rng(seed)
            k = C.MIXTURE_CHUNKS
            parts = [run([nm for nm, p in q.items() if rng.random() > p], n // k, seed + i) for i in range(k)]
            sim = merge_results(parts)
        sim.weather, sim.questionable, sim.line = wx, q, (spread_home, total)
        return sim

    def matchup_report(self, away, home, week):
        """Defensive profile for both sides: funnel, man/zone, blitz, pressure."""
        tp, _ = self.features(week)
        cols = ["plays_pg", "neutral_pass_rate", "def_pass_epa", "def_rush_epa", "funnel_z",
                "pass_epa_adj", "rush_epa_adj", "def_ypt", "def_ypc", "man_rate", "two_high_rate",
                "blitz_rate", "pressure_rate", "tshare_WR", "tshare_TE", "tshare_RB", "yptr_WR", "yptr_TE", "yptr_RB"]
        rep = tp.loc[[away, home], [c for c in cols if c in tp.columns]].round(3)
        # league rank (1 = best defense) for the adjusted ratings
        for c in ["pass_epa_adj", "rush_epa_adj"]:
            if c in tp.columns:
                rep[c.replace("_epa_adj", "_def_rank")] = tp[c].rank().loc[[away, home]].astype(int)
        return rep.T
