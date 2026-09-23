"""Monte Carlo game simulator.

Why simulate instead of projecting each prop separately?  Because same-game parlay legs are
correlated.  Every simulated game draws ONE score margin, ONE set of play counts and ONE
target/carry distribution, and every player stat is derived from that shared draw.  So
"QB 250+ yds AND WR1 80+ yds" is priced from how often both happen in the same simulated game
— which is exactly what FanDuel is (roughly) doing when it prices an SGP.

Pipeline per simulated game, per team:
  margin, total  ~ Normal(spread, MARGIN_SD), Normal(total, TOTAL_SD)
  plays          ~ Normal(avg(own pace, opponent pace allowed), PLAYS_SD)
  pass rate      = neutral pass rate + funnel(opponent defense) + game script(margin)
  targets/carries ~ Multinomial(volume, usage shares of active players)
  receptions     ~ Binomial(targets, catch rate x opponent completion factor)
  rec yards      ~ Gamma(sum of per-catch gammas) x coverage-scheme and defense factors
  rush yards     ~ carries x YPC x run-defense factor + heavy-tailed noise
  TDs            ~ Poisson(sim points / POINTS_PER_TD), split pass/rush, allocated by
                   red-zone target share / inside-10 carry share
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C
import markets as M

OTHER_REC = {"catch_rate": 0.64, "ypr": 10.3}
OTHER_RUSH_YPC = 4.2


@dataclass
class SimResult:
    home: str
    away: str
    n: int
    points: dict = field(default_factory=dict)            # team -> array
    players: dict = field(default_factory=dict)           # full_name -> {stat: array}
    meta: dict = field(default_factory=dict)              # full_name -> {team, pos}

    def stat(self, player, stat):
        key = _match(player, self.players)
        if key is None:
            raise KeyError(f"{player} not in simulation (active players: {sorted(self.players)[:8]}...)")
        return self.players[key][stat]

    def prob(self, player, stat, line, side="over"):
        """P(stat > line | player plays). Half-point lines; alt ladder 'k+' -> line=k-0.5."""
        x = self.stat(player, stat)
        x = x[~np.isnan(x)]
        return float((x > line).mean() if side == "over" else (x < line).mean())

    def mean(self, player, stat):
        return float(np.nanmean(self.stat(player, stat)))

    def p_active(self, player):
        return float(1 - np.isnan(self.stat(player, "targets")).mean())

    def leg_mask(self, leg):
        """leg = dict(player=, stat=, line=, side=) or dict(team=, market='ml'|'spread'|'total', line=, side=)."""
        if "player" in leg:
            x = self.stat(leg["player"], leg["stat"])
            return x > leg["line"] if leg.get("side", "over") == "over" else x < leg["line"]
        h, a = self.points[self.home], self.points[self.away]
        if leg["market"] == "team_total":
            x = h if leg["team"] == self.home else a
            return x > leg["line"] if leg.get("side", "over") == "over" else x < leg["line"]
        if leg["market"] == "total":
            return (h + a) > leg["line"] if leg.get("side", "over") == "over" else (h + a) < leg["line"]
        t = leg["team"]
        own, opp = (h, a) if t == self.home else (a, h)
        if leg["market"] == "ml":
            return own > opp
        return (own - opp + leg["line"]) > 0          # spread: line is the team's number, e.g. -6.5

    def leg_valid(self, leg):
        return ~np.isnan(self.stat(leg["player"], leg["stat"])) if "player" in leg else np.ones(self.n, bool)

    def joint_prob(self, legs):
        """Joint probability, conditional on every player in the legs being active."""
        m = np.ones(self.n, dtype=bool)
        v = np.ones(self.n, dtype=bool)
        for leg in legs:
            m &= self.leg_mask(leg)
            v &= self.leg_valid(leg)
        return float(m[v].mean()) if v.any() else float("nan")

    def summary(self, min_share=0.0):
        rows = []
        for name, s in self.players.items():
            rows.append({"player": name, **self.meta[name],
                         **{f"{k}_mean": float(np.nanmean(v)) for k, v in s.items()},
                         "p_anytime_td": float((s["anytime_td"][~np.isnan(s["anytime_td"])] > 0.5).mean()),
                         "p_active": float(1 - np.isnan(s["targets"]).mean())})
        return pd.DataFrame(rows).sort_values(["team", "rec_yds_mean"], ascending=[True, False])


def _match(name, players):
    if name in players:
        return name
    low = name.lower().replace(".", "").replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", "")
    for k in players:
        kk = k.lower().replace(".", "").replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", "")
        if kk == low:
            return k
    return None


def _t_noise(rng, size, df):
    return rng.standard_t(df, size) / np.sqrt(df / (df - 2))


def weather_effects(weather, lg_ypt, lg_cmp):
    """-> (pass-rate shift, catch-rate shift, yards-per-reception multiplier)."""
    if not weather or not weather.get("outdoor", False):
        return 0.0, 0.0, 1.0
    wind = float(weather.get("wind") or 0)
    temp = weather.get("temp")
    precip = 1.0 if weather.get("precip") else 0.0
    cold = 1.0 if (temp is not None and temp == temp and temp < 32) else 0.0
    d_pr = C.WX_COLD_PASS * cold + C.WX_PRECIP_PASS * precip
    d_cmp = C.WX_PRECIP_CMP * precip
    ypt_mult = 1 + (C.WX_WIND_YPT * wind + C.WX_PRECIP_YPT * precip) / lg_ypt
    cmp_mult = 1 + d_cmp / lg_cmp
    return d_pr, d_cmp, ypt_mult / cmp_mult


def simulate_game(home, away, spread_home, total, tp, usage_home, usage_away, qb_home, qb_away,
                  n=C.N_SIMS, seed=7, weather=None, hist=None):
    """spread_home: points the HOME team is favoured by (Rams -6.5 at home -> 6.5)."""
    rng = np.random.default_rng(seed)
    lg = tp.attrs["league"]
    if C.MARGIN_MODEL == "empirical":
        margin, tot = M.sample_outcomes(spread_home, total, n, rng, hist=hist)
        tot = np.maximum(tot, np.abs(margin))
    else:
        margin = rng.normal(spread_home, C.MARGIN_SD, n)
        tot = np.maximum(rng.normal(total, C.TOTAL_SD, n), 6)
    pts = {home: np.maximum((tot + margin) / 2, 0), away: np.maximum((tot - margin) / 2, 0)}
    exp_pts = {home: (total + spread_home) / 2, away: (total - spread_home) / 2}
    res = SimResult(home, away, n)
    wx_pr, wx_cmp, wx_ypr = weather_effects(weather, lg["ypt"], lg["cmp"])

    for team, opp, usage, qb in [(home, away, usage_home, qb_home), (away, home, usage_away, qb_away)]:
        T, O = tp.loc[team], tp.loc[opp]
        own_margin = pts[team] - pts[opp]
        la = tp.attrs.get("league_adj", {})
        d_cmp, d_ypt, d_ypc, d_fun = O.def_cmp_rate / lg["cmp"], O.def_ypt / lg["ypt"], O.def_ypc / lg["ypc"], O.funnel_z
        if C.USE_OPP_ADJ and "ypt_adj" in O.index:
            a = C.DEF_STRENGTH
            d_cmp = (1 - a) * d_cmp + a * O.cmp_adj / la["cmp_adj"]
            d_ypt = (1 - a) * d_ypt + a * O.ypt_adj / la["ypt_adj"]
            d_ypc = (1 - a) * d_ypc + a * O.ypc_adj / la["ypc_adj"]
            d_fun = (1 - a) * d_fun + a * O.funnel_adj_z

        plays = np.clip(np.round(rng.normal((T.plays_pg + O.plays_allowed_pg) / 2, C.PLAYS_SD, n)), 40, 90).astype(int)
        pr = (T.neutral_pass_rate + C.FUNNEL_BETA * d_fun
              + C.SCRIPT_PASS_BETA * (-own_margin) / 14.0 + wx_pr)
        pr = np.clip(pr, 0.30, 0.80)
        pass_plays = rng.binomial(plays, pr)
        sacks = rng.binomial(pass_plays, T.sack_rate)
        att = pass_plays - sacks
        rushes = plays - pass_plays
        eff = np.clip(((pts[team] + 7) / (exp_pts[team] + 7)) ** C.EFFICIENCY_POINTS_ELASTICITY, 0.6, 1.6)

        u = usage.reset_index(drop=True)
        names = list(u["full_name"])
        k = len(u)

        # ---------------- receiving ----------------
        pos_g = u["pos"].map({"WR": "WR", "TE": "TE", "RB": "RB"})
        tmult = np.ones(k)
        if C.USE_DVP and "tshare_WR" in O.index:
            tmult *= np.array([O.get(f"tshare_{p}", 1.0) if isinstance(p, str) else 1.0 for p in pos_g])
        if C.USE_SHELL and "s2r" in u.columns and "two_high_rate" in O.index:
            h, hl = O.two_high_rate, la.get("two_high", 0.45)
            tmult *= ((h * u["s2r"] + (1 - h) * u["s1r"]) / (hl * u["s2r"] + (1 - hl) * u["s1r"])).values
        tgt_p = np.append(u["s_tgt"].values * tmult ** C.DEF_STRENGTH, 0.0)
        tgt_p[-1] = max(1 - tgt_p[:-1].sum(), 0.03)
        tgt_p = tgt_p / tgt_p.sum()
        tgt_p_sim = rng.dirichlet(tgt_p * C.TARGET_SHARE_CONC, n)          # per-game share volatility
        targets = rng.multinomial(att, tgt_p_sim)                          # (n, k+1)
        cmp_f = d_cmp
        ypc_allowed_ratio = d_ypt / cmp_f                 # yards per completion allowed
        cr = np.append(u["catch_rate"].values, OTHER_REC["catch_rate"]) * cmp_f + wx_cmp
        cov = (O.man_rate * u["ypt_man"] + (1 - O.man_rate) * u["ypt_zone"]) / u["ypt"].replace(0, np.nan)
        cov = np.append(cov.fillna(1.0).clip(0.8, 1.25).values, 1.0)
        pmult = np.ones(k)
        if C.USE_DVP and "yptr_WR" in O.index:
            raw_all = O.def_ypt / lg["ypt"]
            pmult = np.clip(np.array([O.get(f"yptr_{p}", raw_all) / raw_all if isinstance(p, str) else 1.0
                                      for p in pos_g]), 0.8, 1.25)
        ypr = np.append(u["ypr"].values * pmult ** C.DEF_STRENGTH, OTHER_REC["ypr"]) * cov * ypc_allowed_ratio * wx_ypr
        cr_sim = np.clip(cr + rng.normal(0, C.CATCH_RATE_SD, (n, k + 1)), 0.25, 0.95)
        rec = rng.binomial(targets, cr_sim)
        shape = C.YPR_GAMMA_SHAPE
        rec_yds = rng.gamma(np.maximum(rec * shape, 1e-9), (ypr * eff[:, None]) / shape)
        rec_yds = np.where(rec > 0, rec_yds, 0.0)

        # ---------------- rushing ----------------
        car_p = np.append(u["s_car"].values, 0.0)
        car_p[-1] = max(1 - car_p[:-1].sum(), 0.02)
        car_p = car_p / car_p.sum()
        carries = rng.multinomial(rushes, rng.dirichlet(car_p * C.CARRY_SHARE_CONC, n))
        run_f = d_ypc
        ypc = np.append(u["ypc"].values, OTHER_RUSH_YPC) * run_f
        rush_yds = (carries * ypc * np.sqrt(eff)[:, None]
                    + np.sqrt(carries) * C.RUSH_SD_PER_CARRY * _t_noise(rng, carries.shape, C.RUSH_T_DF))
        rush_yds = np.where(carries > 0, rush_yds, 0.0)

        # ---------------- touchdowns ----------------
        tds = rng.poisson(pts[team] / C.POINTS_PER_TD)
        pass_td = rng.binomial(tds, np.clip(T.pass_td_share + 0.02 * d_fun, 0.35, 0.8))
        rush_td = tds - pass_td
        rtd_p = np.append(0.6 * u["s_rz"].values + 0.4 * u["s_tgt"].values, 0.0)
        rtd_p[-1] = max(1 - rtd_p[:-1].sum(), 0.04)
        rtd_p /= rtd_p.sum()
        gtd_p = np.append(0.65 * u["s_gl"].values + 0.35 * u["s_car"].values, 0.0)
        gtd_p[-1] = max(1 - gtd_p[:-1].sum(), 0.03)
        gtd_p /= gtd_p.sum()
        rec_td = rng.multinomial(pass_td, rtd_p)
        rsh_td = rng.multinomial(rush_td, gtd_p)
        # a receiving TD implies at least one catch
        fix = (rec_td > 0) & (rec == 0)
        rec = np.where(fix, 1, rec)
        rec_yds = np.where(fix, np.maximum(rng.gamma(shape, ypr / shape, rec.shape), 1), rec_yds)

        res.points[team] = pts[team]
        pass_yds_team = rec_yds.sum(axis=1)
        for i, nm in enumerate(names):
            res.players[nm] = {
                "targets": targets[:, i], "receptions": rec[:, i], "rec_yds": rec_yds[:, i],
                "carries": carries[:, i], "rush_yds": rush_yds[:, i],
                "rush_rec_yds": rec_yds[:, i] + rush_yds[:, i],
                "anytime_td": rec_td[:, i] + rsh_td[:, i],
                "pass_yds": np.zeros(n), "pass_tds": np.zeros(n), "pass_att": np.zeros(n),
                "completions": np.zeros(n),
            }
            res.meta[nm] = {"team": team, "pos": u.loc[i, "pos"], "pid": u.loc[i, "pid"]}
        qb_row = u[u["pid"] == qb]
        qb_name = qb_row["full_name"]
        if len(qb_name):
            q = res.players[qb_name.iloc[0]]
            # blend the QB's rushing toward his own recent per-game average (see config)
            trail = float(qb_row["trail_rush"].iloc[0]) if "trail_rush" in qb_row.columns else np.nan
            mu = float(np.mean(q["rush_yds"]))
            if trail == trail and mu > 1:
                w = C.QB_RUSH_TRAIL_WEIGHT
                scale = ((1 - w) * mu + w * max(trail, 0)) / mu
                q["rush_yds"] = q["rush_yds"] * scale
                q["carries"] = np.round(q["carries"] * scale)
            # starter plays the whole game unless an early exit is drawn
            exit_ = rng.random(n) < C.P_QB_EXIT
            frac = np.where(exit_, rng.uniform(*C.QB_EXIT_FRAC, n), 1.0)
            q["pass_yds"] = pass_yds_team * frac
            q["pass_tds"] = rng.binomial(pass_td, frac).astype(float)
            q["pass_att"] = np.round(att * frac)
            q["completions"] = np.round(rec.sum(axis=1) * frac)
            q["rush_yds"] = q["rush_yds"] * frac
            q["carries"] = np.round(q["carries"] * frac)
    return res


def merge_results(parts):
    """Stack scenario chunks (e.g. questionable player in / out) into one result; players
    missing from a chunk get NaN there, so their props are conditional on playing."""
    first = parts[0]
    res = SimResult(first.home, first.away, sum(p.n for p in parts))
    res.points = {t: np.concatenate([p.points[t] for p in parts]) for t in first.points}
    names = {nm for p in parts for nm in p.players}
    stats = list(next(iter(first.players.values())).keys())
    for nm in names:
        res.players[nm] = {st: np.concatenate([p.players[nm][st].astype(float) if nm in p.players
                                               else np.full(p.n, np.nan) for p in parts]) for st in stats}
        res.meta[nm] = next(p.meta[nm] for p in parts if nm in p.meta)
    return res
