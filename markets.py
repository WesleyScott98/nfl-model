"""Market-consistent game outcome distributions and game-line pricing.

The closing spread and total are the best available predictors of an NFL game (a pure stats model
lost to them badly in testing: 47.5-48.6% ATS, 2024-25). So instead of second-guessing them, we take
them as given and ask how outcomes are DISTRIBUTED around them. NFL margins pile up on key numbers
(3, 7, 10, 6, 4, 14...), so a smooth bell curve misprices moneylines, alternate spreads and teasers.
Here each simulated game is a real historical game with a similar spread and total, preserving
key numbers and the margin/total correlation.
"""
import numpy as np
import pandas as pd

import data

HIST_SEASONS = range(2003, 2026)
_HIST = None


def history(max_season=None):
    global _HIST
    if _HIST is None:
        frames = []
        for s in HIST_SEASONS:
            try:
                d = data.schedules(s)
                frames.append(d[["season", "week", "game_type", "spread_line", "total_line", "home_score", "away_score"]])
            except Exception:
                pass
        h = pd.concat(frames)
        h = h[h["home_score"].notna() & h["spread_line"].notna() & h["total_line"].notna()].copy()
        h["margin"] = h["home_score"] - h["away_score"]
        h["pts"] = h["home_score"] + h["away_score"]
        h["tot_resid"] = h["pts"] - h["total_line"]
        _HIST = h.reset_index(drop=True)
    return _HIST if max_season is None else _HIST[_HIST["season"] <= max_season]


def sample_outcomes(spread_home, total, n, rng, hist=None, bw=0.6, min_eff=300):
    """Draw (home margin, total points) from historical games with a similar spread.
    Bandwidth widens automatically for rare spreads so every draw rests on >= min_eff games."""
    h = history() if hist is None else hist
    s = h["spread_line"].values
    while True:
        w = np.exp(-0.5 * ((s - spread_home) / bw) ** 2)
        if w.sum() ** 2 / (w ** 2).sum() >= min_eff or bw > 6:
            break
        bw *= 1.4
    idx = rng.choice(len(h), size=n, p=w / w.sum())
    margin = h["margin"].values[idx].astype(float)
    # small integer correction so the mean matches this spread exactly without breaking key numbers
    # integer shift (randomised rounding) so the mean matches this spread without breaking key numbers
    shift = spread_home - np.average(s, weights=w)
    k, frac = int(abs(shift)), abs(shift) - int(abs(shift))
    margin = margin + np.sign(shift) * (k + (rng.random(n) < frac))
    tot = np.maximum(total + h["tot_resid"].values[idx], 0)
    return margin, tot


def normal_outcomes(spread_home, total, n, rng, sd_m=13.5, sd_t=10.0):
    return rng.normal(spread_home, sd_m, n), np.maximum(rng.normal(total, sd_t, n), 0)


# ---------------------------------------------------------------- game-line pricing
def game_market_probs(margin, tot, home, away):
    """Probabilities for the standard game markets from simulated (margin, total)."""
    def cover(team, line):   # team spread line, e.g. -3.5
        m = margin if team == home else -margin
        return float(np.mean(m + line > 0)), float(np.mean(m + line == 0))
    return {"ml": {home: float(np.mean(margin > 0)), away: float(np.mean(margin < 0)), "tie": float(np.mean(margin == 0))},
            "cover": cover,
            "over": lambda line: (float(np.mean(tot > line)), float(np.mean(tot == line)))}


def teaser_leg_prob(margin, team, home, line, points=6):
    m = margin if team == home else -margin
    return float(np.mean(m + line + points > 0)), float(np.mean(m + line + points == 0))
