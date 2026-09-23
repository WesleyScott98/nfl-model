"""Turn simulation output + FanDuel prices into ranked bets.

For every priced outcome:
  p_model   calibrated simulation probability
  p_market  de-vigged fair probability (two-way) or implied minus an assumed margin (one-sided)
  p_final   MODEL_WEIGHT * p_model + (1 - MODEL_WEIGHT) * p_market   <- shrink toward the market
  EV        p_final * decimal_odds - 1
  kelly     fractional Kelly stake as % of bankroll (capped)

Shrinking toward the market is deliberate: the book's number already contains information
the model doesn't (injury news, sharp money). A bet only shows up if the model disagrees
strongly enough that even after deferring 65% to the market it still clears break-even.
"""
import numpy as np
import pandas as pd

import calibrate as K
import config as C
from odds import american_to_decimal, decimal_to_american, devig_two_way, implied_prob


def price_props(sim, props: pd.DataFrame, cal=None) -> pd.DataFrame:
    cal = cal if cal is not None else K.load()
    rows = []
    two_way = props[~props["alternate"]].pivot_table(index=["player", "stat", "line"], columns="side",
                                                      values="price", aggfunc="first")
    for r in props.itertuples():
        try:
            raw = sim.prob(r.player, r.stat, r.line, side=r.side)
        except KeyError:
            continue                                  # player not active / not in sim
        p_over_cal = K.apply(raw if r.side == "over" else 1 - raw, r.stat, cal)
        p_model = p_over_cal if r.side == "over" else 1 - p_over_cal
        imp = implied_prob(r.price)
        key = (r.player, r.stat, r.line)
        if (not r.alternate) and key in two_way.index and two_way.loc[key].notna().all():
            o, u = devig_two_way(implied_prob(two_way.loc[key, "over"]), implied_prob(two_way.loc[key, "under"]))
            p_mkt = o if r.side == "over" else u
        else:
            margin = C.ONE_SIDED_MARGIN["anytime_td" if r.stat == "anytime_td" else "alt"]
            p_mkt = imp / (1 + margin)
        p_final = C.MODEL_WEIGHT * p_model + (1 - C.MODEL_WEIGHT) * p_mkt
        dec = american_to_decimal(r.price)
        ev = p_final * dec - 1
        b = dec - 1
        kelly = max(0.0, (p_final * b - (1 - p_final)) / b) * C.KELLY_FRACTION
        rows.append({"player": r.player, "stat": r.stat, "side": r.side, "line": r.line, "price": r.price,
                     "alt": r.alternate, "sim_mean": round(sim.mean(r.player, r.stat), 1),
                     "p_model": round(p_model, 3), "p_market": round(p_mkt, 3), "p_final": round(p_final, 3),
                     "breakeven": round(imp, 3), "edge": round(p_final - imp, 3), "ev": round(ev, 3),
                     "kelly_pct": round(min(kelly, C.KELLY_CAP) * 100, 2)})
    out = pd.DataFrame(rows)
    return out.sort_values("ev", ascending=False) if len(out) else out


def best_bets(priced: pd.DataFrame, min_edge=C.MIN_EDGE):
    ok = ~priced["stat"].isin(C.EXCLUDE_STATS)
    return priced[ok & (priced["edge"] >= min_edge) & (priced["ev"] > 0)]


def price_parlay(sim, legs, book_american=None, cal=None):
    """Joint probability of an SGP from the simulation (correlation included), compared with
    the product of the individual probabilities (what you'd get if legs were independent) and,
    optionally, FanDuel's quoted SGP price."""
    cal = cal if cal is not None else K.load()
    joint_raw = sim.joint_prob(legs)
    singles = []
    for leg in legs:
        v = sim.leg_valid(leg)
        p = sim.leg_mask(leg)[v].mean()
        if "player" in leg:
            p_over = p if leg.get("side", "over") == "over" else 1 - p
            pc = K.apply(p_over, leg["stat"], cal)
            p = pc if leg.get("side", "over") == "over" else 1 - pc
        singles.append(float(p))
    indep = float(np.prod(singles))
    raw_indep = float(np.prod([sim.leg_mask(l)[sim.leg_valid(l)].mean() for l in legs]))
    # carry the correlation uplift the sim found onto the calibrated singles
    corr_lift = joint_raw / raw_indep if raw_indep > 0 else 1.0
    joint = min(indep * corr_lift, min(singles))
    out = {"legs": len(legs), "p_joint": round(joint, 4), "p_if_independent": round(indep, 4),
           "correlation_lift": round(corr_lift, 2), "fair_american": decimal_to_american(1 / max(joint, 1e-6))}
    if book_american is not None:
        out["book_american"] = book_american
        out["ev"] = round(joint * american_to_decimal(book_american) - 1, 3)
    return out
