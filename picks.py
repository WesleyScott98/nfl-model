"""Slate-wide pick engine: every market type, ranked, plus SGPs, cross-game parlays and teasers.

    from game import Model; import picks
    m = Model(2026)
    out = picks.run_slate(m, week=3)                         # FanDuel via The Odds API
    out = picks.run_slate(m, week=3, manual={("NYG","LA"): {"game": odds.manual_game(...), "props": odds.manual_lines([...])}})
    picks.show(out)

What each category can and can't do (see README for the tests behind this):
  * Spreads / totals (main lines): the closing line beats any stats model we tested, so the
    engine does NOT invent side/total opinions. It flags a main line only when injury news you
    feed it (out/questionable) moves the model away from the market.
  * Moneylines, alternate spreads/totals, team totals: priced from real historical margins around
    the spread (key numbers 3/7/10), so the engine can catch prices that are inconsistent with the
    main line.
  * Teasers: 6-pt legs on +1.5..+2.5 underdogs (76.5% in 2021-25) when the price clears breakeven.
  * Player props / anytime TD / alt ladders: calibrated simulation (validated 2024 + 2025).
  * show(out, mode="best") ranks the top plays in every category whether or not they clear the
    edge bar, tiered VALUE / lean / thin. mode="value" shows only bets that clear it.
  * SGPs: correlation-aware joint probabilities; you must compare the fair price to FanDuel's quote.
  * Cross-game parlays: only of legs that are +EV on their own.
"""
import itertools

import numpy as np
import pandas as pd

import calibrate as K
import config as C
import edge as E
import odds as O
from odds import american_to_decimal, decimal_to_american, devig_two_way, implied_prob
from teams import TEAM_NAMES

TEAM_ABBR = TEAM_NAMES
SGP_MIN_VALUE = 1.10     # joint model prob must beat the market-implied joint prob by 10% (SGP holds are large)


# ------------------------------------------------------------------ game markets
def _game_prob(sim, row):
    """(p_win, p_push) for one game-market outcome."""
    h, a = sim.points[sim.home], sim.points[sim.away]
    mk = row["market"]
    if mk in ("total", "alt_total"):
        t = h + a
        return (float(np.mean(t > row["line"])) if row["side"] == "over" else float(np.mean(t < row["line"])),
                float(np.mean(t == row["line"])))
    own, opp = (h, a) if row["team"] == sim.home else (a, h)
    if mk == "team_total":
        return (float(np.mean(own > row["line"])) if row["side"] == "over" else float(np.mean(own < row["line"]))), 0.0
    if mk == "ml":
        return float(np.mean(own > opp)), float(np.mean(own == opp))
    d = own - opp + row["line"]
    return float(np.mean(d > 0)), float(np.mean(d == 0))


def _pair_key(r):
    if r["market"] in ("ml",):
        return ("ml",)
    if r["market"] in ("spread", "alt_spread"):
        return (r["market"], abs(r["line"]))
    if r["market"] in ("total", "alt_total"):
        return (r["market"], r["line"])
    return (r["market"], r["team"], r["line"])


def price_game_markets(sim, gm: pd.DataFrame, game: str):
    if gm is None or gm.empty:
        return pd.DataFrame()
    gm = gm.dropna(subset=["price"]).copy()
    gm["key"] = gm.apply(_pair_key, axis=1)
    rows = []
    for _, r in gm.iterrows():
        pw, pp = _game_prob(sim, r)
        mates = gm[(gm["key"] == r["key"]) & gm.index.isin(gm.index.difference([r.name]))]
        imp = implied_prob(r["price"])
        suspicious = False
        if len(mates) == 1:
            imp2 = implied_prob(mates["price"].iloc[0])
            if imp + imp2 < 1.0:                 # impossible single-book market (negative vig): stale/typo line
                suspicious, p_mkt = True, pw
            else:
                fair, _ = devig_two_way(imp, imp2)
                p_mkt = fair * (1 - pp)
        else:
            p_mkt = imp / (1 + C.ONE_SIDED_MARGIN["alt"])
        main = r["market"] in ("spread", "total")
        # main spread/total: the model is anchored on them, so it only adds info via injury news
        p_final = C.MODEL_WEIGHT * pw + (1 - C.MODEL_WEIGHT) * p_mkt if not main else 0.5 * (pw + p_mkt)
        dec = american_to_decimal(r["price"])
        ev = p_final * (dec - 1) - (1 - p_final - pp)
        b = dec - 1
        kelly = max(0.0, (p_final * b - (1 - p_final - pp)) / b) * C.KELLY_FRACTION
        mk_ = r["market"]
        if mk_ == "ml":
            sel = f"{r['team']} ML"
        elif mk_ in ("spread", "alt_spread"):
            sel = f"{r['team']} {r['line']:+g}" + (" (alt)" if mk_ == "alt_spread" else "")
        elif mk_ in ("total", "alt_total"):
            sel = f"{r['side'].title()} {r['line']:g}" + (" (alt)" if mk_ == "alt_total" else "")
        else:
            sel = f"{r['team']} team total {r['side']} {r['line']:g}"
        leg = ({"team": r["team"], "market": "ml"} if r["market"] == "ml" else
               {"team": r["team"], "market": "spread", "line": r["line"]} if "spread" in r["market"] else
               {"market": "total", "line": r["line"], "side": r["side"]} if "total" in r["market"] and r["market"] != "team_total" else
               {"team": r["team"], "market": "team_total", "line": r["line"], "side": r["side"]})
        rows.append({"game": game, "category": "Moneyline/Spread" if r["market"] in ("ml", "spread", "alt_spread") else "Totals",
                     "selection": sel, "price": r["price"], "p_model": round(pw, 3), "p_market": round(p_mkt, 3),
                     "p_final": round(p_final, 3), "p_push": round(pp, 3), "breakeven": round(imp, 3),
                     "edge": round(p_final - imp, 3), "ev": round(ev, 3), "kelly_pct": round(min(kelly, C.KELLY_CAP) * 100, 2),
                     "leg": leg, "check": "market looks inconsistent - verify price" if suspicious else ""})
    return pd.DataFrame(rows)


def price_player_props(sim, props, game):
    if props is None or props.empty:
        return pd.DataFrame()
    p = E.price_props(sim, props)
    if p.empty:
        return p
    p = p[~p["stat"].isin(C.EXCLUDE_STATS)].copy()
    p["game"] = game
    p["category"] = np.where(p["stat"] == "anytime_td", "Anytime TD", "Player props")
    p["selection"] = p.apply(lambda r: f"{r['player']} anytime TD" if r["stat"] == "anytime_td" else
                             f"{r['player']} {r['side']} {r['line']:g} {r['stat']}" + (" (alt)" if r["alt"] else ""), axis=1)
    p["leg"] = p.apply(lambda r: {"player": r["player"], "stat": r["stat"], "line": r["line"], "side": r["side"]}, axis=1)
    p["p_push"] = 0.0
    return p[["game", "category", "selection", "price", "p_model", "p_market", "p_final", "p_push", "breakeven",
              "edge", "ev", "kelly_pct", "leg"]]


# ------------------------------------------------------------------ teasers
def teaser_legs(sim, gm, game, teaser_price=-120, points=6):
    be = (1 / american_to_decimal(teaser_price)) ** 0.5
    rows = []
    if gm is None or gm.empty:
        return pd.DataFrame()
    for _, r in gm[gm["market"] == "spread"].iterrows():
        h, a = sim.points[sim.home], sim.points[sim.away]
        own, opp = (h, a) if r["team"] == sim.home else (a, h)
        p = float(np.mean(own - opp + r["line"] + points > 0))
        in_range = C.TEASER_DOG_RANGE[0] <= r["line"] <= C.TEASER_DOG_RANGE[1]
        rows.append({"game": game, "leg": f"{r['team']} {r['line']:+g} -> {r['line'] + points:+g}", "p_leg": round(p, 3),
                     "breakeven_leg": round(be, 3), "historically_validated": in_range})
    return pd.DataFrame(rows)


def teaser_combos(legs, teaser_price=-120, top=5):
    ok = legs[(legs["historically_validated"]) & (legs["p_leg"] > legs["breakeven_leg"])]
    out = []
    dec = american_to_decimal(teaser_price)
    for a, b in itertools.combinations(ok.itertuples(), 2):
        if a.game == b.game:
            continue
        p = a.p_leg * b.p_leg
        out.append({"teaser (6 pt, 2 team)": f"{a.leg}  +  {b.leg}", "price": teaser_price, "p_win": round(p, 3),
                    "ev": round(p * dec - 1, 3)})
    return pd.DataFrame(out).sort_values("ev", ascending=False).head(top) if out else pd.DataFrame()


# ------------------------------------------------------------------ SGPs and parlays
def build_sgps(sim, priced_game, top_legs=10, max_legs=3, top=5):
    cand = priced_game[(priced_game["p_model"] >= priced_game["p_market"]) & priced_game["p_final"].between(0.2, 0.92)]
    cand = cand.assign(tilt=cand["p_model"] - cand["p_market"]).sort_values("tilt", ascending=False).head(top_legs)
    cal = K.load()
    out = []
    for k in range(2, max_legs + 1):
        for combo in itertools.combinations(cand.itertuples(), k):
            legs = [c.leg for c in combo]
            keys = [(l.get("player"), l.get("stat")) if "player" in l else ("game", l["market"]) for l in legs]
            if len(set(keys)) < len(keys):
                continue
            r = E.price_parlay(sim, legs, cal=cal)
            lift = r["correlation_lift"]
            p_final = float(np.prod([c.p_final for c in combo])) * lift
            p_mkt = float(np.prod([c.p_market for c in combo])) * lift
            if p_final <= 0 or p_mkt <= 0:
                continue
            out.append({"game": combo[0].game, "legs": " | ".join(c.selection for c in combo), "n_legs": k,
                        "p_joint": round(min(p_final, 0.99), 4), "fair_odds": decimal_to_american(1 / min(p_final, 0.99)),
                        "value_vs_market": round(p_final / p_mkt, 3), "correlation_lift": lift,
                        "bet_if_fanduel_pays_at_least": decimal_to_american(1.05 / min(p_final, 0.99))})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df[df["value_vs_market"] >= SGP_MIN_VALUE].sort_values("value_vs_market", ascending=False).head(top)


def build_parlays(singles, max_legs=3, top=5):
    good = singles[(singles["edge"] >= C.MIN_EDGE) & (singles["ev"] > 0)].sort_values("ev", ascending=False)
    good = good.drop_duplicates("game").head(8)          # one leg per game -> independent legs
    out = []
    for k in range(2, max_legs + 1):
        for combo in itertools.combinations(good.itertuples(), k):
            dec = float(np.prod([american_to_decimal(c.price) for c in combo]))
            p = float(np.prod([c.p_final for c in combo]))
            out.append({"legs": " | ".join(f"{c.selection} ({c.price:+d})" for c in combo), "n_legs": k,
                        "price": decimal_to_american(dec), "p_win": round(p, 3), "ev": round(p * dec - 1, 3)})
    return pd.DataFrame(out).sort_values("ev", ascending=False).head(top) if out else pd.DataFrame()


# ------------------------------------------------------------------ driver
def run_slate(model, week, games=None, manual=None, out_names=(), questionable=None, snap_override=None,
              weather="auto", teaser_price=-120, n=C.N_SIMS):
    """games: list of (away, home); default = every game that week. manual: {(away, home):
    {"game": odds.manual_game(...), "props": odds.manual_lines(...)}}; otherwise The Odds API."""
    sched = model.sched[model.sched["week"] == week]
    games = games or list(zip(sched["away_team"], sched["home_team"]))
    api_events = None
    all_singles, sgps, tlegs = [], [], []
    for away, home in games:
        game = f"{away}@{home}"
        if manual is not None:
            gm = manual.get((away, home), {}).get("game")
            props = manual.get((away, home), {}).get("props")
        else:
            if api_events is None:
                api_events = O.game_lines()
                api_events["h"], api_events["a"] = api_events["home"].map(TEAM_ABBR), api_events["away"].map(TEAM_ABBR)
            ev = api_events[(api_events["h"] == home) & (api_events["a"] == away)]
            if ev.empty:
                continue
            eid = ev["event_id"].iloc[0]
            gm, props = O.game_markets(eid, TEAM_ABBR), O.player_props(eid)
        # anchor the simulation on FanDuel's own main spread/total when available
        spread_home = total = None
        if gm is not None and not gm.empty:
            sp = gm[(gm["market"] == "spread") & (gm["team"] == home)]
            tt = gm[gm["market"] == "total"]
            spread_home = -float(sp["line"].iloc[0]) if len(sp) else None
            total = float(tt["line"].iloc[0]) if len(tt) else None
        sim = model.simulate(away, home, week, spread_home, total, out_names=out_names, questionable=questionable,
                             snap_override=snap_override, weather=weather, n=n)
        g_priced = pd.concat([price_game_markets(sim, gm, game), price_player_props(sim, props, game)], ignore_index=True)
        all_singles.append(g_priced)
        if len(g_priced):
            sgps.append(build_sgps(sim, g_priced))
        tlegs.append(teaser_legs(sim, gm, game, teaser_price))
    singles = pd.concat(all_singles, ignore_index=True) if all_singles else pd.DataFrame()
    tl = pd.concat(tlegs, ignore_index=True) if tlegs else pd.DataFrame()
    best = singles[(singles["edge"] >= C.MIN_EDGE) & (singles["ev"] > 0)
                   & (singles["price"] >= C.MIN_PRICE_AMERICAN)].sort_values("ev", ascending=False) if len(singles) else singles
    return {"best_singles": best, "all_priced": singles,
            "sgps": pd.concat(sgps, ignore_index=True) if sgps else pd.DataFrame(),
            "parlays": build_parlays(singles) if len(singles) else pd.DataFrame(),
            "teaser_legs": tl, "teasers": teaser_combos(tl, teaser_price) if len(tl) else pd.DataFrame()}


def log_week(out, season, week):
    """Save this week's full priced board for later model-vs-market analysis."""
    import tracking as T
    return T.log_board(out["all_priced"], season, week)


def tier(row):
    """Value = beats break-even by MIN_EDGE after deferring to the market; Lean = positive but
    thin; Thin = the model rates it below the price (best of a bad board)."""
    if row["edge"] >= C.MIN_EDGE and row["ev"] > 0:
        return "VALUE"
    if row["ev"] > 0:
        return "lean"
    return "thin"


def best_available(out, per_cat=5, min_price=None):
    """Top-rated plays per category regardless of the edge threshold, tiered.
    Singles priced shorter than MIN_PRICE_AMERICAN are dropped — too much risk per unit won."""
    s = out["all_priced"].copy()
    if s.empty:
        return s
    floor = C.MIN_PRICE_AMERICAN if min_price is None else min_price
    s = s[s["price"] >= floor]
    s["tier"] = s.apply(tier, axis=1)
    return s.sort_values("ev", ascending=False).groupby("category", observed=True).head(per_cat)


def best_bets(out, top=8, min_prob=0.55, min_price=C.MIN_PRICE_AMERICAN):
    """Two ranked views of the same board:
      value      — best price vs the model, highest EV first (what profits long-run)
      confidence — most likely to WIN at a sane price, highest probability first
    Both carry the tier label so you can see what you're taking."""
    s = out["all_priced"].copy()
    if s.empty:
        return {"value": s, "confidence": s}
    s["tier"] = s.apply(tier, axis=1)
    value = s.sort_values("ev", ascending=False).head(top)
    conf = s[(s["p_final"] >= min_prob) & (s["price"] >= min_price)].sort_values("p_final", ascending=False).head(top)
    return {"value": value, "confidence": conf}


def show(out, per_cat=5, mode="best"):
    """mode="best": top plays in every category, tiered (VALUE / lean / thin).
       mode="value": only bets that clear the edge threshold."""
    cols = ["game", "selection", "price", "p_final", "breakeven", "edge", "ev", "kelly_pct", "tier"]
    if mode == "best":
        bb = best_bets(out)
        print("=== BEST VALUE (highest expected value — these are what profit long run) ===")
        print(bb["value"][cols].to_string(index=False) if len(bb["value"]) else "  nothing priced")
        print("\n=== MOST LIKELY (highest win probability at a reasonable price) ===")
        print(bb["confidence"][cols].to_string(index=False) if len(bb["confidence"]) else "  nothing at 55%+ and better than -300")
    b = best_available(out, per_cat) if mode == "best" else out["best_singles"].assign(tier="VALUE")
    for cat in ["Moneyline/Spread", "Totals", "Player props", "Anytime TD"]:
        x = b[b["category"] == cat] if len(b) else b
        print(f"\n=== {cat} ===")
        if len(x):
            print(x[cols].head(per_cat).to_string(index=False))
        else:
            print("  nothing priced in this category")
    print("\n=== Same-game parlays (compare fair odds to FanDuel's SGP quote) ===")
    print(out["sgps"].drop(columns=["correlation_lift"]).to_string(index=False) if len(out["sgps"]) else "  none clear the value bar")
    print("\n=== Cross-game parlays (legs that are +EV on their own) ===")
    print(out["parlays"].to_string(index=False) if len(out["parlays"]) else "  fewer than two +EV legs on different games")
    print("\n=== Teasers (6 pt, underdog legs +1.5..+2.5 only) ===")
    if len(out["teasers"]):
        print(out["teasers"].to_string(index=False))
    else:
        tl = out["teaser_legs"]
        q = tl[tl["historically_validated"]] if len(tl) else tl
        if len(q):
            q = q.assign(tag=np.where(q["p_leg"] > q["breakeven_leg"], "above", "BELOW"))
            print("  no 2-team combo this slate; legs in the +1.5..+2.5 range: " + ", ".join(
                q["leg"] + " (" + q["p_leg"].astype(str) + ", " + q["tag"] + " breakeven " + q["breakeven_leg"].astype(str) + ")"))
        else:
            print("  no legs in the +1.5..+2.5 underdog range this slate")
