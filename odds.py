"""FanDuel lines via The Odds API (https://the-odds-api.com).

Free tier = 500 credits/month. Cost = (#markets) x (#regions) per call, and player props are
fetched per event, so budget it: one slate of game lines ~3 credits; one game's full prop
board ~9 credits. Set ODDS_API_KEY as an environment variable (or a Databricks secret).

If you don't have a key yet, use manual_lines() to type FanDuel numbers in by hand.
"""
import requests
import pandas as pd

import config as C

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# our stat names <- Odds API market keys
PROP_MARKETS = {
    "player_pass_yds": "pass_yds", "player_pass_yds_alternate": "pass_yds",
    "player_pass_tds": "pass_tds",
    "player_rush_yds": "rush_yds", "player_rush_yds_alternate": "rush_yds",
    "player_reception_yds": "rec_yds", "player_reception_yds_alternate": "rec_yds",
    "player_receptions": "receptions", "player_receptions_alternate": "receptions",
    "player_rush_reception_yds": "rush_rec_yds",
    "player_anytime_td": "anytime_td",
}


# ---------------------------------------------------------------- odds math
def american_to_decimal(a):
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied_prob(a):
    return 1 / american_to_decimal(a)


def decimal_to_american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def devig_two_way(p_over_raw, p_under_raw):
    """Multiplicative de-vig: scale both sides so they sum to 1."""
    s = p_over_raw + p_under_raw
    return p_over_raw / s, p_under_raw / s


# ---------------------------------------------------------------- API
def _get(url, **params):
    import os
    key = C.ODDS_API_KEY or os.environ.get("ODDS_API_KEY")   # read at call time, not import time
    if not key:
        raise RuntimeError("Set ODDS_API_KEY (free key at the-odds-api.com) or use manual_lines().")
    r = requests.get(url, params={"apiKey": key, "regions": "us", "oddsFormat": "american",
                                  "bookmakers": C.BOOKMAKER, **params}, timeout=20)
    r.raise_for_status()
    print(f"[odds-api] credits remaining: {r.headers.get('x-requests-remaining')}")
    return r.json()


def game_lines():
    """Moneyline, spread and total for every upcoming game at FanDuel."""
    rows = []
    for ev in _get(f"{BASE}/odds", markets="h2h,spreads,totals"):
        for bk in ev.get("bookmakers", []):
            for mk in bk["markets"]:
                for o in mk["outcomes"]:
                    rows.append({"event_id": ev["id"], "home": ev["home_team"], "away": ev["away_team"],
                                 "commence": ev["commence_time"], "market": mk["key"],
                                 "name": o["name"], "point": o.get("point"), "price": o["price"]})
    return pd.DataFrame(rows)


def player_props(event_id, markets=None):
    """All FanDuel player props (incl. alternate ladders) for one game, one row per outcome."""
    markets = markets or list(PROP_MARKETS)
    data = _get(f"{BASE}/events/{event_id}/odds", markets=",".join(markets))
    rows = []
    for bk in data.get("bookmakers", []):
        for mk in bk["markets"]:
            stat = PROP_MARKETS.get(mk["key"])
            if stat is None:
                continue
            for o in mk["outcomes"]:
                side = {"Over": "over", "Under": "under", "Yes": "over", "No": "under"}.get(o["name"])
                if side is None:
                    continue
                rows.append({"player": o.get("description"), "stat": stat, "market": mk["key"],
                             "alternate": mk["key"].endswith("_alternate"), "side": side,
                             "line": o.get("point", 0.5), "price": o["price"]})
    return pd.DataFrame(rows)


def manual_lines(rows):
    """Type lines in yourself: [("Malik Nabers", "rec_yds", 63.5, -114, -106), ...]
    (player, stat, line, over_price, under_price or None)."""
    out = []
    for player, stat, line, over, under in rows:
        out.append({"player": player, "stat": stat, "market": "manual", "alternate": under is None,
                    "side": "over", "line": line, "price": over})
        if under is not None:
            out.append({"player": player, "stat": stat, "market": "manual", "alternate": False,
                        "side": "under", "line": line, "price": under})
    return pd.DataFrame(out)


GAME_MARKETS = "h2h,spreads,totals,alternate_spreads,alternate_totals,team_totals"
_MK = {"h2h": "ml", "spreads": "spread", "totals": "total", "alternate_spreads": "alt_spread",
       "alternate_totals": "alt_total", "team_totals": "team_total"}


def game_markets(event_id, team_abbr):
    """Every FanDuel game-level market for one event (main + alternate spreads/totals, team totals).
    team_abbr: dict full team name -> abbreviation. Rows: market, team, side, line, price."""
    data = _get(f"{BASE}/events/{event_id}/odds", markets=GAME_MARKETS)
    rows = []
    for bk in data.get("bookmakers", []):
        for mk in bk["markets"]:
            m = _MK.get(mk["key"])
            for o in mk["outcomes"]:
                if m in ("ml", "spread", "alt_spread"):
                    rows.append({"market": m, "team": team_abbr.get(o["name"]), "side": None,
                                 "line": o.get("point"), "price": o["price"]})
                elif m in ("total", "alt_total"):
                    rows.append({"market": m, "team": None, "side": o["name"].lower(), "line": o["point"], "price": o["price"]})
                elif m == "team_total":
                    rows.append({"market": m, "team": team_abbr.get(o.get("description")), "side": o["name"].lower(),
                                 "line": o["point"], "price": o["price"]})
    return pd.DataFrame(rows)


def manual_game(home, away, spread=None, ml=None, total=None, alt_spreads=(), alt_totals=(), team_totals=()):
    """Type FanDuel game lines in by hand.
    spread=(home_line, home_price, away_price)   e.g. (-6.5, -110, -110)
    ml=(home_price, away_price)                  total=(line, over_price, under_price)
    alt_spreads=[(team, line, price)]  alt_totals=[("over"|"under", line, price)]
    team_totals=[(team, "over"|"under", line, price)]"""
    r = []
    if spread:
        r += [{"market": "spread", "team": home, "side": None, "line": spread[0], "price": spread[1]},
              {"market": "spread", "team": away, "side": None, "line": -spread[0], "price": spread[2]}]
    if ml:
        r += [{"market": "ml", "team": home, "side": None, "line": None, "price": ml[0]},
              {"market": "ml", "team": away, "side": None, "line": None, "price": ml[1]}]
    if total:
        r += [{"market": "total", "team": None, "side": "over", "line": total[0], "price": total[1]},
              {"market": "total", "team": None, "side": "under", "line": total[0], "price": total[2]}]
    r += [{"market": "alt_spread", "team": t, "side": None, "line": l, "price": p} for t, l, p in alt_spreads]
    r += [{"market": "alt_total", "team": None, "side": sd, "line": l, "price": p} for sd, l, p in alt_totals]
    r += [{"market": "team_total", "team": t, "side": sd, "line": l, "price": p} for t, sd, l, p in team_totals]
    return pd.DataFrame(r)
