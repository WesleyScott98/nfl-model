"""Command-line entry point.

  # simulate a game, print projections + matchup report
  python run_game.py --away NYG --home LA --week 2 --spread 6.5 --total 47.5 --out "Puka Nacua"

  # also pull FanDuel props (needs ODDS_API_KEY) and rank +EV bets
  python run_game.py --away NYG --home LA --week 2 --props
"""
import argparse

import pandas as pd

import edge
import odds
from game import Model

from teams import TEAM_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--spread", type=float, help="home team favoured by (e.g. 6.5)")
    ap.add_argument("--total", type=float)
    ap.add_argument("--out", nargs="*", default=[], help="players ruled out (full names)")
    ap.add_argument("--questionable", nargs="*", default=[], help='e.g. "Puka Nacua=0.35" (P plays)')
    ap.add_argument("--wind", type=float, help="override weather: mph")
    ap.add_argument("--temp", type=float, help="override weather: deg F")
    ap.add_argument("--precip", action="store_true", help="override weather: rain/snow expected")
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--props", action="store_true", help="fetch FanDuel props via The Odds API")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    m = Model(a.season)
    print("\nDEFENSIVE / MATCHUP PROFILE (funnel_z > 0 = weaker vs pass than run)")
    print(m.matchup_report(a.away, a.home, a.week).to_string())
    q = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.questionable}
    wx = "auto"
    if a.no_weather:
        wx = None
    elif a.wind is not None or a.temp is not None or a.precip:
        wx = {"outdoor": True, "wind": a.wind or 0, "temp": a.temp, "precip": a.precip}
    sim = m.simulate(a.away, a.home, a.week, a.spread, a.total, out_names=a.out, questionable=q, weather=wx)
    print(f"\nWEATHER: {sim.weather}   QUESTIONABLE (P plays): {sim.questionable}")
    s = sim.summary()
    cols = ["player", "team", "pos", "targets_mean", "receptions_mean", "rec_yds_mean", "carries_mean",
            "rush_yds_mean", "pass_yds_mean", "p_anytime_td", "p_active"]
    print("\nPROJECTIONS (simulation means)")
    print(s[(s["targets_mean"] > 1.5) | (s["carries_mean"] > 3) | (s["pass_yds_mean"] > 0)][cols].round(2).to_string(index=False))

    if a.props:
        ev = odds.game_lines()
        ev["h"], ev["a"] = ev["home"].map(TEAM_NAMES), ev["away"].map(TEAM_NAMES)
        match = ev[(ev["h"] == a.home) & (ev["a"] == a.away)]
        if match.empty:
            raise SystemExit("Game not found in FanDuel feed")
        props = odds.player_props(match["event_id"].iloc[0])
        priced = edge.price_props(sim, props)
        print("\nTOP +EV (after shrinking toward market)")
        print(edge.best_bets(priced).head(25).to_string(index=False))
        if a.csv:
            priced.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
