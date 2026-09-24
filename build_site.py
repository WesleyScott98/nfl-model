"""Build the public site: run the model for the current NFL week, write index.html.

    python build_site.py                 # auto-detects the week
    python build_site.py --week 5        # force a week
    python build_site.py --season 2026 --out site

Everything the page needs is baked into the HTML, so the result is a single static file that
any host will serve. Run it again whenever you want fresh numbers (injuries, inactives, lines).
"""
import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # flat layout: all files in one folder
import calibrate as K          # noqa: E402
import inactives as IN         # noqa: E402
import edge as E               # noqa: E402
from game import Model         # noqa: E402
from odds import decimal_to_american  # noqa: E402

# ---- live FanDuel odds (optional) -------------------------------------------------------
# Set ODDS_API_KEY to pull real prices. Credit cost per run, at the free tier's 500/month:
#   ODDS_MODE=lines  -> 3 credits   (spread, total, moneyline for the whole slate)
#   ODDS_MODE=full   -> ~3 + 4 per game  (adds TD / receptions / receiving / rushing props)
# "full" on a 16-game slate is roughly 67 credits, so budget about 7 full runs a month on the
# free tier — or run "lines" on the early builds and "full" once near kickoff.
ODDS_MODE = os.environ.get("ODDS_MODE", "full" if os.environ.get("ODDS_API_KEY") else "off")
# Props cost 4 credits per game (one per market), so only pull them for games kicking off inside
# this window. At 30 hours, the Saturday-afternoon run covers the whole Sunday slate, the Sunday-
# evening run covers Monday night, and the Wednesday-evening run covers Thursday night — so prices
# are posted roughly a day before each game. Other runs take game lines only (3 credits a slate).
PROPS_WINDOW_HOURS = float(os.environ.get("PROPS_WINDOW_HOURS", 30))
PROP_MARKETS = ["player_anytime_td", "player_receptions", "player_reception_yds", "player_rush_yds",
                "player_pass_yds"]   # passing lines aren't bet here — they identify the starting QB
STAT_TO_MARKET = {"receptions": "receptions", "rec_yds": "rec_yds", "rush_yds": "rush_yds",
                  "anytime TD": "anytime_td"}
# FanDuel posts props only for players it expects to play, so the board itself says who's active.
# Only trusted when the board is well populated, and it downgrades rather than removes.
MARKET_BOARD_MIN = 12        # players with props before we trust an absence
MARKET_ABSENT_P = 0.30       # a projected starter with no props at all: treat as 30% to play
# Absence-based inactive detection is OFF: in testing it flagged backups the book simply hadn't
# priced, and missed a genuinely inactive starter when he was his position's only entry. The
# injury report, live status and snap-collapse checks cover inactives more reliably.
# Reading the STARTING QB off the board is separate, and stays on - it's unambiguous.
MARKET_ABSENT_ENABLED = False
# Anchoring to the market on ROLE, not on edge:
# FanDuel's posted line is the market's view of a player's workload. If the model is within
# MARKET_TRUST_BAND of the line, leave it alone - that's where genuine disagreements (and edges)
# live. If it's outside that band, the model has probably misread the player's role (a promotion,
# a committee change, a new team), so close MARKET_PULL of the gap toward the line.
MARKET_TRUST_BAND = 0.35     # within +/-35% of the posted line, trust the model completely
MARKET_PULL = 0.6            # beyond it, move 60% of the way to the market
# Touchdowns get the same treatment. The model reads scoring chances off recent team performance,
# which misses a change the market already knows about (a new QB, a healthier offense). If the two
# disagree by more than MARKET_TD_BAND, close MARKET_PULL of that gap.
MARKET_TD_BAND = 0.10        # 10 percentage points
ATD_MARGIN = 0.08            # rough vig baked into an anytime-TD price

# QB passing yards is deliberately absent: it's the model's weakest market (2-9% skill in backtests
# vs ~30% for receptions and yards), and the odds feed doesn't price it either.
LINES = {"receptions": [2.5, 3.5, 4.5, 5.5], "rec_yds": [39.5, 49.5, 59.5, 69.5, 79.5],
         "rush_yds": [39.5, 49.5, 59.5, 74.5, 89.5]}
PRICE_FLOOR = -350             # singles shorter than this are dropped by the page anyway


def current_week(model, season):
    """The week whose games are still ahead of us (or just finished today)."""
    now = datetime.now(timezone.utc) - timedelta(hours=6)
    s = model.sched.copy()
    s["kick"] = pd.to_datetime(s["gameday"] + " " + s["gametime"], errors="coerce")
    upcoming = s[s["kick"] >= now.replace(tzinfo=None)]
    return int(upcoming["week"].min()) if len(upcoming) else int(s["week"].max())


def fetch_odds(this_week=None):
    """FanDuel prices for THIS WEEK's games only, keyed by matchup. Returns {} if there's no key or
    the call fails — the site is fully usable either way."""
    if ODDS_MODE == "off":
        return {}
    try:
        import odds as O
        from teams import TEAM_NAMES
    except Exception:
        return {}
    try:
        ev = O.game_lines()
        if ev.empty:
            return {}
        ev["h"], ev["a"] = ev["home"].map(TEAM_NAMES), ev["away"].map(TEAM_NAMES)
        book = {}
        now = datetime.now(timezone.utc)
        skipped_other_week = props_pulled = 0
        for (eid, h, a), grp in ev.groupby(["event_id", "h", "a"]):
            if not h or not a:
                continue
            if this_week is not None and f"{a}@{h}" not in this_week:
                skipped_other_week += 1
                continue                      # a later week's game: don't spend credits on it
            entry = {"lines": {}, "props": {}}
            for r in grp.itertuples():
                if r.market == "h2h":
                    entry["lines"].setdefault("ml", {})[TEAM_NAMES.get(r.name, r.name)] = r.price
                elif r.market == "spreads":
                    entry["lines"].setdefault("spread", {})[TEAM_NAMES.get(r.name, r.name)] = (r.point, r.price)
                elif r.market == "totals":
                    entry["lines"].setdefault("total", {})[r.name.lower()] = (r.point, r.price)
            kick = pd.to_datetime(grp["commence"].iloc[0], utc=True, errors="coerce")
            soon = kick is not pd.NaT and (kick - now).total_seconds() / 3600 <= PROPS_WINDOW_HOURS
            if ODDS_MODE == "full" and soon:
                try:
                    pr = O.player_props(eid, markets=PROP_MARKETS)
                    props_pulled += 1
                    for r in pr.itertuples():
                        if r.side != "over":
                            continue
                        key = (r.player, r.stat, float(r.line))
                        entry["props"][key] = int(r.price)
                except Exception as ex:
                    print(f"  props unavailable for {a}@{h}: {ex}")
            book[f"{a}@{h}"] = entry
        print(f"FanDuel odds: {len(book)} games this week, player props for {props_pulled} of them "
              f"(kicking off within {PROPS_WINDOW_HOURS:g}h), {skipped_other_week} later-week games skipped")
        return book
    except Exception as ex:
        print(f"odds unavailable ({ex}); building with model prices only")
        return {}


def market_signals(book, usage, quiet=False):
    """Read the FanDuel board for who's playing.

    A sportsbook posts props only for players it expects on the field, so:
      * the QB with a posted passing line is the starter for that team
      * a player the model projects as a starter with NO posted props, on a board that's otherwise
        full, is probably inactive -> downgraded to MARKET_ABSENT_P, not deleted
    """
    qb_from_book, downgrade = {}, {}
    if usage is None or not len(usage):
        return qb_from_book, downgrade
    team_of = {str(r.full_name).lower(): r.team for r in usage.itertuples()}
    for matchup, entry in book.items():
        props = entry.get("props") or {}
        if not props:
            continue
        names = {p for (p, _, _) in props}
        teams = matchup.split("@")
        # Starting QB = the highest posted passing line on each team. A book only prices a passing
        # line for someone it expects to start, so this needs no other context.
        best_qb = {}
        for (p, st, ln), _ in props.items():
            if st != "pass_yds":
                continue
            tm = team_of.get(str(p).lower())
            if tm and ln and ln > best_qb.get(tm, (0, None))[0]:
                best_qb[tm] = (ln, p)
        for tm, (_, p) in best_qb.items():
            qb_from_book[tm] = p
        if len(names) < MARKET_BOARD_MIN:
            continue                                  # thin board: an absence means nothing
        # A projected starter missing from the board is only meaningful if the board covers his
        # team AND his position — otherwise the market simply isn't offering that kind of prop.
        by_team = {t: {r.full_name for r in usage.itertuples() if r.team == t and r.full_name in names}
                   for t in teams}
        covered_pos = {t: {r.pos for r in usage.itertuples() if r.team == t and r.full_name in names}
                       for t in teams}
        for r in (usage.itertuples() if MARKET_ABSENT_ENABLED else []):
            if r.team not in teams or r.full_name in names:
                continue
            if len(by_team.get(r.team, ())) < 5:            # thin coverage for this team
                continue
            if r.pos not in covered_pos.get(r.team, set()):  # market isn't pricing his position
                continue
            if getattr(r, "s_tgt", 0) >= 0.12 or getattr(r, "s_car", 0) >= 0.25:
                downgrade[r.full_name] = MARKET_ABSENT_P
    if not quiet:
        if qb_from_book:
            print("[fanduel] starting QB per the board: "
                  + ", ".join(f"{t}={n}" for t, n in sorted(qb_from_book.items())))
        if downgrade:
            print("[fanduel] projected starters with no props posted (treated as "
                  f"{int(MARKET_ABSENT_P*100)}% to play): " + ", ".join(sorted(downgrade)))
    return qb_from_book, downgrade


def attach_price(bet, player, book_props):
    """Add FanDuel's price to a bet when the book posts that exact line."""
    stat = STAT_TO_MARKET.get("anytime TD" if "TD" in bet["bet"] else
                              bet["bet"].split("+ ")[-1].replace(" ", "_").replace("rec_yards", "rec_yds")
                              .replace("rush_yards", "rush_yds").replace("catches", "receptions"))
    if stat is None:
        return
    line = 0.5 if "TD" in bet["bet"] else float(bet["bet"].split("+")[0]) - 0.5
    price = book_props.get((player, stat, line))
    if price is None:
        return
    bet["book"] = price
    from odds import american_to_decimal, implied_prob
    bet["ev"] = round(bet["prob"] * american_to_decimal(price) - 1, 3)
    bet["value"] = bet["ev"] > 0.02


def load_overrides(here):
    """Manual corrections for news the feed hasn't caught: {"out": [...],
    "questionable": {"Name": 0.6}, "snap_limit": {"Name": 0.5}}."""
    path = os.path.join(here, "overrides.json")
    if not os.path.exists(path):
        return [], {}, {}, {}
    try:
        o = json.load(open(path))
        out = [n for n in o.get("out", []) if isinstance(n, str)]
        if out:
            print(f"overrides: forcing out {', '.join(out)}")
        qb = {k: v for k, v in (o.get("qb") or {}).items() if isinstance(v, str)}
        if qb:
            print("overrides: named starting QBs " + ", ".join(f"{k}={v}" for k, v in qb.items()))
        return out, o.get("questionable", {}), o.get("snap_limit", {}), qb
    except Exception as ex:
        print(f"overrides.json unreadable ({ex}); ignoring")
        return [], {}, {}, {}


def auto_injury_watch(m, week):
    """Players whose snap share collapsed last week are probably hurt. Official designations don't
    publish until Wednesday, so until they do, treat these players as questionable (50% to play)
    instead of projecting them as full starters. Anyone the report already covers is skipped —
    the report always wins, and this clears itself automatically."""
    sn = m.snaps[(m.snaps["season"] == m.season) & (m.snaps["week"] == week - 1)]
    prior = m.snaps[(m.snaps["season"] == m.season) & (m.snaps["week"] < week - 1)]
    if sn.empty or prior.empty:
        return {}
    reported = set(m.inj[(m.inj["season"] == m.season) & (m.inj["week"] == week)
                         & m.inj["report_status"].notna()]["gsis_id"].dropna())
    usual = prior.groupby("gsis_id")["offense_pct"].median()
    info = None
    try:
        import data as _d
        info = _d.player_info().set_index("gsis_id")["display_name"]
    except Exception:
        pass
    auto, shown = {}, []
    for r in sn.itertuples():
        u = usual.get(r.gsis_id)
        if not (u and u >= 0.5 and r.offense_pct < 0.5 * u):
            continue
        name = info.get(r.gsis_id) if info is not None else None
        tag = "already on this week's report" if r.gsis_id in reported else "auto-flagged questionable"
        if name and r.gsis_id not in reported:
            auto[name] = 0.5
        shown.append(f"{name or r.gsis_id} ({r.team}, {int(r.offense_pct*100)}% of snaps vs "
                     f"{int(u*100)}% usual) — {tag}")
    if shown:
        print("Injury watch — snap share collapsed last week:")
        for f in shown[:15]:
            print("   ", f)
    return auto


def build(season, week, n_sims=20000, overrides=([], {}, {}, {})):
    role_notes = []
    m = Model(season)
    cal = K.load()
    this_week = {f"{g.away_team}@{g.home_team}" for g in m.sched[m.sched["week"] == week].itertuples()}
    book = fetch_odds(this_week)
    out_names, questionable, snap_limit, qb_named = overrides
    qb_named = dict(qb_named)                      # your overrides stay on top of anything automatic
    auto_q = auto_injury_watch(m, week) or {}
    # live availability (Sleeper) closes the gap between Friday's report and kickoff
    cache = os.environ.get("NFL_EDGE_CACHE", "/tmp/nfl-cache")
    live_out, live_q = IN.availability(cache, m.features(week)[1])
    # the depth chart names the starting QB when it disagrees with whoever took the snaps last week
    depth = IN.depth_starters(cache)
    for tm, roles in depth.items():
        qb_name = roles.get("QB")
        if qb_name and tm not in qb_named:
            qb_named[tm] = qb_name
    # the sportsbook board is the most current signal of all — it outranks the depth chart,
    # though anything you set by hand in overrides.json still wins
    book_qb, book_downgrade = market_signals(book, m.features(week)[1])
    manual_qb = set((overrides[3] or {}).keys())
    for tm, name in book_qb.items():
        if tm not in manual_qb:
            qb_named[tm] = name
    for nm, p in book_downgrade.items():
        questionable.setdefault(nm, p)
    out_names = list(dict.fromkeys(list(out_names) + live_out))
    questionable = {**auto_q, **live_q, **questionable}   # live beats auto; your overrides beat both
    fair = lambda p: decimal_to_american(1 / max(min(p, 0.97), 0.02))
    out = {"season": season, "week": week, "odds": bool(book),
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "games": []}
    sched = m.sched[m.sched["week"] == week]
    for g in sched.itertuples():
        if pd.isna(g.spread_line):
            continue
        bk = book.get(f"{g.away_team}@{g.home_team}", {"lines": {}, "props": {}})
        sp_line, tot_line = float(g.spread_line), float(g.total_line)
        if bk["lines"].get("spread", {}).get(g.home_team):
            sp_line = -float(bk["lines"]["spread"][g.home_team][0])
        if bk["lines"].get("total", {}).get("over"):
            tot_line = float(bk["lines"]["total"]["over"][0])
        sim = m.simulate(g.away_team, g.home_team, week, sp_line, tot_line, n=n_sims,
                         out_names=out_names, questionable=questionable, snap_override=snap_limit,
                         qb=qb_named)
        s = sim.summary()
        mar = sim.points[g.home_team] - sim.points[g.away_team]
        tot = sim.points[g.home_team] + sim.points[g.away_team]
        sp = sp_line
        cover_home = float(((mar - sp) > 0).mean())
        push_sp = float((mar == sp).mean())
        over = float((tot > tot_line).mean())
        push_to = float((tot == tot_line).mean())
        game = {"away": g.away_team, "home": g.home_team, "kickoff": f"{g.gameday} {g.gametime} ET",
                "spread": f"{g.home_team} {-sp:+g}", "total": tot_line,
                "book_ml": bk["lines"].get("ml", {}),
                "ml_fair": {g.home_team: fair(float((mar > 0).mean())), g.away_team: fair(float((mar < 0).mean()))},
                "ml_prob": {g.home_team: round(float((mar > 0).mean()), 3), g.away_team: round(float((mar < 0).mean()), 3)},
                "spread_line": sp, "cover": {g.home_team: round(cover_home, 3), g.away_team: round(1 - cover_home - push_sp, 3)},
                "total_probs": {"over": round(over, 3), "under": round(1 - over - push_to, 3)},
                "proj_total": round(float(tot.mean()), 1),
                "team_totals_proj": {g.home_team: round(float(sim.points[g.home_team].mean()), 1),
                                     g.away_team: round(float(sim.points[g.away_team].mean()), 1)},
                "players": [], "parlays": []}
        for r in s[(s.targets_mean > 2) | (s.carries_mean > 4) | (s.pass_yds_mean > 0)].head(12).itertuples():
            pl = {"name": r.player, "team": r.team, "pos": r.pos,
                  "proj": {"rec": round(r.receptions_mean, 1), "rec_yds": round(r.rec_yds_mean, 1),
                           "rush_yds": round(r.rush_yds_mean, 1), "pass_yds": round(r.pass_yds_mean, 1)},
                  "bets": []}
            p_td = K.apply(sim.prob(r.player, "anytime_td", 0.5), "anytime_td", cal)
            if p_td > 0.08:
                pl["bets"].append({"bet": "anytime TD", "prob": round(p_td, 3), "fair": fair(p_td)})
            for stat, lines in LINES.items():
                if stat in ("receptions", "rec_yds") and r.targets_mean < 2.5:
                    continue
                if stat == "rush_yds" and r.carries_mean < 4:
                    continue
                for ln in lines:
                    p = K.apply(sim.prob(r.player, stat, ln), stat, cal)
                    if 0.35 < p < 0.9:
                        pl["bets"].append({"bet": f"{int(ln + 0.5)}+ {stat.replace('_', ' ')}",
                                           "prob": round(p, 3), "fair": fair(p)})
            # FanDuel posts its own numbers (45.5 yards, 4.5 catches), not our ladder - so price the
            # model against the lines you can actually bet, and drop our ladder for that stat.
            posted = [(st, ln, pr) for (pname, st, ln), pr in bk["props"].items() if pname == r.player]
            if posted:
                covered = {st for st, _, _ in posted}
                pl["bets"] = [b for b in pl["bets"]
                              if ("anytime_td" if "TD" in b["bet"] else
                                  b["bet"].split("+ ")[-1].replace(" ", "_")) not in covered]
                from odds import american_to_decimal
                for st, ln, price in posted:
                    try:
                        if st == "anytime_td":
                            p = K.apply(sim.prob(r.player, st, ln), st, cal)
                            from odds import implied_prob
                            mkt = implied_prob(price) / (1 + ATD_MARGIN)
                            if abs(p - mkt) > MARKET_TD_BAND:
                                before = p
                                p = p + (mkt - p) * MARKET_PULL
                                role_notes.append(f"{r.player} anytime TD: model {before:.0%} vs "
                                                  f"FanDuel {mkt:.0%} -> using {p:.0%}")
                        else:
                            x = sim.stat(r.player, st)
                            x = x[~np.isnan(x)] if hasattr(x, "__len__") else x
                            mu = float(x.mean()) if len(x) else 0.0
                            scale = 1.0
                            if ln > 0 and mu > 0:
                                gap = mu / ln - 1.0
                                if abs(gap) > MARKET_TRUST_BAND:
                                    target = mu + (ln - mu) * MARKET_PULL   # close most of the gap
                                    scale = max(0.4, min(target / mu, 2.5))
                                    role_notes.append(f"{r.player} {st}: model {mu:.1f} vs FanDuel "
                                                      f"{ln} -> using {mu*scale:.1f}")
                            p = K.apply(float((x * scale > ln).mean()), st, cal)
                    except Exception:
                        continue
                    label = "anytime TD" if st == "anytime_td" else f"{int(ln + 0.5)}+ {st.replace('_', ' ')}"
                    ev = round(p * american_to_decimal(price) - 1, 3)
                    pl["bets"].append({"bet": label, "prob": round(p, 3), "fair": fair(p),
                                       "book": int(price), "ev": ev, "value": ev > 0.02})
            if pl["bets"]:
                game["players"].append(pl)
        legs = []
        for pl in game["players"][:8]:
            for b in pl["bets"]:
                if 0.45 < b["prob"] < 0.85:
                    stat = "anytime_td" if "TD" in b["bet"] else b["bet"].split("+ ")[1].replace(" ", "_")
                    ln = 0.5 if "TD" in b["bet"] else float(b["bet"].split("+")[0]) - 0.5
                    legs.append({"desc": f"{pl['name']} {b['bet']}", "book": b.get("book"),
                                 "leg": {"player": pl["name"], "stat": stat, "line": ln}, "p": b["prob"]})
        legs = sorted(legs, key=lambda x: -x["p"])[:8]
        combos = []
        for k in (2, 3):
            for c in itertools.combinations(legs, k):
                if len({(l["leg"]["player"], l["leg"]["stat"]) for l in c}) < k:
                    continue
                r = E.price_parlay(sim, [l["leg"] for l in c], cal=cal)
                entry = {"legs": [l["desc"] for l in c], "prob": r["p_joint"],
                         "fair": r["fair_american"], "correlation_lift": r["correlation_lift"]}
                books = [l.get("book") for l in c]
                if all(b is not None for b in books):
                    from odds import american_to_decimal
                    dec = 1.0
                    for b in books:
                        dec *= american_to_decimal(b)
                    entry["book_price"] = decimal_to_american(dec)   # straight-parlay math
                combos.append(entry)
        game["parlays"] = sorted(combos, key=lambda x: -x["prob"])[:6]
        out["games"].append(game)
        print(f"  {g.away_team}@{g.home_team}", flush=True)
    if role_notes:
        print(f"[fanduel] {len(role_notes)} projections pulled toward the market "
              f"(the model had the player's role wrong):")
        for n in role_notes[:15]:
            print("   ", n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=int(os.environ.get("NFL_SEASON", 2026)))
    ap.add_argument("--week", type=int)
    ap.add_argument("--out", default=".")
    ap.add_argument("--sims", type=int, default=20000)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    t = time.time()
    week = a.week or current_week(Model(a.season), a.season)
    print(f"building season {a.season}, week {week}")
    data = build(a.season, week, a.sims, load_overrides(here))
    if not data["games"]:
        print("no games with lines for this week — leaving the current site in place")
        return
    # "</" is escaped so a stray "</script" in third-party data (odds feed, player names) can't
    # terminate the <script> block early and inject markup into every visitor's page.
    payload = json.dumps(data).replace("</", "<\\/")
    html = open(os.path.join(here, "template.html")).read().replace("__DATA__", payload)
    html = html.replace("Week 3 picks", f"Week {week} picks").replace("Week 3 NFL picks", f"Week {week} NFL picks")
    os.makedirs(os.path.join(here, a.out), exist_ok=True)
    with open(os.path.join(here, a.out, "index.html"), "w") as f:
        f.write(html)
    json.dump(data, open(os.path.join(here, a.out, "picks.json"), "w"))
    print(f"wrote {a.out}/index.html — {len(data['games'])} games, "
          f"{sum(len(g['players']) for g in data['games'])} players, {round(time.time() - t)}s")


if __name__ == "__main__":
    main()
