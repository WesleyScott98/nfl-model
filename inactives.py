"""Live player availability from Sleeper's public API (free, no key).

Why this exists: the official NFL injury report is finalized Friday, so it can't tell you who got
scratched 90 minutes before kickoff. Sleeper is the largest free fantasy platform and its player
feed carries a live `injury_status` per player, which flips to "Out" when someone is declared
inactive. That makes it the one public source that closes the gap between Friday's report and
kickoff.

Treated as best-effort: if the call fails, the shape changes, or a name can't be matched, the build
carries on with the official report alone. It never blocks a build and never silently guesses —
every decision is printed in the log.

Sleeper asks that /players/nfl not be hammered (it's a ~5MB payload), so this caches to disk and
re-fetches at most every REFRESH_MINUTES.
"""
import json
import os
import re
import time

import requests

URL = "https://api.sleeper.app/v1/players/nfl"
REFRESH_MINUTES = 20
TIMEOUT = 60

# injury_status values that mean "not playing"
OUT_STATUS = {"out", "ir", "injured reserve", "pup", "sus", "suspended", "na", "doubtful",
              "non football injury", "physically unable to perform", "did not play"}
# roster status values that mean "not available at all"
OUT_ROSTER = {"injured reserve", "inactive", "suspended", "non football injury", "practice squad",
              "physically unable to perform", "reserve/covid-19", "cut", "retired"}
QUESTIONABLE = {"questionable", "q"}
TEAM_FIX = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def _norm(name):
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    for suf in (" jr", " sr", " iii", " ii", " iv", " v"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return " ".join(n.split())


def _fetch(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "sleeper_players.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < REFRESH_MINUTES * 60:
        return json.load(open(path))
    last = None
    for wait in (0, 3, 8):
        if wait:
            time.sleep(wait)
        try:
            r = requests.get(URL, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or len(data) < 500:
                raise ValueError(f"unexpected payload ({type(data).__name__}, {len(data) if hasattr(data,'__len__') else '?'} entries)")
            json.dump(data, open(path, "w"))
            return data
        except Exception as ex:
            last = ex
    raise RuntimeError(last)


def availability(cache_dir, usage=None, quiet=False):
    """-> (out_names, questionable{name: p_play}) for players the model actually projects.

    usage: the player-usage table, so only relevant players are reported (and names are matched
    against the ones the model uses). Pass None to get everything Sleeper flags."""
    try:
        players = _fetch(cache_dir)
    except Exception as ex:
        print(f"[sleeper] live availability unavailable ({ex}); using the official injury report only")
        return [], {}

    wanted = None
    if usage is not None and len(usage):
        wanted = {}
        for r in usage.itertuples():
            nm = getattr(r, "full_name", None) or getattr(r, "name", None)
            if nm:
                wanted.setdefault(_norm(nm), nm)

    out, quest, unmatched = [], {}, 0
    for p in players.values():
        if not isinstance(p, dict):
            continue
        full = p.get("full_name") or " ".join(filter(None, [p.get("first_name"), p.get("last_name")]))
        key = _norm(full)
        if wanted is not None and key not in wanted:
            continue
        name = wanted[key] if wanted is not None else full
        inj = str(p.get("injury_status") or "").strip().lower()
        ros = str(p.get("status") or "").strip().lower()
        if inj in OUT_STATUS or ros in OUT_ROSTER:
            out.append((name, inj or ros, TEAM_FIX.get(p.get("team"), p.get("team"))))
        elif inj in QUESTIONABLE:
            quest[name] = 0.5

    if not quiet:
        if out:
            print(f"[sleeper] {len(out)} projected players are OUT right now:")
            for nm, why, tm in sorted(out)[:25]:
                print(f"    {nm} ({tm}) — {why}")
        if quest:
            print(f"[sleeper] {len(quest)} listed questionable (simulated at 50% to play): "
                  + ", ".join(sorted(quest)[:12]) + ("..." if len(quest) > 12 else ""))
        if not out and not quest:
            print("[sleeper] no availability flags for any projected player")
    return [nm for nm, _, _ in out], quest
