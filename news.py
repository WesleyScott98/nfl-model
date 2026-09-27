"""Read NFL player news and turn it into model adjustments.

The injury report says Out / Questionable / nothing. It does not say "on a snap count", "expected
to be limited", "lost the job", or "will start". Beat reporting says all of that, hours or days
before it shows up anywhere structured — and sometimes it never does. Zay Flowers in Week 3 2026 is
the example: officially Questionable, but reported as on a snap count if active. Those are very
different bets.

This module fetches NBC Sports / Rotoworld player news (a public page, fetched once per build) and
maps headlines onto three model inputs:

    out          — ruled out, inactive, will not play
    questionable — game-time decision, doubtful, unlikely
    snap_limit   — on a snap count, limited role, eased back in

It is deliberately conservative: only clear phrasing counts, a player must already be projected for
the model to care, and anything it can't parse is ignored. If the fetch fails the build carries on
with the official report — news is an enhancement, never a dependency.
"""
import os
import re
import time

import requests

URL = "https://www.nbcsports.com/fantasy/football/player-news"
# ESPN's public JSON: built for programmatic access, so it's the more reliable of the two.
ESPN_NEWS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=200"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={}"
REFRESH_MINUTES = 30
TIMEOUT = 30

# phrase -> (what it means, value). Order matters: the strongest signal wins.
RULES = [
    (r"ruled out|will (?:not|n't) play|won'?t play|is inactive|declared inactive|out for (?:the )?(?:game|week|season)"
     r"|placed on injured reserve|to ir\b", "out", None),
    (r"snap count|pitch count|limited role|eased back|workload limit|play(?:ing)? limited snaps"
     r"|limited number of snaps|snap limit", "snap_limit", 0.6),
    (r"doubtful|unlikely to play|not expected to play|trending toward(?:s)? (?:sitting|missing)"
     r"|game-?time decision|coin flip", "questionable", 0.4),
    (r"questionable|will be a decision|monitor|uncertain", "questionable", 0.55),
    (r"expected to play|will play|is active|cleared to play|no restrictions|full go|removed from the injury report",
     "clear", None),
]


def _norm(name):
    n = str(name or "").lower().replace(".", "").replace("'", "").replace("-", " ")
    for suf in (" jr", " sr", " iii", " ii", " iv", " v"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return " ".join(n.split())


def _fetch(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "player_news.html")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < REFRESH_MINUTES * 60:
        return open(path, encoding="utf-8", errors="ignore").read()
    r = requests.get(URL, timeout=TIMEOUT, headers={"User-Agent": "nfl-model/1.0 (personal picks site)"})
    r.raise_for_status()
    open(path, "w", encoding="utf-8").write(r.text)
    return r.text


def _items(html):
    """-> [(player name, headline + blurb)] — crude but stable: each news block names its player
    in a heading, and the text that follows is the report."""
    out = []
    # player links look like /nfl/zay-flowers/8491 ; the visible name follows in the same block
    for m in re.finditer(r'/nfl/([a-z0-9\-]+)/\d+[^>]*>([^<]{3,40})</a>(.{0,1200}?)(?=/nfl/[a-z0-9\-]+/\d+|$)',
                         html, re.S | re.I):
        name = m.group(2).strip()
        body = re.sub(r"<[^>]+>", " ", m.group(3))
        body = re.sub(r"\s+", " ", body)
        if len(name.split()) >= 2 and len(body) > 40:
            out.append((name, body[:600]))
    return out


def _espn_json(url, cache_dir, key):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"espn_{key}.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < REFRESH_MINUTES * 60:
        import json as _j
        return _j.load(open(path))
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "nfl-model/1.0"})
    r.raise_for_status()
    data = r.json()
    import json as _j
    _j.dump(data, open(path, "w"))
    return data


def espn_signals(cache_dir, usage=None, week=None, quiet=False):
    """ESPN's public feeds: headlines plus the per-game injury blocks, which carry the short
    comments ('limited snaps', 'game-time decision') that the official report never contains."""
    outs, quest, limits = [], {}, {}
    wanted = None
    if usage is not None and len(usage):
        wanted = {}
        for r in usage.itertuples():
            nm = getattr(r, "full_name", None)
            if nm:
                wanted.setdefault(_norm(nm), nm)

    def apply(name, text):
        key = _norm(name)
        if wanted is not None and key not in wanted:
            return
        shown = wanted[key] if wanted is not None else name
        low = str(text).lower()
        for pattern, kind, value in RULES:
            if re.search(pattern, low):
                if kind == "out" and shown not in outs:
                    outs.append(shown)
                elif kind == "snap_limit":
                    limits.setdefault(shown, value)
                elif kind == "questionable":
                    quest.setdefault(shown, value)
                return

    # 1. headlines
    try:
        for art in _espn_json(ESPN_NEWS, cache_dir, "news").get("articles", []):
            blob = " ".join(filter(None, [art.get("headline"), art.get("description")]))
            for ath in (art.get("categories") or []):
                a = ath.get("athlete") or {}
                if a.get("displayName"):
                    apply(a["displayName"], blob)
    except Exception as ex:
        if not quiet:
            print(f"[espn] news feed unavailable ({ex})")

    # 2. per-game injury blocks, which is where status + comment live
    try:
        board = _espn_json(ESPN_SCOREBOARD + (f"?week={week}&seasontype=2" if week else ""),
                           cache_dir, f"board{week or ''}")
        for ev in board.get("events", [])[:20]:
            try:
                summ = _espn_json(ESPN_SUMMARY.format(ev.get("id")), cache_dir, f"sum{ev.get('id')}")
            except Exception:
                continue
            for team_block in (summ.get("injuries") or []):
                for it in (team_block.get("injuries") or []):
                    ath = (it.get("athlete") or {}).get("displayName")
                    if not ath:
                        continue
                    status = str(it.get("status") or "")
                    detail = (it.get("details") or {})
                    comment = " ".join(str(x) for x in [it.get("longComment"), it.get("shortComment"),
                                                        detail.get("type"), detail.get("detail")] if x)
                    apply(ath, f"{status} {comment}")
    except Exception as ex:
        if not quiet:
            print(f"[espn] injury blocks unavailable ({ex})")

    if not quiet and (outs or quest or limits):
        print(f"[espn] {len(outs)} out, {len(quest)} doubtful/questionable, {len(limits)} snap-limited")
    return outs, quest, limits


def combined_news(cache_dir, usage=None, week=None, quiet=False):
    """Everything the beat is saying, from both sources. The stronger signal wins:
    out > snap limit > doubtful > questionable."""
    n_out, n_q, n_lim = player_news(cache_dir, usage, quiet)
    e_out, e_q, e_lim = espn_signals(cache_dir, usage, week, quiet)
    outs = list(dict.fromkeys(n_out + e_out))
    quest = {**e_q, **n_q}
    limits = {**e_lim, **n_lim}
    for nm in outs:                      # being out beats any softer flag
        quest.pop(nm, None)
        limits.pop(nm, None)
    return outs, quest, limits


def player_news(cache_dir, usage=None, quiet=False):
    """-> (out_names, questionable{name: p}, snap_limit{name: share}) drawn from beat reporting."""
    try:
        html = _fetch(cache_dir)
    except Exception as ex:
        print(f"[news] player news unavailable ({ex}); using the official report only")
        return [], {}, {}

    wanted = None
    if usage is not None and len(usage):
        wanted = {}
        for r in usage.itertuples():
            nm = getattr(r, "full_name", None)
            if nm:
                wanted.setdefault(_norm(nm), nm)

    outs, quest, limits, seen = [], {}, {}, set()
    for name, body in _items(html):
        key = _norm(name)
        if wanted is not None and key not in wanted:
            continue                      # not a player the model projects
        if key in seen:
            continue                      # the page is newest-first, so the first mention wins
        text = body.lower()
        for pattern, kind, value in RULES:
            if not re.search(pattern, text):
                continue
            seen.add(key)
            shown = wanted[key] if wanted is not None else name
            if kind == "out":
                outs.append(shown)
            elif kind == "snap_limit":
                limits[shown] = value
            elif kind == "questionable":
                quest[shown] = value
            break                          # strongest matching rule only

    if not quiet:
        if outs:
            print(f"[news] reported out: {', '.join(sorted(outs))}")
        if limits:
            print(f"[news] reported on a snap count (usage cut to "
                  f"{int(list(limits.values())[0] * 100)}%): {', '.join(sorted(limits))}")
        if quest:
            print(f"[news] reported doubtful / game-time decisions: "
                  + ", ".join(f"{k} ({int(v*100)}%)" for k, v in sorted(quest.items())))
        if not (outs or limits or quest):
            print("[news] no actionable player news matched a projected player")
    return outs, quest, limits
