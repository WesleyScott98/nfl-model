"""Data layer. All football data comes from nflverse (free, open, updated nightly in season):
play-by-play with EPA, player stats, schedules with closing lines, injuries, snap counts,
FTN charting (blitzers, box counts) and NGS participation (man/zone, coverage shell).

Each table is cached to parquet; in-season tables refresh once per CACHE_HOURS."""
import os
import time

import nflreadpy as nfl
import pandas as pd

from config import CACHE_DIR

CACHE_HOURS = 6
PARTICIPATION_MAX_SEASON = 2025  # NGS participation (coverage scheme) lags; not yet published for 2026


def _cached(name: str, season: int, loader):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_{season}.parquet")
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < CACHE_HOURS * 3600
    if fresh:
        return pd.read_parquet(path)
    df = loader(season).to_pandas()
    df.to_parquet(path)
    return df


def pbp(seasons):
    df = pd.concat([_cached("pbp", s, nfl.load_pbp) for s in seasons], ignore_index=True)
    return df[df["season_type"] == "REG"]


def player_stats(seasons):
    df = pd.concat([_cached("pstats", s, nfl.load_player_stats) for s in seasons], ignore_index=True)
    return df[df["season_type"] == "REG"]


def schedules(season):
    return _cached("sched", season, nfl.load_schedules)


def injuries(season):
    try:
        return _cached("inj", season, nfl.load_injuries)
    except Exception:
        return pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "report_status"])


def ftn(seasons):
    out = []
    for s in seasons:
        try:
            out.append(_cached("ftn", s, nfl.load_ftn_charting))
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def participation(seasons):
    out = []
    for s in seasons:
        if s > PARTICIPATION_MAX_SEASON:
            continue
        try:
            df = _cached("part", s, nfl.load_participation)
            out.append(df[["nflverse_game_id", "play_id", "defense_man_zone_type",
                           "defense_coverage_type", "number_of_pass_rushers", "was_pressure"]])
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def pbp_with_scheme(seasons):
    """Play-by-play joined to coverage scheme (man/zone) and FTN blitz counts where available."""
    df = pbp(seasons)
    part = participation(seasons)
    if len(part):
        df = df.merge(part, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
    f = ftn(seasons)
    if len(f):
        df = df.merge(f[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_defense_box"]],
                      left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"],
                      how="left", suffixes=("", "_ftn"))
    return df


def snaps(seasons):
    """Offensive snap counts keyed by gsis player id (appearance + snap share)."""
    df = pd.concat([_cached("snaps", s, nfl.load_snap_counts) for s in seasons], ignore_index=True)
    df = df[df["game_type"] == "REG"]
    players = _cached("players", 0, lambda _: nfl.load_players())[["gsis_id", "pfr_id"]].dropna()
    df = df.merge(players, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return df[["game_id", "season", "week", "team", "gsis_id", "offense_snaps", "offense_pct"]]


def player_info():
    """gsis_id -> full display name + position (used for position groups and sportsbook name matching)."""
    p = _cached("players", 0, lambda _: nfl.load_players())
    return p[["gsis_id", "display_name", "position"]].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")


# ---------------------------------------------------------------- weather
# (lat, lon, roof) — roof: "dome" (never weather), "retractable" (treated as closed unless the
# schedule says open), "outdoors".
STADIUMS = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4008, "retractable"),
    "BAL": (39.2780, -76.6227, "outdoors"), "BUF": (42.7738, -78.7870, "outdoors"),
    "CAR": (35.2258, -80.8528, "outdoors"), "CHI": (41.8623, -87.6167, "outdoors"),
    "CIN": (39.0955, -84.5161, "outdoors"), "CLE": (41.5061, -81.6995, "outdoors"),
    "DAL": (32.7473, -97.0945, "retractable"), "DEN": (39.7439, -105.0201, "outdoors"),
    "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "outdoors"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "outdoors"), "KC": (39.0489, -94.4839, "outdoors"),
    "LV": (36.0909, -115.1833, "dome"), "LA": (33.9535, -118.3392, "dome"),
    "LAC": (33.9535, -118.3392, "dome"), "MIA": (25.9580, -80.2389, "outdoors"),
    "MIN": (44.9737, -93.2577, "dome"), "NE": (42.0909, -71.2643, "outdoors"),
    "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8135, -74.0745, "outdoors"),
    "NYJ": (40.8135, -74.0745, "outdoors"), "PHI": (39.9008, -75.1675, "outdoors"),
    "PIT": (40.4468, -80.0158, "outdoors"), "SF": (37.4030, -121.9700, "outdoors"),
    "SEA": (47.5952, -122.3316, "outdoors"), "TB": (27.9759, -82.5033, "outdoors"),
    "TEN": (36.1665, -86.7713, "outdoors"), "WAS": (38.9078, -76.8645, "outdoors"),
}
PRECIP_WORDS = "rain|snow|shower|drizzle|sleet"


def weather_forecast(home, gameday, gametime):
    """Kickoff-hour forecast from Open-Meteo (free, no key). Returns
    {"outdoor", "temp" (F), "wind" (mph), "precip" (bool)} or None on failure.
    gameday 'YYYY-MM-DD', gametime 'HH:MM' Eastern (nflverse schedule format)."""
    import datetime as dt
    from zoneinfo import ZoneInfo
    import requests
    lat, lon, roof = STADIUMS[home]
    if roof != "outdoors":
        return {"outdoor": False}
    try:
        kick = dt.datetime.fromisoformat(f"{gameday}T{gametime}").replace(tzinfo=ZoneInfo("America/New_York"))
        kick = kick.astimezone(dt.timezone.utc)
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation,wind_speed_10m",
                  "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "timezone": "UTC",
                  "start_date": kick.date().isoformat(),
                  "end_date": (kick + dt.timedelta(hours=4)).date().isoformat()}
        r = None
        for attempt, wait in enumerate((0, 2, 5)):      # the free forecast API is often slow, so be patient
            if wait:
                import time as _t
                _t.sleep(wait)
            try:
                r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=40, params=params)
                r.raise_for_status()
                break
            except Exception:
                r = None
                if attempt == 2:
                    raise
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["time"] = pd.to_datetime(h["time"], utc=True)
        w = h[(h["time"] >= kick.replace(minute=0)) & (h["time"] < kick + dt.timedelta(hours=3))]
        return {"outdoor": True, "temp": float(w["temperature_2m"].mean()),
                "wind": float(w["wind_speed_10m"].mean()), "precip": bool(w["precipitation"].sum() >= 1.0)}
    except Exception as ex:
        print(f"[weather] forecast unavailable ({ex}); running without weather")
        return None


def weather_from_schedule(g, pbp_weather_text=None):
    """Observed weather for a completed game (nflverse schedule + pbp weather text)."""
    if g is None or pd.isna(g.get("temp")) and pd.isna(g.get("wind")):
        return None
    outdoor = g.get("roof") in ("outdoors", "open")
    txt = (pbp_weather_text or "").lower()
    import re
    return {"outdoor": outdoor, "temp": g.get("temp"), "wind": g.get("wind") if pd.notna(g.get("wind")) else 0,
            "precip": bool(re.search(PRECIP_WORDS, txt))}


def roster_status(season):
    """Weekly official roster status: ACT active, RES injured reserve, EXE commissioner exempt,
    SUS suspended, CUT released, RET retired, DEV practice squad, INA game-day inactive."""
    df = _cached("rostw", season, nfl.load_rosters_weekly)
    return df[["season", "week", "team", "gsis_id", "full_name", "status"]]
