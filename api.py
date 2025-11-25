"""Lightweight FastAPI wrapper around the NHL analyzer logic.

Endpoints:
- GET /health
- GET /nhl/matchup?home=TOR&away=BOS[&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD]
  Fetches recent games (default last 45 days), builds team profiles, runs lean + EV,
  and returns JSON suitable for a front-end.
"""

import logging
import os
import json
import threading
import time as time_module
import shutil
from datetime import date, datetime, timedelta, time, timezone
from pathlib import Path
from typing import Optional

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials as firebase_credentials
except Exception:
    firebase_admin = None
    firebase_auth = None
    firebase_credentials = None
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Header, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_sources import (
    CACHE_DIR,
    load_games_from_espn_date_range,
    load_current_odds_for_matchup,
    load_schedule_for_date,
    TEAM_CODE_TO_ID,
    read_pick_cache,
    write_pick_cache,
    load_injuries_rotowire,
    load_moneypuck_player_stats,
    load_projected_goalies,
    get_probable_goalie,
    get_team_adv_stats,
    load_nhl_team_stats,
    load_moneypuck_team_stats,
    load_moneypuck_goalie_stats,
    load_team_roster,
    _normalize_name_key,
)
from nhl_analyzer import (
    get_team_profile,
    implied_prob_american,
    profit_on_win_for_1_unit,
    model_probs_from_scores,
    confidence_grade,
)
from nfl_analyzer import (
    lean_matchup as nfl_lean_matchup,
    calculate_stats as nfl_calculate_stats,
)
from nfl_data_sources import (
    load_games_from_espn_date_range as load_games_nfl_range,
    load_schedule_for_date as load_schedule_nfl,
    load_current_odds_for_matchup as load_odds_nfl,
    load_games_from_espn_scoreboard as load_games_nfl_today,
)

app = FastAPI(title="NHL Betting Analyzer API", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nhl_analyzer_api")

SPORTS = {
    "nhl": {
        "title": "NHL",
        "teams": sorted(TEAM_CODE_TO_ID.keys()),
    },
    "nfl": {
        "title": "NFL",
        "teams": [
            "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
            "DET","GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN",
            "NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
        ],
    },
}

ALIASES = {
    "ARI": "UTA",  # legacy Arizona to Utah
}

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/New_York")
PICK_GATE_HOUR = int(os.getenv("PICK_GATE_HOUR", "12"))
PICK_GATE_MINUTE = int(os.getenv("PICK_GATE_MINUTE", "0"))
DEFAULT_LOOKBACK_DAYS = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "45"))
API_ADMIN_TOKEN = os.getenv("API_ADMIN_TOKEN")
MANUAL_OVERRIDES_PATH = CACHE_DIR / "manual_overrides.json"
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", CACHE_DIR / "snapshots"))
AUTO_SNAPSHOT_ENABLED = os.getenv("AUTO_SNAPSHOT_ENABLED", "true").lower() == "true"
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_ENFORCE_AUTH = os.getenv("FIREBASE_ENFORCE_AUTH", "false").lower() == "true"
FIREBASE_REQUIRE_PRO_FOR_PICK = os.getenv("FIREBASE_REQUIRE_PRO_FOR_PICK", "false").lower() == "true"
FIREBASE_PRO_CLAIM_VALUE = os.getenv("FIREBASE_PRO_CLAIM_VALUE", "pro")
FIREBASE_APP = None


def _cleanup_daily_cache():
    """
    On startup, clear projected goalie cache and all matchup snapshots.
    Prevents stale projected starters/snapshots after redeploys.
    """
    try:
        for path in CACHE_DIR.glob("projected_goalies_*.json"):
            try:
                path.unlink()
            except Exception:
                continue
        snap_dir = CACHE_DIR / "snapshots" / "matchups"
        if snap_dir.exists():
            shutil.rmtree(snap_dir, ignore_errors=True)
    except Exception:
        logger.warning("Cache cleanup skipped (non-fatal).")


_cleanup_daily_cache()


def _init_firebase():
    """Initialize Firebase admin if credentials are provided."""
    global FIREBASE_APP
    if FIREBASE_APP or not FIREBASE_CREDENTIALS_PATH or not firebase_admin or not firebase_credentials:
        return
    try:
        cred = firebase_credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        FIREBASE_APP = firebase_admin.initialize_app(cred, {"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None)
        logger.info("Firebase initialized")
    except Exception as e:
        logger.warning("Firebase init failed: %s", e)
        FIREBASE_APP = None


_init_firebase()


# ---------- Helpers ----------

def _parse_date_or_400(value: Optional[str], fallback: date, label: str) -> date:
    """
    Parse YYYY-MM-DD or raise a 400 with a clear message. Returns fallback if value is None.
    """
    if value is None:
        return fallback
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be YYYY-MM-DD (got '{value}').",
        )


def _validate_team_or_400(team: str, label: str) -> str:
    """
    Ensure team is 2-4 letters and in the known NHL code set.
    """
    cleaned = (team or "").strip().upper()
    if not cleaned.isalpha() or not (2 <= len(cleaned) <= 4):
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a team abbreviation (2-4 letters).",
        )
    cleaned = ALIASES.get(cleaned, cleaned)

    if cleaned not in TEAM_CODE_TO_ID:
        raise HTTPException(
            status_code=400,
            detail=f"{label} '{cleaned}' is not a known NHL team code.",
        )
    return cleaned


def _now_in_zone():
    """Timezone-aware now in the configured app timezone."""
    try:
        zone = ZoneInfo(APP_TIMEZONE)
    except Exception:
        try:
            zone = ZoneInfo("UTC")
        except Exception:
            # Fall back to naive UTC if tzdata is missing
            return datetime.now(timezone.utc)
    return datetime.now(zone)


def _decode_firebase_token(id_token: str) -> Optional[dict]:
    if not FIREBASE_APP or not firebase_auth:
        return None
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        logger.info("Firebase token verify failed: %s", e)
        return None


def _extract_authorization_token(header_val: Optional[str]) -> Optional[str]:
    if not header_val:
        return None
    try:
        parts = header_val.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    except Exception:
        return None
    return None


async def current_user_optional(authorization: str = Header(None)):
    """
    Return decoded firebase user dict or None.
    If FIREBASE_ENFORCE_AUTH is True and verification fails, raise 401.
    """
    if not FIREBASE_APP:
        return None
    token = _extract_authorization_token(authorization)
    decoded = _decode_firebase_token(token) if token else None
    if FIREBASE_ENFORCE_AUTH and not decoded:
        raise HTTPException(status_code=401, detail="Auth required")
    return decoded


def _is_pro(user: Optional[dict]) -> bool:
    if not user:
        return False
    claims = user.get("claims") or user
    tier = claims.get("tier") or claims.get("plan") or claims.get("role")
    return tier == FIREBASE_PRO_CLAIM_VALUE


def _read_overrides():
    if not MANUAL_OVERRIDES_PATH.exists():
        return {"injuries": [], "goalies": []}
    try:
        with MANUAL_OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"injuries": [], "goalies": []}
        data.setdefault("injuries", [])
        data.setdefault("goalies", [])
        return data
    except Exception:
        return {"injuries": [], "goalies": []}


def _write_overrides(data: dict):
    try:
        MANUAL_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MANUAL_OVERRIDES_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _require_admin(request: Request):
    if not API_ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token not configured.")
    token = request.headers.get("x-admin-token") or request.headers.get("X-Admin-Token")
    if token != API_ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _matchup_snapshot_path(date_str: str, home: str, away: str) -> Path:
    safe_home = (home or "").upper()
    safe_away = (away or "").upper()
    return SNAPSHOT_DIR / "matchups" / date_str / f"{safe_away}_at_{safe_home}.json"


def _read_matchup_snapshot(date_str: str, home: str, away: str) -> Optional[dict]:
    path = _matchup_snapshot_path(date_str, home, away)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_matchup_snapshot(date_str: str, home: str, away: str, data: dict) -> None:
    path = _matchup_snapshot_path(date_str, home, away)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _read_pick_snapshot(date_str: str) -> Optional[dict]:
    path = SNAPSHOT_DIR / "pick" / f"{date_str}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_pick_snapshot(date_str: str, data: dict) -> None:
    path = SNAPSHOT_DIR / "pick" / f"{date_str}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _injuries_by_team(force_refresh: bool = False, player_stats: Optional[dict] = None) -> dict:
    data = load_injuries_rotowire(force_refresh=force_refresh, player_stats=player_stats) or {}
    items = data.get("items") or []
    by_team = {}
    for entry in items:
        team = entry.get("team")
        if not team:
            continue
        by_team.setdefault(team, []).append(entry)
    return {
        "source": data.get("source"),
        "fetched_at": data.get("fetched_at"),
        "items": by_team,
    }


def _snapshot_all_matchups_for_date(
    date_str: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    force_refresh: bool = False,
) -> dict:
    """Compute and write snapshots for all matchups on a given date."""
    player_stats = load_moneypuck_player_stats(force_refresh=force_refresh)
    schedule = load_schedule_for_date(date_str, force_refresh=force_refresh)
    projected_goalies = load_projected_goalies(date_str, force_refresh=force_refresh)
    now = _now_in_zone()
    end_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_dt = end_dt - timedelta(days=lookback_days)
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    injuries_blob = _injuries_by_team(force_refresh=force_refresh, player_stats=player_stats)
    injuries_by_team = (injuries_blob or {}).get("items") or {}
    injuries_meta = {
        "source": (injuries_blob or {}).get("source"),
        "fetched_at": (injuries_blob or {}).get("fetched_at"),
    }

    if not schedule:
        return {
            "date": date_str,
            "games": [],
            "computed_at": now.isoformat(),
            "status": "no_schedule",
        }

    try:
        games = load_games_from_espn_date_range(start_str, end_str, force_refresh=force_refresh)
    except Exception as e:
        logger.exception("Snapshot: data fetch failed for %s-%s | %s", start_str, end_str, e)
        raise

    results = []
    for g in schedule:
        home = g["home_team"]
        away = g["away_team"]
        try:
            home_inj_list = injuries_by_team.get(home)
            away_inj_list = injuries_by_team.get(away)
            home_score, away_score, reasons = _lean_scores(
                home,
                away,
                games,
                force_refresh=force_refresh,
                injuries_home=home_inj_list,
                injuries_away=away_inj_list,
                projected_goalies=projected_goalies.get("items") if projected_goalies else None,
            )
        except Exception:
            continue

        model_home_prob, model_away_prob = model_probs_from_scores(home_score, away_score, temperature=1.15)
        ev = _ev_block(home, away, home_score, away_score, force_refresh=force_refresh)

        diff = home_score - away_score
        if diff > 1.5:
            side_lean = f"{home} ML (strong lean)"
        elif diff > 0.5:
            side_lean = f"{home} ML (slight lean)"
        elif diff < -1.5:
            side_lean = f"{away} ML (strong lean)"
        elif diff < -0.5:
            side_lean = f"{away} ML (slight lean)"
        else:
            side_lean = "No clear edge"

        # Build response payload identical to /nhl/matchup with snapshot flags
        response = {
            "params": {
                "home": home,
                "away": away,
                "start_date": start_str,
                "end_date": end_str,
                "force_refresh": force_refresh,
                "use_snapshot": True,
                "include_injuries": True,
            },
            "side": {
                "home_score": home_score,
                "away_score": away_score,
                "lean": side_lean,
                "reasons": reasons,
                "goalie": reasons.get("goalie"),
            },
            "model_prob": {"home": model_home_prob, "away": model_away_prob},
            "ev": ev,
            "profiles": {
                "home": _profile_summary(get_team_profile(home, games, today=end_dt)),
                "away": _profile_summary(get_team_profile(away, games, today=end_dt)),
            },
            "computed_at": now.isoformat(),
            "snapshot_used": True,
        }
        response["injuries"] = {
            "home": injuries_by_team.get(home),
            "away": injuries_by_team.get(away),
            "source": injuries_meta.get("source"),
            "fetched_at": injuries_meta.get("fetched_at"),
        }
        try:
            _write_matchup_snapshot(end_str, home, away, response)
        except Exception:
            pass
        results.append(response)

    return {
        "date": date_str,
        "computed_at": now.isoformat(),
        "games": results,
        "status": "ok",
        "force_refresh": force_refresh,
        "lookback_days": lookback_days,
    }


def _run_daily_snapshot_once(date_str: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS):
    """
    Internal: compute matchup snapshots for the date and POTD (observing gate time).
    """
    try:
        logger.info("[AUTO_SNAPSHOT] Starting snapshot for %s", date_str)
        snapshot = _snapshot_all_matchups_for_date(date_str, lookback_days=lookback_days, force_refresh=False)
        # Respect pick gate timing: only compute if now past gate
        now = _now_in_zone()
        gate = time(PICK_GATE_HOUR, PICK_GATE_MINUTE)
        pick_result = None
        if now.time() >= gate:
            try:
                pick_result = nhl_pick(lookback_days=lookback_days, force_refresh=False, ignore_pick_gate=False, cache=True)
            except Exception:
                logger.exception("[AUTO_SNAPSHOT] Failed to compute POTD snapshot for %s", date_str)
        else:
            logger.info("[AUTO_SNAPSHOT] Skipping POTD; gate not reached (gate=%s)", gate)
        logger.info("[AUTO_SNAPSHOT] Completed snapshot for %s | games=%s", date_str, len(snapshot.get("games", [])))
        return {"matchups": snapshot, "pick": pick_result}
    except Exception:
        logger.exception("[AUTO_SNAPSHOT] Unexpected failure for %s", date_str)
        return None


def _injury_penalty(entries):
    """
    Return (penalty, reasons) given a list of injury entries.
    Uses importance flag + position to scale.
    """
    if not entries:
        return 0.0, []

    penalty = 0.0
    reasons = []
    important_goalie = False
    important_skaters = 0
    other_skaters = 0

    for item in entries:
        pos = (item.get("position") or "").upper()
        status = (item.get("status") or "").lower()
        important = bool(item.get("important"))

        # Apply only to meaningful absences; treat out/IR/day-to-day/suspension as impacting
        if status and all(k not in status for k in ["out", "ir", "day", "susp", "injured"]):
            continue

        if important:
            if pos == "G":
                penalty += 0.6
                important_goalie = True
            else:
                penalty += 0.25
                important_skaters += 1
        else:
            penalty += 0.1
            if pos == "G":
                # non-important goalie: light bump
                reasons.append(f"Depth goalie out: {item.get('player')}")
            else:
                other_skaters += 1

    # Cap to avoid runaway
    penalty = min(penalty, 1.2)

    if important_goalie:
        reasons.append("Missing starting goalie")
    if important_skaters:
        reasons.append(f"{important_skaters} key skater(s) out")
    elif other_skaters:
        reasons.append(f"{other_skaters} depth skater(s) out")

    return penalty, reasons


GOALIE_LEAGUE_AVG_SV = 0.905  # conservative league baseline
GOALIE_SHRINK_STARTS = 10.0
GOALIE_SAMPLE_FLOOR_STARTS = 8.0
GOALIE_RATING_CAP = 0.03  # cap contribution in sv% points
GOALIE_SCORE_K = 16.0  # maps rating delta to score (0.02 -> 0.32)
GOALIE_SCORE_CAP = 0.45
GOALIE_GSAX_PER60_WEIGHT = 0.008  # converts gsax/60 to sv%-like points
GOALIE_GSAX_PER60_CAP = 1.5
GOALIE_B2B_RATING_PENALTY = 0.006  # ~3-4% win-prob swing when scaled


def _start_prob_from_status(status: Optional[str]) -> float:
    """Map projected goalie status to a start probability."""
    if not status:
        return 0.9
    s = status.lower()
    if "confirm" in s:
        return 1.0
    if "prob" in s or "likely" in s or "expect" in s:
        return 0.78
    if "question" in s or "doubt" in s:
        return 0.55
    if "possible" in s or "tbd" in s:
        return 0.65
    return 0.8


def _regress_sv(save_pct: Optional[float], games_started: Optional[float]) -> Optional[float]:
    """Shrink save% toward league average based on starts."""
    if save_pct is None:
        return None
    gs = games_started or 0.0
    weight = gs / (gs + GOALIE_SHRINK_STARTS) if gs >= 0 else 0.0
    return GOALIE_LEAGUE_AVG_SV + weight * (save_pct - GOALIE_LEAGUE_AVG_SV)


def _goalie_rating_for_team(
    team_code: str,
    rest_info: dict,
    projected_goalies: Optional[dict],
    mp_goalies: Optional[dict],
    force_refresh: bool = False,
):
    """
    Build a goalie rating offset (vs league avg) for a team.
    Uses projected starter when available; otherwise the probable goalie from roster stats.
    Returns (rating, reasons, detail).
    """
    proj_entry = None
    if projected_goalies:
        proj_entry = projected_goalies.get(team_code)
    probable = get_probable_goalie(team_code, force_refresh=force_refresh, projected=projected_goalies)

    # Fallback: if no probable goalie found but a projected name exists, synthesize a stub to allow rating
    if not probable and proj_entry and proj_entry.get("goalie"):
        probable = {
            "id": None,
            "name": proj_entry.get("goalie"),
            "name_norm": _normalize_name_key(proj_entry.get("goalie")),
            "stats": {},
            "projected": True,
        }
    if not probable:
        return 0.0, ["Goalie data unavailable"], {"start_prob": 0.0}

    stats = probable.get("stats") or {}
    mp = None
    try:
        name_norm = probable.get("name_norm")
        mp = mp_goalies.get(name_norm) if mp_goalies else None
    except Exception:
        mp = None

    save_pct = (
        (mp or {}).get("save_pct")
        if (mp or {}).get("save_pct") is not None
        else stats.get("save_pct")
    )
    games_started = (
        stats.get("games_started")
        or stats.get("games_played")
        or (mp or {}).get("games_started")
        or (mp or {}).get("games_played")
    )
    reg_sv = _regress_sv(save_pct, games_started)

    gsax_per60 = None
    if mp:
        gsax_per60 = mp.get("gsax_per60")
        if gsax_per60 is None and mp.get("gsax") is not None and mp.get("icetime"):
            try:
                gsax_per60 = (mp["gsax"] / mp["icetime"]) * 60.0
            except Exception:
                gsax_per60 = None
    gsax_component = 0.0
    if gsax_per60 is not None:
        capped = max(min(gsax_per60, GOALIE_GSAX_PER60_CAP), -GOALIE_GSAX_PER60_CAP)
        gsax_component = capped * GOALIE_GSAX_PER60_WEIGHT

    rating = 0.0
    reasons = []
    details = {
        "goalie": probable.get("name"),
        "save_pct": save_pct,
        "regressed_sv": reg_sv,
        "games_started": games_started,
        "gsax_per60": gsax_per60,
        "gsax_component": gsax_component,
        "rest_penalty": 0.0,
    }

    if reg_sv is not None:
        rating += reg_sv - GOALIE_LEAGUE_AVG_SV
        reasons.append(f"{probable.get('name')} regressed sv% input")
    if gsax_component:
        rating += gsax_component
        reasons.append("GSAx contribution")

    # Small-sample shrink
    sample_weight = 1.0
    if games_started is not None and games_started < GOALIE_SAMPLE_FLOOR_STARTS:
        sample_weight = games_started / GOALIE_SAMPLE_FLOOR_STARTS
        rating *= sample_weight
        reasons.append("Small-sample shrink applied")
    details["sample_weight"] = sample_weight

    # Rest penalty (goalie-specific B2B)
    if rest_info.get("is_back_to_back"):
        rating -= GOALIE_B2B_RATING_PENALTY
        details["rest_penalty"] = GOALIE_B2B_RATING_PENALTY
        reasons.append("Zero-rest goalie penalty")

    # Start probability blend
    start_prob = _start_prob_from_status(proj_entry.get("status") if proj_entry else None)
    rating *= start_prob
    details["start_prob"] = start_prob
    details["raw_rating_capped"] = rating
    if start_prob < 1.0:
        reasons.append(f"Start prob {start_prob:.0%} applied")

    # Cap rating to prevent runaway impact
    rating = max(min(rating, GOALIE_RATING_CAP), -GOALIE_RATING_CAP)
    details["rating_capped"] = rating
    details["projected_status"] = proj_entry.get("status") if proj_entry else None
    return rating, reasons, details


def _lean_scores(
    home_team: str,
    away_team: str,
    games,
    force_refresh: bool = False,
    injuries_home=None,
    injuries_away=None,
    projected_goalies=None,
) -> Tuple[float, float, dict]:
    """Pure version of the lean logic: returns scores + reasons."""
    today = _now_in_zone().date()
    home_profile = get_team_profile(home_team, games, today=today)
    away_profile = get_team_profile(away_team, games, today=today)
    adv_home = get_team_adv_stats(home_team, force_refresh=force_refresh) or {}
    adv_away = get_team_adv_stats(away_team, force_refresh=force_refresh) or {}
    try:
        mp_goalies = load_moneypuck_goalie_stats(force_refresh=force_refresh)
    except Exception:
        mp_goalies = {}

    if home_profile is None or away_profile is None:
        raise ValueError("Not enough data to build team profiles.")

    def win_pct(stats):
        g = stats["games"]
        return stats["wins"] / g if g > 0 else 0.0

    home_score = 0.0
    away_score = 0.0
    reasons_home = []
    reasons_away = []

    # Recency: use last 10 record as a single modest factor
    home_win10 = win_pct(home_profile["last10"])
    away_win10 = win_pct(away_profile["last10"])
    home_score += home_win10 * 1.0
    away_score += away_win10 * 1.0
    if home_win10 > away_win10 + 0.1:
        reasons_home.append("Better recent form (last 10)")
    elif away_win10 > home_win10 + 0.1:
        reasons_away.append("Better recent form (last 10)")

    # Streaks: light and capped
    for profile, is_home_flag in ((home_profile, True), (away_profile, False)):
        streak = profile["streak"]
        length = streak["length"]
        stype = streak["type"]
        if length > 0 and stype:
            delta_raw = 0.25 * length if stype == "W" else -0.25 * length
            delta = max(min(delta_raw, 0.75), -0.75)
            if is_home_flag:
                home_score += delta
                if stype == "W" and length >= 3:
                    reasons_home.append(f"On a {length}-game winning streak")
                if stype == "L" and length >= 3:
                    reasons_home.append(f"On a {length}-game losing streak")
            else:
                away_score += delta
                if stype == "W" and length >= 3:
                    reasons_away.append(f"On a {length}-game winning streak")
                if stype == "L" and length >= 3:
                    reasons_away.append(f"On a {length}-game losing streak")

    # Flat home-ice bonus (softened)
    home_score += 0.25
    reasons_home.append("Home-ice advantage")

    # Home/Road splits
    def pct(stats):
        return stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

    home_home = home_profile["home"]
    away_road = away_profile["away"]
    home_home_win_pct = pct(home_home)
    away_road_win_pct = pct(away_road)
    if home_home["games"] > 0:
        home_score += home_home_win_pct * 0.8
        if home_home_win_pct >= 0.55:
            reasons_home.append("Strong home record")
        elif home_home_win_pct <= 0.45:
            reasons_away.append("Home team weaker at home")
    if away_road["games"] > 0:
        away_score += away_road_win_pct * 0.8
        if away_road_win_pct >= 0.55:
            reasons_away.append("Strong road record")
        elif away_road_win_pct <= 0.45:
            reasons_home.append("Away team weaker on the road")

    # Defensive edge (lower GA last 10) with modest weight
    home_ga10 = home_profile["last10"]["avg_against"]
    away_ga10 = away_profile["last10"]["avg_against"]
    if home_ga10 + 0.5 < away_ga10:
        home_score += 0.2
        reasons_home.append("Better defensive numbers (fewer goals against)")
    elif away_ga10 + 0.5 < home_ga10:
        away_score += 0.2
        reasons_away.append("Better defensive numbers (fewer goals against)")

    # Advanced stats edge (xG/HDCF) from MoneyPuck if available
    xgf_pct_home = adv_home.get("xgf_pct")
    xgf_pct_away = adv_away.get("xgf_pct")
    if xgf_pct_home is not None and xgf_pct_away is not None:
        delta = xgf_pct_home - xgf_pct_away
        if delta >= 3.0:
            home_score += 0.55
            reasons_home.append("Better xG share (season-to-date)")
        elif delta <= -3.0:
            away_score += 0.55
            reasons_away.append("Better xG share (season-to-date)")

    # Special teams edge (PP/PK) if meaningful
    pp_home = adv_home.get("pp")
    pk_home = adv_home.get("pk")
    pp_away = adv_away.get("pp")
    pk_away = adv_away.get("pk")
    if pp_home and pk_home and pp_away and pk_away:
        # simple net special teams strength
        net_home = (pp_home or 0) - (100 - (pk_home or 0))
        net_away = (pp_away or 0) - (100 - (pk_away or 0))
        delta = net_home - net_away
        if delta >= 5:
            home_score += 0.15
            reasons_home.append("Special teams edge")
        elif delta <= -5:
            away_score += 0.15
            reasons_away.append("Special teams edge")

    # Rest / back-to-back
    rest_home = home_profile.get("rest") or {}
    rest_away = away_profile.get("rest") or {}
    home_days = rest_home.get("days_since_last")
    away_days = rest_away.get("days_since_last")
    if rest_home.get("is_back_to_back") and not rest_away.get("is_back_to_back"):
        home_score -= 0.40
        reasons_away.append("Home on back-to-back; away more rested")
    if rest_away.get("is_back_to_back") and not rest_home.get("is_back_to_back"):
        away_score -= 0.40
        reasons_home.append("Away on back-to-back; home more rested")
    if home_days is not None and away_days is not None:
        diff = home_days - away_days
        if diff >= 2:
            home_score += 0.20
            reasons_home.append(f"More rest ({home_days}d vs {away_days}d)")
        elif diff <= -2:
            away_score += 0.20
            reasons_away.append(f"More rest ({away_days}d vs {home_days}d)")

    # Goalie edge (continuous + capped)
    goalie_meta = {}
    try:
        home_rating, home_g_reasons, home_goalie_meta = _goalie_rating_for_team(
            home_team,
            rest_home,
            projected_goalies,
            mp_goalies,
            force_refresh=force_refresh,
        )
        away_rating, away_g_reasons, away_goalie_meta = _goalie_rating_for_team(
            away_team,
            rest_away,
            projected_goalies,
            mp_goalies,
            force_refresh=force_refresh,
        )
        rating_diff = home_rating - away_rating
        goalie_score = max(min(rating_diff * GOALIE_SCORE_K, GOALIE_SCORE_CAP), -GOALIE_SCORE_CAP)
        goalie_meta = {
            "home_rating": home_rating,
            "away_rating": away_rating,
            "rating_diff": rating_diff,
            "score_contrib": goalie_score,
            "home": home_goalie_meta,
            "away": away_goalie_meta,
        }
        # Attach granular reasons to the appropriate side
        if home_g_reasons:
            reasons_home.extend([f"Goalie: {r}" for r in home_g_reasons])
        if away_g_reasons:
            reasons_away.extend([f"Goalie: {r}" for r in away_g_reasons])

        if goalie_score > 0:
            home_score += goalie_score
            reasons_home.append("Goalie edge (scaled/capped)")
        elif goalie_score < 0:
            away_score += abs(goalie_score)
            reasons_away.append("Goalie edge (scaled/capped)")
    except Exception as e:
        logger.exception("Goalie rating failed for %s vs %s", home_team, away_team)
        goalie_meta = {"error": "goalie_rating_failed", "detail": str(e)}

    return home_score, away_score, {
        "home_reasons": reasons_home,
        "away_reasons": reasons_away,
        "goalie": goalie_meta,
    }


def _ev_block(home_team, away_team, home_score, away_score, force_refresh=False):
    """Return EV calculations if odds available; else None."""
    odds = load_current_odds_for_matchup(
        home_team, away_team, force_refresh=force_refresh
    )
    if odds is None:
        logger.info("[EV] No odds for %s vs %s", home_team, away_team)
        return None

    home_odds = odds.get("home_ml")
    away_odds = odds.get("away_ml")
    if home_odds is None or away_odds is None:
        logger.info("[EV] Odds missing fields for %s vs %s", home_team, away_team)
        return None

    p_home_model, p_away_model = model_probs_from_scores(home_score, away_score, temperature=1.15)
    p_home_market = implied_prob_american(home_odds)
    p_away_market = implied_prob_american(away_odds)
    home_profit = profit_on_win_for_1_unit(home_odds)
    away_profit = profit_on_win_for_1_unit(away_odds)

    home_ev = p_home_model * home_profit - (1 - p_home_model)
    away_ev = p_away_model * away_profit - (1 - p_away_model)
    home_edge_pct = (p_home_model - p_home_market) * 100.0
    away_edge_pct = (p_away_model - p_away_market) * 100.0

    return {
        "odds": {"home_ml": home_odds, "away_ml": away_odds},
        "model_prob": {"home": p_home_model, "away": p_away_model},
        "market_prob": {"home": p_home_market, "away": p_away_market},
        "edge_pct": {"home": home_edge_pct, "away": away_edge_pct},
        "ev_units": {"home": home_ev, "away": away_ev},
        "grades": {
            "home": confidence_grade(home_edge_pct, home_ev),
            "away": confidence_grade(away_edge_pct, away_ev),
        },
    }


def _profile_summary(profile):
    """Select only summary fields for API output."""
    return {
        "team": profile["team"],
        "full": profile["full"],
        "last5": profile["last5"],
        "last10": profile["last10"],
        "home": profile["home"],
        "away": profile["away"],
        "high_scoring_last5": profile["high_scoring_last5"],
        "streak": profile["streak"],
        "rest": profile.get("rest"),
    }


def _team_stats_snapshot(force_refresh: bool = False):
    nhl_stats = load_nhl_team_stats(force_refresh=force_refresh)
    adv_stats = load_moneypuck_team_stats(force_refresh=force_refresh)
    teams = []
    all_codes = set(nhl_stats.keys()) | set(adv_stats.keys())
    for code in sorted(all_codes):
        teams.append(
            {
                "team": code,
                "nhl": nhl_stats.get(code) or {},
                "adv": adv_stats.get(code) or {},
            }
        )
    return teams


class InjuryEntry(BaseModel):
    team: str
    player: str
    status: Optional[str] = None
    note: Optional[str] = None
    probable_goalie: Optional[bool] = None
    updated_at: Optional[str] = None


class GoalieEntry(BaseModel):
    team: str
    goalie: str
    expected_starter: bool = False
    updated_at: Optional[str] = None


class OverridePayload(BaseModel):
    injuries: list[InjuryEntry] = []
    goalies: list[GoalieEntry] = []

# ---------- Pick of the Day helpers ----------

def _pick_candidate_from_ev(home, away, ev, reasons, slate_size: int = 1):
    """Return the best side candidate from an EV block or None, applying filters."""
    if ev is None:
        return None

    # Dynamic thresholds based on slate size
    if slate_size <= 3:
        min_ev = 0.05
        min_edge = 2.0
    elif slate_size <= 8:
        min_ev = 0.08
        min_edge = 3.0
    else:
        min_ev = 0.10
        min_edge = 4.0
    min_model_prob = 0.55
    max_juice = 150  # cap at -150/+150

    goalie_meta = reasons.get("goalie") if isinstance(reasons, dict) else None

    best = None
    for side_key, team in (("home", home), ("away", away)):
        ev_units = ev["ev_units"][side_key]
        edge_pct = ev["edge_pct"][side_key]
        grade = ev["grades"][side_key]
        model_prob = ev["model_prob"][side_key]
        market_prob = ev["market_prob"][side_key]
        odds = ev["odds"][f"{side_key}_ml"]
        side_reasons = []
        try:
            if isinstance(reasons, dict):
                if side_key == "home":
                    side_reasons = reasons.get("home_reasons") or []
                else:
                    side_reasons = reasons.get("away_reasons") or []
        except Exception:
            side_reasons = []

        # Odds cap
        if odds is None or odds < -max_juice or odds > max_juice:
            continue
        # Filters for a viable pick
        if ev_units < min_ev:
            continue
        if edge_pct < min_edge:
            continue
        if model_prob < min_model_prob:
            continue
        if grade == "No Bet":
            continue

        # Skip if goalie data missing/uncertain for the chosen side
        if goalie_meta and isinstance(goalie_meta, dict):
            if goalie_meta.get("error"):
                continue
            side_goalie = goalie_meta.get(side_key)
            if side_goalie:
                start_prob = side_goalie.get("start_prob")
                if start_prob is not None and start_prob < 0.5:
                    continue

        candidate = {
            "matchup": {"home": home, "away": away},
            "side": team,
            "side_type": side_key,
            "odds": odds,
            "ev_units": ev_units,
            "edge_pct": edge_pct,
            "grade": grade,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "reasons": side_reasons,
        }
        if best is None or candidate["ev_units"] > best["ev_units"]:
            best = candidate
    return best


# ---------- Routes ----------


@app.get("/auth/me")
def auth_me():
    """Return basic app status (auth not enforced yet)."""
    return {"auth_configured": bool(FIREBASE_CREDENTIALS_PATH)}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("REQUEST %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("RESPONSE %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sports")
def sports():
    """List supported sports and team codes (multisport groundwork)."""
    return {
        "sports": [
            {"id": key, "name": value["title"], "teams": value["teams"]}
            for key, value in SPORTS.items()
        ]
    }


@app.get("/teams")
def list_teams(sport: str = Query("nhl", description="Sport id, e.g. nhl or nfl")):
    """List team abbreviations for UI dropdowns."""
    sport = sport.lower()
    if sport not in SPORTS:
        raise HTTPException(status_code=400, detail="Unknown sport.")
    teams = SPORTS[sport]["teams"]
    return {"sport": sport, "teams": teams}


# ---------- NFL Routes ----------

@app.get("/nfl/today")
def nfl_today(force_refresh: bool = Query(False), user: Optional[dict] = Depends(current_user_optional)):
    now = _now_in_zone()
    today = now.date().isoformat()
    schedule = load_schedule_nfl(today)
    enriched = []
    for g in schedule:
        odds = load_odds_nfl(g["home_team"], g["away_team"])
        enriched.append({"home_team": g["home_team"], "away_team": g["away_team"], "odds": odds})
    return {"date": today, "games": enriched, "force_refresh": force_refresh}


@app.get("/nfl/team")
def nfl_team(
    team: str = Query(..., description="Team abbreviation, e.g. SF"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    force_refresh: bool = Query(False),
    user: Optional[dict] = Depends(current_user_optional),
):
    today = _now_in_zone().date()
    default_start = today - timedelta(days=120)
    start_dt = _parse_date_or_400(start_date, default_start, "start_date")
    end_dt = _parse_date_or_400(end_date, today, "end_date")
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()
    games = load_games_nfl_range(start_str, end_str)
    if not games:
        raise HTTPException(status_code=400, detail="No games in range")
    stats = nfl_calculate_stats(team.upper(), games)
    return {"params": {"team": team.upper(), "start_date": start_str, "end_date": end_str}, "profile": stats}


@app.get("/nfl/matchup")
def nfl_matchup(
    home: str = Query(...),
    away: str = Query(...),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    force_refresh: bool = Query(False),
    user: Optional[dict] = Depends(current_user_optional),
):
    today = _now_in_zone().date()
    default_start = today - timedelta(days=120)
    start_dt = _parse_date_or_400(start_date, default_start, "start_date")
    end_dt = _parse_date_or_400(end_date, today, "end_date")
    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()
    games = load_games_nfl_range(start_str, end_str)
    if not games:
        raise HTTPException(status_code=400, detail="No games in range")
    home_score, away_score, reasons = nfl_lean_matchup(home.upper(), away.upper(), games)
    if home_score is None:
        raise HTTPException(status_code=400, detail="Not enough data")
    model_home_prob, model_away_prob = model_probs_from_scores(home_score, away_score, temperature=1.15)
    ev = None
    odds = load_odds_nfl(home.upper(), away.upper())
    if odds and odds.get("home_ml") is not None and odds.get("away_ml") is not None:
        p_home_market = implied_prob_american(odds["home_ml"])
        p_away_market = implied_prob_american(odds["away_ml"])
        home_profit = profit_on_win_for_1_unit(odds["home_ml"])
        away_profit = profit_on_win_for_1_unit(odds["away_ml"])
        home_ev = model_home_prob * home_profit - (1 - model_home_prob)
        away_ev = model_away_prob * away_profit - (1 - model_away_prob)
        home_edge = (model_home_prob - p_home_market) * 100
        away_edge = (model_away_prob - p_away_market) * 100
        ev = {
            "odds": odds,
            "model_prob": {"home": model_home_prob, "away": model_away_prob},
            "market_prob": {"home": p_home_market, "away": p_away_market},
            "edge_pct": {"home": home_edge, "away": away_edge},
            "ev_units": {"home": home_ev, "away": away_ev},
            "grades": {"home": confidence_grade(home_edge, home_ev), "away": confidence_grade(away_edge, away_ev)},
        }
    diff = home_score - away_score
    side_lean = reasons.get("lean") or "No clear edge"
    return {
        "params": {"home": home.upper(), "away": away.upper(), "start_date": start_str, "end_date": end_str},
        "side": {"home_score": home_score, "away_score": away_score, "lean": side_lean, "reasons": reasons},
        "model_prob": {"home": model_home_prob, "away": model_away_prob},
        "ev": ev,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/nfl/pick")
def nfl_pick():
    raise HTTPException(status_code=501, detail="NFL pick not implemented yet.")


@app.get("/nhl/today")
def nhl_today(force_refresh: bool = Query(False), user: Optional[dict] = Depends(current_user_optional)):
    """Return today's NHL schedule (home/away abbreviations)."""
    now = _now_in_zone()
    today = now.date().isoformat()
    schedule = load_schedule_for_date(today, force_refresh=force_refresh)
    enriched = []
    for g in schedule:
        odds = load_current_odds_for_matchup(
            g["home_team"], g["away_team"], force_refresh=force_refresh
        )
        enriched.append(
            {
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "odds": odds,
            }
        )
    logger.info(
        "[TODAY] Games: %s (force_refresh=%s)",
        len(enriched),
        force_refresh,
    )
    return {
        "date": today,
        "timezone": APP_TIMEZONE,
        "games": [
            {"home": g["home_team"], "away": g["away_team"], "odds": g["odds"]}
            for g in enriched
        ],
        "force_refresh": force_refresh,
    }


@app.get("/", include_in_schema=False)
def root():
    """Serve the quick test UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "UI not found. Ensure static/index.html exists."}


@app.get("/admin", include_in_schema=False)
def admin_console():
    """Serve the admin overrides UI."""
    admin_path = STATIC_DIR / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)
    return {"message": "Admin UI not found. Ensure static/admin.html exists."}


@app.get("/nhl/matchup")
def nhl_matchup(
    home: str = Query(..., description="Home team abbreviation, e.g. BOS"),
    away: str = Query(..., description="Away team abbreviation, e.g. TOR"),
    start_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to 45 days ago"
    ),
    end_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to today"
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
    use_snapshot: bool = Query(
        True,
        description="If true and using default range for today, return/write a persisted daily snapshot instead of recomputing.",
    ),
    include_injuries: bool = Query(
        True,
        description="If true, include injury lists per team (rotowire).",
    ),
    user: Optional[dict] = Depends(current_user_optional),
    ):
    # Defaults: last 45 days to keep fetches fast and recent
    now = _now_in_zone()
    today = now.date()
    pick_gate = time(PICK_GATE_HOUR, PICK_GATE_MINUTE)
    default_start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    start_dt = _parse_date_or_400(start_date, default_start, "start_date")
    end_dt = _parse_date_or_400(end_date, today, "end_date")

    if end_dt < start_dt:
        raise HTTPException(
            status_code=400,
            detail="end_date must be on or after start_date.",
        )

    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    home = _validate_team_or_400(home, "home")
    away = _validate_team_or_400(away, "away")

    is_default_range_today = (
        start_date is None
        and end_date is None
        and end_dt == today
    )
    snapshot_allowed = (
        use_snapshot
        and not force_refresh
        and is_default_range_today
        and now.time() >= pick_gate
    )

    if snapshot_allowed:
        snap = _read_matchup_snapshot(end_str, home, away)
        if snap:
            snap["snapshot_used"] = True
            return snap

    try:
        games = load_games_from_espn_date_range(
            start_str, end_str, force_refresh=force_refresh
        )
    except Exception as e:
        logger.exception("Data fetch failed for %s-%s | %s", start_str, end_str, e)
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not games:
        logger.info(
            "[MATCHUP] No games for %s vs %s in range %s-%s",
            home,
            away,
            start_str,
            end_str,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "No games available for the chosen date range. "
                "Try widening the dates."
            ),
        )

    today = _now_in_zone().date()
    home_profile = get_team_profile(home, games, today=today)
    away_profile = get_team_profile(away, games, today=today)
    if home_profile is None or away_profile is None:
        logger.info(
            "[MATCHUP] Insufficient data for %s or %s in range %s-%s",
            home,
            away,
            start_str,
            end_str,
        )
        raise HTTPException(
            status_code=400,
            detail="Not enough data for one or both teams in this range.",
        )

    injuries = None
    player_stats = None
    projected_goalies = None
    if include_injuries:
        try:
            player_stats = load_moneypuck_player_stats(force_refresh=False)
            injuries = _injuries_by_team(force_refresh=force_refresh, player_stats=player_stats)
        except Exception:
            injuries = None
    try:
        projected_goalies = load_projected_goalies(end_str, force_refresh=force_refresh)
    except Exception:
        projected_goalies = None

    try:
        home_inj_list = injuries["items"].get(home) if injuries and injuries.get("items") else None
        away_inj_list = injuries["items"].get(away) if injuries and injuries.get("items") else None
        home_score, away_score, reasons = _lean_scores(
            home,
            away,
            games,
            force_refresh=force_refresh,
            injuries_home=home_inj_list,
            injuries_away=away_inj_list,
            projected_goalies=projected_goalies["items"] if projected_goalies else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Always expose model win probabilities, even if odds/EV are unavailable.
    model_home_prob, model_away_prob = model_probs_from_scores(home_score, away_score, temperature=1.15)
    model_probs = {"home": model_home_prob, "away": model_away_prob}

    ev = _ev_block(home, away, home_score, away_score, force_refresh=force_refresh)

    diff = home_score - away_score
    side_lean = None
    if diff > 1.5:
        side_lean = f"{home} ML (strong lean)"
    elif diff > 0.5:
        side_lean = f"{home} ML (slight lean)"
    elif diff < -1.5:
        side_lean = f"{away} ML (strong lean)"
    elif diff < -0.5:
        side_lean = f"{away} ML (slight lean)"
    else:
        side_lean = "No clear edge"

    response = {
        "params": {
            "home": home,
            "away": away,
            "start_date": start_str,
            "end_date": end_str,
            "force_refresh": force_refresh,
            "use_snapshot": use_snapshot,
            "include_injuries": include_injuries,
        },
        "side": {
            "home_score": home_score,
            "away_score": away_score,
            "lean": side_lean,
            "reasons": reasons,
            "goalie": reasons.get("goalie"),
        },
        "model_prob": model_probs,
        "ev": ev,
        "profiles": {
            "home": _profile_summary(home_profile),
            "away": _profile_summary(away_profile),
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_used": False,
    }
    if injuries:
        response["injuries"] = {
            "home": injuries["items"].get(home) if injuries.get("items") else None,
            "away": injuries["items"].get(away) if injuries.get("items") else None,
            "source": injuries.get("source"),
            "fetched_at": injuries.get("fetched_at"),
        }
        response.setdefault("goalies", {})  # ensure key exists for snapshots without projected info
    if projected_goalies:
        response["goalies"] = {
            "home": projected_goalies["items"].get(home) if projected_goalies.get("items") else None,
            "away": projected_goalies["items"].get(away) if projected_goalies.get("items") else None,
            "source": projected_goalies.get("source"),
            "fetched_at": projected_goalies.get("fetched_at"),
        }
    if projected_goalies:
        response["goalies"] = {
            "home": projected_goalies["items"].get(home) if projected_goalies.get("items") else None,
            "away": projected_goalies["items"].get(away) if projected_goalies.get("items") else None,
            "source": projected_goalies.get("source"),
            "fetched_at": projected_goalies.get("fetched_at"),
        }

    return response


@app.get("/nhl/team")
def nhl_team(
    team: str = Query(..., description="Team abbreviation, e.g. BOS"),
    start_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to 45 days ago"
    ),
    end_date: Optional[str] = Query(
        None, description="YYYY-MM-DD; defaults to today"
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
    user: Optional[dict] = Depends(current_user_optional),
):
    now = _now_in_zone()
    today = now.date()
    default_start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    start_dt = _parse_date_or_400(start_date, default_start, "start_date")
    end_dt = _parse_date_or_400(end_date, today, "end_date")

    if end_dt < start_dt:
        raise HTTPException(
            status_code=400,
            detail="end_date must be on or after start_date.",
        )

    start_str = start_dt.isoformat()
    end_str = end_dt.isoformat()

    try:
        games = load_games_from_espn_date_range(
            start_str, end_str, force_refresh=force_refresh
        )
    except Exception as e:
        logger.exception("Data fetch failed for team %s | %s", team, e)
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not games:
        logger.info(
            "[TEAM] No games for %s in range %s-%s",
            team,
            start_str,
            end_str,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "No games available for the chosen date range. "
                "Try widening the dates."
            ),
        )

    team = _validate_team_or_400(team, "team")
    today = _now_in_zone().date()
    profile = get_team_profile(team, games, today=today)
    if profile is None:
        raise HTTPException(
            status_code=400,
            detail="Not enough data for that team in this range.",
        )

    return {
        "params": {
            "team": team,
            "start_date": start_str,
            "end_date": end_str,
            "force_refresh": force_refresh,
        },
        "profile": _profile_summary(profile),
    }


@app.get("/nhl/pick")
def nhl_pick(
    lookback_days: int = Query(
        DEFAULT_LOOKBACK_DAYS,
        ge=7,
        le=200,
        description="Days of history to use for profiles (min 7, max 200).",
    ),
    force_refresh: bool = Query(
        False,
        description="If true, bypass cached ESPN responses for this request.",
    ),
    ignore_pick_gate: bool = Query(
        False,
        description="If true, skip the noon gating (testing/internal).",
    ),
    cache: bool = Query(
        True,
        description="If true, returns cached pick for the day when available.",
    ),
    user: Optional[dict] = Depends(current_user_optional),
):
    if FIREBASE_ENFORCE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Auth required")
    if FIREBASE_REQUIRE_PRO_FOR_PICK and not _is_pro(user):
        raise HTTPException(status_code=403, detail="Pro tier required")
    now = _now_in_zone()
    today = now.date()
    start_dt = today - timedelta(days=lookback_days)
    start_str = start_dt.isoformat()
    end_str = today.isoformat()

    pick_gate = time(PICK_GATE_HOUR, PICK_GATE_MINUTE)
    if cache and not force_refresh:
        cached = read_pick_cache(end_str)
        if cached:
            cached["cached"] = True
            return cached
        snap = _read_pick_snapshot(end_str)
        if snap:
            snap["cached"] = True
            snap["snapshot_used"] = True
            return snap

    if not ignore_pick_gate and now.time() < pick_gate:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": f"Pick available after {pick_gate.strftime('%I:%M %p')} {APP_TIMEZONE}.",
            "candidates": [],
            "params": {
                "lookback_days": lookback_days,
                "force_refresh": force_refresh,
                "ignore_pick_gate": ignore_pick_gate,
            },
        }

    schedule = load_schedule_for_date(end_str, force_refresh=force_refresh)
    if not schedule:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": "No games scheduled today.",
            "candidates": [],
            "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
        }

    try:
        games = load_games_from_espn_date_range(
            start_str, end_str, force_refresh=force_refresh
        )
    except Exception as e:
        logger.exception("Data fetch failed for pick of day %s-%s | %s", start_str, end_str, e)
        raise HTTPException(status_code=502, detail=f"Data fetch failed: {e}")

    if not games:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": "No historical games available to build profiles.",
            "candidates": [],
            "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
        }

    # Load projected goalies for uncertainty filtering
    try:
        projected_goalies = load_projected_goalies(end_str, force_refresh=force_refresh)
    except Exception:
        projected_goalies = None

    candidates = []
    for g in schedule:
        home = g["home_team"]
        away = g["away_team"]
        try:
            home_score, away_score, reasons = _lean_scores(
                home,
                away,
                games,
                force_refresh=force_refresh,
                projected_goalies=projected_goalies["items"] if projected_goalies else None,
            )
        except ValueError:
            continue

        ev = _ev_block(home, away, home_score, away_score, force_refresh=force_refresh)
        candidate = _pick_candidate_from_ev(home, away, ev, reasons, slate_size=len(schedule))
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return {
            "date": end_str,
            "timezone": APP_TIMEZONE,
            "pick": None,
            "reason": "No qualifying pick met the thresholds.",
            "candidates": [],
            "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
        }

    best = max(candidates, key=lambda c: c["ev_units"])
    result = {
        "date": end_str,
    "timezone": APP_TIMEZONE,
    "pick": best,
    "candidates": candidates,
    "params": {"lookback_days": lookback_days, "force_refresh": force_refresh},
    }
    if cache:
        try:
            write_pick_cache(end_str, result)
        except Exception:
            pass
        try:
            result_with_flag = dict(result)
            result_with_flag["snapshot_used"] = True
            _write_pick_snapshot(end_str, result_with_flag)
        except Exception:
            pass
    return result


@app.post("/nhl/snapshot_today")
def snapshot_today(
    request: Request,
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS, ge=7, le=200),
    force_refresh: bool = Query(False),
):
    """
    Admin: compute snapshots for all today's matchups and POTD. Intended to be called once daily (e.g., noon).
    """
    _require_admin(request)
    today = _now_in_zone().date().isoformat()
    snapshot = _snapshot_all_matchups_for_date(today, lookback_days=lookback_days, force_refresh=force_refresh)
    # compute pick with cache + snapshot write
    pick = nhl_pick(lookback_days=lookback_days, force_refresh=force_refresh, ignore_pick_gate=True, cache=True)
    return {
        "date": today,
        "matchups": snapshot,
        "pick": pick,
    }


# ---------- Background auto-snapshot (runs daily after gate) ----------
_AUTO_SNAPSHOT_LAST_DATE = None


def _auto_snapshot_loop():
    global _AUTO_SNAPSHOT_LAST_DATE
    while True:
        try:
            now = _now_in_zone()
            gate = time(PICK_GATE_HOUR, PICK_GATE_MINUTE)
            today = now.date()
            if _AUTO_SNAPSHOT_LAST_DATE != today and now.time() >= gate:
                _run_daily_snapshot_once(today.isoformat(), lookback_days=DEFAULT_LOOKBACK_DAYS)
                _AUTO_SNAPSHOT_LAST_DATE = today
            time_to_sleep = 60  # check roughly every minute
            time_module.sleep(time_to_sleep)
        except Exception:
            logger.exception("[AUTO_SNAPSHOT] Loop failure; will retry")
            try:
                time_module.sleep(60)
            except Exception:
                pass


if AUTO_SNAPSHOT_ENABLED:
    try:
        t = threading.Thread(target=_auto_snapshot_loop, name="auto-snapshot", daemon=True)
        t.start()
        logger.info("[AUTO_SNAPSHOT] Background loop started (gate %02d:%02d)", PICK_GATE_HOUR, PICK_GATE_MINUTE)
    except Exception:
        logger.exception("[AUTO_SNAPSHOT] Failed to start background loop")


@app.get("/nhl/stats")
def nhl_stats(force_refresh: bool = Query(False, description="Refresh cached stats sources")):
    """
    Return merged team stats from NHL API (PP/PK/FO/shots) and MoneyPuck (xG/HDCF/PP/PK).
    """
    teams = _team_stats_snapshot(force_refresh=force_refresh)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "force_refresh": force_refresh,
        "teams": teams,
    }


@app.get("/nhl/roster")
def nhl_roster(
    team: str = Query(..., description="Team abbreviation, e.g. BOS"),
    force_refresh: bool = Query(False, description="Refresh roster cache"),
):
    team = _validate_team_or_400(team, "team")
    roster = load_team_roster(team, force_refresh=force_refresh)
    if not roster:
        raise HTTPException(status_code=404, detail="Roster not available for that team.")
    players = []
    try:
        athletes = roster.get("athletes", [])
    except Exception:
        athletes = []
    for group in athletes:
        try:
            position = group.get("position", {}).get("abbreviation")
        except Exception:
            position = None
        for item in group.get("items", []):
            try:
                name = item.get("fullName") or item.get("displayName")
            except Exception:
                name = None
            if not name:
                continue
            players.append({"name": name, "position": position})
    return {"team": team, "players": players}


@app.get("/admin/overrides")
def get_overrides(request: Request):
    _require_admin(request)
    return _read_overrides()


@app.post("/admin/overrides")
def update_overrides(payload: OverridePayload, request: Request):
    _require_admin(request)
    data = {
        "injuries": [entry.dict() for entry in payload.injuries],
        "goalies": [entry.dict() for entry in payload.goalies],
    }
    _write_overrides(data)
    return {"status": "ok", "overrides": data}


@app.get("/nhl/injuries")
def nhl_injuries(force_refresh: bool = Query(False, description="Refresh cached injuries")):
    player_stats = load_moneypuck_player_stats(force_refresh=force_refresh)
    injuries = load_injuries_rotowire(force_refresh=force_refresh, player_stats=player_stats)
    if not injuries:
        raise HTTPException(status_code=502, detail="Injury feed unavailable.")
    return injuries
