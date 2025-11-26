"""
Advanced NFL stats using nfl_data_py.
Provides EPA, injuries, and other advanced metrics.
"""

from functools import lru_cache
from typing import Dict, Optional
import logging

import nfl_data_py as nfl

logger = logging.getLogger(__name__)


# Cache play-by-play data (large dataset)
@lru_cache(maxsize=4)
def load_pbp_data(season: int):
    """Load and cache play-by-play data for a season."""
    try:
        return nfl.import_pbp_data([season])
    except Exception as e:
        logger.warning(f"Failed to load PBP data for {season}: {e}")
        return None


@lru_cache(maxsize=4)
def load_injuries(season: int):
    """Load and cache injury data for a season."""
    try:
        return nfl.import_injuries([season])
    except Exception as e:
        logger.warning(f"Failed to load injury data for {season}: {e}")
        return None


@lru_cache(maxsize=4)
def load_weekly_data(season: int):
    """Load and cache weekly player stats."""
    try:
        return nfl.import_weekly_data([season])
    except Exception as e:
        logger.warning(f"Failed to load weekly data for {season}: {e}")
        return None


def get_team_epa(team: str, season: int, weeks: int = 4) -> Optional[Dict]:
    """
    Calculate team's offensive and defensive EPA/play.
    """
    pbp = load_pbp_data(season)
    if pbp is None:
        return None

    try:
        plays = pbp[
            (pbp["play_type"].isin(["pass", "run"]))
            & (pbp["epa"].notna())
        ]

        if len(plays) == 0:
            return None

        max_week = plays["week"].max()
        min_week = max(1, max_week - weeks + 1)
        recent = plays[plays["week"] >= min_week]

        off_plays = recent[recent["posteam"] == team]
        off_epa = off_plays["epa"].mean() if len(off_plays) > 0 else 0.0

        def_plays = recent[recent["defteam"] == team]
        def_epa = def_plays["epa"].mean() if len(def_plays) > 0 else 0.0

        return {
            "off_epa": round(off_epa, 3),
            "def_epa": round(def_epa, 3),
            "net_epa": round(off_epa - def_epa, 3),
            "off_plays": len(off_plays),
            "def_plays": len(def_plays),
        }
    except Exception as e:
        logger.warning(f"Error calculating EPA for {team}: {e}")
        return None


def get_qb_status(team: str, season: int) -> Optional[Dict]:
    """
    Check if starting QB is healthy.
    """
    injuries = load_injuries(season)
    if injuries is None:
        return None

    try:
        team_qb = injuries[
            (injuries["team"] == team)
            & (injuries["position"] == "QB")
        ]

        if len(team_qb) == 0:
            return {"qb_name": "Unknown", "status": "healthy", "is_backup": False}

        if "report_date" in team_qb.columns:
            latest = team_qb.sort_values("report_date", ascending=False).iloc[0]
        else:
            latest = team_qb.iloc[0]

        report_status = latest.get("report_status", "")
        is_out = report_status in ["Out", "Doubtful", "IR"]
        is_questionable = report_status in ["Questionable", "Probable"]

        if is_out:
            status = "out"
        elif is_questionable:
            status = "questionable"
        else:
            status = "healthy"

        return {
            "qb_name": latest.get("full_name", "Unknown"),
            "status": status,
            "is_backup": is_out,
            "injury": latest.get("report_primary_injury", ""),
        }
    except Exception as e:
        logger.warning(f"Error checking QB status for {team}: {e}")
        return None


def get_team_success_rate(team: str, season: int, weeks: int = 4) -> Optional[float]:
    """
    Calculate offensive success rate.
    Success = gaining expected yards based on down/distance.
    """
    pbp = load_pbp_data(season)
    if pbp is None:
        return None

    try:
        plays = pbp[
            (pbp["play_type"].isin(["pass", "run"]))
            & (pbp["posteam"] == team)
            & (pbp["success"].notna())
        ]

        if len(plays) == 0:
            return None

        max_week = plays["week"].max()
        min_week = max(1, max_week - weeks + 1)
        recent = plays[plays["week"] >= min_week]

        if len(recent) == 0:
            return None

        return round(recent["success"].mean(), 3)
    except Exception as e:
        logger.warning(f"Error calculating success rate for {team}: {e}")
        return None


def get_advanced_matchup_data(home_team: str, away_team: str, season: int) -> Dict:
    """
    Get all advanced stats for a matchup.
    """
    return {
        "home_epa": get_team_epa(home_team, season),
        "away_epa": get_team_epa(away_team, season),
        "home_qb": get_qb_status(home_team, season),
        "away_qb": get_qb_status(away_team, season),
        "home_success": get_team_success_rate(home_team, season),
        "away_success": get_team_success_rate(away_team, season),
    }
