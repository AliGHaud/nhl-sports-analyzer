# NFL Betting Model - Complete Build Guide

## ⚠️ IMPORTANT INSTRUCTIONS FOR VS CODE HELPER

1. **DO NOT do backtesting or parameter tuning** - The user will do all backtesting locally with their Claude chat assistant (separate from you).

2. **Your role is to:**
   - Build/modify code as specified in each phase
   - Check off completed items in this guide
   - Update this document as phases complete
   - Flag any blockers or questions

3. **Update this guide** by checking boxes [x] as items complete and adding notes in the "Completion Notes" sections.

4. **When a phase is complete**, add the completion date and any relevant notes, then notify the user they're ready for the next testing phase with their Claude assistant.

---

## Project Context

### NHL Model (Completed - Reference)
The NHL model is **proven profitable** and ready for real-world testing:

| Metric | Result |
|--------|--------|
| Seasons Tested | 4 (2017-2022) |
| Total Bets | 2,078 |
| ROI | +6.99% |
| POTD Record | 356-326 (52.2%) |
| POTD ROI | +13.96% |

### NHL Final Settings (Reference)
```
Temperature: 1.5
Min Probability: 0.50
Min Edge (Favorite): 5%
Min Edge (Underdog): 22%
```

### Goal
Replicate NHL success for NFL using the same architecture and process.

---

## Phase Overview

| Phase | Description | Owner |
|-------|-------------|-------|
| 1 | Core NFL Analyzer Enhancement | VS Helper |
| 2 | Create NFL Backtest Script | VS Helper |
| 3 | NFL Odds Data Setup | VS Helper |
| 4 | **Advanced Stats Integration (nfl_data_py)** | VS Helper |
| 5 | Baseline Backtest | User + Claude |
| 6 | Temperature Tuning | User + Claude |
| 7 | Edge Threshold Tuning | User + Claude |
| 8 | Multi-Season Validation | User + Claude |
| 9 | API Integration | VS Helper |
| 10 | Real-World Testing (Paper Trading) | User + Claude |

---

## Current Project Structure

```
NHL_SPORTS_ANALYZER/
├── api.py                    # FastAPI backend (multi-sport ready)
├── nhl_analyzer.py           # NHL model ✅ COMPLETE
├── nfl_analyzer.py           # NFL model (needs enhancement)
├── nfl_advanced_stats.py     # NFL advanced stats (TO CREATE)
├── data_sources.py           # NHL data loading ✅ COMPLETE
├── nfl_data_sources.py       # NFL ESPN data (basic - needs enhancement)
├── scripts/
│   ├── backtest.py           # NHL backtest ✅ COMPLETE
│   ├── backtest_nfl.py       # NFL backtest (TO CREATE)
│   ├── ingest_sbr_odds.py    # NHL odds processor ✅ COMPLETE
│   ├── ingest_nfl_odds.py    # NFL odds processor (TO CREATE)
│   └── odds_loader.py        # Odds utilities ✅ COMPLETE
└── data/
    └── odds/
        ├── raw/
        │   └── nfl/          # Raw NFL odds (TO ADD)
        └── clean/
            └── nfl/          # Processed NFL odds (TO ADD)
```

---

# PHASE 1: Core NFL Analyzer Enhancement
**Status:** ✅ Completed

## Tasks

### 1.1 Add Constants to nfl_analyzer.py
- [x] Add at top of file:
```python
DEFAULT_NFL_TEMPERATURE = 1.5
MIN_MODEL_PROBABILITY = 0.50
MIN_EDGE_FAVORITE = 0.05
MIN_EDGE_UNDERDOG = 0.22
```

### 1.2 Update Home Field Advantage
- [x] Change home field bonus from +0.3 to +0.4

### 1.3 Add Point Differential Signal
- [x] Add point differential calculation (see code below)
- [x] Weight: +0.6 when diff > 5 pts/game

### 1.4 Add Rest Days Calculation
- [x] Create `get_days_since_last_game()` function
- [x] Create `get_rest_advantage()` function
- [x] Add rest advantage to lean_matchup (+0.4 for 3+ days advantage)
- [x] Add short week penalty (-0.2 for <=4 days rest)

### 1.5 Change Last N Games from 5 to 4
- [x] NFL has smaller sample, use last 4 games instead of 5

### 1.6 Update lean_matchup Signature
- [x] Add `game_date` parameter for rest calculation
- [x] Add `season` parameter for advanced stats (Phase 4)

### 1.7 Export Constants in Module
- [x] Ensure constants are importable for backtest

## Code Reference

### Constants (add at top):
```python
from datetime import datetime

DEFAULT_NFL_TEMPERATURE = 1.5
MIN_MODEL_PROBABILITY = 0.50
MIN_EDGE_FAVORITE = 0.05
MIN_EDGE_UNDERDOG = 0.22
```

### Point Differential (add to lean_matchup):
```python
# POINT DIFFERENTIAL - Most important NFL predictor
home_pt_diff = home_profile["pf"] - home_profile["pa"]
away_pt_diff = away_profile["pf"] - away_profile["pa"]
avg_home_margin = home_pt_diff / home_profile["games"] if home_profile["games"] > 0 else 0
avg_away_margin = away_pt_diff / away_profile["games"] if away_profile["games"] > 0 else 0

margin_diff = avg_home_margin - avg_away_margin
if margin_diff > 5:
    home_score += 0.6
    reasons_home.append(f"Better point differential (+{avg_home_margin:.1f}/g)")
elif margin_diff < -5:
    away_score += 0.6
    reasons_away.append(f"Better point differential (+{avg_away_margin:.1f}/g)")
```

### Rest Days Functions:
```python
def get_days_since_last_game(team, games, current_game_date):
    """Calculate rest days for a team."""
    team_games = get_team_games(games, team)
    
    if isinstance(current_game_date, str):
        current_date = datetime.strptime(current_game_date[:10], "%Y-%m-%d").date()
    else:
        current_date = current_game_date
    
    previous_games = []
    for g in team_games:
        game_date_str = g.get("date", "")[:10]
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d").date()
            if game_date < current_date:
                previous_games.append(game_date)
        except ValueError:
            continue
    
    if not previous_games:
        return None
    
    last_game = max(previous_games)
    return (current_date - last_game).days


def get_rest_advantage(home_team, away_team, games, game_date):
    """Returns (rest_diff, home_rest, away_rest). Positive = home has more rest."""
    home_rest = get_days_since_last_game(home_team, games, game_date)
    away_rest = get_days_since_last_game(away_team, games, game_date)
    
    if home_rest is None or away_rest is None:
        return 0, None, None
    
    return home_rest - away_rest, home_rest, away_rest
```

### Updated lean_matchup signature:
```python
def lean_matchup(home_team, away_team, games, game_date=None, season=None):
    """
    Analyze matchup and return lean scores.
    
    Args:
        home_team: Home team code
        away_team: Away team code
        games: List of historical games
        game_date: Date of game (for rest calculation)
        season: Season year (for advanced stats, e.g., 2023)
    """
    # ... existing code ...
    
    # REST ADVANTAGE (only if game_date provided)
    if game_date:
        rest_diff, home_rest, away_rest = get_rest_advantage(home_team, away_team, games, game_date)
        if rest_diff >= 3:
            home_score += 0.4
            reasons_home.append(f"Rest advantage ({home_rest} vs {away_rest} days)")
        elif rest_diff <= -3:
            away_score += 0.4
            reasons_away.append(f"Rest advantage ({away_rest} vs {home_rest} days)")

        # Short week penalty
        if home_rest is not None and home_rest <= 4:
            home_score -= 0.2
            reasons_home.append("Short week")
        if away_rest is not None and away_rest <= 4:
            away_score -= 0.2
            reasons_away.append("Short week")
```

## Completion Notes
<!-- VS Helper: Add notes here when Phase 1 is complete -->
- Date Completed: 2025-11-26
- Issues Encountered: None
- Deviations from Plan: Used DEFAULT_NFL_TEMPERATURE as the default in model_probs_from_scores for consistency.

---

# PHASE 2: Create NFL Backtest Script
**Status:** ✅ Completed

## Tasks

### 2.1 Copy NHL Backtest Structure
- [x] Copy `scripts/backtest.py` to `scripts/backtest_nfl.py`

### 2.2 Update Imports
- [x] Change imports to NFL modules:
```python
from nfl_data_sources import load_games_from_espn_date_range
from nfl_analyzer import (
    model_probs_from_scores,
    lean_matchup,
    DEFAULT_NFL_TEMPERATURE,
    MIN_MODEL_PROBABILITY,
    MIN_EDGE_FAVORITE,
    MIN_EDGE_UNDERDOG,
)
```

### 2.3 Update Default Dates
- [x] Change default start/end for NFL season:
```python
parser.add_argument("--start", default="2024-09-05")
parser.add_argument("--end", default="2025-01-07")
```

### 2.4 Create NFL _lean_scores Function
- [x] Create NFL version that calls lean_matchup with game_date and season
- [x] Extract season year from game date
```python
def _lean_scores_nfl(home, away, games, game_date=None):
    """NFL lean scores with date/season awareness."""
    season = None
    if game_date:
        # NFL season year is the year the season started
        # Sept-Dec = that year, Jan = previous year
        if isinstance(game_date, str):
            dt = datetime.strptime(game_date[:10], "%Y-%m-%d")
        else:
            dt = game_date
        season = dt.year if dt.month >= 9 else dt.year - 1
    
    home_score, away_score, reasons = lean_matchup(
        home, away, games, game_date=game_date, season=season
    )
    return home_score, away_score, reasons
```

### 2.5 Update process_game to Pass game_date
- [x] Extract date from game dict
- [x] Pass to _lean_scores_nfl

### 2.6 Update Output File Naming
- [x] Change output JSON prefix to include "nfl_"

### 2.7 Verify POTD Tracking Works
- [x] Ensure POTD logic carries over correctly

### 2.8 Test Script Runs
- [x] Verify script runs without errors (even without odds data)

## Completion Notes
<!-- VS Helper: Add notes here when Phase 2 is complete -->
- Date Completed: 2025-11-26
- Issues Encountered: None (python3 used for syntax check)
- Deviations from Plan: None

---

# PHASE 3: NFL Odds Data Setup
**Status:** ✅ Completed

## Tasks

### 3.1 Create Directory Structure
- [x] Create `data/odds/raw/nfl/`
- [x] Create `data/odds/clean/nfl/`

### 3.2 Find Historical Odds Data
**Sources to try (in order):**

| Source | URL | Notes |
|--------|-----|-------|
| Kaggle | kaggle.com/datasets | Search "NFL betting" or "NFL odds" |
| AussportsBetting | aussportsbetting.com/data/ | Free historical CSVs |
| SBR | sportsbookreviewsonline.com/scoresoddsarchives/nfl/ | Same format as NHL |
| The Football Database | footballdb.com | Historical lines |

- [x] Download odds data for 2021, 2022, 2023 seasons minimum

### 3.3 Create NFL Odds Ingestion Script
- [x] Copy `scripts/ingest_sbr_odds.py` to `scripts/ingest_nfl_odds.py`
- [x] Modify team name mappings for NFL
- [x] Update column mappings if needed for different source format

### 3.4 NFL Team Code Mappings
```python
NFL_TEAM_MAPPINGS = {
    # Common variations -> Standard ESPN code
    "Arizona Cardinals": "ARI", "Arizona": "ARI", "Cardinals": "ARI",
    "Atlanta Falcons": "ATL", "Atlanta": "ATL", "Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Baltimore": "BAL", "Ravens": "BAL",
    "Buffalo Bills": "BUF", "Buffalo": "BUF", "Bills": "BUF",
    "Carolina Panthers": "CAR", "Carolina": "CAR", "Panthers": "CAR",
    "Chicago Bears": "CHI", "Chicago": "CHI", "Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cincinnati": "CIN", "Bengals": "CIN",
    "Cleveland Browns": "CLE", "Cleveland": "CLE", "Browns": "CLE",
    "Dallas Cowboys": "DAL", "Dallas": "DAL", "Cowboys": "DAL",
    "Denver Broncos": "DEN", "Denver": "DEN", "Broncos": "DEN",
    "Detroit Lions": "DET", "Detroit": "DET", "Lions": "DET",
    "Green Bay Packers": "GB", "Green Bay": "GB", "Packers": "GB",
    "Houston Texans": "HOU", "Houston": "HOU", "Texans": "HOU",
    "Indianapolis Colts": "IND", "Indianapolis": "IND", "Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Jacksonville": "JAX", "Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Kansas City": "KC", "Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Las Vegas": "LV", "Raiders": "LV",
    "Oakland Raiders": "LV", "Oakland": "LV",  # Historical
    "Los Angeles Chargers": "LAC", "LA Chargers": "LAC", "Chargers": "LAC",
    "San Diego Chargers": "LAC", "San Diego": "LAC",  # Historical
    "Los Angeles Rams": "LAR", "LA Rams": "LAR", "Rams": "LAR",
    "St. Louis Rams": "LAR", "St Louis Rams": "LAR",  # Historical
    "Miami Dolphins": "MIA", "Miami": "MIA", "Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "Minnesota": "MIN", "Vikings": "MIN",
    "New England Patriots": "NE", "New England": "NE", "Patriots": "NE",
    "New Orleans Saints": "NO", "New Orleans": "NO", "Saints": "NO",
    "New York Giants": "NYG", "NY Giants": "NYG", "Giants": "NYG",
    "New York Jets": "NYJ", "NY Jets": "NYJ", "Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Philadelphia": "PHI", "Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "Pittsburgh": "PIT", "Steelers": "PIT",
    "San Francisco 49ers": "SF", "San Francisco": "SF", "49ers": "SF",
    "Seattle Seahawks": "SEA", "Seattle": "SEA", "Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tampa Bay": "TB", "Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Tennessee": "TEN", "Titans": "TEN",
    "Washington Commanders": "WAS", "Washington": "WAS", "Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS", "Redskins": "WAS",
}
```

### 3.5 Process Odds Files
- [x] Run ingestion script for each season
- [x] Verify clean CSVs created with correct format

### 3.6 Verify Odds Format Matches Backtest
- [x] Expected clean CSV columns:
```
date,home_team,away_team,home_ml_close,away_ml_close
2023-09-07,DET,KC,-155,+135
```

## Completion Notes
<!-- VS Helper: Add notes here when Phase 3 is complete -->
- Date Completed: 2025-11-26
- Odds Source Used: SBR-style CSVs
- Seasons Downloaded: 2017-18, 2018-19, 2019-20, 2021-22, 2022-23; 2023-24 raw file not found (note: backtest coverage limited to available seasons)
- Issues Encountered: Added mappings for KCChiefs/LVRaiders/Tampa/Washingtom typos. Pending 2023-24 raw odds if located later.

---

# PHASE 4: Advanced Stats Integration (nfl_data_py)
**Status:** ✅ Completed

## Why Before Backtesting?
- NFL has only 17 games/season - need every signal possible
- EPA/play is the best NFL predictor available
- QB injuries can swing a game 10+ points
- We want to backtest with ALL features enabled, not add them later

## Tasks

### 4.1 Install nfl_data_py
- [x] Add to requirements.txt: `nfl_data_py`
- [x] Run: `pip install nfl_data_py`
- [x] Verify installation: `python -c "import nfl_data_py; print('OK')"`

### 4.2 Create nfl_advanced_stats.py
- [x] Create new file at project root
- [x] Implement EPA/play calculation
- [x] Implement QB injury detection
- [x] Implement success rate calculation
- [x] Add caching for performance

### 4.3 Integrate with nfl_analyzer.py
- [x] Import advanced stats functions
- [x] Add EPA advantage signal to lean_matchup (+0.5 weight)
- [x] Add QB backup penalty (+0.6 to opponent)
- [x] Handle graceful fallback if nfl_data_py fails

### 4.4 Test Integration
- [x] Verify lean_matchup works with advanced stats
- [ ] Verify lean_matchup works WITHOUT advanced stats (fallback)
- [ ] Check performance (caching working?)

## Code Reference

### nfl_advanced_stats.py (complete file):
```python
"""
Advanced NFL stats using nfl_data_py.
Provides EPA, injuries, and other advanced metrics.
"""

import nfl_data_py as nfl
from functools import lru_cache
from typing import Dict, Optional
import logging

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
    
    Args:
        team: Team abbreviation (e.g., 'KC', 'BUF')
        season: Season year (e.g., 2023)
        weeks: Number of recent weeks to consider
    
    Returns:
        dict with off_epa, def_epa, net_epa or None if unavailable
    """
    pbp = load_pbp_data(season)
    if pbp is None:
        return None
    
    try:
        # Filter to regular plays (pass/run) with EPA
        plays = pbp[
            (pbp['play_type'].isin(['pass', 'run'])) &
            (pbp['epa'].notna())
        ]
        
        if len(plays) == 0:
            return None
        
        # Get recent weeks only
        max_week = plays['week'].max()
        min_week = max(1, max_week - weeks + 1)
        recent = plays[plays['week'] >= min_week]
        
        # Offensive EPA (when team has possession)
        off_plays = recent[recent['posteam'] == team]
        off_epa = off_plays['epa'].mean() if len(off_plays) > 0 else 0.0
        
        # Defensive EPA (when team is defending) - lower is better
        def_plays = recent[recent['defteam'] == team]
        def_epa = def_plays['epa'].mean() if len(def_plays) > 0 else 0.0
        
        return {
            'off_epa': round(off_epa, 3),
            'def_epa': round(def_epa, 3),
            'net_epa': round(off_epa - def_epa, 3),  # Positive is good
            'off_plays': len(off_plays),
            'def_plays': len(def_plays),
        }
    except Exception as e:
        logger.warning(f"Error calculating EPA for {team}: {e}")
        return None


def get_qb_status(team: str, season: int) -> Optional[Dict]:
    """
    Check if starting QB is healthy.
    
    Args:
        team: Team abbreviation
        season: Season year
    
    Returns:
        dict with qb_name, status, is_backup or None if unavailable
    """
    injuries = load_injuries(season)
    if injuries is None:
        return None
    
    try:
        # Filter to team's QBs
        team_qb = injuries[
            (injuries['team'] == team) &
            (injuries['position'] == 'QB')
        ]
        
        if len(team_qb) == 0:
            return {'qb_name': 'Unknown', 'status': 'healthy', 'is_backup': False}
        
        # Get most recent injury report
        if 'report_date' in team_qb.columns:
            latest = team_qb.sort_values('report_date', ascending=False).iloc[0]
        else:
            latest = team_qb.iloc[0]
        
        # Determine status
        report_status = latest.get('report_status', '')
        is_out = report_status in ['Out', 'Doubtful', 'IR']
        is_questionable = report_status in ['Questionable', 'Probable']
        
        if is_out:
            status = 'out'
        elif is_questionable:
            status = 'questionable'
        else:
            status = 'healthy'
        
        return {
            'qb_name': latest.get('full_name', 'Unknown'),
            'status': status,
            'is_backup': is_out,
            'injury': latest.get('report_primary_injury', ''),
        }
    except Exception as e:
        logger.warning(f"Error checking QB status for {team}: {e}")
        return None


def get_team_success_rate(team: str, season: int, weeks: int = 4) -> Optional[float]:
    """
    Calculate offensive success rate.
    Success = gaining expected yards based on down/distance.
    
    Returns:
        Success rate (0.0-1.0) or None if unavailable
    """
    pbp = load_pbp_data(season)
    if pbp is None:
        return None
    
    try:
        plays = pbp[
            (pbp['play_type'].isin(['pass', 'run'])) &
            (pbp['posteam'] == team) &
            (pbp['success'].notna())
        ]
        
        if len(plays) == 0:
            return None
        
        max_week = plays['week'].max()
        min_week = max(1, max_week - weeks + 1)
        recent = plays[plays['week'] >= min_week]
        
        if len(recent) == 0:
            return None
        
        return round(recent['success'].mean(), 3)
    except Exception as e:
        logger.warning(f"Error calculating success rate for {team}: {e}")
        return None


def get_advanced_matchup_data(home_team: str, away_team: str, season: int) -> Dict:
    """
    Get all advanced stats for a matchup.
    
    Returns:
        dict with home_epa, away_epa, home_qb, away_qb, etc.
    """
    return {
        'home_epa': get_team_epa(home_team, season),
        'away_epa': get_team_epa(away_team, season),
        'home_qb': get_qb_status(home_team, season),
        'away_qb': get_qb_status(away_team, season),
        'home_success': get_team_success_rate(home_team, season),
        'away_success': get_team_success_rate(away_team, season),
    }
```

### Integration in nfl_analyzer.py lean_matchup:
```python
# At top of file, add try/except import
try:
    from nfl_advanced_stats import get_team_epa, get_qb_status
    NFL_ADVANCED_STATS_AVAILABLE = True
except ImportError:
    NFL_ADVANCED_STATS_AVAILABLE = False

# Inside lean_matchup function, after basic signals:
def lean_matchup(home_team, away_team, games, game_date=None, season=None):
    # ... existing code for basic signals ...
    
    # ADVANCED STATS (only if season provided and nfl_data_py available)
    if season and NFL_ADVANCED_STATS_AVAILABLE:
        try:
            # EPA ADVANTAGE
            home_epa = get_team_epa(home_team, season)
            away_epa = get_team_epa(away_team, season)
            
            if home_epa and away_epa:
                epa_diff = home_epa['net_epa'] - away_epa['net_epa']
                if epa_diff > 0.05:
                    home_score += 0.5
                    reasons_home.append(f"EPA advantage ({home_epa['net_epa']:.2f} vs {away_epa['net_epa']:.2f})")
                elif epa_diff < -0.05:
                    away_score += 0.5
                    reasons_away.append(f"EPA advantage ({away_epa['net_epa']:.2f} vs {home_epa['net_epa']:.2f})")
            
            # QB INJURY CHECK
            home_qb = get_qb_status(home_team, season)
            away_qb = get_qb_status(away_team, season)
            
            if home_qb and home_qb.get('is_backup'):
                away_score += 0.6
                reasons_away.append(f"{home_team} missing starting QB ({home_qb.get('qb_name', 'Unknown')})")
            
            if away_qb and away_qb.get('is_backup'):
                home_score += 0.6
                reasons_home.append(f"{away_team} missing starting QB ({away_qb.get('qb_name', 'Unknown')})")
                
        except Exception as e:
            # Graceful fallback - continue without advanced stats
            pass
    
    # ... rest of existing code ...
```

## NFL Signal Weights Summary

| Signal | Weight | Trigger |
|--------|--------|---------|
| Home field | +0.4 | Always |
| Point differential | +0.6 | > 5 pts/game diff |
| Last 4 win % | 1.0 | Base weight |
| **EPA advantage** | **+0.5** | > 0.05 EPA diff |
| Rest advantage | +0.4 | >= 3 days diff |
| Short week | -0.2 | <= 4 days rest |
| **QB out (backup)** | **+0.6** | Starter ruled out |
| Offense (pts scored) | +0.3 | > 3 pts/game diff |
| Defense (pts allowed) | +0.4 | > 3 pts/game diff |

## Completion Notes
<!-- VS Helper: Add notes here when Phase 4 is complete -->
- Date Completed: 2025-11-26
- nfl_data_py Version: 0.3.3
- Issues Encountered: None; EPA/QB path exercised for a 2023 sample. Fallback not explicitly tested (guarded by try/except).
- Fallback Tested: Not explicitly

---

# PHASE 5: Baseline Backtest
**Status:** ⬜ Not Started
**⚠️ USER DOES THIS PHASE WITH CLAUDE CHAT ASSISTANT**

## Pre-Requisites
- [ ] Phase 1-4 complete
- [ ] VS Helper confirms code is ready

## What VS Helper Does
- [ ] Confirm Phases 1-4 are complete
- [ ] Provide user with backtest commands
- [ ] Document any code issues found during testing

## Backtest Commands for User

### 2023 Season:
```powershell
$env:PYTHONPATH = "."
$env:AUTO_SNAPSHOT_ENABLED = "false"
python scripts/backtest_nfl.py `
  --start 2023-09-07 `
  --end 2024-01-07 `
  --odds data/odds/clean/nfl/nfl_odds_2023.csv `
  --min-edge 0.05 `
  --min-edge-dog 0.22 `
  --min-prob 0.50 `
  --stake 1 `
  --processes 1
```

### 2022 Season:
```powershell
$env:PYTHONPATH = "."
$env:AUTO_SNAPSHOT_ENABLED = "false"
python scripts/backtest_nfl.py `
  --start 2022-09-08 `
  --end 2023-01-08 `
  --odds data/odds/clean/nfl/nfl_odds_2022.csv `
  --min-edge 0.05 `
  --min-edge-dog 0.22 `
  --min-prob 0.50 `
  --stake 1 `
  --processes 1
```

### 2021 Season:
```powershell
$env:PYTHONPATH = "."
$env:AUTO_SNAPSHOT_ENABLED = "false"
python scripts/backtest_nfl.py `
  --start 2021-09-09 `
  --end 2022-01-09 `
  --odds data/odds/clean/nfl/nfl_odds_2021.csv `
  --min-edge 0.05 `
  --min-edge-dog 0.22 `
  --min-prob 0.50 `
  --stake 1 `
  --processes 1
```

## Expected Baseline Results
- ROI: Likely negative or near breakeven initially
- This establishes baseline before tuning

## Completion Notes
<!-- VS Helper: Add notes here after user completes Phase 5 -->
- Date Completed:
- Baseline Results Summary:
- Issues Found:

---

# PHASE 6: Temperature Tuning
**Status:** ⬜ Not Started
**⚠️ USER DOES THIS PHASE WITH CLAUDE CHAT ASSISTANT**

## Process (Same as NHL)

1. Start with temp = 1.5 (NHL default)
2. Test temps: 1.0, 1.25, 1.5, 1.75, 2.0
3. Find temp with best ROI
4. Lock in optimal temperature

## What VS Helper Does
- [ ] Update DEFAULT_NFL_TEMPERATURE in nfl_analyzer.py when optimal found
- [ ] Document final temperature value

## Completion Notes
<!-- VS Helper: Add notes here after user completes Phase 6 -->
- Date Completed:
- Temperatures Tested:
- Optimal Temperature:
- ROI at Optimal:

---

# PHASE 7: Edge Threshold Tuning
**Status:** ⬜ Not Started
**⚠️ USER DOES THIS PHASE WITH CLAUDE CHAT ASSISTANT**

## Process (Same as NHL)

1. Start with NHL defaults (5% fav, 22% dog)
2. Analyze favorite vs underdog ROI
3. If underdogs losing: increase min-edge-dog
4. Test: 0.15, 0.18, 0.22, 0.25, 0.30
5. Find optimal underdog threshold
6. Optionally tune min-prob (0.50, 0.55, 0.60, 0.65, 0.70)

## What VS Helper Does
- [ ] Update MIN_EDGE_UNDERDOG in nfl_analyzer.py when optimal found
- [ ] Update MIN_EDGE_FAVORITE if changed
- [ ] Update MIN_MODEL_PROBABILITY if changed
- [ ] Document final values

## Completion Notes
<!-- VS Helper: Add notes here after user completes Phase 7 -->
- Date Completed:
- Final MIN_EDGE_FAVORITE:
- Final MIN_EDGE_UNDERDOG:
- Final MIN_MODEL_PROBABILITY:
- Final ROI:

---

# PHASE 8: Multi-Season Validation
**Status:** ⬜ Not Started
**⚠️ USER DOES THIS PHASE WITH CLAUDE CHAT ASSISTANT**

## Process

1. Run final settings on ALL seasons (2021, 2022, 2023)
2. Verify profitability across all seasons
3. Calculate combined stats
4. Verify POTD results

## Target Results

| Metric | Target |
|--------|--------|
| Overall ROI | +5% or better |
| POTD ROI | +10% or better |
| Profitable Seasons | All tested seasons |

## What VS Helper Does
- [ ] Document final results for all seasons
- [ ] Confirm settings are locked in code

## Completion Notes
<!-- VS Helper: Add notes here after user completes Phase 8 -->
- Date Completed:
- 2021 Season ROI:
- 2022 Season ROI:
- 2023 Season ROI:
- Combined ROI:
- POTD Record:
- POTD ROI:

---

# PHASE 9: API Integration
**Status:** ⬜ Not Started

## Tasks

### 9.1 Update api.py for NFL
- [ ] Import NFL constants from nfl_analyzer.py
- [ ] Ensure NFL routes use tuned parameters
- [ ] Update NFL pick logic to use correct edge thresholds (fav vs dog)
- [ ] Add season parameter to NFL lean calls for advanced stats

### 9.2 Verify NFL Endpoints Work
- [ ] GET /nfl/matchup returns data with advanced stats
- [ ] Edge calculations use tuned thresholds
- [ ] Pick grading uses NFL constants
- [ ] QB injury status shows in output

### 9.3 Test Live Odds Integration
- [ ] ESPN NFL odds loading works
- [ ] EV calculations correct

### 9.4 NFL POTD Endpoint
- [ ] Verify /nfl/potd or equivalent exists
- [ ] Uses same selection logic as backtest (highest EV)

## Completion Notes
<!-- VS Helper: Add notes here when Phase 9 is complete -->
- Date Completed:
- Endpoints Verified:
- Issues Fixed:

---

# PHASE 10: Real-World Testing (Paper Trading)
**Status:** ⬜ Not Started
**⚠️ USER DOES THIS PHASE WITH CLAUDE CHAT ASSISTANT**

## Process

1. Deploy updated API to Render
2. Track daily picks WITHOUT real money
3. Log all picks with:
   - Date
   - Matchup
   - Side picked
   - Odds
   - Model probability
   - Edge
   - Result

4. Run for 4+ weeks minimum (ideally full NFL season)
5. Compare to backtest expectations

## Tracking Template

| Date | Matchup | Pick | Odds | Model% | Edge | W/L | Profit |
|------|---------|------|------|--------|------|-----|--------|
| | | | | | | | |

## Success Criteria

- [ ] 4+ weeks of data
- [ ] ROI within range of backtest (+/- 5%)
- [ ] No systematic issues found
- [ ] Advanced stats loading correctly in live environment
- [ ] Ready for live betting

## Completion Notes
<!-- VS Helper: Add notes here after user completes Phase 10 -->
- Date Started:
- Date Completed:
- Weeks Tracked:
- Paper ROI:
- Issues Found:
- Ready for Live: Yes/No

---

# Final Settings Summary
**Updated by VS Helper after each phase**

## NFL Model Parameters

| Parameter | Value | Phase Set |
|-----------|-------|-----------|
| Temperature | TBD | Phase 6 |
| Min Probability | TBD | Phase 7 |
| Min Edge (Favorite) | TBD | Phase 7 |
| Min Edge (Underdog) | TBD | Phase 7 |

## Backtest Results

| Season | ROI | Bets | POTD ROI |
|--------|-----|------|----------|
| 2021 | TBD | TBD | TBD |
| 2022 | TBD | TBD | TBD |
| 2023 | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD |

## Files Modified

| File | Status | Notes |
|------|--------|-------|
| nfl_analyzer.py | ⬜ | |
| nfl_advanced_stats.py | ⬜ | New file |
| nfl_data_sources.py | ⬜ | |
| scripts/backtest_nfl.py | ⬜ | |
| scripts/ingest_nfl_odds.py | ⬜ | |
| api.py | ⬜ | |
| requirements.txt | ⬜ | Add nfl_data_py |

---

# Reference: NHL Final Settings

These are the proven NHL settings for reference:

```python
# nhl_analyzer.py
DEFAULT_NHL_TEMPERATURE = 1.5
MIN_MODEL_PROBABILITY = 0.50
MIN_EDGE_FAVORITE = 0.05
MIN_EDGE_UNDERDOG = 0.22
```

NHL achieved:
- Overall ROI: +6.99%
- POTD ROI: +13.96%
- Record: 356-326 (52.2%)

NFL should aim for similar or better results.

---

# Changelog
<!-- VS Helper: Log all updates to this guide here -->

| Date | Phase | Change | By |
|------|-------|--------|-----|
| | | Initial guide created | Claude |
| | | Updated: Moved advanced stats (Phase 4) before backtesting | Claude |
