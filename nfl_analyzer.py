"""
Simple NFL lean engine (separate from NHL).
"""

from datetime import datetime
import math

from nfl_data_sources import (
    load_games_from_espn_scoreboard,
    load_games_from_espn_date_range,
    load_current_odds_for_matchup,
    load_schedule_for_date,
)
try:
    from nfl_advanced_stats import get_team_epa, get_qb_status

    NFL_ADVANCED_STATS_AVAILABLE = True
except ImportError:
    NFL_ADVANCED_STATS_AVAILABLE = False

DEFAULT_NFL_TEMPERATURE = 1.5
MIN_MODEL_PROBABILITY = 0.55
MIN_EDGE_FAVORITE = 0.30  # Effectively dogs only
MIN_EDGE_UNDERDOG = 0.15
POTD_MIN_PROB = 0.60


def get_team_games(games, team):
    return [g for g in games if g["home_team"] == team or g["away_team"] == team]


def get_last_n_games(team, games, n=4):
    tg = get_team_games(games, team)
    return tg[-n:] if len(tg) >= n else tg


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


def calculate_stats(team, games):
    wins = 0
    losses = 0
    pf = pa = 0
    for g in games:
        hg = int(g["home_goals"])
        ag = int(g["away_goals"])
        if g["home_team"] == team:
            pf += hg
            pa += ag
            wins += hg > ag
            losses += hg < ag
        else:
            pf += ag
            pa += hg
            wins += ag > hg
            losses += ag < hg
    n = len(games)
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "pf": pf,
        "pa": pa,
        "avg_pf": pf / n if n else 0,
        "avg_pa": pa / n if n else 0,
    }


def implied_prob_american(odds):
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def profit_on_win_for_1_unit(odds):
    odds = float(odds)
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0


def model_probs_from_scores(home_score, away_score, temperature=DEFAULT_NFL_TEMPERATURE):
    t = max(float(temperature), 1e-6)
    eh = math.exp(home_score / t)
    ea = math.exp(away_score / t)
    total = eh + ea
    return eh / total, ea / total


def confidence_grade(edge_pct, ev_units):
    if ev_units <= 0:
        return "No Bet"
    edge = abs(edge_pct)
    if edge >= 7.5 and ev_units >= 0.10:
        return "A"
    if edge >= 5.0 and ev_units >= 0.06:
        return "B+"
    if edge >= 3.0 and ev_units >= 0.03:
        return "B"
    if edge >= 1.5 and ev_units >= 0.01:
        return "C+"
    if edge >= 0.5 and ev_units > 0:
        return "C"
    return "D"


def lean_matchup(home_team, away_team, games, game_date=None, season=None):
    """
    Analyze matchup and return lean scores. game_date is used for rest days; season is kept for future advanced stats.
    """
    home_profile = calculate_stats(home_team, get_team_games(games, home_team))
    away_profile = calculate_stats(away_team, get_team_games(games, away_team))
    if home_profile["games"] == 0 or away_profile["games"] == 0:
        return None, None, {"home_reasons": [], "away_reasons": []}

    home_score = 0.0
    away_score = 0.0
    reasons_home = []
    reasons_away = []

    def win_pct(stats):
        return stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

    # POINT DIFFERENTIAL - most important NFL predictor
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

    # Last 4 win pct
    home_last4 = calculate_stats(home_team, get_last_n_games(home_team, games, 4))
    away_last4 = calculate_stats(away_team, get_last_n_games(away_team, games, 4))
    home_score += win_pct(home_last4) * 1.0
    away_score += win_pct(away_last4) * 1.0
    if win_pct(home_last4) > win_pct(away_last4) + 0.1:
        reasons_home.append("Better recent form (last 4)")
    elif win_pct(away_last4) > win_pct(home_last4) + 0.1:
        reasons_away.append("Better recent form (last 4)")

    # Points allowed (defense)
    if home_last4["avg_pa"] + 2 < away_last4["avg_pa"]:
        home_score += 0.5
        reasons_home.append("Better defense (points allowed)")
    elif away_last4["avg_pa"] + 2 < home_last4["avg_pa"]:
        away_score += 0.5
        reasons_away.append("Better defense (points allowed)")

    # Points for (offense)
    if home_last4["avg_pf"] > away_last4["avg_pf"] + 2:
        home_score += 0.5
        reasons_home.append("Better offense (scoring)")
    elif away_last4["avg_pf"] > home_last4["avg_pf"] + 2:
        away_score += 0.5
        reasons_away.append("Better offense (scoring)")

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

    # ADVANCED STATS (only if season provided and package available)
    if season and NFL_ADVANCED_STATS_AVAILABLE:
        try:
            home_epa = get_team_epa(home_team, season)
            away_epa = get_team_epa(away_team, season)

            if home_epa and away_epa:
                epa_diff = home_epa["net_epa"] - away_epa["net_epa"]
                if epa_diff > 0.05:
                    home_score += 0.5
                    reasons_home.append(
                        f"EPA advantage ({home_epa['net_epa']:.2f} vs {away_epa['net_epa']:.2f})"
                    )
                elif epa_diff < -0.05:
                    away_score += 0.5
                    reasons_away.append(
                        f"EPA advantage ({away_epa['net_epa']:.2f} vs {home_epa['net_epa']:.2f})"
                    )

            home_qb = get_qb_status(home_team, season)
            away_qb = get_qb_status(away_team, season)

            if home_qb and home_qb.get("is_backup"):
                away_score += 0.6
                reasons_away.append(
                    f"{home_team} missing starting QB ({home_qb.get('qb_name', 'Unknown')})"
                )

            if away_qb and away_qb.get("is_backup"):
                home_score += 0.6
                reasons_home.append(
                    f"{away_team} missing starting QB ({away_qb.get('qb_name', 'Unknown')})"
                )
        except Exception:
            # Graceful fallback if advanced stats fail
            pass

    # Home field (light)
    home_score += 0.4
    reasons_home.append("Home field")

    diff = home_score - away_score
    lean = "No clear edge"
    if diff > 1.0:
        lean = f"{home_team} ML (strong lean)"
    elif diff > 0.3:
        lean = f"{home_team} ML (slight lean)"
    elif diff < -1.0:
        lean = f"{away_team} ML (strong lean)"
    elif diff < -0.3:
        lean = f"{away_team} ML (slight lean)"

    return home_score, away_score, {"home_reasons": reasons_home, "away_reasons": reasons_away, "lean": lean}
