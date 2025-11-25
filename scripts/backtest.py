"""
Simple backtest harness for the NHL lean engine.

Usage (default full season window):
  python scripts/backtest.py

Custom dates:
  python scripts/backtest.py --start 2024-10-01 --end 2025-04-15
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from data_sources import load_games_from_espn_date_range
from api import _lean_scores
from nhl_analyzer import model_probs_from_scores

# Local helper for odds data
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from odds_loader import load_clean_odds  # noqa: E402


def log_loss(prob, outcome):
    """Binary log loss."""
    # Clamp to avoid log(0)
    p = min(max(prob, 1e-9), 1 - 1e-9)
    return -math.log(p if outcome == 1 else (1 - p))


def brier(prob, outcome):
    return (prob - outcome) ** 2


def bucket(prob, width=0.1):
    """Return bucket string like '0.5-0.6'."""
    lo = math.floor(prob / width) * width
    hi = lo + width
    return f"{lo:.1f}-{hi:.1f}"


def american_to_prob(us: float) -> float:
    if us is None:
        raise ValueError("missing odds value")
    if us > 0:
        return 100.0 / (us + 100.0)
    return -us / (-us + 100.0)


def american_to_decimal(us: float) -> float:
    if us > 0:
        return us / 100.0 + 1.0
    return 100.0 / (-us) + 1.0


def calc_ev(prob: float, us_line: float) -> float:
    """Expected value in units for a 1u bet."""
    dec = american_to_decimal(us_line)
    return prob * (dec - 1.0) - (1.0 - prob)


def backtest(start_date: str, end_date: str, odds_file: str | None = None):
    games = load_games_from_espn_date_range(start_date, end_date, force_refresh=False)
    if not games:
        print("No games loaded for range", start_date, end_date)
        return

    total = 0
    ll_sum = 0.0
    brier_sum = 0.0
    correct = 0
    buckets = defaultdict(lambda: {"n": 0, "wins": 0})
    failures = 0
    odds_hits = 0
    ev_sum = 0.0
    edge_sum = 0.0
    ev_count = 0

    odds_lookup = {}
    if odds_file:
        odds_lookup = load_clean_odds(odds_file)

    for g in games:
        try:
            home = g["home_team"]
            away = g["away_team"]
            hg = int(g["home_goals"])
            ag = int(g["away_goals"])
        except Exception:
            continue

        outcome = 1 if hg > ag else 0  # home win = 1

        try:
            home_score, away_score, _ = _lean_scores(
                home,
                away,
                games,
                force_refresh=False,
                injuries_home=None,
                injuries_away=None,
                projected_goalies=None,
            )
        except Exception:
            failures += 1
            continue

        prob_home, prob_away = model_probs_from_scores(home_score, away_score)
        total += 1
        correct += 1 if (prob_home >= 0.5 and outcome == 1) or (prob_home < 0.5 and outcome == 0) else 0
        ll_sum += log_loss(prob_home, outcome)
        brier_sum += brier(prob_home, outcome)

        b = bucket(prob_home)
        buckets[b]["n"] += 1
        buckets[b]["wins"] += outcome

        if odds_lookup:
            from datetime import datetime, timedelta

            # Try exact date first, then ±1 day for timezone mismatches
            game_date = datetime.strptime(g["date"], "%Y-%m-%d")
            day_before = (game_date - timedelta(days=1)).strftime("%Y-%m-%d")
            day_after = (game_date + timedelta(days=1)).strftime("%Y-%m-%d")

            odds = None
            for date_key in [g["date"], day_before, day_after]:
                key = (date_key, home, away)
                if key in odds_lookup:
                    odds = odds_lookup[key]
                    break

            if odds and odds.get("home_ml_close") and odds.get("away_ml_close"):
                odds_hits += 1
                home_ml = float(odds["home_ml_close"])
                implied_home = american_to_prob(home_ml)
                edge = prob_home - implied_home
                edge_sum += edge
                ev_sum += calc_ev(prob_home, home_ml)
                ev_count += 1

    if total == 0:
        print("No usable games found.")
        return

    print(f"Backtest {start_date} to {end_date}")
    print(f"Games evaluated: {total} (failures: {failures})")
    print(f"Accuracy: {correct/total*100:.2f}%")
    print(f"Avg log loss: {ll_sum/total:.4f}")
    print(f"Avg Brier score: {brier_sum/total:.4f}")
    print("\nCalibration (home prob buckets):")
    for k in sorted(buckets.keys()):
        n = buckets[k]["n"]
        if n == 0:
            continue
        win_rate = buckets[k]["wins"] / n
        print(f"  {k}: n={n}, actual_home_win={win_rate*100:.1f}%")

    if odds_lookup:
        print(f"\nOdds coverage: {odds_hits}/{total}")
        if ev_count:
            print(
                f"Avg edge: {edge_sum/ev_count*100:.2f}% | "
                f"Avg EV (1u): {ev_sum/ev_count:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest NHL lean engine.")
    parser.add_argument("--start", default="2024-10-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-04-15", help="End date YYYY-MM-DD")
    parser.add_argument("--odds", help="Path to clean odds CSV (optional)")
    args = parser.parse_args()
    backtest(args.start, args.end, odds_file=args.odds)
