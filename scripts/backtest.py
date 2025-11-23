"""
Simple backtest harness for the NHL lean engine.

Usage (default full season window):
  python scripts/backtest.py

Custom dates:
  python scripts/backtest.py --start 2024-10-01 --end 2025-04-15
"""

import argparse
import math
from collections import defaultdict

from data_sources import load_games_from_espn_date_range
from api import _lean_scores
from nhl_analyzer import model_probs_from_scores


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


def backtest(start_date: str, end_date: str):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest NHL lean engine.")
    parser.add_argument("--start", default="2024-10-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-04-15", help="End date YYYY-MM-DD")
    args = parser.parse_args()
    backtest(args.start, args.end)
