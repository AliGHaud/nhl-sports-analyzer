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
import json
from pathlib import Path

from datetime import datetime, timedelta
from typing import Dict, Any, List
from functools import partial
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
except ImportError:  # fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

from data_sources import load_games_from_espn_date_range
from api import _lean_scores
from nhl_analyzer import model_probs_from_scores

# Local helper for odds data
SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from odds_loader import load_clean_odds  # noqa: E402


def validate_backtest_input(odds_data: Dict, start_date: str, end_date: str) -> None:
    """Basic validation that the odds file has content and required fields."""
    if odds_data is None:
        raise ValueError("Odds data is None; cannot validate.")
    if not odds_data:
        raise ValueError("Empty odds file; no rows found.")
    sample = next(iter(odds_data.values()))
    required_keys = {"home_ml_close", "away_ml_close", "date", "home_team", "away_team"}
    missing = [k for k in required_keys if k not in sample]
    if missing:
        raise ValueError(f"Odds rows missing required columns: {missing}")
    print(
        f"Validated odds file with {len(odds_data)} rows "
        f"for range {start_date} to {end_date}."
    )


def calibration_bucket_label(prob: float, step: float = 0.05) -> str:
    """Bucket probabilities into strings like '0.50-0.55'."""
    lo = max(0.0, math.floor(prob / step) * step)
    hi = min(1.0, lo + step)
    return f"{lo:.2f}-{hi:.2f}"


def find_odds_for_game(game: dict, odds_lookup: Dict) -> dict | None:
    """Return odds entry for a game, checking +/- one day for timezone drift."""
    if not odds_lookup:
        return None
    home = game.get("home_team")
    away = game.get("away_team")
    try:
        game_date = datetime.strptime(game["date"], "%Y-%m-%d")
    except Exception:
        return None

    for offset in [0, -1, 1]:
        check_date = (game_date + timedelta(days=offset)).strftime("%Y-%m-%d")
        key = (check_date, home, away)
        odds = odds_lookup.get(key)
        if odds and odds.get("home_ml_close") and odds.get("away_ml_close"):
            return odds
    return None


def prefilter_games_with_odds(games: List[dict], odds_lookup: Dict) -> List[dict]:
    """Filter the games list to only those where we find a matching odds entry."""
    if not odds_lookup:
        return games
    filtered = []
    for g in games:
        if find_odds_for_game(g, odds_lookup):
            filtered.append(g)
    print(f"Pre-filtered to {len(filtered)} games with odds data")
    return filtered


def process_game(
    g: dict,
    games: List[dict],
    odds_lookup: Dict,
    min_edge: float,
    min_edge_dog: float,
    prob_threshold: float,
    stake: float,
    cache_scores: bool = False,
    score_cache: Dict | None = None,
):
    """Process a single game's metrics and optional bet."""
    try:
        home = g["home_team"]
        away = g["away_team"]
        hg = int(g["home_goals"])
        ag = int(g["away_goals"])
    except Exception:
        return {"skip": True, "failure": False}

    outcome = 1 if hg > ag else 0

    cache_key = (home, away, g.get("date"))
    try:
        if cache_scores and score_cache is not None and cache_key in score_cache:
            home_score, away_score = score_cache[cache_key]
        else:
            home_score, away_score, _ = _lean_scores(
                home,
                away,
                games,
                force_refresh=False,
                injuries_home=None,
                injuries_away=None,
                projected_goalies=None,
            )
            if cache_scores and score_cache is not None:
                score_cache[cache_key] = (home_score, away_score)
    except Exception:
        return {"skip": True, "failure": True}

    prob_home, prob_away = model_probs_from_scores(home_score, away_score)
    odds_entry = find_odds_for_game(g, odds_lookup) if odds_lookup else None
    odds_hit = bool(odds_entry)
    bet_result = None

    if odds_entry:
        home_ml = float(odds_entry["home_ml_close"])
        away_ml = float(odds_entry["away_ml_close"])
        implied_home = american_to_prob(home_ml)
        implied_away = american_to_prob(away_ml)

        home_edge = prob_home - implied_home
        away_edge = prob_away - implied_away

        bet_side = None
        edge_value = 0.0
        prob_value = 0.0
        implied_value = 0.0
        line_value = 0.0
        win_flag = False
        other_implied = 0.0

        home_is_favorite = implied_home >= implied_away
        away_is_favorite = implied_away >= implied_home
        home_edge_required = min_edge if home_is_favorite else min_edge_dog
        away_edge_required = min_edge if away_is_favorite else min_edge_dog

        if (
            prob_home >= prob_threshold
            and home_edge > home_edge_required
            and home_edge > away_edge
        ):
            bet_side = "home"
            edge_value = home_edge
            prob_value = prob_home
            implied_value = implied_home
            line_value = home_ml
            win_flag = outcome == 1
            other_implied = implied_away
        elif (
            prob_away >= prob_threshold
            and away_edge > away_edge_required
            and away_edge > home_edge
        ):
            bet_side = "away"
            edge_value = away_edge
            prob_value = prob_away
            implied_value = implied_away
            line_value = away_ml
            win_flag = outcome == 0
            other_implied = implied_home

        if bet_side:
            decimal_odds = american_to_decimal(line_value)
            profit = stake * (decimal_odds - 1.0) if win_flag else -stake
            fav_key = "favorite" if implied_value >= other_implied else "underdog"
            ev_units = prob_value * (decimal_odds - 1.0) - (1 - prob_value)
            bet_result = {
                "side": bet_side,
                "edge": edge_value,
                "win": win_flag,
                "profit": profit,
                "fav_key": fav_key,
                "prob": prob_value,
                "stake": stake,
                "ev": ev_units,
                "decimal_odds": decimal_odds,
            }

    result = {
        "skip": False,
        "failure": False,
        "correct": (prob_home >= 0.5 and outcome == 1)
        or (prob_home < 0.5 and outcome == 0),
        "ll": log_loss(prob_home, outcome),
        "brier": brier(prob_home, outcome),
        "bucket": bucket(prob_home),
        "outcome": outcome,
        "odds_hit": odds_hit,
        "bet": bet_result,
        "date": g.get("date"),
        "home": home,
        "away": away,
    }
    return result


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


def calculate_potd_results(all_bets_by_date: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Select highest EV bet per day and summarize results."""
    potd_picks: List[Dict] = []

    for game_date in sorted(all_bets_by_date.keys()):
        day_bets = all_bets_by_date[game_date]
        if not day_bets:
            continue
        best_bet = max(day_bets, key=lambda x: x.get("ev", 0))
        best_bet = best_bet.copy()
        best_bet["date"] = game_date
        potd_picks.append(best_bet)

    if not potd_picks:
        return {
            "total_days": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "avg_odds": 0.0,
            "best_streak": 0,
            "worst_streak": 0,
            "picks": [],
        }

    wins = sum(1 for p in potd_picks if p.get("win"))
    losses = len(potd_picks) - wins
    profit = sum(p.get("profit", 0.0) for p in potd_picks)
    roi = (profit / len(potd_picks)) * 100 if potd_picks else 0.0
    win_rate = (wins / len(potd_picks)) * 100 if potd_picks else 0.0
    avg_odds = (
        sum(p.get("decimal_odds", 0.0) for p in potd_picks) / len(potd_picks)
        if potd_picks
        else 0.0
    )

    best_streak = 0
    worst_streak = 0
    current_win_streak = 0
    current_lose_streak = 0

    for pick in potd_picks:
        if pick.get("win"):
            current_win_streak += 1
            current_lose_streak = 0
            best_streak = max(best_streak, current_win_streak)
        else:
            current_lose_streak += 1
            current_win_streak = 0
            worst_streak = max(worst_streak, current_lose_streak)

    return {
        "total_days": len(potd_picks),
        "wins": wins,
        "losses": losses,
        "profit": profit,
        "roi": roi,
        "win_rate": win_rate,
        "avg_odds": avg_odds,
        "best_streak": best_streak,
        "worst_streak": worst_streak,
        "picks": potd_picks,
    }


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


def backtest(
    start_date: str,
    end_date: str,
    odds_file: str | None = None,
    min_edge: float = 0.05,
    min_edge_dog: float = 0.22,
    min_prob: float = 0.5,
    stake: float = 1.0,
    processes: int = 1,
    quiet: bool = False,
) -> Dict[str, Any]:
    games = load_games_from_espn_date_range(start_date, end_date, force_refresh=False)
    if not games:
        print("No games loaded for range", start_date, end_date)
        return {}
    score_cache: Dict = {}

    total = 0
    ll_sum = 0.0
    brier_sum = 0.0
    correct = 0
    buckets = defaultdict(lambda: {"n": 0, "wins": 0})
    failures = 0
    odds_hits = 0
    edge_sum = 0.0
    bet_count = 0
    bet_wins = 0
    total_profit = 0.0
    total_staked = 0.0
    fav_split = {
        "favorite": {"bets": 0, "wins": 0, "profit": 0.0},
        "underdog": {"bets": 0, "wins": 0, "profit": 0.0},
    }
    side_split = {
        "home": {"bets": 0, "wins": 0, "profit": 0.0},
        "away": {"bets": 0, "wins": 0, "profit": 0.0},
    }

    all_bets_by_date: Dict[str, List[Dict]] = defaultdict(list)

    odds_lookup = {}
    bet_calibration: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"bets": 0, "wins": 0}
    )

    if odds_file:
        odds_lookup = load_clean_odds(odds_file)
        validate_backtest_input(odds_lookup, start_date, end_date)
        games = prefilter_games_with_odds(games, odds_lookup)

    prob_threshold = max(min_prob, 0.5)

    processes = int(processes or 1)
    if processes <= 0:
        processes = cpu_count()

    def apply_result(res: Dict[str, Any]):
        nonlocal total, correct, ll_sum, brier_sum, failures, odds_hits, bet_count, bet_wins, edge_sum, total_profit, total_staked
        if not res or res.get("skip"):
            if res.get("failure"):
                failures += 1
            return
        total += 1
        correct += 1 if res["correct"] else 0
        ll_sum += res["ll"]
        brier_sum += res["brier"]
        bucket_label = res["bucket"]
        buckets[bucket_label]["n"] += 1
        buckets[bucket_label]["wins"] += res["outcome"]
        if res["odds_hit"]:
            odds_hits += 1
        bet_info = res.get("bet")
        if bet_info:
            bet_count += 1
            edge_sum += bet_info["edge"]
            total_staked += bet_info["stake"]
            total_profit += bet_info["profit"]
            if bet_info["win"]:
                bet_wins += 1
            fav_key = bet_info["fav_key"]
            fav_split[fav_key]["bets"] += 1
            fav_split[fav_key]["profit"] += bet_info["profit"]
            if bet_info["win"]:
                fav_split[fav_key]["wins"] += 1
            side_key = bet_info["side"]
            side_split[side_key]["bets"] += 1
            side_split[side_key]["profit"] += bet_info["profit"]
            if bet_info["win"]:
                side_split[side_key]["wins"] += 1
            bucket_key = calibration_bucket_label(bet_info["prob"])
            bet_calibration[bucket_key]["bets"] += 1
            if bet_info["win"]:
                bet_calibration[bucket_key]["wins"] += 1

            game_date = res.get("date")
            if game_date:
                all_bets_by_date[game_date].append(
                    {
                        "home": res.get("home"),
                        "away": res.get("away"),
                        "side": bet_info["side"],
                        "ev": bet_info.get("ev", 0.0),
                        "edge": bet_info["edge"],
                        "prob": bet_info["prob"],
                        "win": bet_info["win"],
                        "profit": bet_info["profit"],
                        "fav_key": bet_info["fav_key"],
                        "decimal_odds": bet_info.get("decimal_odds", 0.0),
                    }
                )

    use_pool = processes > 1
    if use_pool:
        process_func = partial(
            process_game,
            games=games,
            odds_lookup=odds_lookup,
            min_edge=min_edge,
            min_edge_dog=min_edge_dog,
            prob_threshold=prob_threshold,
            stake=stake,
            cache_scores=False,
            score_cache=None,
        )
        with Pool(processes=processes) as pool:
            pool_iter = pool.imap(process_func, games)
            if not quiet:
                pool_iter = tqdm(
                    pool_iter, total=len(games), desc="Backtesting", unit="game"
                )
            for res in pool_iter:
                apply_result(res)
    else:
        iterator = games
        if not quiet:
            iterator = tqdm(games, desc="Backtesting", unit="game")
        for g in iterator:
            res = process_game(
                g,
                games,
                odds_lookup,
                min_edge,
                min_edge_dog,
                prob_threshold,
                stake,
                cache_scores=True,
                score_cache=score_cache,
            )
            apply_result(res)

    if total == 0:
        print("No usable games found.")
        return {}

    summary: Dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "total_games": total,
        "failures": failures,
        "accuracy": correct / total if total else 0.0,
        "avg_log_loss": ll_sum / total if total else None,
        "avg_brier": brier_sum / total if total else None,
        "odds_coverage": odds_hits,
        "bet_filters": {
            "stake": stake,
            "min_prob": prob_threshold,
            "min_edge": min_edge,
        },
        "bet_results": {
            "bets": bet_count,
            "wins": bet_wins,
            "avg_edge": edge_sum / bet_count if bet_count else 0.0,
            "profit_units": total_profit,
            "total_staked": total_staked,
            "roi": (total_profit / total_staked) if total_staked else 0.0,
        },
        "favorite_split": fav_split,
        "side_split": side_split,
        "calibration": {k: v for k, v in buckets.items()},
        "bet_calibration": {
            k: {
                "bets": v["bets"],
                "wins": v["wins"],
                "hit_rate": (v["wins"] / v["bets"]) if v["bets"] else None,
            }
            for k, v in sorted(bet_calibration.items())
        },
        "potd": None,
    }

    results_path = f"backtest_results_{start_date}_to_{end_date}_edge_{min_edge:.3f}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not quiet:
        print(f"\nSaved results to {results_path}")

    if not quiet:
        print(f"\nBacktest {start_date} to {end_date}")
        print(f"Games evaluated: {total} (failures: {failures})")
        print(f"Accuracy: {summary['accuracy']*100:.2f}%")
        print(f"Avg log loss: {summary['avg_log_loss']:.4f}")
        print(f"Avg Brier score: {summary['avg_brier']:.4f}")
        print("\nCalibration (home prob buckets):")
        for k in sorted(buckets.keys()):
            n = buckets[k]["n"]
            if n == 0:
                continue
            win_rate = buckets[k]["wins"] / n
            print(f"  {k}: n={n}, actual_home_win={win_rate*100:.1f}%")

        if odds_lookup:
            print(f"\nOdds coverage: {odds_hits}/{total}")
            print(
                f"Bet filter -> stake: {stake:.2f}u, min_prob: {prob_threshold:.2f}, "
                f"min_edge: {min_edge*100:.1f}% (fav), {min_edge_dog*100:.1f}% (dog)"
            )
            if bet_count:
                avg_edge_pct = summary["bet_results"]["avg_edge"] * 100.0
                hit_rate = bet_wins / bet_count * 100.0
                roi_pct = summary["bet_results"]["roi"] * 100.0
                print(f"Bets meeting filter: {bet_count}")
                print(
                    f"Avg edge: {avg_edge_pct:.2f}% | Hit rate: {hit_rate:.2f}% | "
                    f"Profit: {total_profit:.2f}u on {total_staked:.2f}u staked "
                    f"(ROI {roi_pct:.2f}%)"
                )
                print("Favorite vs. Underdog results:")
                for label, stats in fav_split.items():
                    if not stats["bets"]:
                        continue
                    fav_roi = (
                        stats["profit"] / (stats["bets"] * stake) * 100.0
                        if stake
                        else 0.0
                    )
                    fav_hit = stats["wins"] / stats["bets"] * 100.0
                    print(
                        f"  {label.capitalize()}: {stats['bets']} bets, "
                        f"hit {fav_hit:.1f}%, profit {stats['profit']:.2f}u "
                        f"(ROI {fav_roi:.2f}%)"
                    )
                print("Home vs. Away results:")
                for label, stats in side_split.items():
                    if not stats["bets"]:
                        continue
                    side_roi = (
                        stats["profit"] / (stats["bets"] * stake) * 100.0
                        if stake
                        else 0.0
                    )
                    side_hit = stats["wins"] / stats["bets"] * 100.0
                    print(
                        f"  {label.capitalize()}: {stats['bets']} bets, "
                        f"hit {side_hit:.1f}%, profit {stats['profit']:.2f}u "
                        f"(ROI {side_roi:.2f}%)"
                    )
                if bet_calibration:
                    print("\nBet calibration buckets:")
                    for label, stats in summary["bet_calibration"].items():
                        if not stats["bets"]:
                            continue
                        print(
                            f"  {label}: bets={stats['bets']}, "
                            f"hit={stats['hit_rate']*100:.1f}%"
                        )

                potd_results = calculate_potd_results(all_bets_by_date)
                if potd_results["total_days"]:
                    print("\n" + "=" * 50)
                    print("PICK OF THE DAY RESULTS (Highest EV per day)")
                    print("=" * 50)
                    print(
                        f"POTD Record: {potd_results['wins']}-{potd_results['losses']}"
                    )
                    print(f"Win Rate: {potd_results['win_rate']:.1f}%")
                    print(f"Profit: {potd_results['profit']:+.2f}u")
                    print(f"ROI: {potd_results['roi']:+.2f}%")
                    print(f"Avg Odds: {potd_results['avg_odds']:.2f}")
                    print(f"Best Win Streak: {potd_results['best_streak']}")
                    print(f"Worst Lose Streak: {potd_results['worst_streak']}")
                    summary["potd"] = {
                        "record": f"{potd_results['wins']}-{potd_results['losses']}",
                        "wins": potd_results["wins"],
                        "losses": potd_results["losses"],
                        "win_rate": potd_results["win_rate"],
                        "profit": potd_results["profit"],
                        "roi": potd_results["roi"],
                        "avg_odds": potd_results["avg_odds"],
                        "best_streak": potd_results["best_streak"],
                        "worst_streak": potd_results["worst_streak"],
                    }
            else:
                print("No games met the betting filter.")

    return summary


def run_edge_sweep(
    start_date: str,
    end_date: str,
    odds_file: str,
    edges: List[float],
    min_prob: float,
    stake: float,
    processes: int = 1,
    min_edge_dog: float = 0.22,
) -> List[Dict[str, Any]]:
    print(f"\nRunning edge sweep for {edges}")
    results = []
    for edge in edges:
        result = backtest(
            start_date,
            end_date,
            odds_file=odds_file,
            min_edge=edge,
            min_edge_dog=min_edge_dog,
            min_prob=min_prob,
            stake=stake,
            processes=processes,
            quiet=True,
        )
        if result:
            results.append(result)
            roi_pct = result["bet_results"]["roi"] * 100.0
            print(
                f"Edge {edge:.3f}: ROI={roi_pct:.2f}% | Bets={result['bet_results']['bets']} "
                f"| Profit={result['bet_results']['profit_units']:.2f}u"
            )
        else:
            print(f"Edge {edge:.3f}: No results")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest NHL lean engine.")
    parser.add_argument("--start", default="2024-10-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-04-15", help="End date YYYY-MM-DD")
    parser.add_argument("--odds", help="Path to clean odds CSV (optional)")
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum model edge (prob - market) required to count a bet (default 0.05)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.5,
        help="Minimum model probability threshold required to count a bet (default 0.5)",
    )
    parser.add_argument(
        "--min-edge-dog",
        type=float,
        default=0.22,
        help="Minimum edge for underdog bets (default 0.22 = 22%)",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="Flat stake (units) for each qualifying bet (default 1.0)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes for multiprocessing (default 1).",
    )
    parser.add_argument(
        "--edge-sweep",
        help="Comma-separated list of edge thresholds to evaluate (e.g., 0.01,0.03,0.05).",
    )
    args = parser.parse_args()
    if args.edge_sweep:
        sweep_values = []
        for chunk in args.edge_sweep.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                sweep_values.append(float(chunk))
            except ValueError:
                print(f"Skipping invalid edge value '{chunk}'")
        if not sweep_values:
            print("No valid edge values supplied for sweep. Running default backtest.")
            backtest(
                args.start,
                args.end,
                odds_file=args.odds,
                min_edge=args.min_edge,
                min_edge_dog=args.min_edge_dog,
                min_prob=args.min_prob,
                stake=args.stake,
                processes=args.processes,
            )
        else:
            run_edge_sweep(
                args.start,
                args.end,
                odds_file=args.odds,
                edges=sweep_values,
                min_prob=args.min_prob,
                stake=args.stake,
                processes=args.processes,
                min_edge_dog=args.min_edge_dog,
            )
    else:
        backtest(
            args.start,
            args.end,
            odds_file=args.odds,
            min_edge=args.min_edge,
            min_edge_dog=args.min_edge_dog,
            min_prob=args.min_prob,
            stake=args.stake,
            processes=args.processes,
        )
