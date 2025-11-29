# NFL Spread Model Rebuild Guide

## Overview

The current spread model is fundamentally flawed - it predicts margins using point differentials, converts to cover probability using a normal distribution, and consistently bets favorites that lose money. This guide outlines a complete rebuild based on sports betting research and best practices.

**Current Problems:**
- Model only finds "value" on favorites (157/157 bets on favorites)
- Massively overconfident (predicts 70%+ cover, hits ~48%)
- Using σ=35 when NFL actual margin σ≈13-14
- Margin predictions too extreme (±17 points for some games)
- No edge over the market

**Target Outcome:**
- Well-calibrated probabilities (when model says 60%, it hits ~60%)
- Finds value on both favorites AND underdogs
- Positive ROI on backtests across multiple seasons

---

## Phase 1: Elo Rating System

### Why Elo?
Research shows: `spread ≈ rating_home – rating_away + HFA` explains NFL spreads with R²≈0.98. Elo ratings naturally regress, update weekly, and avoid the extreme predictions from raw point differentials.

### Implementation

**File:** `nfl_elo.py`

```python
"""
NFL Elo Rating System

Each team starts at 1500. Ratings update after each game based on:
- Margin of victory (capped to avoid blowout overweighting)
- Home field advantage
- Expected vs actual outcome
"""

DEFAULT_ELO = 1500
K_FACTOR = 20  # How quickly ratings change
HFA_ELO = 48   # Home field advantage in Elo points (~2.5-3 points on spread)
SPREAD_MULTIPLIER = 25  # Elo difference / 25 ≈ spread

def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected win probability for team A."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def margin_multiplier(margin: int) -> float:
    """
    Adjust K-factor based on margin of victory.
    Caps blowouts to prevent overreaction.
    """
    abs_margin = abs(margin)
    if abs_margin <= 1:
        return 1.0
    elif abs_margin <= 8:
        return 1.0 + 0.1 * (abs_margin - 1)
    else:
        # Log scaling for blowouts (diminishing returns)
        return 1.7 + 0.1 * math.log(abs_margin - 7)

def update_elo(winner_elo: float, loser_elo: float, margin: int, 
               winner_home: bool = False) -> tuple[float, float]:
    """
    Update Elo ratings after a game.
    Returns (new_winner_elo, new_loser_elo)
    """
    # Adjust for home field
    if winner_home:
        adjusted_winner = winner_elo + HFA_ELO
        adjusted_loser = loser_elo
    else:
        adjusted_winner = winner_elo
        adjusted_loser = loser_elo + HFA_ELO
    
    expected = expected_score(adjusted_winner, adjusted_loser)
    mult = margin_multiplier(margin)
    
    change = K_FACTOR * mult * (1 - expected)
    
    return winner_elo + change, loser_elo - change

def elo_to_spread(home_elo: float, away_elo: float) -> float:
    """
    Convert Elo difference to predicted spread.
    Positive = home favored, negative = away favored.
    """
    elo_diff = (home_elo + HFA_ELO) - away_elo
    return elo_diff / SPREAD_MULTIPLIER

def build_elo_history(games: list[dict], season_start_regress: float = 0.33) -> dict:
    """
    Build Elo ratings from game history.
    
    Args:
        games: List of games sorted by date
        season_start_regress: Regress toward mean at season start (1/3 typical)
    
    Returns:
        Dict mapping (team, date) -> elo_rating
    """
    ratings = defaultdict(lambda: DEFAULT_ELO)
    history = {}
    current_season = None
    
    for game in sorted(games, key=lambda g: g['date']):
        game_date = game['date']
        season = get_season(game_date)
        
        # Regress ratings at season start
        if season != current_season:
            if current_season is not None:
                for team in ratings:
                    ratings[team] = DEFAULT_ELO + (ratings[team] - DEFAULT_ELO) * (1 - season_start_regress)
            current_season = season
        
        home = game['home_team']
        away = game['away_team']
        
        # Store pre-game ratings
        history[(home, game_date)] = ratings[home]
        history[(away, game_date)] = ratings[away]
        
        # Update based on result
        home_score = game['home_goals']
        away_score = game['away_goals']
        margin = home_score - away_score
        
        if margin > 0:  # Home win
            ratings[home], ratings[away] = update_elo(
                ratings[home], ratings[away], margin, winner_home=True
            )
        elif margin < 0:  # Away win
            ratings[away], ratings[home] = update_elo(
                ratings[away], ratings[home], -margin, winner_home=False
            )
        # Ties: no update (rare in NFL)
    
    return history, ratings
```

### Key Parameters (Research-Backed Values):
- `K_FACTOR = 20`: FiveThirtyEight uses K=20, optimal for NFL's ~270 games/year
- `HFA_ELO = 48-65`: Home field advantage (~2.5-3 points spread equivalent)
- `SPREAD_MULTIPLIER = 25`: Elo diff / 25 ≈ spread
- `season_start_regress = 0.33`: Regress 1/3 toward mean at season start

### FiveThirtyEight Situational Adjustments:
- **Bye week**: +25 Elo for teams off bye
- **Short rest**: -5 Elo for road team on short rest (Thursday games)
- **Travel**: -4 Elo per 1000 miles traveled
- **Denver altitude**: -10 Elo when visiting Denver
- **Backup QB**: -80 Elo when backup starts (simplified QB adjustment)
- **Weather**: 538 tested and found NO measurable improvement - skip this
- **Coaching**: 538 tested and found NO payoff - skip this

### Early Season Strategy (Weeks 1-6):
Don't start everyone at 1500! Two approaches:
1. **Carry over + regress**: Use last season's final Elo, regressed 1/3 toward 1500
2. **Vegas seeding**: Use preseason win totals to set initial Elo (strong teams ~1650, weak ~1350)

Combining both is ideal: blend last year's rating with Vegas win total adjustment.

---

## Phase 2: Cover Probability Model

### Option A: Logistic Regression (RECOMMENDED FOR START)

**Research finding**: With <1500 rows, logistic regression often matches or outperforms XGBoost because it's simpler and more stable. A Carnegie Mellon NFL project chose logistic "due to its speed and ability to handle arbitrary features."

Predict cover directly rather than predicting margin then converting.

**File:** `nfl_cover_model.py`

```python
"""
Binary classification model: Does home team cover the spread?

Features:
- Elo difference (home - away, adjusted for HFA)
- Spread itself (market's view)
- Elo vs spread disagreement (key signal!)
- Rest advantage
- Home underdog flag (known bias)
- Divisional game flag
- Weather factors (if available)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

class NFLCoverModel:
    def __init__(self):
        # Use calibrated classifier for better probability estimates
        base_model = LogisticRegression(C=1.0, max_iter=1000)
        self.model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        self.feature_names = []
    
    def extract_features(self, game: dict, elo_home: float, elo_away: float, 
                         spread: float) -> np.ndarray:
        """
        Extract features for a single game.
        """
        features = []
        
        # 1. Elo-based predicted spread
        elo_spread = elo_to_spread(elo_home, elo_away)
        features.append(elo_spread)
        
        # 2. Market spread
        features.append(spread)
        
        # 3. Model vs market disagreement (KEY FEATURE)
        # Positive = model thinks home is better than market says
        disagreement = elo_spread - spread
        features.append(disagreement)
        
        # 4. Home underdog flag (known profitable bias)
        home_is_underdog = spread > 0
        features.append(1.0 if home_is_underdog else 0.0)
        
        # 5. Big favorite flag (favorites often don't cover big spreads)
        big_favorite = abs(spread) >= 7
        features.append(1.0 if big_favorite else 0.0)
        
        # 6. Rest advantage (if available)
        rest_diff = game.get('rest_diff', 0)
        features.append(rest_diff)
        
        # 7. Divisional game (more competitive, closer to spread)
        is_divisional = game.get('is_divisional', False)
        features.append(1.0 if is_divisional else 0.0)
        
        return np.array(features)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train on historical data. y = 1 if home covered, 0 otherwise."""
        self.model.fit(X, y)
    
    def predict_cover_prob(self, X: np.ndarray) -> np.ndarray:
        """Return P(home covers)."""
        return self.model.predict_proba(X)[:, 1]
```

### Option B: XGBoost + Logistic Ensemble

For better performance, ensemble tree-based and linear models:

```python
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

def build_ensemble():
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    lr = LogisticRegression(C=1.0, max_iter=1000)
    
    # Soft voting averages probabilities
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb), ('lr', lr)],
        voting='soft',
        weights=[0.6, 0.4]  # Weight XGB slightly more
    )
    
    # Wrap in calibrator
    return CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
```

---

## Phase 3: Calibration Pipeline

### Critical Research Finding:
- **Isotonic regression**: Works well with ≥1000 samples, but produces erratic "step" functions with fewer
- **Platt scaling**: More robust with limited data (parametric, fewer parameters)
- **With 800-1000 NFL games**: Isotonic is borderline - use Platt scaling OR pool data across seasons

**Recommendation**: Start with Platt scaling for safety, try isotonic only if pooling 4+ seasons.

**File:** `nfl_calibration.py`

```python
"""
Probability calibration using Platt scaling and isotonic regression.
"""

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import numpy as np

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        """
        method: 'isotonic' (non-parametric) or 'platt' (logistic)
        """
        self.method = method
        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            self.calibrator = LogisticRegression()
    
    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray):
        """
        Fit calibrator on validation set.
        raw_probs: Model's predicted probabilities
        outcomes: Actual outcomes (0 or 1)
        """
        if self.method == 'isotonic':
            self.calibrator.fit(raw_probs, outcomes)
        else:
            # Platt scaling: logistic regression on log-odds
            log_odds = np.log(raw_probs / (1 - raw_probs + 1e-10))
            self.calibrator.fit(log_odds.reshape(-1, 1), outcomes)
    
    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated ones."""
        if self.method == 'isotonic':
            return self.calibrator.predict(raw_probs)
        else:
            log_odds = np.log(raw_probs / (1 - raw_probs + 1e-10))
            return self.calibrator.predict_proba(log_odds.reshape(-1, 1))[:, 1]

def evaluate_calibration(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """
    Compute calibration metrics.
    Returns dict with Brier score, calibration curve data, and reliability stats.
    """
    # Brier score (lower is better, 0 = perfect)
    brier = np.mean((probs - outcomes) ** 2)
    
    # Calibration curve
    fraction_positive, mean_predicted = calibration_curve(
        outcomes, probs, n_bins=n_bins, strategy='uniform'
    )
    
    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
    ece = np.sum(np.abs(fraction_positive - mean_predicted) * bin_counts / len(probs))
    
    return {
        'brier_score': brier,
        'ece': ece,
        'calibration_curve': {
            'predicted': mean_predicted.tolist(),
            'actual': fraction_positive.tolist()
        }
    }
```

---

## Phase 4: Training Pipeline with Walk-Forward Validation

### Retraining Schedule (Research-Backed):
- **Weekly**: Update Elo ratings after each game (automatic with K=20)
- **Preseason**: Retrain the ML model (logistic/XGBoost) on all past seasons
- **Mid-season (optional)**: Recalibrate around Week 8 if performance drifts
- **Never**: Don't retrain weekly - too little data, will chase noise

**CRITICAL:** Never train on future data. Use strict time-ordered validation.

**File:** `nfl_spread_training.py`

```python
"""
Walk-forward training pipeline for NFL spread model.
"""

def walk_forward_validation(games: list[dict], odds_data: dict):
    """
    Train on past seasons, predict next season.
    
    Example splits:
    - Train 2017-2019, predict 2020
    - Train 2017-2020, predict 2021
    - Train 2017-2021, predict 2022
    """
    seasons = [2017, 2018, 2019, 2020, 2021, 2022]
    results = []
    
    for test_season in seasons[3:]:  # Start predicting from 2020
        train_seasons = [s for s in seasons if s < test_season]
        
        print(f"Training on {train_seasons}, testing on {test_season}")
        
        # Split data
        train_games = [g for g in games if get_season(g['date']) in train_seasons]
        test_games = [g for g in games if get_season(g['date']) == test_season]
        
        # Build Elo from training data
        elo_history, current_elos = build_elo_history(train_games)
        
        # Extract features and labels for training
        X_train, y_train = [], []
        for game in train_games:
            spread = get_spread_for_game(game, odds_data)
            if spread is None:
                continue
            
            home_elo = elo_history.get((game['home_team'], game['date']), DEFAULT_ELO)
            away_elo = elo_history.get((game['away_team'], game['date']), DEFAULT_ELO)
            
            features = extract_features(game, home_elo, away_elo, spread)
            label = 1 if did_home_cover(game, spread) else 0
            
            X_train.append(features)
            y_train.append(label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train model
        model = build_ensemble()
        model.fit(X_train, y_train)
        
        # Calibrate on last portion of training data
        calibrator = ProbabilityCalibrator(method='isotonic')
        val_probs = model.predict_proba(X_train[-200:])[:, 1]
        calibrator.fit(val_probs, y_train[-200:])
        
        # Test on holdout season
        X_test, y_test = [], []
        test_game_info = []
        
        # Continue building Elo into test season (simulating real-time)
        for game in sorted(test_games, key=lambda g: g['date']):
            spread = get_spread_for_game(game, odds_data)
            if spread is None:
                continue
            
            home_elo = current_elos[game['home_team']]
            away_elo = current_elos[game['away_team']]
            
            features = extract_features(game, home_elo, away_elo, spread)
            label = 1 if did_home_cover(game, spread) else 0
            
            X_test.append(features)
            y_test.append(label)
            test_game_info.append({
                'game': game,
                'spread': spread,
                'elo_spread': elo_to_spread(home_elo, away_elo)
            })
            
            # Update Elo after game (for next predictions)
            update_elos_for_game(current_elos, game)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Predict and calibrate
        raw_probs = model.predict_proba(X_test)[:, 1]
        calibrated_probs = calibrator.calibrate(raw_probs)
        
        # Evaluate
        cal_metrics = evaluate_calibration(calibrated_probs, y_test)
        
        results.append({
            'test_season': test_season,
            'n_games': len(y_test),
            'raw_brier': np.mean((raw_probs - y_test) ** 2),
            'calibrated_brier': cal_metrics['brier_score'],
            'calibration_data': cal_metrics
        })
    
    return results
```

---

## Phase 5: Betting Strategy

### Bet Sizing (Research-Backed):
- **Full Kelly**: Mathematically optimal but TOO AGGRESSIVE - leads to wild bankroll swings
- **Half Kelly**: Recommended - reduces volatility significantly
- **Quarter Kelly**: Even safer, good when edge estimates are uncertain
- **Flat betting**: Simplest, most robust when edge estimates are noisy

**Recommendation**: Start with flat 1-2% of bankroll per bet. Once model is validated with positive ROI across multiple seasons, consider quarter or half Kelly.

### Value Betting with Calibrated Probabilities

```python
def find_spread_bets(game: dict, calibrated_prob: float, spread: float,
                     home_price: float = -110, away_price: float = -110,
                     min_edge: float = 0.03) -> dict | None:
    """
    Find value bets where calibrated probability exceeds market implied.
    
    Args:
        calibrated_prob: P(home covers) after calibration
        spread: Market spread (negative = home favored)
        home_price: American odds for home spread
        away_price: American odds for away spread
        min_edge: Minimum edge required to bet (3% default)
    """
    # Implied probabilities from odds
    implied_home = american_to_prob(home_price)
    implied_away = american_to_prob(away_price)
    
    # Calibrated probabilities
    home_cover_prob = calibrated_prob
    away_cover_prob = 1 - calibrated_prob
    
    # Calculate edges
    home_edge = home_cover_prob - implied_home
    away_edge = away_cover_prob - implied_away
    
    # Check for value
    if home_edge >= min_edge:
        return {
            'side': 'home',
            'spread': spread,
            'prob': home_cover_prob,
            'implied': implied_home,
            'edge': home_edge,
            'odds': home_price
        }
    elif away_edge >= min_edge:
        return {
            'side': 'away',
            'spread': -spread,  # Away spread is opposite
            'prob': away_cover_prob,
            'implied': implied_away,
            'edge': away_edge,
            'odds': away_price
        }
    
    return None
```

### Home Underdog Boost

Research shows home underdogs historically cover >50%. Add explicit adjustment:

```python
def apply_home_underdog_boost(prob: float, spread: float, boost: float = 0.02) -> float:
    """
    Boost home cover probability when home team is underdog.
    
    Args:
        prob: Raw/calibrated P(home covers)
        spread: Market spread (positive = home is underdog)
        boost: Probability boost (2% default based on research)
    """
    if spread > 0:  # Home is underdog
        return min(prob + boost, 0.95)
    return prob
```

---

## Phase 6: Features to Add

### High-Value Features (PROVEN by FiveThirtyEight):

1. **Elo difference** - Core signal (R²≈0.98 for spread prediction)
2. **Elo vs spread disagreement** - Where model differs from market (KEY!)
3. **Home underdog flag** - Known bias, historically covers >50%
4. **Rest/bye week** - +25 Elo post-bye
5. **Travel distance** - -4 Elo per 1000 miles
6. **Denver altitude** - -10 Elo for visitors
7. **Short rest** - -5 Elo for Thursday road teams

### Medium-Value (add if data available):

8. **Backup QB flag** - -80 Elo adjustment
9. **Total (O/U)** - Low totals = more variance = underdog value
10. **Big spread flag** - Favorites often don't cover spreads ≥7

### SKIP THESE (538 tested, no improvement):

- ❌ Weather (rain/wind/snow)
- ❌ Coaching changes
- ❌ Revenge spots (too subjective)
- ❌ Divisional games (mixed evidence)

---

## Implementation Order

### Week 1: Elo Foundation
- [ ] Implement `nfl_elo.py` with basic Elo system
- [ ] Build historical Elo ratings for 2017-2022
- [ ] Validate Elo spread predictions vs actual spreads (should correlate well)
- [ ] Test: Elo predicted spread vs market spread correlation

### Week 2: Cover Model
- [ ] Implement `nfl_cover_model.py` with logistic regression
- [ ] Extract features (Elo diff, spread, disagreement, home dog)
- [ ] Train with walk-forward validation
- [ ] Test: Raw model accuracy and Brier score

### Week 3: Calibration
- [ ] Implement `nfl_calibration.py`
- [ ] Apply isotonic regression calibration
- [ ] Compare raw vs calibrated Brier scores
- [ ] Test: Calibration curves should be diagonal

### Week 4: Betting Integration
- [ ] Integrate with existing backtest framework
- [ ] Add value bet detection with calibrated probs
- [ ] Backtest across all seasons
- [ ] Test: Should find both favorite AND underdog value

### Week 5: Tuning & Enhancement
- [ ] Tune Elo parameters (K, HFA, regress)
- [ ] Add home underdog boost
- [ ] Try XGBoost ensemble
- [ ] Add weather/rest features if available

---

## Success Metrics

### Realistic Expectations (from research):
- **~55% ATS wins** can be profitable at -110 odds
- **>60% ATS** is extremely difficult in NFL spread betting
- Focus on ROI and CLV (Closing Line Value), not just accuracy

### Calibration (most important):
- Brier score < 0.24 (better than naive 0.25)
- ECE < 0.05 (expected calibration error)
- 60% bucket hits 58-62% of time

### Betting Performance:
- Positive ROI across 2+ seasons in walk-forward test
- Finds value on BOTH sides (not just favorites)
- Consistent CLV (beating closing lines)

### Comparison to Current Model:
| Metric | Current | Target |
|--------|---------|--------|
| Brier Score | 0.196 | < 0.190 |
| Calibration | Way off | ECE < 0.05 |
| Bet Distribution | 100% favorites | Mixed |
| ATS Win Rate | ~48% | >52% |
| ROI | -3% to -6% | > 0% |

---

## Files to Create

```
nfl_spread_v2/
├── nfl_elo.py              # Elo rating system
├── nfl_cover_model.py      # Binary cover prediction model
├── nfl_calibration.py      # Probability calibration
├── nfl_spread_training.py  # Walk-forward training pipeline
├── nfl_spread_features.py  # Feature extraction
├── nfl_spread_betting.py   # Value bet detection
└── backtest_spread_v2.py   # New backtest harness
```

---

## References

- FiveThirtyEight NFL Elo methodology
- NFelo margin probability analysis
- ATSwins AI calibration guide
- Poliscidata NFL betting market analysis
- Academic: CMU sports prediction project

---

## Key Research Findings Summary (ChatGPT Deep Search)

### Elo System
- **K=20** is optimal (FiveThirtyEight, Nate Silver, multiple public models)
- Grid-search K and HFA to minimize prediction error
- Seed initial Elo from last season + Vegas win totals (not flat 1500)

### Calibration
- **Isotonic**: Needs ≥1000 samples, overfits with less
- **Platt scaling**: More robust with limited data
- Always calibrate on held-out data (never same data as training)

### Model Choice
- **Logistic regression** often matches XGBoost with <1500 samples
- Start simple, add complexity only if validated improvement
- Use L1/L2 regularization to prevent overfitting

### Features
- 538 tested weather and coaching - **NO improvement**
- Focus on: Elo diff, rest, travel, altitude, QB status
- Include (Elo_diff - Spread) as key feature for mispricing

### Retraining
- Elo: Update weekly (automatic)
- ML model: Retrain preseason only, maybe mid-season recalibration
- Never retrain weekly - chases noise

### Bet Sizing
- Full Kelly too aggressive for uncertain edges
- Half Kelly or quarter Kelly recommended
- Flat 1-3% of bankroll is safest

### Validation
- Walk-forward only (train 2017-2019, test 2020, etc.)
- Track ROI and CLV, not just accuracy
- 55% ATS is profitable; 60%+ is extremely rare

---

## Notes

- Start simple (Elo + logistic) before adding complexity
- Calibration is more important than raw accuracy
- Home underdog bias is real - exploit it
- Always use walk-forward validation (no lookahead)
- Monitor for overfitting with limited NFL data (~270 games/season)
