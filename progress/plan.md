# Progress / Plan

## Goal
Build a production-ready web app that analyzes NHL (first), using real stats/odds to surface leans and a “pick of the day,” with mobile-friendly UI. Make it monetizable (start with ads; add a paid tier for better picks/features/no ads), robust, and hardened against exploits. Future: expand to other sports, richer trends, and a dedicated odds API (paid if needed).

## Implemented
- FastAPI endpoints: /health, /sports, /teams, /nhl/matchup (profile + lean + EV + reasons), /nhl/team (profile), /nhl/today (schedule + odds), root UI.
- Analyzer logic: team profiles (full/last5/last10/home/away/high-scoring/streak), balanced lean engine with reasons and total lean, EV vs market (implied probs, edges, confidence grades, pick summary), CLI workflows (today, date range, future schedule mode).
- Data layer: ESPN scoreboard fetch with caching; date-range loader (completed games only); schedule loader; odds loader for today’s games; cache under data/cache; NHL Stats API stub.
- UI: dark SPA tester at `/` with matchup analysis, team analysis, today’s games with odds, raw JSON view; mobile-friendly styling; odds shown when present.
- Pick of the Day: API endpoint with EV/edge/model thresholds, noon gate in app timezone; UI card with date/tz and no-pick/early gate states.
- Pick/Pick gate persistence: daily pick cached; matchup snapshots now only created by the noon job (not on-demand) and served post-gate; all snapshots share the noon timestamp.
- Timezone-aware “today”: uses configurable `APP_TIMEZONE` (default America/New_York) for schedules/pick; includes timezone in responses/UI; tzdata added for Windows.
- Default lookback: analysis/pick windows now default to 45 days (configurable via `DEFAULT_LOOKBACK_DAYS`; was 60) to emphasize recency while keeping enough sample.
- Lean weighting refresh: single modest recency factor with capped streaks, flat home bonus, reduced home/road split weight, higher xG impact, lower GA weight, HDF removed, higher goalie edge weight, special teams kept modest.
- Rest/Back-to-back signals: profiles now include rest days and B2B flag; lean reasons adjust for rest gaps and B2B; UI team cards show rest.
- Pick caching: pick is cached per-day (in data/cache) and gated until configured time (default 12:00 in APP_TIMEZONE), with force_refresh to rerun.
- MoneyPuck integration: pull team-level xG% and high-danger share from MoneyPuck CSV (per season), filter to all-situations row, scale percentages correctly, and add modest lean adjustments/reasons when deltas are meaningful.
- Goalie heuristic: ESPN roster/stats to pick a probable starter by starts/sv%, apply small lean nudge when sv% gap is >= ~1%, with reasons.
- API-Sports (optional): API-Sports Hockey fixtures/odds via `API_SPORTS_KEY` (league=57), fallback to ESPN when unset/unavailable.
- Admin overrides: `/admin` console (token-protected) to manage manual injuries/goalie overrides stored in `data/cache/manual_overrides.json`; team/roster lookups exposed via `/teams` and `/nhl/roster`; unlock now enforced by valid token; remove buttons visible; persists when volume mounted on data/cache.
- ESPN abbreviation normalization: map 2-letter variants to 3-letter codes for schedule/odds matching.
- UI: modal popup for Today’s “Analyze” that shows matchup summary/reasons/EV/teams; Today and modal calls default to force_refresh for fresher data.
- Injuries: Rotowire JSON injury feed with TTL cache; `/nhl/injuries` endpoint; matchup responses include injuries (per team) + source/timestamp; UI card shows injuries. Injury entries now include `important` flag (softer thresholds: skater gp>=5 & toi>12; goalie gp/starts>=5 & sv%>=0.895) when stats provided.
- Tests: FastAPI TestClient coverage for matchup (with/without odds), bad date, unknown team, team endpoint, today endpoint with odds, sports endpoint.
- Docs: README with setup, endpoints, deploy notes, and ESPN/caching info.

## Decisions
- Scope now: NHL only; keep ESPN data for games/odds short term.
- Odds: plan to move to a dedicated odds API when ready (paid ok).
- Picks: add “pick of the day.”
- Trends: include home/away, last-N form, pace/possession, injuries, rest/back-to-back; add more per sport later.
- UX: must be mobile-friendly; include today’s board, matchup detail, team page, trends; keep logs/persistence.
- Monetization: lean toward ads first; paid tier for better picks/features/no ads. Requires user accounts/auth to gate features.
- Security: emphasize input validation, rate limiting (protect APIs from abusive traffic), logging/monitoring, and WAF/CDN shielding.

## Model signals, sources, and current weights
- xG share (MoneyPuck, season-long): cornerstone (~50% influence target). Recent xG trend is a light tweak.
- Goalie edge (ESPN roster/stats, sv% gap; plan GSAx/long-run SV%): heavier weight (~20–25%).
- Recency (last 10 W-L) + capped streak: modest (1.0 * win%, streak ±0.25 per game capped at ±0.75).
- Home: flat +0.75 bonus; home/road split weight trimmed (0.8 * win%).
- Rest/B2B: situational only (-0.5 B2B, +0.4 for 2+ extra rest days).
- Defense: GA last 10 with small weight (0.5); plan to swap to xGA/shot suppression.
- Special teams: small PP/PK net factor (~0.3); season-long, low weight.
- HDF%: de-emphasized (dropped) to avoid double-counting xG.
- Data feeds: ESPN (games/odds, fallback), API-Sports Hockey (fixtures/odds when `API_SPORTS_KEY` is set), MoneyPuck (xG/adv), ESPN roster/stats for goalies.
- Persistence: manual overrides persist across redeploys when data/cache is mounted on a Render volume; otherwise overrides are ephemeral.

## Next Actions
1) Monitor/tune new lean weights and de-duplication: watch outputs and adjust recency/home/GA/xG/goalie/special-teams balances if needed.
2) Pick-of-day tuning: monitor 45-day window impact and adjust EV/edge/model thresholds or recency weighting if needed.
3) UI polish for mobile: tighten layout, larger tap targets, quick filters, loading/error states; keep raw JSON hidden by default.
4) Odds reliability: research odds APIs (e.g., TheOddsAPI, OddsAPI, paid feeds) and plan the integration point.
5) Goalie metric upgrade: add GSAx/long-run SV% and starter/backup identification; weight goalie gap toward ~20–25% influence.
6) Defense metric upgrade: replace GA-based nudge with xGA/shot suppression when available; keep weight modest to avoid goalie double-counting.
7) Special teams refinement: use season-long ST goal differential or PPxG-PKxG (when available) as the small ST factor; keep weight low.
8) Injury/lineup adjustments: incorporate key player/goalie absences via simple ratings or flags to adjust lean scores.
9) Trends data: extend profiles to include rest/back-to-back flags and recent pace (skate to future multi-sport). Integrate free advanced stats (MoneyPuck) for xG/HDCF/GSAx/PP/PK; cache locally and merge into profiles with tiered weights (modest to avoid double counting). Defer injuries/line-history until a paid/reliable feed is chosen.
10) Persistence/logs: add request/response logging and a lightweight store for recent outputs (or file/DB when storage is enabled).
11) Auth/monetization: add user accounts, free vs. paid tier feature flags, and (later) ad placements vs. Stripe/Paddle for payments.
12) Security hardening: add rate limiting, stricter input validation, error handling, and WAF/CDN once on paid hosting.
13) Deploy: move to paid Render to avoid autosleep once ready.
14) Optional/context signals: monitor PDO/regression flags, schedule strength, penalty differential, finishing talent vs xG, and deserve-to-win deltas as qualitative overlays or small tweaks (avoid overweighting vs core signals).
15) Docs: keep README in sync (new endpoints/timezone/pick gate/lookback/recency weighting/home bonus/rest weighting/defense process/HDF stance/special teams/goalie weighting/de-duplication/context signals/admin console/overrides) and add UI usage notes.
16) Hook player stats into injury importance: pass skater/goalie stats into injury loader so `important` flags reflect real usage; highlight in UI.
17) Scheduled snapshots only: ensure only the noon job/admin endpoint creates matchup snapshots; pre-noon analyses always fresh, post-noon always from the common snapshot time.

## Advanced stats roadmap (tiered weighting)
- Tier 1 (heavy impact): starter/goalie quality (xSV%/GSAx), xG/HDCF differential, PP vs PK mismatch, rest/travel (B2B/3-in-4/time zones), rush xGA, key injuries.
- Tier 2 (medium): O-zone faceoff%, zone entry/exit success, top D-pair TOI, forecheck vs breakout quality, goalie recent form (last 5).
- Tier 3 (light/optional): team chemistry/line stability, referee penalty rate, reverse line movement (only with reliable line-history feed).
- Data requirement: ESPN feed lacks these; need richer stat/injury/odds sources (e.g., xG/HDCF/goalie/injury/line-history APIs). Until sourced, keep missing signals off or very lightly weighted.
## Recommendations (for pick-of-day thresholds)
- Only consider sides with EV > +0.02 to +0.03 units and edge >= +2% vs market.
- Require model win prob >= 52–53% unless longshot logic is explicit.
- Prefer grades B/B+ and above; allow C+ only if edge >= 5% and EV >= 0.05u.
- Break ties by higher EV, then higher model prob, then smaller juice.
- If no candidate meets criteria: return “no pick today” explicitly.
- Use a freshness window (e.g., odds timestamp <= 15 minutes) before surfacing a pick.

## Nice-to-haves (UI/UX)
- “Today’s board” tab with compact cards: odds, model prob, edge, grade badge, and a tap-to-expand for reasons.
- “Pick of the day” featured card at top if available; otherwise a “no pick” state.
- Team search and recent teams shortcuts.
- Mobile: single-column stack, sticky action buttons, hide raw JSON behind a toggle.
