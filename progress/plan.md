# Progress / Plan

## Goal
Build a production-ready web app that analyzes NHL (first), using real stats/odds to surface leans and a “pick of the day,” with mobile-friendly UI. Make it monetizable (start with ads; add a paid tier for better picks/features/no ads), robust, and hardened against exploits. Future: expand to other sports, richer trends, and a dedicated odds API (paid if needed).

## Implemented
- FastAPI endpoints: /health, /sports, /teams, /nhl/matchup (profile + lean + EV + reasons), /nhl/team (profile), /nhl/today (schedule + odds), root UI.
- Analyzer logic: team profiles (full/last5/last10/home/away/high-scoring/streak), balanced lean engine with reasons and total lean, EV vs market (implied probs, edges, confidence grades, pick summary), CLI workflows (today, date range, future schedule mode).
- Data layer: ESPN scoreboard fetch with caching; date-range loader (completed games only); schedule loader; odds loader for today’s games; cache under data/cache; NHL Stats API stub.
- UI: dark SPA tester at `/` with matchup analysis, team analysis, today’s games with odds, raw JSON view; mobile-friendly styling; odds shown when present.
- Pick of the Day: API endpoint with EV/edge/model thresholds, noon gate in app timezone; UI card with date/tz and no-pick/early gate states.
- Timezone-aware “today”: uses configurable `APP_TIMEZONE` (default America/New_York) for schedules/pick; includes timezone in responses/UI; tzdata added for Windows.
- Rest/Back-to-back signals: profiles now include rest days and B2B flag; lean reasons adjust for rest gaps and B2B; UI team cards show rest.
- Pick caching: pick is cached per-day (in data/cache) and gated until configured time (default 12:00 in APP_TIMEZONE), with force_refresh to rerun.
- MoneyPuck integration: pull team-level xG% and high-danger share from MoneyPuck CSV (per season), filter to all-situations row, scale percentages correctly, and add modest lean adjustments/reasons when deltas are meaningful.
- Goalie heuristic: ESPN roster/stats to pick a probable starter by starts/sv%, apply small lean nudge when sv% gap is >= ~1%, with reasons.
- ESPN abbreviation normalization: map 2-letter variants to 3-letter codes for schedule/odds matching.
- UI: modal popup for Today’s “Analyze” that shows matchup summary/reasons/EV/teams; Today and modal calls default to force_refresh for fresher data.
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

## Next Actions
1) Pick-of-day logic: implement a daily selector using confidence grade + EV thresholds (see Recommendations below) and expose it via API and UI card.
2) UI polish for mobile: tighten layout, larger tap targets, quick filters, loading/error states; keep raw JSON hidden by default.
3) Odds reliability: research odds APIs (e.g., TheOddsAPI, OddsAPI, paid feeds) and plan the integration point.
4) Trends data: extend profiles to include rest/back-to-back flags and recent pace (skate to future multi-sport). Integrate free advanced stats (MoneyPuck) for xG/HDCF/GSAx/PP/PK; cache locally and merge into profiles with tiered weights (modest to avoid double counting). Defer injuries/line-history until a paid/reliable feed is chosen.
5) Persistence/logs: add request/response logging and a lightweight store for recent outputs (or file/DB when storage is enabled).
6) Auth/monetization: add user accounts, free vs. paid tier feature flags, and (later) ad placements vs. Stripe/Paddle for payments.
7) Security hardening: add rate limiting, stricter input validation, error handling, and WAF/CDN once on paid hosting.
8) Deploy: move to paid Render to avoid autosleep once ready.
9) Docs: keep README in sync (new endpoints/timezone/pick gate) and add UI usage notes.

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
