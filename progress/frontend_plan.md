# Frontend Plan (React UI revamp)

## Goals
- Rebuild the UI to match the reference demo (`frontend/Reference html/sports_predictor_full_demo.html`):
  - Landing page
  - Login page
  - App experience (free/pro tiers)
  - Detailed matchup modal (reasons, goalie impact, injuries, EV/odds)
  - Bet tracker
  - Alerts
- Integrate existing API data (picks, spreads, today’s games, matchup analysis, injuries/goalies/EV where available).
- Keep Tailwind styling consistent with the demo (glass panels, gradients, motion).

## Phases
1) Routing & Page Shells (React Router)
   - Add routes for Landing, Login, App, Bet Tracker, Alerts.
   - Preserve existing Picks experience under `/app`.
   - Shared layout (header/nav) to switch sections.

2) App (Free/Pro) Experience
   - Surface picks, spreads, today’s games, analysis modal.
   - Pro/Free gating (UI-level stub until auth is wired).
   - Better empty/error states.

3) Analysis Modal Parity
   - Render full matchup payload: scores, model probs, odds/edge/EV, reasons, goalie impact, injuries.
   - Mirror old static modal layout.

4) Landing & Login Pages
   - Landing hero, value props, CTA.
   - Login page (stub auth state; real wiring later).

5) Bet Tracker & Alerts
   - Build UI shells and mock data.
   - Wire to backend when endpoints exist.

6) Polish & Theming
   - Match demo gradients, cards, motion.
   - Mobile responsiveness, hover/focus states.

## Notes from Implementation Guide
- Firebase is already configured for the old static UI; Google auth works there. Keep static UI as fallback until React auth is wired.
- All frontend changes remain local until ready to swap; do not touch production static files.
- Current API picks: `/nhl/pick`, `/nhl/picks` (wraps POTD), `/nfl/spread-picks`; auth/pro checks are in place server-side.
- Goal: match the reference demo (landing/login/App free/App pro/tracker/alerts) and eventually wire Firebase auth into the React UI.

## Next Actions
- Add React Router and page shells (Landing, Login, App/Picks, Bet Tracker, Alerts).
- Keep current Picks data flow under `/app`.
- Expand analysis modal with richer fields (reasons, goalie, injuries, EV).
