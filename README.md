# NHL Betting Analyzer API (starter)

Minimal FastAPI wrapper around your NHL analysis code. Uses ESPN public data to fetch games/odds, compute leans/EV, and return JSON for a front-end.

## Quick start (Windows/PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
.\start-api.ps1
```
Open http://127.0.0.1:8000/docs for interactive docs (Swagger UI).

## Endpoints
- `GET /health` - simple status check.
- `GET /nhl/matchup?home=BOS&away=TOR[&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&force_refresh=true]`
  - Returns lean scores/reasons, EV (if odds available), and team profiles. Defaults to last 60 days if no dates provided. `force_refresh=true` bypasses cache.
- `GET /nhl/team?team=BOS[&start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&force_refresh=true]`
  - Returns a single team profile (full stats for the range, last5/10, home/away splits, streak, high-scoring trend).
- `GET /nhl/today[?force_refresh=true]`
  - Returns today's schedule (home/away abbreviations) for quick UI buttons.
- `GET /sports`
  - Lists supported sports and team codes (foundation for multi-sport).

## Notes
- Data comes from ESPN's public NHL endpoint; odds may be missing for some games.
- Caching: responses from ESPN are cached briefly to avoid hammering the API. Use `force_refresh=true` to bypass.
- Stop the server with Ctrl+C. Restart via `.\start-api.ps1`.

## Deploy (Render/Fly or similar)
- Ensure `Procfile` is present (this repo includes it): `web: uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}`.
- Set `PORT` env if your host requires it; most free tiers provide it automatically.
- Use `pip install -r requirements.txt` during build; run command `uvicorn api:app --host 0.0.0.0 --port $PORT`.
- Make sure your host allows outbound HTTP calls to ESPN (no extra credentials needed).

## Next steps (suggested)
- Add logging/metrics shipping (now basic logging exists).
- Add input validation and clearer error messages.
- Expand tests with FastAPI TestClient and mocked ESPN responses.
- Add a public-facing front-end and auth/paywall when ready.
