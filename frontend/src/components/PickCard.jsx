function formatPct(val) {
  if (val === undefined || val === null) return 'N/A'
  const num = typeof val === 'number' ? val : parseFloat(val)
  if (Number.isNaN(num)) return 'N/A'
  return `${(num * (num < 1 ? 100 : 1)).toFixed(1)}%`
}

export default function PickCard({ pick = {}, rank, betType }) {
  const {
    home_team,
    away_team,
    pick_side,
    spread,
    edge,
    ev,
    model_prob,
    elo_spread,
    market_spread,
    disagreement,
  } = pick

  const matchup = `${away_team || 'AWY'} @ ${home_team || 'HOME'}`
  const pickText =
    pick_side === 'home'
      ? `${home_team || 'HOME'} ${spread ? `${spread > 0 ? '+' : ''}${spread}` : ''}`
      : `${away_team || 'AWY'} ${spread ? `${spread > 0 ? '+' : ''}${spread}` : ''}`

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/70 px-4 py-3 shadow-sm hover:border-indigo-400/60 hover:-translate-y-0.5 transition fade-in">
      <div className="flex items-center justify-between mb-1">
        <div className="text-sm text-slate-400">#{rank}</div>
        <div className="text-xs px-2 py-1 rounded-full bg-indigo-500/20 text-indigo-200 border border-indigo-500/40">
          {betType === 'spreads' ? 'Spread' : 'Moneyline'}
        </div>
      </div>
      <div className="text-sm text-slate-400">{matchup}</div>
      <div className="text-xl font-semibold mb-2">{pickText}</div>
      <div className="flex flex-wrap gap-2 text-sm">
        <Chip label="Model" value={formatPct(model_prob)} />
        <Chip
          label="Edge"
          value={
            edge !== undefined && edge !== null
              ? `${(edge * (Math.abs(edge) < 1 ? 100 : 1)).toFixed(2)}%`
              : disagreement !== undefined
              ? `${disagreement > 0 ? '+' : ''}${disagreement.toFixed(1)} pts`
              : 'N/A'
          }
        />
        <Chip label="EV" value={ev !== undefined ? `${ev.toFixed(3)}u` : 'N/A'} />
        {betType === 'spreads' && (
          <>
            <Chip
              label="Elo"
              value={elo_spread !== undefined ? elo_spread.toFixed(1) : 'N/A'}
            />
            <Chip
              label="Market"
              value={market_spread !== undefined ? market_spread.toFixed(1) : 'N/A'}
            />
          </>
        )}
      </div>
    </div>
  )
}

function Chip({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-800/80 border border-slate-700 px-2 py-1">
      <span className="text-xs text-slate-400 mr-1">{label}:</span>
      <span className="font-semibold">{value}</span>
    </div>
  )
}
