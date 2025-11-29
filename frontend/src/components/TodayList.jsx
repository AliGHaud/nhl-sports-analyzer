import GameCard from './GameCard'

export default function TodayList({ sport, games, isLoading, error, onRefresh, onAnalysis }) {
  return (
    <div className="mt-6">
      <div className="flex items-center justify-between mb-2">
        <div className="text-lg font-semibold text-slate-100">Today's Games</div>
        <button
          onClick={onRefresh}
          className="rounded-lg border border-indigo-500/50 px-3 py-1 text-sm text-indigo-200 hover:bg-indigo-500/10"
        >
          Refresh
        </button>
      </div>
      {error && (
        <div className="mb-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
          {error}
        </div>
      )}
      {isLoading ? (
        <div className="text-sm text-slate-400">Loading games...</div>
      ) : (
        <div className="grid gap-3">
          {games.map((g, idx) => (
            <GameCard
              key={`${g.home_team || g.home}-${g.away_team || g.away}-${idx}`}
              sport={sport}
              game={g}
              onAnalysis={onAnalysis}
            />
          ))}
          {!games.length && <div className="text-sm text-slate-400">No games found for today.</div>}
        </div>
      )}
    </div>
  )
}
