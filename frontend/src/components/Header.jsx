import { classNames } from '../utils/classNames'

export default function Header({ onRefresh }) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4 fade-in">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center font-black text-lg shadow-lg shadow-indigo-500/40">
          SP
        </div>
        <div>
          <div className="text-xl font-semibold">Sports Predictor</div>
          <div className="text-sm text-slate-400">AI-Powered Picks</div>
        </div>
        <span
          className={classNames(
            'ml-2 inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-semibold',
            'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30',
            'pulse'
          )}
        >
          <span className="h-2 w-2 rounded-full bg-emerald-300 animate-pulse" />
          Live
        </span>
      </div>
      <button
        type="button"
        onClick={onRefresh}
        className="self-start sm:self-auto inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-600 px-4 py-2 text-sm font-semibold shadow-lg shadow-indigo-500/30 hover:from-indigo-400 hover:to-purple-500 transition transform hover:-translate-y-0.5"
      >
        <span>Refresh</span>
      </button>
    </div>
  )
}
