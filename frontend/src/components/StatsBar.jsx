const statItems = [
  { key: 'record', label: 'Record' },
  { key: 'roi', label: 'ROI' },
  { key: 'streak', label: 'Streak' },
  { key: 'picksCount', label: 'Picks' },
]

export default function StatsBar({ record, roi, streak, picksCount }) {
  const values = { record, roi, streak, picksCount }
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
      {statItems.map((item) => (
        <div
          key={item.key}
          className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-3 shadow-sm hover:border-indigo-400/60 transition"
        >
          <div className="text-xs uppercase tracking-wide text-slate-400">{item.label}</div>
          <div className="text-lg font-semibold mt-1">{values[item.key] ?? 'N/A'}</div>
        </div>
      ))}
    </div>
  )
}
