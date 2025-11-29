import { classNames } from '../utils/classNames'

const sports = [
  { id: 'nhl', label: 'NHL', icon: '🏒' },
  { id: 'nfl', label: 'NFL', icon: '🏈' },
]

export default function SportTabs({ active, onChange }) {
  return (
    <div className="flex gap-3 mb-4">
      {sports.map((s) => {
        const isActive = active === s.id
        return (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            className={classNames(
              'flex-1 rounded-2xl px-4 py-3 text-center font-semibold transition transform',
              'border border-slate-700',
              isActive
                ? 'bg-gradient-to-r from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/30 text-white hover:-translate-y-0.5'
                : 'bg-slate-900 hover:border-indigo-500/60 hover:-translate-y-0.5'
            )}
          >
            <span className="text-lg mr-2">{s.icon}</span>
            {s.label}
          </button>
        )
      })}
    </div>
  )
}
