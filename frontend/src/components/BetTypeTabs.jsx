import { classNames } from '../utils/classNames'

const betTypes = [
  { id: 'moneyline', label: '💰 Moneyline' },
  { id: 'spreads', label: '📊 Spreads' },
]

export default function BetTypeTabs({ active, onChange }) {
  return (
    <div className="flex gap-2 mb-4">
      {betTypes.map((b) => {
        const isActive = active === b.id
        return (
          <button
            key={b.id}
            onClick={() => onChange(b.id)}
            className={classNames(
              'flex-1 rounded-xl px-3 py-2 text-sm font-semibold border transition',
              isActive
                ? 'bg-indigo-500/20 border-indigo-500 text-white'
                : 'bg-slate-900 border-slate-700 hover:border-indigo-400 hover:text-indigo-200'
            )}
          >
            {b.label}
          </button>
        )
      })}
    </div>
  )
}
