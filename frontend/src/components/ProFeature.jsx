import { classNames } from '../utils/classNames'

export default function ProFeature({ children, userTier, blur = false, fallback = null }) {
  const isPro = userTier === 'pro'
  if (isPro) return children

  if (blur) {
    return (
      <div className="relative">
        <div className="blur-sm pointer-events-none select-none">{children}</div>
        <div
          className={classNames(
            'absolute inset-0 flex items-center justify-center rounded-xl',
            'bg-slate-900/70 border border-slate-700 text-slate-200'
          )}
        >
          <div className="text-center text-sm">
            <div className="font-semibold mb-1">Pro feature</div>
            <div className="text-slate-400 mb-2">Upgrade to unlock</div>
            <a
              href="/upgrade"
              className="inline-flex items-center justify-center rounded-lg bg-gradient-to-r from-indigo-500 to-purple-600 px-3 py-1 text-xs font-semibold text-white shadow"
            >
              Upgrade
            </a>
          </div>
        </div>
      </div>
    )
  }

  if (fallback) return fallback

  return null
}
