export default function ProBadge({ label = 'Pro' }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/20 border border-indigo-400/40 px-2 py-0.5 text-[11px] font-semibold text-indigo-100">
      {label}
    </span>
  )
}
