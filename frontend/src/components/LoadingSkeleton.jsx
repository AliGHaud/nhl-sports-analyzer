export default function LoadingSkeleton() {
  return (
    <div className="grid gap-3">
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="h-24 rounded-2xl border border-slate-800 bg-slate-900/70 shimmer"
        />
      ))}
    </div>
  )
}
