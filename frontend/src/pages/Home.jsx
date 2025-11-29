const stats = [
  { label: 'NHL ROI', value: '+12.4%' },
  { label: 'NFL Spread ROI', value: '+10.9%' },
  { label: 'Win Rate', value: '58.2%' },
  { label: 'Picks Tracked', value: '2,400+' },
]

const bullets = ['No credit card required', 'Cancel anytime', 'Full transparency']

const features = [
  { title: 'Elo-based models', desc: 'Time-tested, calibrated models that find edges the market misses.' },
  { title: 'Live odds & edges', desc: 'Updated edges vs. market lines with clear EV and confidence grades.' },
  { title: 'Detailed analysis', desc: 'Goalie impact, injuries, reasons, and EV breakdowns per matchup.' },
  { title: 'POTD & spreads', desc: 'Moneyline and spread picks with thresholds you control.' },
  { title: 'Alerts (soon)', desc: 'Be notified on line moves, new picks, and thresholds.' },
  { title: 'Bet tracker (soon)', desc: 'Track wagers, ROI, bankroll, and performance over time.' },
]

const tiers = [
  {
    name: 'Free',
    price: '$0',
    cta: 'Explore Free',
    perks: ['Today’s board (limited)', 'Single pick view', 'Basic matchup analysis'],
  },
  {
    name: 'Pro',
    price: '$19/mo',
    highlight: 'Best value',
    cta: 'Start Pro Trial',
    perks: ['Full board + POTD', 'Spreads + moneyline', 'Detailed modal (reasons/goalies)', 'Alerts & tracker (when ready)'],
  },
]

export default function Home() {
  return (
    <div className="fade-in relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -left-40 top-10 h-72 w-72 rounded-full bg-indigo-600/20 blur-3xl" />
        <div className="absolute right-0 top-1/4 h-80 w-80 rounded-full bg-purple-500/20 blur-3xl" />
      </div>

      <div className="relative space-y-8">
        <div className="rounded-3xl border border-slate-800 bg-gradient-to-br from-slate-900/90 to-slate-950 p-10 shadow-2xl shadow-indigo-500/10">
          <div className="inline-flex items-center gap-2 rounded-full border border-indigo-400/40 bg-indigo-500/10 px-3 py-1 text-xs font-semibold text-indigo-200 mb-4">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            Verified +12% ROI since 2023
          </div>
          <h1 className="text-4xl md:text-5xl font-black leading-tight text-slate-50">
            Data-Driven <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">Sports</span>
            <br />
            Betting Edge
          </h1>
          <p className="mt-4 text-lg text-slate-300 max-w-3xl">
            Elo-based models that find edges the market misses. No hype, no &quot;locks&quot; — just math, live odds, and transparent performance.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a
              href="/login"
              className="rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 hover:from-indigo-400 hover:to-purple-500 transition inline-flex items-center justify-center"
            >
              Start Free Trial →
            </a>
            <a
              href="/login"
              className="rounded-xl border border-slate-700 bg-slate-900 px-5 py-3 text-sm font-semibold text-slate-200 hover:border-indigo-400 hover:text-indigo-200 transition inline-flex items-center justify-center"
            >
              View Demo
            </a>
          </div>
          <div className="mt-5 flex flex-wrap gap-4 text-sm text-emerald-300">
            {bullets.map((b) => (
              <div key={b} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                {b}
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-4">
          {stats.map((s) => (
            <div
              key={s.label}
              className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4 shadow-sm shadow-slate-900/40"
            >
              <div className="text-3xl font-bold text-slate-50">{s.value}</div>
              <div className="text-sm text-slate-400 mt-1">{s.label}</div>
            </div>
          ))}
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-900/30">
          <div className="text-xs uppercase text-indigo-300 mb-3">Why Sports Predictor</div>
          <div className="grid gap-4 md:grid-cols-3">
            {features.map((f) => (
              <div key={f.title} className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
                <div className="text-sm font-semibold text-slate-100 mb-1">{f.title}</div>
                <div className="text-sm text-slate-400">{f.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-900/30">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-xs uppercase text-indigo-300 mb-1">Pricing</div>
              <div className="text-2xl font-bold text-slate-50">Choose your plan</div>
            </div>
            <div className="text-xs text-slate-400">Cancel anytime • No hidden fees</div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {tiers.map((t) => (
              <div
                key={t.name}
                className={`rounded-2xl border p-5 bg-slate-900/70 ${t.highlight ? 'border-indigo-400/60 shadow-lg shadow-indigo-500/20' : 'border-slate-800'}`}
              >
                {t.highlight && (
                  <div className="inline-flex items-center gap-1 rounded-full bg-indigo-500/20 border border-indigo-400/40 px-3 py-1 text-xs font-semibold text-indigo-200 mb-3">
                    {t.highlight}
                  </div>
                )}
                <div className="text-lg font-semibold text-slate-100">{t.name}</div>
                <div className="text-3xl font-bold text-slate-50 mt-1">{t.price}</div>
                <ul className="mt-3 space-y-2 text-sm text-slate-300">
                  {t.perks.map((perk) => (
                    <li key={perk} className="flex items-center gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      {perk}
                    </li>
                  ))}
                </ul>
                <a
                  href="/login"
                  className="mt-4 w-full rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-indigo-500/30 hover:from-indigo-400 hover:to-purple-500 transition inline-flex items-center justify-center"
                >
                  {t.cta}
                </a>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-8 shadow-xl shadow-slate-900/30">
          <div className="text-xs uppercase text-indigo-300 mb-2">Track Record</div>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-sm text-slate-400">NHL</div>
              <div className="text-2xl font-bold text-slate-50">+12.4% ROI</div>
              <div className="text-xs text-slate-500 mt-1">Across multiple seasons</div>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-sm text-slate-400">NFL Spreads</div>
              <div className="text-2xl font-bold text-slate-50">+10.9% ROI</div>
              <div className="text-xs text-slate-500 mt-1">Validated in backtests</div>
            </div>
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-4">
              <div className="text-sm text-slate-400">Confidence</div>
              <div className="text-2xl font-bold text-slate-50">58.2% Win Rate</div>
              <div className="text-xs text-slate-500 mt-1">Transparent performance</div>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 p-8 shadow-2xl shadow-indigo-500/20 text-center">
          <h3 className="text-2xl font-bold text-slate-50 mb-2">Ready to get started?</h3>
          <p className="text-slate-300 mb-4">Join today and see today&apos;s board, POTD, and spread picks with transparent edges.</p>
          <div className="flex justify-center gap-3">
            <a href="/login" className="rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 hover:from-indigo-400 hover:to-purple-500 transition inline-flex items-center justify-center">
              Start Free Trial
            </a>
            <a href="/login" className="rounded-xl border border-slate-700 bg-slate-900 px-5 py-3 text-sm font-semibold text-slate-200 hover:border-indigo-400 hover:text-indigo-200 transition inline-flex items-center justify-center">
              View Demo
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
