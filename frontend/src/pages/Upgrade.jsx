import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const tiers = [
  {
    name: 'Free',
    price: '$0',
    period: 'forever',
    description: 'Get started with basic picks',
    features: [
      'Daily POTD (2hr delay)',
      'Basic game schedules',
      'Single sport access',
    ],
    notIncluded: [
      'Real-time picks',
      'Full analysis breakdown',
      'All sports access',
      'Bet tracker',
      'Discord alerts',
    ],
    cta: 'Current Plan',
    current: true,
  },
  {
    name: 'Pro',
    price: '$29',
    period: '/month',
    description: 'Full access to all features',
    features: [
      'Real-time POTD & picks',
      'Full analysis breakdown',
      'All sports (NHL, NFL, NBA, MLB)',
      'Unlimited bet tracker',
      'Discord alerts',
      'Priority support',
    ],
    notIncluded: [],
    cta: 'Upgrade to Pro',
    current: false,
    highlighted: true,
  },
]

export default function Upgrade() {
  const { user, userTier } = useAuth()
  const isPro = userTier === 'pro'

  return (
    <div className="fade-in max-w-4xl mx-auto py-8">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-slate-100 mb-3">
          Upgrade to Pro
        </h1>
        <p className="text-slate-400 text-lg">
          Unlock full access to all picks, analysis, and features
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {tiers.map((tier) => {
          const isCurrentTier =
            (tier.name === 'Free' && !isPro) || (tier.name === 'Pro' && isPro)

          return (
            <div
              key={tier.name}
              className={[
                'rounded-2xl border p-6 flex flex-col',
                tier.highlighted
                  ? 'border-indigo-500 bg-gradient-to-b from-indigo-500/10 to-slate-900/50 shadow-lg shadow-indigo-500/20'
                  : 'border-slate-700 bg-slate-900/50',
              ].join(' ')}
            >
              {tier.highlighted && (
                <div className="text-xs font-semibold text-indigo-400 uppercase tracking-wide mb-2">
                  Most Popular
                </div>
              )}
              <h2 className="text-xl font-bold text-slate-100">{tier.name}</h2>
              <div className="mt-2 flex items-baseline gap-1">
                <span className="text-4xl font-bold text-slate-100">
                  {tier.price}
                </span>
                <span className="text-slate-400">{tier.period}</span>
              </div>
              <p className="text-slate-400 text-sm mt-2">{tier.description}</p>

              <ul className="mt-6 space-y-3 flex-1">
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-2 text-sm">
                    <svg
                      className="w-5 h-5 text-green-400 shrink-0 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-slate-200">{feature}</span>
                  </li>
                ))}
                {tier.notIncluded.map((feature) => (
                  <li
                    key={feature}
                    className="flex items-start gap-2 text-sm opacity-50"
                  >
                    <svg
                      className="w-5 h-5 text-slate-500 shrink-0 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                    <span className="text-slate-400">{feature}</span>
                  </li>
                ))}
              </ul>

              <div className="mt-6">
                {isCurrentTier ? (
                  <div className="w-full py-2.5 px-4 rounded-xl border border-slate-600 bg-slate-800 text-center text-sm font-semibold text-slate-400">
                    Current Plan
                  </div>
                ) : tier.name === 'Pro' ? (
                  <button
                    className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold text-sm shadow-lg shadow-indigo-500/30 hover:from-indigo-400 hover:to-purple-500 transition"
                    onClick={() => {
                      // Stripe checkout will be wired in Phase 4
                      alert('Stripe checkout coming in Phase 4!')
                    }}
                  >
                    Upgrade to Pro
                  </button>
                ) : (
                  <div className="w-full py-2.5 px-4 rounded-xl border border-slate-600 bg-slate-800 text-center text-sm font-semibold text-slate-400">
                    Free Forever
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {!user && (
        <div className="mt-8 text-center">
          <p className="text-slate-400 text-sm mb-3">
            Already have an account?
          </p>
          <Link
            to="/login"
            className="text-indigo-400 hover:text-indigo-300 font-semibold transition"
          >
            Sign in to upgrade
          </Link>
        </div>
      )}

      <div className="mt-12 text-center text-slate-500 text-sm">
        <p>Cancel anytime. No questions asked.</p>
      </div>
    </div>
  )
}
