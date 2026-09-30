'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  LayoutDashboard, Map, Building2, Wind,
  TrendingUp, BarChart3, Zap, Clock
} from 'lucide-react'

const nav = [
  { href: '/',            label: 'Dashboard',        icon: LayoutDashboard },
  { href: '/forecast',    label: 'Forecast Map',     icon: Map },
  { href: '/district',    label: 'District Analysis', icon: Building2 },
  { href: '/regime',      label: 'Weather Regime',   icon: Wind },
  { href: '/probability', label: 'Probability',      icon: TrendingUp },
  { href: '/lead',        label: 'Multi-Lead (D1–5)',icon: Clock },
  { href: '/performance', label: 'Performance',      icon: BarChart3 },
]

export default function Sidebar() {
  const path = usePathname()
  return (
    <aside
      className="fixed left-0 top-0 h-screen w-60 flex flex-col z-40"
      style={{
        background: 'linear-gradient(180deg, #070f20 0%, #050c1a 100%)',
        borderRight: '1px solid #1a3050',
      }}
    >
      {/* Logo */}
      <div className="px-5 pt-6 pb-5" style={{ borderBottom: '1px solid #1a3050' }}>
        <div className="flex items-center gap-3">
          <div className="relative w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ background: 'linear-gradient(135deg, #2563eb, #7c3aed)' }}>
            <Zap size={16} className="text-white" />
          </div>
          <div>
            <div className="font-bold text-white text-sm tracking-widest">VARSH-AI</div>
            <div className="text-[10px] text-slate-500 tracking-wide mt-0.5">NWP Post-Processing</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        <p className="text-[9px] text-slate-600 font-semibold tracking-widest uppercase px-3 mb-2">Navigation</p>
        {nav.map(({ href, label, icon: Icon }) => {
          const active = path === href
          return (
            <Link key={href} href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                active
                  ? 'nav-active text-blue-400'
                  : 'text-slate-500 hover:text-slate-200 hover:bg-white/[0.04]'
              }`}>
              <Icon size={16} className={active ? 'text-blue-400' : ''} strokeWidth={active ? 2.5 : 1.8} />
              {label}
              {active && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400" />
              )}
            </Link>
          )
        })}
      </nav>

      {/* Status footer */}
      <div className="px-4 py-4 mx-3 mb-3 rounded-xl" style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-2 mb-2">
          <div className="relative w-2 h-2">
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-60" />
          </div>
          <span className="text-xs text-emerald-400 font-semibold">LIVE FORECAST</span>
        </div>
        <div className="text-[10px] text-slate-500 space-y-0.5">
          <div>30 Sep 2025 · +24h Lead</div>
          <div>Source: NCUM-G (12 km)</div>
          <div className="text-slate-600">MoES / NCMRWF · SIH PS 26080</div>
        </div>
      </div>
    </aside>
  )
}
