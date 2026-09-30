'use client'
import React from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  LayoutDashboard,
  MapPin,
  Building2,
  Wind,
  Flame,
  BarChart3,
  SlidersHorizontal,
  Clock
} from 'lucide-react'

const navItems = [
  { href: '/',            label: 'National Overview',  icon: LayoutDashboard, badge: null },
  { href: '/forecast',    label: 'GeoMap Explorer',    icon: MapPin,          badge: 'GIS' },
  { href: '/district',    label: 'District Analytics', icon: Building2,       badge: '50' },
  { href: '/regime',      label: 'Synoptic Regimes',   icon: Wind,            badge: 'AI' },
  { href: '/probability', label: 'Risk & Alerts',      icon: Flame,           badge: 'Live' },
  { href: '/lead',        label: 'Multi-Lead (D1–5)',  icon: Clock,           badge: 'Lead' },
  { href: '/performance', label: 'Verification',       icon: BarChart3,       badge: 'IMD' },
]

export default function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="w-full bg-[#004d66] text-white border-b border-[#00384d] sticky top-0 z-50 shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-12 py-1">
          {/* Nav Links */}
          <div className="flex items-center space-x-1 sm:space-x-2 overflow-x-auto scrollbar-none">
            {navItems.map(({ href, label, icon: Icon, badge }) => {
              const active = pathname === href
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all whitespace-nowrap ${
                    active
                      ? 'bg-white text-[#004d66] shadow-sm font-bold'
                      : 'text-cyan-50 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon size={14} className={active ? 'text-[#004d66]' : 'text-cyan-200'} />
                  <span>{label}</span>
                  {badge && (
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-extrabold uppercase tracking-wider ${
                        active
                          ? 'bg-[#004d66]/15 text-[#004d66]'
                          : badge === 'Live'
                          ? 'bg-red-500 text-white'
                          : 'bg-white/20 text-white'
                      }`}
                    >
                      {badge}
                    </span>
                  )}
                </Link>
              )
            })}
          </div>

          {/* Quick Actions / Role Badge */}
          <div className="hidden lg:flex items-center gap-3">
            {/* National Alert Indicator */}
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-red-500/20 border border-red-300/40 text-xs">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-400"></span>
              </span>
              <span className="text-red-100 font-bold text-[11px]">8 Districts on Heavy Rain Alert</span>
            </div>

            {/* Model Tag */}
            <div className="flex items-center gap-1.5 text-xs text-cyan-100 bg-[#003d52] px-2.5 py-1 rounded-md border border-[#002f40]">
              <SlidersHorizontal size={12} className="text-cyan-300" />
              <span className="text-white font-medium text-[11px]">Varsh-AI (FiLM + QRF)</span>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
