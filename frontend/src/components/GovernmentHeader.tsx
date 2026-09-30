'use client'
import React from 'react'
import { ShieldCheck, Clock, Radio, Award } from 'lucide-react'

export default function GovernmentHeader() {
  return (
    <header className="w-full bg-white text-slate-800 border-b border-slate-200 select-none shadow-xs">
      {/* Top Tricolor Ribbon */}
      <div className="h-1 w-full flex">
        <div className="flex-1 bg-[#FF9933]" />
        <div className="flex-1 bg-[#FFFFFF]" />
        <div className="flex-1 bg-[#138808]" />
      </div>

      {/* Main Government Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        {/* Left: Emblem & Ministry Branding */}
        <div className="flex items-center gap-3.5">
          {/* Ashoka Stambh Emblem SVG */}
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-[#005f80]/10 to-[#15803d]/10 border border-slate-200 shadow-xs flex-shrink-0">
            <svg viewBox="0 0 24 24" className="w-7 h-7 text-[#005f80]" fill="currentColor">
              <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3zm0 2.18l6 2.25v4.66c0 4.1-2.67 7.9-6 8.89-3.33-.99-6-4.79-6-8.89V6.43l6-2.25zM11 7h2v6h-2zm0 8h2v2h-2z" />
            </svg>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold text-[#b45309] tracking-wider uppercase">
                भारत सरकार • Government of India
              </span>
              <span className="text-slate-300 text-xs">•</span>
              <span className="text-[10px] text-slate-500 font-semibold bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                SIH PS 26080
              </span>
            </div>
            <h1 className="text-lg sm:text-xl font-extrabold text-[#004d66] tracking-tight flex items-center gap-2">
              <span>VARSH-AI</span>
              <span className="hidden sm:inline-block text-xs font-normal text-slate-500">
                — National Monsoon NWP Post-Processing
              </span>
            </h1>
            <p className="text-[11px] text-slate-600 font-medium">
              Ministry of Earth Sciences (MoES) • National Centre for Medium Range Weather Forecasting (NCMRWF)
            </p>
          </div>
        </div>

        {/* Right: Live Status & Metadata */}
        <div className="flex items-center gap-3 sm:gap-4 flex-wrap">
          {/* Live Operational Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 border border-emerald-200 text-xs shadow-xs">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-600"></span>
            </span>
            <span className="text-emerald-800 font-bold tracking-wide">OPERATIONAL</span>
            <span className="text-slate-300">|</span>
            <span className="text-slate-600 font-mono text-[11px] font-semibold">+24h LEAD</span>
          </div>

          {/* Clock & Ground Truth benchmark */}
          <div className="hidden md:flex flex-col text-right text-[11px] text-slate-600 leading-tight">
            <div className="flex items-center justify-end gap-1.5 text-slate-800 font-semibold">
              <Clock size={12} className="text-[#005f80]" />
              <span>30 Sep 2025 · 12:00 UTC</span>
            </div>
            <span className="text-slate-500 text-[10px]">Benchmark: IMD 0.25° Gridded Data (2021–2025)</span>
          </div>

          {/* Official Badge */}
          <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-blue-50 border border-blue-200 text-[11px] text-blue-900 font-medium">
            <ShieldCheck size={13} className="text-[#005f80]" />
            <span>MoES / NCMRWF</span>
          </div>
        </div>
      </div>
    </header>
  )
}
