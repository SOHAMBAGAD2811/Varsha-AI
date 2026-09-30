'use client'
import React, { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import Link from 'next/link'
import { DISTRICTS, REGIME_META, CURRENT_FORECAST } from '@/lib/mockData'
import { api, DistrictDetail } from '@/lib/api'
import {
  MapPin,
  Layers,
  Filter,
  ShieldAlert,
  ChevronRight,
  RotateCcw,
  Search,
  ExternalLink,
  Compass,
  FileSpreadsheet
} from 'lucide-react'
import NDMABulletinModal from '@/components/NDMABulletinModal'

const IndiaMap = dynamic(() => import('@/components/IndiaMap'), { ssr: false })

const LAYERS = [
  { key: 'varsh',           label: 'Varsh-AI AI Corrected',     desc: 'Regime-aware post-processed precipitation' },
  { key: 'nwp',             label: 'Raw NWP (NCUM-G)',          desc: 'Raw Numerical Model forecast' },
  { key: 'qdm',             label: 'QDM Baseline',              desc: 'Traditional Quantile Delta Mapping' },
  { key: 'heavy_prob',      label: 'P(R > 64.5 mm)',            desc: 'QRF Heavy Rainfall Probability' },
  { key: 'very_heavy_prob', label: 'P(R > 115.5 mm)',          desc: 'Very Heavy Rainfall Risk' },
  { key: 'threshold',       label: 'Threshold Risk Mask',       desc: 'Binary exceedance warning mask' },
] as const

type LayerKey = typeof LAYERS[number]['key']

export default function GeoMapExplorer() {
  const [selectedLayer, setSelectedLayer] = useState<LayerKey>('varsh')
  const [selectedState, setSelectedState] = useState<string>('all')
  const [selectedRisk, setSelectedRisk] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [selectedDistrict, setSelectedDistrict] = useState<string>('Nashik')
  const [liveDistricts, setLiveDistricts] = useState<any[]>([])
  const [liveDistrictDetail, setLiveDistrictDetail] = useState<any | null>(null)
  const [alertSummary, setAlertSummary] = useState<any>(null)
  const [apiOnline, setApiOnline] = useState<boolean>(false)
  const [bulletinOpen, setBulletinOpen] = useState<boolean>(false)

  // Fetch all 594+ districts on mount
  useEffect(() => {
    api.allDistrictsExpanded()
      .then(res => {
        setLiveDistricts(res.districts)
        setAlertSummary(res.alert_summary)
        setApiOnline(true)
      })
      .catch(() => {
        api.allDistricts()
          .then(res => {
            setLiveDistricts(res.districts)
            setApiOnline(true)
          })
          .catch(() => setApiOnline(false))
      })
  }, [])

  // Fetch selected district details
  useEffect(() => {
    if (selectedDistrict) {
      api.district(selectedDistrict)
        .then(d => setLiveDistrictDetail(d))
        .catch(() => {
          // If not in 50-core list, look up from 594 expanded list
          const match = liveDistricts.find(d => d.name?.toLowerCase() === selectedDistrict.toLowerCase())
          if (match) setLiveDistrictDetail(match)
        })
    }
  }, [selectedDistrict, liveDistricts])

  const districtList = liveDistricts.length > 0
    ? liveDistricts
    : Object.entries(DISTRICTS).map(([name, d], idx) => ({
        district_id: idx,
        name,
        state: d.state,
        nwp_mm: d.nwp,
        qdm_mm: d.qdm,
        varsh_mm: d.varsh,
        obs_mm: d.obs,
        p_light: 0.92,
        p_heavy: d.p_heavy,
        p_very_heavy: d.p_very_heavy,
        lat: 20.0,
        lon: 78.0,
        regime: d.regime,
      }))

  const states = Array.from(new Set(districtList.map(d => d.state))).filter(Boolean).sort()

  const filteredDistricts = districtList.filter(d => {
    const matchState = selectedState === 'all' || d.state === selectedState
    const isRed    = d.alert_level === 'RED'    || d.p_heavy >= 0.70
    const isOrange = d.alert_level === 'ORANGE' || (d.p_heavy >= 0.40 && d.p_heavy < 0.70)
    const isYellow = d.alert_level === 'YELLOW' || (d.p_heavy >= 0.20 && d.p_heavy < 0.40)
    const isNormal = d.alert_level === 'GREEN'  || d.p_heavy < 0.20

    const matchRisk =
      selectedRisk === 'all' ||
      (selectedRisk === 'red' && isRed) ||
      (selectedRisk === 'orange' && isOrange) ||
      (selectedRisk === 'yellow' && isYellow) ||
      (selectedRisk === 'normal' && isNormal)

    const matchSearch =
      !searchQuery ||
      d.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.state?.toLowerCase().includes(searchQuery.toLowerCase())

    return matchState && matchRisk && matchSearch
  })

  const activeDetail = liveDistrictDetail || (
    districtList.find(d => d.name?.toLowerCase() === selectedDistrict.toLowerCase()) || districtList[0]
  )
  const regimeKey = (activeDetail?.regime || 'depression') as keyof typeof REGIME_META
  const regimeMeta = REGIME_META[regimeKey] || REGIME_META.depression

  // Summary counts
  const redCount    = alertSummary?.RED    ?? districtList.filter(d => d.p_heavy >= 0.70).length
  const orangeCount = alertSummary?.ORANGE ?? districtList.filter(d => d.p_heavy >= 0.40 && d.p_heavy < 0.70).length
  const yellowCount = alertSummary?.YELLOW ?? districtList.filter(d => d.p_heavy >= 0.20 && d.p_heavy < 0.40).length
  const greenCount  = alertSummary?.GREEN  ?? districtList.filter(d => d.p_heavy < 0.20).length

  return (
    <div className="space-y-6">
      {/* 1. Header Bar with Neermalya GeoMap styling */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 gov-card p-5 bg-white">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-blue-50 text-[#005f80] border border-blue-200">
              <MapPin size={18} />
            </span>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">GeoMap GIS Explorer — All India Districts</h2>
            <span className="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded bg-emerald-100 text-emerald-800">
              Phase 4 Active
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Zonal raster statistics & IMD 4-tier warning criteria evaluated across 594 official administrative districts
          </p>
        </div>

        {/* Quick Filter Counters */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
            <span className="text-slate-500">Districts: </span>
            <span className="font-bold text-slate-900">{districtList.length}</span>
          </div>
          <div className="px-2.5 py-1.5 rounded-lg bg-red-50 border border-red-200 text-xs">
            <span className="text-red-700 font-semibold">🔴 Red: </span>
            <span className="font-bold text-red-700">{redCount}</span>
          </div>
          <div className="px-2.5 py-1.5 rounded-lg bg-orange-50 border border-orange-200 text-xs">
            <span className="text-orange-700 font-semibold">🟠 Orange: </span>
            <span className="font-bold text-orange-700">{orangeCount}</span>
          </div>
          <div className="px-2.5 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-xs">
            <span className="text-amber-800 font-semibold">🟡 Yellow: </span>
            <span className="font-bold text-amber-800">{yellowCount}</span>
          </div>
          <div className="px-2.5 py-1.5 rounded-lg bg-emerald-50 border border-emerald-200 text-xs">
            <span className="text-emerald-800 font-semibold">🟢 Green: </span>
            <span className="font-bold text-emerald-800">{greenCount}</span>
          </div>
        </div>
      </div>

      {/* 2. Interactive Filter Bar (Dropdowns + Layer Switcher) */}
      <div className="gov-card p-4 bg-white flex flex-wrap items-center justify-between gap-4">
        {/* Dropdowns */}
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <Filter size={14} className="text-[#005f80]" />
            <span className="font-bold uppercase tracking-wider text-[10px]">Filter By:</span>
          </div>

          {/* State Filter */}
          <select
            value={selectedState}
            onChange={e => setSelectedState(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-lg px-3 py-1.5 focus:ring-1 focus:ring-[#005f80] focus:outline-none"
          >
            <option value="all">All States ({states.length})</option>
            {states.map(st => (
              <option key={st} value={st}>{st}</option>
            ))}
          </select>

          {/* Risk Level Filter */}
          <select
            value={selectedRisk}
            onChange={e => setSelectedRisk(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-lg px-3 py-1.5 focus:ring-1 focus:ring-[#005f80] focus:outline-none"
          >
            <option value="all">All Risk Levels</option>
            <option value="red">Red Alert (&gt;70% risk)</option>
            <option value="orange">Orange Alert (40–70%)</option>
            <option value="yellow">Yellow Alert (20–40%)</option>
            <option value="normal">Normal / Green (&lt;20%)</option>
          </select>

          {/* District Quick Select */}
          <select
            value={selectedDistrict}
            onChange={e => setSelectedDistrict(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-lg px-3 py-1.5 focus:ring-1 focus:ring-[#005f80] focus:outline-none max-w-[190px]"
          >
            {filteredDistricts.map(d => (
              <option key={d.name} value={d.name}>{d.name} ({d.state})</option>
            ))}
          </select>

          {/* District Search Box */}
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-2 text-slate-400" />
            <input
              type="text"
              placeholder="Search 594 districts..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-lg pl-8 pr-3 py-1.5 focus:ring-1 focus:ring-[#005f80] focus:outline-none w-44"
            />
          </div>

          {/* Reset Filters */}
          {(selectedState !== 'all' || selectedRisk !== 'all' || searchQuery) && (
            <button
              onClick={() => { setSelectedState('all'); setSelectedRisk('all'); setSearchQuery('') }}
              className="flex items-center gap-1 text-[11px] text-[#005f80] hover:text-[#004d66] px-2 py-1 rounded bg-blue-50 border border-blue-200"
            >
              <RotateCcw size={10} />
              <span>Reset</span>
            </button>
          )}
        </div>

        {/* Active Layer Pill */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-500 font-medium">Active Layer:</span>
          <span className="text-xs font-bold text-[#005f80] px-2.5 py-1 rounded-md bg-blue-50 border border-blue-200">
            {LAYERS.find(l => l.key === selectedLayer)?.label}
          </span>
        </div>
      </div>

      {/* 3. Main Map & District Side Panel Container */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        {/* Map Container (3 cols) */}
        <div className="lg:col-span-3 gov-card bg-white overflow-hidden flex flex-col">
          {/* Layer Tab Bar */}
          <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between gap-2 overflow-x-auto scrollbar-none">
            <div className="flex items-center gap-1.5">
              <Layers size={14} className="text-slate-500 mr-1" />
              {LAYERS.map(l => (
                <button
                  key={l.key}
                  onClick={() => setSelectedLayer(l.key)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold whitespace-nowrap transition-all ${
                    selectedLayer === l.key
                      ? 'bg-[#005f80] text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                  }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive Map */}
          <div className="h-[560px] w-full relative">
            <IndiaMap
              layer={selectedLayer}
              selectedDistrict={selectedDistrict}
              onDistrictClick={name => setSelectedDistrict(name)}
            />

            {/* Floating Legend */}
            <div className="absolute bottom-4 left-4 z-10 bg-white/95 backdrop-blur-md p-3.5 rounded-xl border border-slate-300 shadow-lg text-xs space-y-1.5">
              <span className="text-[10px] font-extrabold uppercase tracking-wider text-slate-700 block mb-1">
                Rainfall Severity (mm/day)
              </span>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#991b1b]" />
                <span className="text-slate-800 text-[11px] font-medium">Extreme (&gt;115 mm)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#dc2626]" />
                <span className="text-slate-800 text-[11px] font-medium">Very Heavy (64.5 – 115 mm)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#ea580c]" />
                <span className="text-slate-800 text-[11px] font-medium">Heavy (35 – 64.5 mm)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#0284c7]" />
                <span className="text-slate-800 text-[11px] font-medium">Moderate (15 – 35 mm)</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#0891b2]" />
                <span className="text-slate-800 text-[11px] font-medium">Light (&lt;15 mm)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Drawer: Selected District Detailed Card (1 col) */}
        <div className="gov-card p-5 bg-white space-y-4">
          <div className="pb-3 border-b border-slate-100">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-[#005f80] uppercase tracking-wider">District Inspector</span>
              <span className="text-[10px] text-slate-400 font-mono">ID: {activeDetail?.district_id}</span>
            </div>
            <h3 className="text-xl font-extrabold text-slate-900 mt-1">{activeDetail?.name}</h3>
            <p className="text-xs text-slate-500">{activeDetail?.state} • India</p>
          </div>

          {/* Current Regime Tag */}
          <div className="p-3 rounded-xl border border-blue-200 bg-blue-50/60 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 text-[10px] font-bold uppercase">Weather Regime</span>
              <Compass size={14} className="text-[#005f80]" />
            </div>
            <p className="font-bold text-sm mt-0.5 text-[#005f80]">{regimeMeta.label}</p>
            <p className="text-[11px] text-slate-600 mt-1">{regimeMeta.summary}</p>
          </div>

          {/* Phase 4: Official IMD 4-Tier Alert Badge & Advisory */}
          {activeDetail?.alert_level && (
            <div
              className="p-3 rounded-xl border text-xs space-y-1"
              style={{
                backgroundColor: activeDetail.alert_level === 'RED' ? '#fef2f2' :
                                 activeDetail.alert_level === 'ORANGE' ? '#fff7ed' :
                                 activeDetail.alert_level === 'YELLOW' ? '#fffbeb' : '#f0fdf4',
                borderColor: activeDetail.alert_color || '#cbd5e1',
              }}
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-extrabold uppercase tracking-wider text-slate-700">
                  IMD Warning Level
                </span>
                <span
                  className="px-2 py-0.5 rounded text-[10px] font-extrabold text-white"
                  style={{ backgroundColor: activeDetail.alert_color || '#16a34a' }}
                >
                  {activeDetail.alert_level} ALERT
                </span>
              </div>
              <p className="text-[11px] text-slate-700 leading-snug font-medium">
                {activeDetail.advisory || 'Normal weather conditions. Follow standard precautions.'}
              </p>
              {activeDetail.varsh_p90_mm && (
                <div className="flex items-center justify-between pt-1 border-t border-slate-200/60 text-[10px] text-slate-500 font-mono">
                  <span>90th Pctl: <strong className="text-slate-800">{activeDetail.varsh_p90_mm} mm</strong></span>
                  <span>Peak: <strong className="text-slate-800">{activeDetail.varsh_max_mm} mm</strong></span>
                  <span>Cells: <strong className="text-slate-800">{activeDetail.grid_cells_n ?? 1}</strong></span>
                </div>
              )}
            </div>
          )}

          {/* Model Comparisons 2x2 Grid */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
              Multi-Model Forecast Comparison
            </span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[10px] text-slate-500 block uppercase font-bold">Raw NWP</span>
                <span className="text-base font-bold text-slate-800">{activeDetail?.nwp_mm}</span>
                <span className="text-[10px] text-slate-500 ml-1">mm</span>
              </div>
              <div className="p-2.5 rounded-xl bg-purple-50/60 border border-purple-200">
                <span className="text-[10px] text-purple-700 block uppercase font-bold">QDM Baseline</span>
                <span className="text-base font-bold text-purple-900">{activeDetail?.qdm_mm}</span>
                <span className="text-[10px] text-slate-500 ml-1">mm</span>
              </div>
              <div className="p-2.5 rounded-xl bg-blue-50 border border-blue-300">
                <span className="text-[10px] text-[#005f80] font-bold block uppercase">Varsh-AI</span>
                <span className="text-lg font-black text-[#004d66]">{activeDetail?.varsh_mm}</span>
                <span className="text-[10px] text-[#005f80] ml-1">mm</span>
              </div>
              <div className="p-2.5 rounded-xl bg-emerald-50 border border-emerald-200">
                <span className="text-[10px] text-emerald-700 block uppercase font-bold">Observed (IMD)</span>
                <span className="text-base font-bold text-emerald-900">{activeDetail?.obs_mm}</span>
                <span className="text-[10px] text-slate-500 ml-1">mm</span>
              </div>
            </div>
          </div>

          {/* Probabilistic Exceedance Bars */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">
              Exceedance Probabilities (QRF)
            </span>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-600 font-medium">P(R &gt; 10 mm / Light)</span>
                <span className="font-bold text-slate-900 font-mono">{Math.round((activeDetail?.p_light || 0.9) * 100)}%</span>
              </div>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div className="bg-[#005f80] h-2 rounded-full" style={{ width: `${(activeDetail?.p_light || 0.9) * 100}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-600 font-medium">P(R &gt; 64.5 mm / Heavy)</span>
                <span className={`font-bold font-mono ${activeDetail?.p_heavy > 0.6 ? 'text-red-700' : 'text-amber-700'}`}>
                  {Math.round((activeDetail?.p_heavy || 0) * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div
                  className={`h-2 rounded-full ${activeDetail?.p_heavy > 0.6 ? 'bg-red-600' : 'bg-amber-500'}`}
                  style={{ width: `${(activeDetail?.p_heavy || 0) * 100}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-600 font-medium">P(R &gt; 115.5 mm / Very Heavy)</span>
                <span className="font-bold font-mono text-purple-800">
                  {Math.round((activeDetail?.p_very_heavy || 0) * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                <div className="bg-purple-600 h-2 rounded-full" style={{ width: `${(activeDetail?.p_very_heavy || 0) * 100}%` }} />
              </div>
            </div>
          </div>

          <div className="pt-2 space-y-2">
            <button
              onClick={() => setBulletinOpen(true)}
              className="w-full py-2.5 px-3 rounded-xl bg-gradient-to-r from-red-600 to-amber-600 hover:from-red-700 hover:to-amber-700 text-white font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-xs"
            >
              <FileSpreadsheet size={13} />
              <span>Generate NDMA CAP Warning Bulletin (SOP)</span>
            </button>

            <Link
              href={`/district`}
              className="w-full py-2.5 px-3 rounded-xl bg-[#004d66] hover:bg-[#003b50] text-white font-bold text-xs flex items-center justify-center gap-1.5 transition-all shadow-xs"
            >
              <span>Inspect Time Series & Quantiles</span>
              <ExternalLink size={12} />
            </Link>
          </div>
        </div>
      </div>

      {/* NDMA CAP Bulletin Modal */}
      <NDMABulletinModal
        districtName={activeDetail?.name || 'Nashik'}
        isOpen={bulletinOpen}
        onClose={() => setBulletinOpen(false)}
      />
    </div>
  )
}
