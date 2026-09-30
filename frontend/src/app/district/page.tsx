'use client'
import React, { useState, useEffect } from 'react'
import { DISTRICTS, TIME_SERIES, REGIME_META } from '@/lib/mockData'
import { api, DistrictDetail } from '@/lib/api'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip,
  LineChart, Line, ResponsiveContainer, Legend
} from 'recharts'
import {
  Building2,
  Compass,
  Search,
  FileSpreadsheet
} from 'lucide-react'
import NDMABulletinModal from '@/components/NDMABulletinModal'

const COLORS = { nwp: '#64748b', qdm: '#9333ea', varsh: '#0284c7', obs: '#16a34a' }

export default function DistrictAnalyticsPage() {
  const [selected, setSelected] = useState('Nashik')
  const [liveData, setLiveData] = useState<DistrictDetail | null>(null)
  const [allLive, setAllLive] = useState<DistrictDetail[]>([])
  const [apiOnline, setApiOnline] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [bulletinOpen, setBulletinOpen] = useState(false)

  useEffect(() => {
    api.allDistricts()
      .then(r => {
        setAllLive(r.districts)
        setApiOnline(true)
      })
      .catch(() => setApiOnline(false))
  }, [])

  useEffect(() => {
    if (selected) {
      api.district(selected)
        .then(d => setLiveData(d))
        .catch(() => setLiveData(null))
    }
  }, [selected])

  const mockD = DISTRICTS[selected] ?? Object.values(DISTRICTS)[0]
  const d = liveData ?? ({
    name: selected,
    state: mockD.state,
    regime: mockD.regime,
    nwp_mm: mockD.nwp,
    qdm_mm: mockD.qdm,
    varsh_mm: mockD.varsh,
    obs_mm: mockD.obs,
    p_light: 0.92,
    p_heavy: mockD.p_heavy,
    p_very_heavy: mockD.p_very_heavy,
    lat: 19.99,
    lon: 73.78,
  } as DistrictDetail)

  const timeSeries = liveData?.time_series ?? TIME_SERIES
  const regime = (d.regime ?? 'depression') as keyof typeof REGIME_META
  const meta = REGIME_META[regime] ?? REGIME_META.depression

  const compareData = [
    { name: 'Raw NWP',  value: d.nwp_mm,   fill: COLORS.nwp },
    { name: 'QDM',      value: d.qdm_mm,   fill: COLORS.qdm },
    { name: 'Varsh-AI', value: d.varsh_mm, fill: COLORS.varsh },
    { name: 'Observed', value: d.obs_mm,   fill: COLORS.obs },
  ]

  const districtNames = allLive.length > 0
    ? allLive.map(item => item.name)
    : Object.keys(DISTRICTS)

  const filteredNames = districtNames.filter(name =>
    name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const isHighRisk = (d.p_heavy || 0) > 0.6
  const alertColor = isHighRisk ? 'text-red-800 bg-red-100 border-red-200' : 'text-amber-800 bg-amber-100 border-amber-200'

  return (
    <div className="space-y-6">
      {/* Top Banner with Search and Selection */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 gov-card p-5 bg-white">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-blue-50 text-[#005f80] border border-blue-200">
              <Building2 size={18} />
            </span>
            <h2 className="text-xl font-bold text-slate-900 tracking-tight">District Meteorological Intelligence</h2>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Detailed time series, quantile exceedance probability, and ground truth benchmark comparison
          </p>
        </div>

        {/* District Selector & Search */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search district..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-xs text-slate-800 pl-8 pr-3 py-1.5 rounded-lg focus:outline-none focus:ring-1 focus:ring-[#005f80] w-48"
            />
          </div>

          <select
            value={selected}
            onChange={e => setSelected(e.target.value)}
            className="bg-slate-50 border border-slate-300 text-xs text-slate-800 px-3 py-1.5 rounded-lg focus:outline-none focus:ring-1 focus:ring-[#005f80]"
          >
            {filteredNames.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Main District Hero & Metrics Card */}
      <div className="gov-card p-6 bg-white space-y-6">
        {/* District Title Row */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-3">
              <h3 className="text-2xl font-black text-slate-900">{d.name}</h3>
              <span className={`text-xs font-bold px-2.5 py-0.5 rounded-full border ${alertColor}`}>
                {isHighRisk ? 'RED ALERT WARNING' : 'MODERATE ADVISORY'}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1 font-medium">
              State: <span className="text-slate-800 font-bold">{d.state}</span> • Coordinates: {d.lat?.toFixed(2)}°N, {d.lon?.toFixed(2)}°E
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* NDMA Bulletin Button */}
            <button
              onClick={() => setBulletinOpen(true)}
              className="px-3.5 py-2.5 rounded-xl bg-gradient-to-r from-red-600 to-amber-600 hover:from-red-700 hover:to-amber-700 text-white font-bold text-xs flex items-center gap-1.5 transition-all shadow-xs"
            >
              <FileSpreadsheet size={15} />
              <span>NDMA CAP Bulletin (SOP)</span>
            </button>

            {/* Regime Badge */}
            <div className={`p-2.5 px-3.5 rounded-xl border text-xs flex items-center gap-2 ${meta.bg} ${meta.border}`}>
              <Compass size={16} className={meta.color} />
              <div>
                <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-bold">Synoptic Regime</span>
                <span className={`font-bold ${meta.color}`}>{meta.label}</span>
              </div>
            </div>
          </div>
        </div>

        {/* 4 Models Metric Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider block">Raw NWP Forecast</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-black text-slate-800">{d.nwp_mm}</span>
              <span className="text-xs text-slate-500 font-medium">mm/day</span>
            </div>
            <span className="text-[10px] text-slate-500 block mt-1 font-medium">NCUM-G 12km Grid</span>
          </div>

          <div className="p-4 rounded-xl bg-purple-50/60 border border-purple-200">
            <span className="text-[10px] text-purple-700 font-bold uppercase tracking-wider block">QDM Corrected</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-black text-purple-900">{d.qdm_mm}</span>
              <span className="text-xs text-slate-500 font-medium">mm/day</span>
            </div>
            <span className="text-[10px] text-slate-500 block mt-1 font-medium">Quantile Delta Mapping</span>
          </div>

          <div className="p-4 rounded-xl bg-blue-50 border border-blue-300">
            <span className="text-[10px] text-[#005f80] font-bold uppercase tracking-wider block">VARSH-AI Prediction</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-3xl font-black text-[#004d66]">{d.varsh_mm}</span>
              <span className="text-xs text-[#005f80] font-bold">mm/day</span>
            </div>
            <span className="text-[10px] text-[#005f80] block mt-1 font-semibold">FiLM + Spatial MoE AI</span>
          </div>

          <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200">
            <span className="text-[10px] text-emerald-700 font-bold uppercase tracking-wider block">Observed Benchmark</span>
            <div className="flex items-baseline gap-1 mt-1">
              <span className="text-2xl font-black text-emerald-900">{d.obs_mm}</span>
              <span className="text-xs text-slate-500 font-medium">mm/day</span>
            </div>
            <span className="text-[10px] text-slate-500 block mt-1 font-medium">IMD 0.25° Ground Truth</span>
          </div>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-4 border-t border-slate-100">
          {/* Comparison Bar Chart (1 col) */}
          <div className="bg-slate-50/60 p-4 rounded-xl border border-slate-200 flex flex-col justify-between">
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">Model Comparison</h4>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={compareData} barSize={36}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 11 }} unit="mm" />
                  <Tooltip
                    contentStyle={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: 8 }}
                    formatter={(v: any) => [`${Number(v).toFixed(1)} mm/day`]}
                  />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                    {compareData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-slate-500 text-center mt-2 font-medium">
              Varsh-AI corrects raw model suppression, closely matching IMD observations.
            </p>
          </div>

          {/* Time Series Multi-Line Chart (2 cols) */}
          <div className="lg:col-span-2 bg-slate-50/60 p-4 rounded-xl border border-slate-200 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                10-Day Forecast Progression & Verification
              </h4>
              <span className="text-[10px] text-slate-500 font-medium">Values in mm/day</span>
            </div>
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 10 }} />
                  <YAxis tick={{ fill: '#64748b', fontSize: 10 }} unit="mm" />
                  <Tooltip
                    contentStyle={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: 8 }}
                    formatter={(v: any, name: string) => [`${Number(v).toFixed(1)} mm`, name]}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: '#334155' }} />
                  <Line type="monotone" dataKey="nwp"   stroke={COLORS.nwp}   strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="Raw NWP" />
                  <Line type="monotone" dataKey="qdm"   stroke={COLORS.qdm}   strokeWidth={1.5} dot={false} name="QDM" />
                  <Line type="monotone" dataKey="varsh" stroke={COLORS.varsh} strokeWidth={2.5} dot={false} name="Varsh-AI" />
                  <Line type="monotone" dataKey="obs"   stroke={COLORS.obs}   strokeWidth={2} dot={{ r: 3, fill: COLORS.obs }} name="IMD Observed" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-slate-500 text-center mt-2 font-medium">
              Solid blue line (Varsh-AI) aligns closely with real green observation dots across time.
            </p>
          </div>
        </div>

        {/* Quantile Exceedance Breakdown */}
        <div className="pt-4 border-t border-slate-100">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wider">
              Quantile Regression Forest (QRF) Uncertainty Bounds
            </span>
            <span className="text-[10px] text-[#005f80] font-mono font-bold">P(R &gt; T | X)</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-600">P(R &gt; 10 mm / Light)</span>
                <span className="font-bold text-slate-900 font-mono">{Math.round((d.p_light || 0.92) * 100)}%</span>
              </div>
              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                <div className="bg-[#005f80] h-2 rounded-full" style={{ width: `${(d.p_light || 0.92) * 100}%` }} />
              </div>
              <span className="text-[10px] text-slate-500 font-medium">Wet day threshold</span>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-600">P(R &gt; 64.5 mm / Heavy)</span>
                <span className={`font-bold font-mono ${d.p_heavy > 0.6 ? 'text-red-700' : 'text-amber-700'}`}>
                  {Math.round((d.p_heavy || 0) * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                <div
                  className={`h-2 rounded-full ${d.p_heavy > 0.6 ? 'bg-red-600' : 'bg-amber-500'}`}
                  style={{ width: `${(d.p_heavy || 0) * 100}%` }}
                />
              </div>
              <span className="text-[10px] text-slate-500 font-medium">Operational disaster trigger</span>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-600">P(R &gt; 115.5 mm / Very Heavy)</span>
                <span className="font-bold font-mono text-purple-800">
                  {Math.round((d.p_very_heavy || 0) * 100)}%
                </span>
              </div>
              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                <div className="bg-purple-600 h-2 rounded-full" style={{ width: `${(d.p_very_heavy || 0) * 100}%` }} />
              </div>
              <span className="text-[10px] text-slate-500 font-medium">Extreme flood warning</span>
            </div>
          </div>
        </div>
      </div>

      {/* NDMA CAP Bulletin Modal */}
      <NDMABulletinModal
        districtName={d.name}
        isOpen={bulletinOpen}
        onClose={() => setBulletinOpen(false)}
      />
    </div>
  )
}
