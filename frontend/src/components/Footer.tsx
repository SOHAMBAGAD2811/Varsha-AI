'use client'
import React from 'react'
import { ExternalLink, Database, Shield } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="w-full bg-[#f1f5f9] border-t border-slate-200 text-slate-600 text-xs py-8 px-4 sm:px-6 lg:px-8 mt-12">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8 pb-8 border-b border-slate-200">
          {/* Col 1: System Info */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-[#004d66] flex items-center justify-center text-white font-bold text-xs">
                V
              </div>
              <span className="font-bold text-slate-900 text-sm tracking-wide">VARSH-AI</span>
            </div>
            <p className="text-slate-600 text-[11px] leading-relaxed">
              AI-based post-processing system for Numerical Weather Prediction (NWP) monsoon precipitation forecasts.
              Developed under Smart India Hackathon PS 26080 for MoES & NCMRWF.
            </p>
            <div className="flex items-center gap-2 text-[10px] text-slate-500 font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
              <span>Operational Models Online • +24h Lead</span>
            </div>
          </div>

          {/* Col 2: Institutional Partners */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Governing Bodies</h4>
            <ul className="space-y-1.5 text-[11px]">
              <li>
                <a href="https://www.moes.gov.in" target="_blank" rel="noreferrer" className="hover:text-[#005f80] flex items-center gap-1 font-medium">
                  Ministry of Earth Sciences (MoES) <ExternalLink size={10} />
                </a>
              </li>
              <li>
                <a href="https://www.ncmrwf.gov.in" target="_blank" rel="noreferrer" className="hover:text-[#005f80] flex items-center gap-1 font-medium">
                  NCMRWF, Noida <ExternalLink size={10} />
                </a>
              </li>
              <li>
                <a href="https://mausam.imd.gov.in" target="_blank" rel="noreferrer" className="hover:text-[#005f80] flex items-center gap-1 font-medium">
                  India Meteorological Department (IMD) <ExternalLink size={10} />
                </a>
              </li>
              <li>
                <span className="text-slate-500">National Monsoon Mission (NMM-II)</span>
              </li>
            </ul>
          </div>

          {/* Col 3: Technical Architecture */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">AI Pipeline Modules</h4>
            <ul className="space-y-1.5 text-[11px] text-slate-600">
              <li className="flex items-center gap-1.5">
                <span className="text-[#005f80] font-bold">•</span> Quantile Delta Mapping (QDM) + SSR
              </li>
              <li className="flex items-center gap-1.5">
                <span className="text-[#005f80] font-bold">•</span> Synoptic Weather Regime Classifier
              </li>
              <li className="flex items-center gap-1.5">
                <span className="text-[#005f80] font-bold">•</span> Quantile Regression Forests (QRF)
              </li>
              <li className="flex items-center gap-1.5">
                <span className="text-[#005f80] font-bold">•</span> Multi-Scale Fractions Skill Score (FSS)
              </li>
              <li className="flex items-center gap-1.5">
                <span className="text-[#005f80] font-bold">•</span> Threshold-Weighted CRPS (twCRPS)
              </li>
            </ul>
          </div>

          {/* Col 4: Ground Truth Datasets */}
          <div className="space-y-2">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">Ground Truth Data</h4>
            <div className="p-3 rounded-lg bg-white border border-slate-200 text-[11px] space-y-1 shadow-xs">
              <div className="flex items-center justify-between text-slate-800 font-bold">
                <span className="flex items-center gap-1.5">
                  <Database size={13} className="text-[#005f80]" />
                  IMD 0.25° Gridded Rainfall
                </span>
                <span className="text-[10px] text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded font-bold border border-emerald-200">2021–2025</span>
              </div>
              <p className="text-[10px] text-slate-500 font-medium">
                1,826 daily precipitation benchmarks across India. Conservative remapping to standard 0.25° grid.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-500 font-medium">
          <p>© {new Date().getFullYear()} Government of India • Ministry of Earth Sciences. All rights reserved.</p>
          <div className="flex items-center gap-3">
            <span className="hover:text-slate-800 cursor-pointer">Privacy Policy</span>
            <span>•</span>
            <span className="hover:text-slate-800 cursor-pointer">Terms of Service</span>
            <span>•</span>
            <span className="hover:text-slate-800 cursor-pointer">NWP Model Metadata</span>
            <span>•</span>
            <span className="text-[#005f80] font-bold">SIH PS 26080</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
