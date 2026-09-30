import type { Metadata } from 'next'
import './globals.css'
import GovernmentHeader from '@/components/GovernmentHeader'
import Navbar from '@/components/Navbar'
import Footer from '@/components/Footer'

export const metadata: Metadata = {
  title: 'VARSH-AI — National Regime-Aware Monsoon Forecasting System',
  description: "Government of India Ministry of Earth Sciences (MoES) and NCMRWF's AI-based post-processing system for monsoon rainfall forecasts (SIH PS 26080)",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-[#f8fafc] text-slate-900 antialiased selection:bg-[#005f80] selection:text-white">
        <GovernmentHeader />
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
