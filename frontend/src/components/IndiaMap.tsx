'use client'
import React, { useEffect, useRef } from 'react'
import { DISTRICTS, REGIME_META } from '@/lib/mockData'
import { API_BASE } from '@/lib/api'

const DISTRICT_COORDS: Record<string, [number, number]> = {
  'Mumbai':         [19.076,  72.878],
  'Nashik':         [19.997,  73.789],
  'Pune':           [18.520,  73.856],
  'Kolhapur':       [16.705,  74.243],
  'Ratnagiri':      [16.994,  73.300],
  'Bhubaneswar':    [20.296,  85.824],
  'Cuttack':        [20.462,  85.883],
  'Visakhapatnam':  [17.686,  83.218],
  'Hyderabad':      [17.385,  78.487],
  'Chennai':        [13.083,  80.270],
  'Patna':          [25.594,  85.137],
  'Guwahati':       [26.144,  91.736],
  'Shimla':         [31.104,  77.173],
  'Dehradun':       [30.316,  78.032],
  'Ahmedabad':      [23.023,  72.572],
  'Surat':          [21.170,  72.831],
  'Jaipur':         [26.913,  75.788],
  'Bengaluru':      [12.972,  77.594],
  'Mangaluru':      [12.866,  74.842],
  'Bhopal':         [23.259,  77.413],
}

type LayerType = 'varsh' | 'nwp' | 'qdm' | 'heavy_prob' | 'very_heavy_prob' | 'threshold'

function getRainValue(district: any, layer: LayerType): number {
  switch (layer) {
    case 'nwp':             return district.nwp
    case 'qdm':             return district.qdm
    case 'heavy_prob':      return district.p_heavy * 100
    case 'very_heavy_prob': return district.p_very_heavy * 100
    case 'threshold':       return district.varsh > 64.5 ? 100 : 0
    default:                return district.varsh
  }
}

function getCircleColor(val: number, layer: LayerType): string {
  if (layer === 'threshold') return val > 0 ? '#dc2626' : '#16a34a'
  if (layer === 'heavy_prob' || layer === 'very_heavy_prob') {
    if (val > 80) return '#b91c1c'
    if (val > 60) return '#ea580c'
    if (val > 40) return '#d97706'
    if (val > 20) return '#0284c7'
    return '#16a34a'
  }
  // Rainfall in mm
  if (val > 115) return '#991b1b'
  if (val > 64)  return '#dc2626'
  if (val > 35)  return '#ea580c'
  if (val > 15)  return '#0284c7'
  if (val > 5)   return '#0891b2'
  return '#64748b'
}

function getCircleRadius(val: number, layer: LayerType): number {
  if (layer === 'threshold') return val > 0 ? 16 : 10
  if (layer.includes('prob')) return 8 + (val / 100) * 16
  return 8 + Math.min(val / 8, 18)
}

interface Props {
  layer: LayerType
  onDistrictClick?: (name: string) => void
  selectedDistrict?: string
}

export default function IndiaMap({ layer, onDistrictClick, selectedDistrict }: Props) {
  const mapRef = useRef<any>(null)
  const markerRef = useRef<Record<string, any>>({})
  const geoJsonLayerRef = useRef<any>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (typeof window === 'undefined') return

    let isMounted = true

    import('leaflet').then(L => {
      if (!isMounted || !containerRef.current) return
      if (mapRef.current) return

      // Clean up previous instance on container if React StrictMode remounted
      if ((containerRef.current as any)._leaflet_id != null) {
        (containerRef.current as any)._leaflet_id = null
      }

      delete (L.Icon.Default.prototype as any)._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
        iconUrl:       'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
        shadowUrl:     'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
      })

      const map = L.map(containerRef.current, {
        center: [21.5, 78.5],
        zoom: 5,
        zoomControl: true,
        attributionControl: true,
        minZoom: 4,
        maxZoom: 10,
      })
      mapRef.current = map

      // Base Canvas
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16,
      }).addTo(map)

      // Reference overlay (labels and boundary lines)
      L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        attribution: '',
        maxZoom: 16,
      }).addTo(map)

      setTimeout(() => {
        map.invalidateSize()
      }, 150)

      // Phase 4: Fetch and render official 594 district polygon boundaries
      fetch(`${API_BASE}/api/forecast/districts-geojson`)
        .then(res => (res.ok ? res.json() : null))
        .then(geojson => {
          if (!mapRef.current || !geojson?.features) return
          const gjLayer = L.geoJSON(geojson, {
            style: (feature: any) => {
              const p = feature.properties
              let fillColor = p.alert_color || '#16a34a'
              if (layer === 'varsh') {
                fillColor = getCircleColor(p.varsh_mm, 'varsh')
              } else if (layer === 'nwp') {
                fillColor = getCircleColor(p.nwp_mm, 'nwp')
              } else if (layer === 'qdm') {
                fillColor = getCircleColor(p.qdm_mm, 'qdm')
              } else if (layer === 'heavy_prob') {
                fillColor = getCircleColor(p.p_heavy * 100, 'heavy_prob')
              } else if (layer === 'very_heavy_prob') {
                fillColor = getCircleColor(p.p_very_heavy * 100, 'very_heavy_prob')
              } else if (layer === 'threshold') {
                fillColor = p.alert_color
              }
              const isSelected = selectedDistrict?.toLowerCase() === p.name?.toLowerCase()
              return {
                fillColor,
                weight: isSelected ? 2.5 : 0.8,
                opacity: 0.9,
                color: isSelected ? '#00384d' : '#94a3b8',
                fillOpacity: isSelected ? 0.75 : 0.45,
              }
            },
            onEachFeature: (feature: any, layerObj: any) => {
              const p = feature.properties
              layerObj.bindTooltip(`
                <div style="font-family:system-ui;padding:4px 6px">
                  <div style="font-weight:800;font-size:12px;color:#0f172a">${p.name}</div>
                  <div style="font-size:10px;color:#64748b;font-weight:600">${p.state}</div>
                  <div style="font-size:11px;font-weight:700;color:#005f80;margin-top:3px">
                    Varsh: ${p.varsh_mm} mm · NWP: ${p.nwp_mm} mm
                  </div>
                  <div style="font-size:10px;font-weight:800;color:${p.alert_color};margin-top:1px">
                    IMD Alert: ${p.alert_level} (P_Heavy: ${Math.round(p.p_heavy * 100)}%)
                  </div>
                </div>
              `, { sticky: true, opacity: 0.95 })

              layerObj.on('click', () => {
                onDistrictClick?.(p.name)
              })
            }
          }).addTo(map)
          geoJsonLayerRef.current = gjLayer
        })
        .catch(() => {})

      // Landmark station circle markers for high-contrast visibility
      Object.entries(DISTRICT_COORDS).forEach(([name, coords]) => {
        const district = (DISTRICTS as any)[name]
        if (!district) return

        const val = getRainValue(district, layer)
        const color = getCircleColor(val, layer)
        const radius = getCircleRadius(val, layer)
        const meta = REGIME_META[district.regime as keyof typeof REGIME_META]

        const marker = L.circleMarker(coords as [number, number], {
          radius: radius * 0.85,
          fillColor: color,
          color: '#ffffff',
          weight: selectedDistrict === name ? 3 : 2,
          opacity: 1.0,
          fillOpacity: 0.9,
        })

        const isRed = district.p_heavy > 0.7
        const isOrange = district.p_heavy > 0.4 && district.p_heavy <= 0.7
        const alertLabel = isRed ? 'RED ALERT' : isOrange ? 'ORANGE' : district.p_heavy > 0.2 ? 'YELLOW' : 'GREEN'
        const alertClass = isRed ? 'background:#fee2e2;color:#991b1b;border:1px solid #f87171' :
                           isOrange ? 'background:#ffedd5;color:#9a3412;border:1px solid #fb923c' :
                           'background:#ecfdf5;color:#065f46;border:1px solid #34d399'

        marker.bindPopup(`
          <div style="font-family:system-ui;min-width:210px;padding:4px">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:4px">
              <div>
                <div style="font-size:15px;font-weight:800;color:#0f172a;line-height:1.2">${name}</div>
                <div style="font-size:11px;color:#64748b;font-weight:500">${district.state}</div>
              </div>
              <span style="font-size:9px;font-weight:800;padding:2px 6px;border-radius:99px;${alertClass}">${alertLabel}</span>
            </div>

            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0">
              <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:6px;border-radius:6px">
                <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase">Raw NWP</div>
                <div style="font-size:15px;font-weight:800;color:#334155">${district.nwp}<span style="font-size:10px;font-weight:400"> mm</span></div>
              </div>
              <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:6px;border-radius:6px">
                <div style="font-size:9px;color:#64748b;font-weight:700;text-transform:uppercase">QDM</div>
                <div style="font-size:15px;font-weight:800;color:#7c3aed">${district.qdm}<span style="font-size:10px;font-weight:400"> mm</span></div>
              </div>
              <div style="background:#eff6ff;border:1px solid #bfdbfe;padding:6px;border-radius:6px">
                <div style="font-size:9px;color:#1d4ed8;font-weight:700;text-transform:uppercase">Varsh-AI</div>
                <div style="font-size:17px;font-weight:900;color:#1e40af">${district.varsh}<span style="font-size:10px;font-weight:500"> mm</span></div>
              </div>
              <div style="background:#f0fdf4;border:1px solid #bbf7d0;padding:6px;border-radius:6px">
                <div style="font-size:9px;color:#15803d;font-weight:700;text-transform:uppercase">Observed</div>
                <div style="font-size:15px;font-weight:800;color:#166534">${district.obs}<span style="font-size:10px;font-weight:500"> mm</span></div>
              </div>
            </div>

            <div style="background:#f1f5f9;border-radius:6px;padding:6px 8px;margin-bottom:6px">
              <div style="display:flex;justify-content:space-between;font-size:11px;font-weight:600;margin-bottom:2px">
                <span style="color:#475569">P(R > 64.5mm)</span>
                <span style="color:${isRed ? '#b91c1c' : '#ea580c'}">${Math.round(district.p_heavy * 100)}%</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:11px;font-weight:600">
                <span style="color:#475569">P(R > 115.5mm)</span>
                <span style="color:#64748b">${Math.round(district.p_very_heavy * 100)}%</span>
              </div>
            </div>

            <div style="font-size:10px;color:#64748b;display:flex;justify-content:space-between">
              <span>Regime: <strong style="color:#0f172a">${meta?.label || district.regime}</strong></span>
              <span style="color:#005f80;font-weight:600">Click for Analytics &rarr;</span>
            </div>
          </div>
        `, { maxWidth: 240 })

        marker.on('click', () => onDistrictClick?.(name))
        marker.addTo(map)
        markerRef.current[name] = marker
      })
    })

    return () => {
      isMounted = false
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
      if (containerRef.current && (containerRef.current as any)._leaflet_id != null) {
        (containerRef.current as any)._leaflet_id = null
      }
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps


  // Update styles on layer or district change
  useEffect(() => {
    if (!mapRef.current) return

    // Update GeoJSON polygon styles
    if (geoJsonLayerRef.current) {
      geoJsonLayerRef.current.eachLayer((layerObj: any) => {
        const p = layerObj.feature?.properties
        if (!p) return
        let fillColor = p.alert_color || '#16a34a'
        if (layer === 'varsh') {
          fillColor = getCircleColor(p.varsh_mm, 'varsh')
        } else if (layer === 'nwp') {
          fillColor = getCircleColor(p.nwp_mm, 'nwp')
        } else if (layer === 'qdm') {
          fillColor = getCircleColor(p.qdm_mm, 'qdm')
        } else if (layer === 'heavy_prob') {
          fillColor = getCircleColor(p.p_heavy * 100, 'heavy_prob')
        } else if (layer === 'very_heavy_prob') {
          fillColor = getCircleColor(p.p_very_heavy * 100, 'very_heavy_prob')
        } else if (layer === 'threshold') {
          fillColor = p.alert_color
        }
        const isSelected = selectedDistrict?.toLowerCase() === p.name?.toLowerCase()
        layerObj.setStyle({
          fillColor,
          weight: isSelected ? 2.5 : 0.8,
          opacity: 0.9,
          color: isSelected ? '#00384d' : '#94a3b8',
          fillOpacity: isSelected ? 0.75 : 0.45,
        })
      })
    }

    // Update landmark markers
    Object.entries(DISTRICT_COORDS).forEach(([name, _]) => {
      const district = (DISTRICTS as any)[name]
      if (!district || !markerRef.current[name]) return
      const val = getRainValue(district, layer)
      const color = getCircleColor(val, layer)
      const radius = getCircleRadius(val, layer)
      const isSelected = selectedDistrict === name

      markerRef.current[name].setStyle({
        fillColor: color,
        color: isSelected ? '#0f172a' : '#ffffff',
        radius: isSelected ? radius + 2 : radius * 0.85,
        weight: isSelected ? 3.5 : 2,
        fillOpacity: isSelected ? 1.0 : 0.9,
      })
    })
  }, [layer, selectedDistrict])

  return (
    <div ref={containerRef} className="w-full h-full rounded-xl overflow-hidden" />
  )
}
