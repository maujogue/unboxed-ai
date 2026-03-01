import { useEffect, useState } from 'react'

const COLORS = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#bc8cff']
const M = { top: 24, right: 20, bottom: 44, left: 52 }
const W = 680
const H = 240

function parseMm(v) {
  if (typeof v === 'number') return v
  const m = String(v || '').match(/([\d.]+)/)
  return m ? parseFloat(m[1]) : null
}

function fmtDate(d) {
  const s = String(d || '')
  if (s.length === 8) return `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`
  return s
}

function TrendBadge({ points, color }) {
  if (points.length < 2) return null
  const delta = points[points.length - 1].v - points[0].v
  const label = delta > 1 ? '↑ Progression' : delta < -1 ? '↓ Régression' : '→ Stable'
  const c = delta > 1 ? '#f85149' : delta < -1 ? '#3fb950' : '#8b949e'
  return (
    <span className="nodule-trend-badge" style={{ color: c, borderColor: c }}>
      {label}
    </span>
  )
}

export function NoduleTimeline({ patientId }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!patientId) return
    setLoading(true)
    setData(null)
    fetch(`/api/nodule-timeline/${encodeURIComponent(patientId)}`)
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [patientId])

  if (loading) return <div className="nodule-timeline-empty">Chargement de l'évolution…</div>
  if (!data) return null

  const timeline = (data.timeline || []).filter((t) => t.date)
  if (!timeline.length) return null

  // Collect all nodule numbers present in at least one timepoint
  const noduleNums = [
    ...new Set(timeline.flatMap((t) => t.nodules.map((n) => n.number))),
  ].sort((a, b) => a - b)

  // Build one series per nodule
  const series = noduleNums
    .map((num, i) => {
      const points = timeline
        .map((t, xi) => {
          const n = t.nodules.find((n) => n.number === num)
          const v = n ? parseMm(n.diameter_mm) : null
          return v !== null ? { xi, date: t.date, v } : null
        })
        .filter(Boolean)
      return { num, color: COLORS[i % COLORS.length], points }
    })
    .filter((s) => s.points.length > 0)

  if (!series.length) return null

  const innerW = W - M.left - M.right
  const innerH = H - M.top - M.bottom

  const xOf = (xi) =>
    timeline.length === 1 ? innerW / 2 : (xi / (timeline.length - 1)) * innerW

  const allVals = series.flatMap((s) => s.points.map((p) => p.v))
  const minV = Math.max(0, Math.min(...allVals) - 8)
  const maxV = Math.max(...allVals) + 12
  const yOf = (v) => innerH - ((v - minV) / (maxV - minV)) * innerH

  const yTicks = Array.from({ length: 5 }, (_, i) => minV + (i / 4) * (maxV - minV))

  return (
    <div className="nodule-timeline">
      <div className="nodule-timeline-header">
        <span className="nodule-timeline-title">Evolution temporelle des lésions</span>
        <div className="nodule-timeline-legend">
          {series.map((s) => (
            <span key={s.num} className="nodule-legend-item">
              <span className="nodule-legend-dot" style={{ background: s.color }} />
              Lésion {s.num}
              <TrendBadge points={s.points} color={s.color} />
            </span>
          ))}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="nodule-timeline-svg"
        aria-label="Graphe d'évolution temporelle des lésions"
      >
        <g transform={`translate(${M.left},${M.top})`}>
          {/* Y grid + ticks */}
          {yTicks.map((v, i) => (
            <g key={i}>
              <line
                x1={0} y1={yOf(v)} x2={innerW} y2={yOf(v)}
                stroke="#30363d" strokeWidth={1} strokeDasharray="4 3"
              />
              <text x={-8} y={yOf(v)} textAnchor="end" dominantBaseline="middle" fontSize={10} fill="#8b949e">
                {Math.round(v)}
              </text>
            </g>
          ))}

          {/* Y axis label */}
          <text
            transform={`translate(-40,${innerH / 2}) rotate(-90)`}
            textAnchor="middle" fontSize={10} fill="#8b949e"
          >
            Diamètre (mm)
          </text>

          {/* Axes */}
          <line x1={0} y1={0} x2={0} y2={innerH} stroke="#30363d" strokeWidth={1.5} />
          <line x1={0} y1={innerH} x2={innerW} y2={innerH} stroke="#30363d" strokeWidth={1.5} />

          {/* X ticks + labels */}
          {timeline.map((t, xi) => (
            <g key={xi} transform={`translate(${xOf(xi)},${innerH})`}>
              <line y2={5} stroke="#30363d" />
              <text y={18} textAnchor="middle" fontSize={10} fill="#8b949e">
                {fmtDate(t.date)}
              </text>
            </g>
          ))}

          {/* Series */}
          {series.map((s) => (
            <g key={s.num}>
              {s.points.length > 1 && (
                <polyline
                  points={s.points.map((p) => `${xOf(p.xi)},${yOf(p.v)}`).join(' ')}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  opacity={0.9}
                />
              )}
              {s.points.map((p, i) => (
                <g key={i}>
                  <circle cx={xOf(p.xi)} cy={yOf(p.v)} r={5} fill={s.color} />
                  <text
                    x={xOf(p.xi)}
                    y={yOf(p.v) - 10}
                    textAnchor="middle"
                    fontSize={11}
                    fontWeight={700}
                    fill={s.color}
                  >
                    {p.v}mm
                  </text>
                </g>
              ))}
            </g>
          ))}
        </g>
      </svg>
    </div>
  )
}
