import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './App.css'

/* ================================================================
   CONSTANTS
================================================================ */
const PHASES = ['ready', 'inputs', 'connections', 'weights', 'signal', 'sum', 'step', 'output']
const PHASE_LABELS = ['Ready', 'Inputs', 'Wires', 'Weights', 'Signal', 'Sum', 'f(z)', 'Output']
const PHASE_MS = 1100

const P = {
  x1:   { x: 90,  y: 145 },
  x2:   { x: 90,  y: 295 },
  bias: { x: 90,  y: 445 },
  sum:  { x: 430, y: 220 },
  step: { x: 600, y: 220 },
  out:  { x: 750, y: 220 },
}

/* ================================================================
   MATH HELPERS
================================================================ */
const calcZ   = (x1, x2, w1, w2, b) => w1 * x1 + w2 * x2 + b
const stepFn  = z => z >= 0 ? 1 : 0

function cubicPath(from, to) {
  const cx = from.x + (to.x - from.x) * 0.5
  return `M ${from.x + 28},${from.y} C ${cx},${from.y} ${cx},${to.y} ${to.x - 30},${to.y}`
}

function linePath(from, to) {
  return `M ${from.x},${from.y - 22} L ${to.x - 28},${to.y + 14}`
}

/* ================================================================
   PRESETS
================================================================ */
const PRESETS = {
  AND:  { w1: 0.5,  w2: 0.5,  b: -0.7, x1: 1, x2: 1 },
  OR:   { w1: 0.6,  w2: 0.6,  b: -0.3, x1: 0, x2: 1 },
  NAND: { w1: -0.5, w2: -0.5, b:  0.7, x1: 1, x2: 1 },
  XOR:  { w1: 0.5,  w2: 0.5,  b: -0.3, x1: 1, x2: 0 },
}

/* ================================================================
   STEP CHART (plain SVG, no Framer Motion inside)
================================================================ */
function StepChart({ z, visible }) {
  if (!visible) return null
  const w = 200, h = 100, mx = w / 2, my = h / 2
  const dotX = mx + Math.max(-mx + 14, Math.min(mx - 14, z * 20))
  const dotY = z >= 0 ? my - 22 : my + 22

  return (
    <motion.div
      className="step-chart-wrap"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h}>
        <line x1={10} y1={my} x2={w-10} y2={my} stroke="rgba(255,255,255,0.12)" strokeWidth={1}/>
        <line x1={mx} y1={8}  x2={mx}   y2={h-8} stroke="rgba(255,255,255,0.12)" strokeWidth={1}/>
        <polyline
          points={`10,${my+22} ${mx},${my+22} ${mx},${my-22} ${w-10},${my-22}`}
          fill="none" stroke="#8b5cf6" strokeWidth={2.5} strokeLinejoin="round"
        />
        <circle cx={dotX} cy={dotY} r={6} fill={z >= 0 ? '#10b981' : '#ef4444'}/>
        <line x1={mx} y1={6} x2={mx} y2={h-6} stroke="rgba(245,158,11,0.5)" strokeWidth={1} strokeDasharray="4 3"/>
        <text x={mx+4} y={14} fill="#f59e0b" fontSize="9" fontFamily="Inter">z=0</text>
        <text x={12} y={my-25} fill="#10b981" fontSize="9" fontFamily="Inter">y=1</text>
        <text x={12} y={my+32} fill="#ef4444" fontSize="9" fontFamily="Inter">y=0</text>
      </svg>
    </motion.div>
  )
}

/* ================================================================
   NETWORK DIAGRAM — pure SVG + Framer Motion wrappers outside SVG
================================================================ */
function NetworkDiagram({ x1, x2, w1, w2, b, phase }) {
  const idx   = PHASES.indexOf(phase)
  const after = (name) => idx >= PHASES.indexOf(name)

  const z    = calcZ(x1, x2, w1, w2, b)
  const pred = stepFn(z)
  const outColor = pred === 1 ? '#10b981' : '#ef4444'

  const pathX1Sum  = cubicPath(P.x1,   P.sum)
  const pathX2Sum  = cubicPath(P.x2,   P.sum)
  const pathBSum   = linePath( P.bias, P.sum)
  const pathSumStep = `M ${P.sum.x+32},${P.sum.y} L ${P.step.x-32},${P.step.y}`
  const pathStepOut = `M ${P.step.x+32},${P.step.y} L ${P.out.x-32},${P.out.y}`

  return (
    <div className="diagram-svg-wrap">
      <svg
        viewBox="0 0 840 530"
        preserveAspectRatio="xMidYMid meet"
        style={{ width: '100%', height: '100%', maxWidth: 840 }}
      >
        <defs>
          <filter id="glow-blue">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#00d4ff" floodOpacity="0.8"/>
          </filter>
          <filter id="glow-purple">
            <feDropShadow dx="0" dy="0" stdDeviation="7" floodColor="#8b5cf6" floodOpacity="0.8"/>
          </filter>
          <filter id="glow-green">
            <feDropShadow dx="0" dy="0" stdDeviation="7" floodColor="#10b981" floodOpacity="0.9"/>
          </filter>
          <filter id="glow-red">
            <feDropShadow dx="0" dy="0" stdDeviation="7" floodColor="#ef4444" floodOpacity="0.9"/>
          </filter>
          <filter id="glow-amber">
            <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#f59e0b" floodOpacity="0.8"/>
          </filter>
          <linearGradient id="wire-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.8"/>
          </linearGradient>

          {/* Path defs for animateMotion */}
          <path id="pm-x1"   d={pathX1Sum}/>
          <path id="pm-x2"   d={pathX2Sum}/>
          <path id="pm-bias" d={pathBSum}/>
          <path id="pm-ss"   d={pathSumStep}/>
          <path id="pm-so"   d={pathStepOut}/>
        </defs>

        {/* Background grid */}
        <defs>
          <pattern id="bg-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.025)" strokeWidth="1"/>
          </pattern>
        </defs>
        <rect width="840" height="530" fill="url(#bg-grid)" rx="16"/>

        {/* ---- WIRES ---- */}
        {after('connections') && (
          <>
            <path d={pathX1Sum}  fill="none" stroke="url(#wire-grad)" strokeWidth="2" opacity="0.7"/>
            <path d={pathX2Sum}  fill="none" stroke="url(#wire-grad)" strokeWidth="2" opacity="0.7"/>
            <path d={pathBSum}   fill="none" stroke="#f59e0b" strokeWidth="2" opacity="0.6"/>
          </>
        )}
        {after('sum') && (
          <path d={pathSumStep} fill="none" stroke="#8b5cf6" strokeWidth="2.5" opacity="0.9"/>
        )}
        {after('step') && (
          <path d={pathStepOut} fill="none" stroke={outColor} strokeWidth="2.5" opacity="0.9"/>
        )}

        {/* ---- SIGNAL PARTICLES ---- */}
        {after('signal') && (
          <>
            <circle r="6" fill="#00d4ff" filter="url(#glow-blue)">
              <animateMotion dur="1.1s" repeatCount="indefinite" begin="0s">
                <mpath href="#pm-x1"/>
              </animateMotion>
            </circle>
            <circle r="6" fill="#00d4ff" filter="url(#glow-blue)">
              <animateMotion dur="1.1s" repeatCount="indefinite" begin="0.4s">
                <mpath href="#pm-x2"/>
              </animateMotion>
            </circle>
            <circle r="5" fill="#f59e0b" filter="url(#glow-amber)">
              <animateMotion dur="1.3s" repeatCount="indefinite" begin="0.7s">
                <mpath href="#pm-bias"/>
              </animateMotion>
            </circle>
          </>
        )}
        {after('sum') && (
          <circle r="6" fill="#8b5cf6" filter="url(#glow-purple)">
            <animateMotion dur="0.8s" repeatCount="indefinite" begin="0s">
              <mpath href="#pm-ss"/>
            </animateMotion>
          </circle>
        )}
        {after('step') && (
          <circle r="6" fill={outColor} filter={`url(#glow-${pred ? 'green' : 'red'})`}>
            <animateMotion dur="0.8s" repeatCount="indefinite" begin="0s">
              <mpath href="#pm-so"/>
            </animateMotion>
          </circle>
        )}

        {/* ---- WEIGHT BADGES ---- */}
        {after('weights') && (
          <>
            <rect x={220} y={P.x1.y - 24} width={78} height={22} rx={6}
              fill="#12122a" stroke="rgba(0,212,255,0.35)" strokeWidth={1}/>
            <text x={259} y={P.x1.y - 8} textAnchor="middle" fill="#00d4ff"
              fontSize="11" fontFamily="JetBrains Mono, monospace">
              w1={w1.toFixed(2)}
            </text>
            <rect x={220} y={P.x2.y + 8} width={78} height={22} rx={6}
              fill="#12122a" stroke="rgba(0,212,255,0.35)" strokeWidth={1}/>
            <text x={259} y={P.x2.y + 24} textAnchor="middle" fill="#00d4ff"
              fontSize="11" fontFamily="JetBrains Mono, monospace">
              w2={w2.toFixed(2)}
            </text>
          </>
        )}

        {/* ---- INPUT NODES ---- */}
        {after('inputs') && (
          <>
            {/* x1 */}
            <circle cx={P.x1.x} cy={P.x1.y} r={32}
              fill="rgba(0,212,255,0.1)" stroke="#00d4ff" strokeWidth={2}
              filter="url(#glow-blue)"/>
            <text x={P.x1.x} y={P.x1.y - 7} textAnchor="middle"
              fill="#00d4ff" fontSize="13" fontWeight="600" fontFamily="Inter">x1</text>
            <text x={P.x1.x} y={P.x1.y + 10} textAnchor="middle"
              fill="#e2e8f0" fontSize="12" fontFamily="JetBrains Mono, monospace">
              {x1.toFixed(1)}
            </text>

            {/* x2 */}
            <circle cx={P.x2.x} cy={P.x2.y} r={32}
              fill="rgba(0,212,255,0.1)" stroke="#00d4ff" strokeWidth={2}
              filter="url(#glow-blue)"/>
            <text x={P.x2.x} y={P.x2.y - 7} textAnchor="middle"
              fill="#00d4ff" fontSize="13" fontWeight="600" fontFamily="Inter">x2</text>
            <text x={P.x2.x} y={P.x2.y + 10} textAnchor="middle"
              fill="#e2e8f0" fontSize="12" fontFamily="JetBrains Mono, monospace">
              {x2.toFixed(1)}
            </text>

            {/* bias */}
            <circle cx={P.bias.x} cy={P.bias.y} r={26}
              fill="rgba(245,158,11,0.1)" stroke="#f59e0b" strokeWidth={2}
              filter="url(#glow-amber)"/>
            <text x={P.bias.x} y={P.bias.y - 5} textAnchor="middle"
              fill="#f59e0b" fontSize="13" fontWeight="600" fontFamily="Inter">b</text>
            <text x={P.bias.x} y={P.bias.y + 11} textAnchor="middle"
              fill="#e2e8f0" fontSize="11" fontFamily="JetBrains Mono, monospace">
              {b.toFixed(2)}
            </text>
          </>
        )}

        {/* ---- SUM NODE ---- */}
        {after('sum') && (
          <>
            <circle cx={P.sum.x} cy={P.sum.y} r={40}
              fill="rgba(139,92,246,0.18)" stroke="#8b5cf6" strokeWidth={2.5}
              filter="url(#glow-purple)"/>
            <text x={P.sum.x} y={P.sum.y - 8} textAnchor="middle"
              fill="#8b5cf6" fontSize="22" fontWeight="700" fontFamily="Inter">
              {'\u03A3'}
            </text>
            <text x={P.sum.x} y={P.sum.y + 14} textAnchor="middle"
              fill="#e2e8f0" fontSize="11" fontFamily="JetBrains Mono, monospace">
              {z.toFixed(3)}
            </text>
            {/* z equation below */}
            <text x={P.sum.x} y={P.sum.y + 60} textAnchor="middle"
              fill="rgba(139,92,246,0.65)" fontSize="10"
              fontFamily="JetBrains Mono, monospace">
              {(w1*x1).toFixed(2)} + {(w2*x2).toFixed(2)} + ({b.toFixed(2)})
            </text>
          </>
        )}

        {/* ---- STEP NODE ---- */}
        {after('step') && (
          <>
            <rect x={P.step.x - 34} y={P.step.y - 34} width={68} height={68}
              rx={10} fill="rgba(139,92,246,0.15)" stroke="#8b5cf6" strokeWidth={2}
              filter="url(#glow-purple)"/>
            <text x={P.step.x} y={P.step.y - 10} textAnchor="middle"
              fill="#8b5cf6" fontSize="12" fontWeight="600" fontFamily="Inter">f(z)</text>
            <text x={P.step.x} y={P.step.y + 8} textAnchor="middle"
              fill={z >= 0 ? '#10b981' : '#ef4444'} fontSize="12"
              fontFamily="JetBrains Mono, monospace">
              {z >= 0 ? '\u2265 0' : '< 0'}
            </text>
            <text x={P.step.x} y={P.step.y + 24} textAnchor="middle"
              fill={z >= 0 ? '#10b981' : '#ef4444'} fontSize="11"
              fontFamily="JetBrains Mono, monospace">
              {z >= 0 ? '\u2713 fire' : '\u2717 silent'}
            </text>
          </>
        )}

        {/* ---- OUTPUT NODE ---- */}
        {after('output') && (
          <>
            <circle cx={P.out.x} cy={P.out.y} r={40}
              fill={pred ? 'rgba(16,185,129,0.18)' : 'rgba(239,68,68,0.18)'}
              stroke={outColor} strokeWidth={3}
              filter={`url(#glow-${pred ? 'green' : 'red'})`}/>
            <text x={P.out.x} y={P.out.y - 10} textAnchor="middle"
              fill={outColor} fontSize="13" fontWeight="600" fontFamily="Inter">
              {'\u0177'}
            </text>
            <text x={P.out.x} y={P.out.y + 14} textAnchor="middle"
              fill={outColor} fontSize="26" fontWeight="700"
              fontFamily="JetBrains Mono, monospace">
              {pred}
            </text>
          </>
        )}

        {/* ---- LAYER LABELS ---- */}
        {after('inputs') && (
          <text x={90} y={500} textAnchor="middle" fill="rgba(255,255,255,0.2)"
            fontSize="9" fontFamily="Inter" letterSpacing="2">INPUT</text>
        )}
        {after('sum') && (
          <text x={430} y={290} textAnchor="middle" fill="rgba(255,255,255,0.2)"
            fontSize="9" fontFamily="Inter" letterSpacing="2">WEIGHTED SUM</text>
        )}
        {after('step') && (
          <text x={600} y={290} textAnchor="middle" fill="rgba(255,255,255,0.2)"
            fontSize="9" fontFamily="Inter" letterSpacing="2">ACTIVATION</text>
        )}
        {after('output') && (
          <text x={750} y={290} textAnchor="middle" fill="rgba(255,255,255,0.2)"
            fontSize="9" fontFamily="Inter" letterSpacing="2">OUTPUT</text>
        )}
      </svg>
    </div>
  )
}

/* ================================================================
   SLIDER ROW
================================================================ */
function SliderRow({ label, value, min, max, step, color, sublabel, onChange }) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div className="slider-row">
      <div className="slider-header">
        <span className="slider-label" style={{ color }}>{label}</span>
        <span className="slider-value mono" style={{ color }}>{value.toFixed(2)}</span>
      </div>
      {sublabel && <div className="slider-sublabel">{sublabel}</div>}
      <div className="slider-track-wrap">
        <div className="slider-fill" style={{ width: `${Math.max(0, pct)}%`, background: color }}/>
        <input
          type="range" min={min} max={max} step={step}
          value={value} onChange={e => onChange(parseFloat(e.target.value))}
          className="slider-input"
        />
      </div>
    </div>
  )
}

/* ================================================================
   TRUTH TABLE
================================================================ */
function TruthTable({ w1, w2, b }) {
  const cases = [[0,0],[0,1],[1,0],[1,1]]
  return (
    <div className="truth-table">
      <div className="truth-title">Truth Table</div>
      <table>
        <thead>
          <tr><th>x1</th><th>x2</th><th>z</th><th>y</th></tr>
        </thead>
        <tbody>
          {cases.map(([cx, cy]) => {
            const z = calcZ(cx, cy, w1, w2, b)
            const p = stepFn(z)
            return (
              <tr key={`${cx}${cy}`}>
                <td className="mono">{cx}</td>
                <td className="mono">{cy}</td>
                <td className="mono" style={{ color: z >= 0 ? '#8b5cf6' : '#64748b' }}>
                  {z.toFixed(2)}
                </td>
                <td className="mono" style={{ color: p ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                  {p}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ================================================================
   PHASE PROGRESS BAR
================================================================ */
function PhaseBar({ phase }) {
  const idx = PHASES.indexOf(phase)
  const pct = PHASES.length > 1 ? (idx / (PHASES.length - 1)) * 100 : 0
  return (
    <div className="phase-bar-wrap">
      {PHASES.map((p, i) => (
        <div key={p}
          className={[
            'phase-dot',
            i <= idx ? 'phase-dot-active' : '',
            i === idx ? 'phase-dot-current' : '',
          ].join(' ')}
        >
          <div className="phase-dot-label">{PHASE_LABELS[i]}</div>
        </div>
      ))}
      <div className="phase-track">
        <motion.div
          className="phase-fill"
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
        />
      </div>
    </div>
  )
}

/* ================================================================
   COMPUTATION PANEL
================================================================ */
function ComputationPanel({ x1, x2, w1, w2, b, phase }) {
  const idx   = PHASES.indexOf(phase)
  const after = (name) => idx >= PHASES.indexOf(name)
  const z     = calcZ(x1, x2, w1, w2, b)
  const pred  = stepFn(z)
  const predColor = pred === 1 ? '#10b981' : '#ef4444'

  return (
    <div className="comp-panel">
      <h3 className="comp-title">Computation Trace</h3>

      {/* Pre-activation */}
      {after('sum') && (
        <motion.div className="comp-section"
          initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="comp-label text-muted">z = w·x + b</div>
          <div className="comp-expr">
            <span className="text-purple mono">{w1.toFixed(2)}</span>
            <span className="text-muted"> x </span>
            <span className="text-blue mono">{x1.toFixed(2)}</span>
            <span className="text-muted"> + </span>
            <span className="text-purple mono">{w2.toFixed(2)}</span>
            <span className="text-muted"> x </span>
            <span className="text-blue mono">{x2.toFixed(2)}</span>
            <span className="text-muted"> + </span>
            <span className="text-amber mono">{b.toFixed(2)}</span>
          </div>
          <div className="comp-result" style={{ borderColor: z >= 0 ? '#8b5cf650' : '#ef444450' }}>
            <span className="text-muted mono">z = </span>
            <span className="mono" style={{ color: '#8b5cf6', fontSize: '1.25em', fontWeight: 700 }}>
              {z.toFixed(4)}
            </span>
          </div>
        </motion.div>
      )}

      {/* Step function */}
      {after('step') && (
        <>
          <div className="comp-divider"/>
          <motion.div className="comp-section"
            initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            <div className="comp-label text-muted">f(z) — Step function</div>
            <div className="comp-expr" style={{ color: z >= 0 ? '#10b981' : '#ef4444' }}>
              {z >= 0 ? 'z >= 0  -->  predict 1' : 'z < 0   -->  predict 0'}
            </div>
            <StepChart z={z} visible={true}/>
          </motion.div>
        </>
      )}

      {/* Output */}
      {after('output') && (
        <motion.div
          className="comp-output"
          style={{ borderColor: predColor, background: `${predColor}10` }}
          initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: [0.175, 0.885, 0.32, 1.275] }}
        >
          <div className="comp-output-label" style={{ color: predColor }}>PREDICTION</div>
          <motion.div
            className="comp-output-value" style={{ color: predColor }}
            key={pred}
            initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 400, damping: 15 }}
          >
            y = {pred}
          </motion.div>
          <div className="comp-output-interp" style={{ color: predColor }}>
            {pred === 1 ? 'NEURON FIRES' : 'NEURON SILENT'}
          </div>
        </motion.div>
      )}

      {!after('sum') && (
        <div className="comp-placeholder">
          <span className="comp-placeholder-text">
            Press <span className="mono text-blue">Run</span> to trace
          </span>
        </div>
      )}

      {/* Reference */}
      <div className="eq-ref">
        <div className="eq-ref-title text-muted">Equations</div>
        <div className="eq-line">
          <span className="text-muted">z =</span>
          <span className="mono text-purple"> w1*x1 + w2*x2 + b</span>
        </div>
        <div className="eq-line">
          <span className="mono text-green">y=1</span>
          <span className="text-muted"> if z &ge; 0 else </span>
          <span className="mono text-red">y=0</span>
        </div>
        <div className="eq-line">
          <span className="text-muted">update: </span>
          <span className="mono text-blue">w += lr*(y-yhat)*x</span>
        </div>
      </div>
    </div>
  )
}

/* ================================================================
   ROOT APP
================================================================ */
export default function App() {
  const [x1, setX1] = useState(1)
  const [x2, setX2] = useState(1)
  const [w1, setW1] = useState(0.5)
  const [w2, setW2] = useState(0.5)
  const [b,  setB]  = useState(-0.7)
  const [phase,     setPhase]   = useState('ready')
  const [isPlaying, setPlaying] = useState(false)
  const timerRef = useRef(null)

  const z    = calcZ(x1, x2, w1, w2, b)
  const pred = stepFn(z)
  const predColor = pred === 1 ? '#10b981' : '#ef4444'

  const stopAnim = useCallback(() => {
    clearTimeout(timerRef.current)
    setPlaying(false)
    setPhase('ready')
  }, [])

  const runAnim = useCallback(() => {
    clearTimeout(timerRef.current)
    setPhase('ready')
    setPlaying(true)
    let i = 1
    const tick = () => {
      if (i >= PHASES.length) { setPlaying(false); return }
      setPhase(PHASES[i++])
      timerRef.current = setTimeout(tick, PHASE_MS)
    }
    timerRef.current = setTimeout(tick, 250)
  }, [])

  const applyPreset = (name) => {
    const p = PRESETS[name]
    setW1(p.w1); setW2(p.w2); setB(p.b); setX1(p.x1); setX2(p.x2)
    stopAnim()
    setTimeout(runAnim, 80)
  }

  const handleSlider = (setter) => (v) => {
    setter(v)
    stopAnim()
    setTimeout(runAnim, 80)
  }

  // Auto-play on mount
  useEffect(() => {
    const t = setTimeout(runAnim, 700)
    return () => { clearTimeout(t); clearTimeout(timerRef.current) }
  }, []) // eslint-disable-line

  return (
    <div className="app">
      {/* HEADER */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-left">
            <div className="module-badge">MODULE 001</div>
            <h1 className="header-title">Perceptron</h1>
            <span className="header-sub">Animated Forward Pass</span>
          </div>
          <div className="header-right">
            <div className="prediction-chip" style={{
              borderColor: `${predColor}50`,
              background:  `${predColor}12`,
              color: predColor,
            }}>
              <span className="chip-label">LIVE</span>
              <span className="chip-value mono">y = {pred}</span>
            </div>
          </div>
        </div>
        <PhaseBar phase={phase}/>
      </header>

      {/* MAIN */}
      <main className="app-main">

        {/* LEFT: Controls */}
        <aside className="panel panel-controls">
          <div className="panel-section">
            <h3 className="panel-heading">Inputs</h3>
            <SliderRow label="x1" value={x1} min={0} max={1} step={0.1}
              color="#00d4ff" sublabel="Feature 1"
              onChange={handleSlider(setX1)}/>
            <SliderRow label="x2" value={x2} min={0} max={1} step={0.1}
              color="#00d4ff" sublabel="Feature 2"
              onChange={handleSlider(setX2)}/>
          </div>

          <div className="panel-section">
            <h3 className="panel-heading">Parameters</h3>
            <SliderRow label="w1" value={w1} min={-2} max={2} step={0.05}
              color="#8b5cf6" sublabel="Weight for x1"
              onChange={handleSlider(setW1)}/>
            <SliderRow label="w2" value={w2} min={-2} max={2} step={0.05}
              color="#8b5cf6" sublabel="Weight for x2"
              onChange={handleSlider(setW2)}/>
            <SliderRow label="b" value={b} min={-2} max={2} step={0.05}
              color="#f59e0b" sublabel="Bias term"
              onChange={handleSlider(setB)}/>
          </div>

          <div className="panel-section">
            <h3 className="panel-heading">Presets</h3>
            <div className="preset-grid">
              {Object.keys(PRESETS).map(name => (
                <button key={name} className="preset-btn" onClick={() => applyPreset(name)}>
                  {name}
                </button>
              ))}
            </div>
          </div>

          <div className="panel-section">
            <TruthTable w1={w1} w2={w2} b={b}/>
          </div>
        </aside>

        {/* CENTER: Diagram */}
        <div className="panel panel-diagram">
          <NetworkDiagram x1={x1} x2={x2} w1={w1} w2={w2} b={b} phase={phase}/>

          <div className="diagram-controls">
            <motion.button
              className={`btn-run ${isPlaying ? 'btn-run-playing' : ''}`}
              onClick={isPlaying ? stopAnim : runAnim}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              <span className="btn-icon">{isPlaying ? '⏹' : '▶'}</span>
              {isPlaying ? 'Stop' : 'Run Animation'}
            </motion.button>

            <motion.button
              className="btn-reset"
              onClick={stopAnim}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              Reset
            </motion.button>

            <span className="phase-label">
              Phase: <span className="mono text-blue">{phase}</span>
            </span>
          </div>
        </div>

        {/* RIGHT: Computation */}
        <aside className="panel panel-computation">
          <ComputationPanel x1={x1} x2={x2} w1={w1} w2={w2} b={b} phase={phase}/>
        </aside>
      </main>

      {/* FOOTER */}
      <footer className="app-footer">
        <span>AI Engineering Academy · Module 001 · Perceptron From Scratch</span>
        <span className="text-muted">0/1 labels · step function · single layer · MIT</span>
      </footer>
    </div>
  )
}
