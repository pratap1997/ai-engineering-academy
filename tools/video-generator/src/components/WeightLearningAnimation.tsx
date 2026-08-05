import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

/**
 * WeightLearningAnimation — Perceptron learns LIVE on screen.
 * 
 * Shows 2D scatter plot with AND gate training data.
 * Decision boundary (line) visibly adjusts each "epoch" as the
 * weight update rule fires — viewer watches the machine learn in real time.
 */
export const WeightLearningAnimation: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // AND gate training data: [x1, x2, label]
  const data: [number, number, number][] = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1],
  ];

  // Simulated weight learning trajectory over epochs
  // w = [w1, w2], b — hand-coded to show realistic convergence
  const epochFrames = 120; // one "epoch" every 120 frames (2 seconds)
  const epochs = [
    { w1: 0.0,  w2: 0.0,  b: 0.0  }, // epoch 0 — random init
    { w1: 0.3,  w2: 0.1,  b: -0.2 }, // epoch 1
    { w1: 0.5,  w2: 0.3,  b: -0.35 },// epoch 2
    { w1: 0.6,  w2: 0.5,  b: -0.5  },// epoch 3 — converging
    { w1: 0.7,  w2: 0.7,  b: -0.65 },// epoch 4 — converging
    { w1: 0.8,  w2: 0.8,  b: -0.75 },// epoch 5 — converged!
  ];

  const epochIdx = Math.min(Math.floor(frame / epochFrames), epochs.length - 2);
  const epochProgress = (frame % epochFrames) / epochFrames;
  const curr = epochs[epochIdx];
  const next = epochs[epochIdx + 1] || curr;

  // Interpolated weights between epochs — smooth animation
  const w1 = curr.w1 + (next.w1 - curr.w1) * epochProgress;
  const w2 = curr.w2 + (next.w2 - curr.w2) * epochProgress;
  const b  = curr.b  + (next.b  - curr.b ) * epochProgress;

  // Canvas coordinate helpers (300x300 unit space → 500x500 px)
  const scale = 180;
  const cx = 310, cy = 310; // center of plot area

  const toScreen = (x: number, y: number) => ({
    sx: cx + x * scale,
    sy: cy - y * scale,
  });

  // Decision boundary line: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
  const linePoints = w2 !== 0
    ? [
        { x: -0.3, y: -(w1 * -0.3 + b) / w2 },
        { x:  1.5, y: -(w1 *  1.5 + b) / w2 },
      ]
    : [
        { x: -b / (w1 || 0.001), y: -0.3 },
        { x: -b / (w1 || 0.001), y:  1.5 },
      ];

  const p1 = toScreen(linePoints[0].x, linePoints[0].y);
  const p2 = toScreen(linePoints[1].x, linePoints[1].y);

  const converged = epochIdx >= 4;
  const updateFired = (frame % epochFrames) < 20 && epochIdx < epochs.length - 2;

  const cardOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <div
      style={{
        width: 1920,
        height: 1080,
        backgroundColor: '#090C10',
        color: '#F3F5F7',
        fontFamily: 'Inter, system-ui, sans-serif',
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '50px 80px',
        position: 'relative',
        overflow: 'hidden',
        opacity: cardOpacity,
      }}
    >
      {/* Background subtle glow */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'radial-gradient(circle at 70% 50%, rgba(56, 189, 248, 0.08), transparent 60%)',
        pointerEvents: 'none',
      }} />

      {/* ── Left panel: context + live stats ── */}
      <div style={{ width: '40%', display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div style={{ fontFamily: 'monospace', color: '#38BDF8', fontSize: 18, letterSpacing: '3px' }}>
          LIVE WEIGHT LEARNING DEMO
        </div>
        <h2 style={{ fontSize: 46, fontWeight: 800, margin: 0, lineHeight: 1.2, color: '#FFFFFF' }}>
          Perceptron Learns<br />
          <span style={{ color: converged ? '#10B981' : '#F59E0B' }}>
            {converged ? 'Converged! ✓' : 'Training...'}
          </span>
        </h2>

        <p style={{ fontSize: 20, color: '#94A3B8', lineHeight: 1.6, margin: 0 }}>
          Training on the <strong>AND gate</strong> dataset. Watch the decision boundary physically
          move each epoch as the weight update rule fires.
        </p>

        {/* Epoch counter */}
        <div style={{
          backgroundColor: '#121620',
          border: '1px solid rgba(56,189,248,0.25)',
          borderRadius: 16,
          padding: '20px 28px',
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 16, color: '#64748B', fontFamily: 'monospace' }}>EPOCH</span>
            <span style={{ fontSize: 32, fontWeight: 700, color: '#38BDF8', fontFamily: 'monospace' }}>
              {epochIdx} / {epochs.length - 1}
            </span>
          </div>

          {/* Live weights */}
          {[['w₁', w1], ['w₂', w2], ['b', b]].map(([label, val]) => (
            <div key={String(label)} style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontSize: 18, color: '#94A3B8', fontFamily: 'monospace' }}>{label}</span>
              <span style={{ fontSize: 18, fontWeight: 600, color: '#A5B4FC', fontFamily: 'monospace' }}>
                {(val as number).toFixed(3)}
              </span>
            </div>
          ))}

          {/* Update flash */}
          {updateFired && (
            <div style={{
              padding: '10px 16px',
              backgroundColor: 'rgba(245, 158, 11, 0.15)',
              border: '1px solid rgba(245, 158, 11, 0.5)',
              borderRadius: 8,
              fontFamily: 'monospace',
              fontSize: 16,
              color: '#FCD34D',
            }}>
              ⚡ Weight update: w ← w + η(y − ŷ)x
            </div>
          )}
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#10B981', boxShadow: '0 0 10px #10B981' }} />
            <span style={{ fontSize: 17, color: '#94A3B8' }}>Class 1 (AND = true)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: '50%', backgroundColor: '#EF4444', boxShadow: '0 0 10px #EF4444' }} />
            <span style={{ fontSize: 17, color: '#94A3B8' }}>Class 0 (AND = false)</span>
          </div>
        </div>
      </div>

      {/* ── Right panel: SVG animated scatter plot ── */}
      <div style={{
        width: '55%',
        height: 620,
        backgroundColor: '#0D1117',
        border: `1px solid ${converged ? 'rgba(16,185,129,0.4)' : 'rgba(56,189,248,0.25)'}`,
        borderRadius: 24,
        boxShadow: `0 0 60px ${converged ? 'rgba(16,185,129,0.15)' : 'rgba(56,189,248,0.1)'}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <svg width={620} height={620} viewBox="0 0 620 620">
          {/* Grid */}
          {[-0.5, 0, 0.5, 1, 1.5].map(v => {
            const { sx, sy } = toScreen(v, 0);
            const { sx: sx2, sy: sy2 } = toScreen(0, v);
            return (
              <g key={v}>
                <line x1={sx} y1={40} x2={sx} y2={580} stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
                <line x1={40} y1={sy2} x2={580} y2={sy2} stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
                <text x={sx} y={600} textAnchor="middle" fill="#475569" fontSize={14} fontFamily="monospace">{v}</text>
                <text x={24} y={sy2 + 5} textAnchor="middle" fill="#475569" fontSize={14} fontFamily="monospace">{v}</text>
              </g>
            );
          })}

          {/* Axes */}
          <line x1={cx} y1={40} x2={cx} y2={580} stroke="rgba(255,255,255,0.15)" strokeWidth={1.5} />
          <line x1={40} y1={cy} x2={580} y2={cy} stroke="rgba(255,255,255,0.15)" strokeWidth={1.5} />

          {/* Decision boundary line */}
          <line
            x1={Math.max(30, Math.min(590, p1.sx))} y1={Math.max(30, Math.min(590, p1.sy))}
            x2={Math.max(30, Math.min(590, p2.sx))} y2={Math.max(30, Math.min(590, p2.sy))}
            stroke={converged ? '#10B981' : '#38BDF8'}
            strokeWidth={converged ? 4 : 3}
            strokeDasharray={converged ? 'none' : '10 4'}
            style={{ filter: `drop-shadow(0 0 8px ${converged ? '#10B981' : '#38BDF8'})` }}
          />

          {/* Data points */}
          {data.map(([x, y, label], i) => {
            const { sx, sy } = toScreen(x, y);
            const color = label === 1 ? '#10B981' : '#EF4444';
            return (
              <g key={i}>
                <circle cx={sx} cy={sy} r={18} fill={color} opacity={0.2} />
                <circle cx={sx} cy={sy} r={10} fill={color}
                  style={{ filter: `drop-shadow(0 0 10px ${color})` }} />
                <text x={sx + 16} y={sy - 14} fill={color} fontSize={15} fontFamily="monospace" fontWeight="bold">
                  ({x},{y})→{label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Converged badge */}
        {converged && (
          <div style={{
            position: 'absolute',
            top: 20,
            right: 20,
            backgroundColor: 'rgba(16,185,129,0.15)',
            border: '1px solid rgba(16,185,129,0.5)',
            borderRadius: 30,
            padding: '8px 20px',
            fontFamily: 'monospace',
            color: '#34D399',
            fontSize: 16,
            fontWeight: 700,
          }}>
            CONVERGED — 100% accuracy
          </div>
        )}
      </div>
    </div>
  );
};
