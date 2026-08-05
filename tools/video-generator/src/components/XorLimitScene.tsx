import React from 'react';
import { interpolate, useCurrentFrame } from 'remotion';

export const XorLimitScene: React.FC = () => {
  const frame = useCurrentFrame();

  // Bouncing line trying and failing to separate XOR points
  const lineAngle = Math.sin(frame * 0.08) * 60;

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
        padding: '60px 100px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Left Column: XOR Problem explanation */}
      <div style={{ width: '48%' }}>
        <div style={{ fontFamily: 'monospace', color: '#EF4444', fontSize: 18, letterSpacing: '3px' }}>
          CHAPTER 05 — EMPIRICAL EXPERIMENTS & XOR LIMITATION
        </div>
        <h2 style={{ fontSize: 48, fontWeight: 800, margin: '16px 0', color: '#FFFFFF' }}>
          The XOR Non-Linearity Bottleneck
        </h2>
        <p style={{ fontSize: 22, color: '#94A3B8', lineHeight: 1.6 }}>
          In 1969, <strong>Marvin Minsky & Seymour Papert</strong> proved that a single-layer Perceptron cannot learn the <strong>XOR (Exclusive OR)</strong> function.
        </p>

        <div style={{ marginTop: 24, padding: 24, backgroundColor: 'rgba(239, 68, 68, 0.12)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 16 }}>
          <div style={{ fontWeight: 700, color: '#F87171', fontSize: 20, marginBottom: 8 }}>
            ⚠️ Geometric Proof of Single-Layer Failure
          </div>
          <div style={{ color: '#CBD5E1', fontSize: 18, lineHeight: 1.5 }}>
            No straight 2D line can separate (0,0) and (1,1) from (0,1) and (1,0). This architectural limitation triggered the first AI Winter and proved the necessity of Multi-Layer Neural Networks.
          </div>
        </div>
      </div>

      {/* Right Column: Animated XOR Plot showing non-separable points */}
      <div
        style={{
          width: '46%',
          height: 580,
          backgroundColor: '#121620',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          borderRadius: 24,
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          boxShadow: '0 25px 50px -12px rgba(239, 68, 68, 0.2)',
        }}
      >
        {/* Grid Background */}
        <div
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        />

        {/* Oscillating Line Failing to Separate XOR */}
        <div
          style={{
            position: 'absolute',
            width: '140%',
            height: 4,
            backgroundColor: '#EF4444',
            boxShadow: '0 0 20px #EF4444',
            transform: `rotate(${lineAngle}deg)`,
            transformOrigin: 'center center',
          }}
        />

        {/* XOR Point 1: (0,0) -> Red Class 0 */}
        <div style={{ position: 'absolute', left: 160, top: 400, textAlign: 'center' }}>
          <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: '#EF4444', boxShadow: '0 0 20px #EF4444', margin: '0 auto' }} />
          <span style={{ fontSize: 16, fontFamily: 'monospace', color: '#94A3B8', marginTop: 4, display: 'block' }}>(0,0) = 0</span>
        </div>

        {/* XOR Point 2: (1,1) -> Red Class 0 */}
        <div style={{ position: 'absolute', left: 440, top: 140, textAlign: 'center' }}>
          <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: '#EF4444', boxShadow: '0 0 20px #EF4444', margin: '0 auto' }} />
          <span style={{ fontSize: 16, fontFamily: 'monospace', color: '#94A3B8', marginTop: 4, display: 'block' }}>(1,1) = 0</span>
        </div>

        {/* XOR Point 3: (0,1) -> Green Class 1 */}
        <div style={{ position: 'absolute', left: 160, top: 140, textAlign: 'center' }}>
          <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: '#10B981', boxShadow: '0 0 20px #10B981', margin: '0 auto' }} />
          <span style={{ fontSize: 16, fontFamily: 'monospace', color: '#94A3B8', marginTop: 4, display: 'block' }}>(0,1) = 1</span>
        </div>

        {/* XOR Point 4: (1,0) -> Green Class 1 */}
        <div style={{ position: 'absolute', left: 440, top: 400, textAlign: 'center' }}>
          <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: '#10B981', boxShadow: '0 0 20px #10B981', margin: '0 auto' }} />
          <span style={{ fontSize: 16, fontFamily: 'monospace', color: '#94A3B8', marginTop: 4, display: 'block' }}>(1,0) = 1</span>
        </div>

        <div style={{ position: 'absolute', top: 20, fontFamily: 'monospace', color: '#EF4444', fontSize: 16, backgroundColor: 'rgba(0,0,0,0.6)', padding: '6px 16px', borderRadius: 20 }}>
          ❌ IMPOSSIBLE TO SEPARATE WITH ONE LINE
        </div>
      </div>
    </div>
  );
};
