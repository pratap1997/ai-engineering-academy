import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

export const MentalModelScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animation for line slope angle rotating to separate points
  const lineAngle = interpolate(frame, [0, 90], [15, -45], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const lineYIntercept = interpolate(frame, [0, 90], [50, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

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
        padding: '60px 100px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Left Column: Explanatory Concept Cards */}
      <div style={{ width: '45%', opacity: cardOpacity }}>
        <div style={{ fontFamily: 'monospace', color: '#38BDF8', fontSize: 18, letterSpacing: '3px' }}>
          CHAPTER 02 — GEOMETRIC MENTAL MODEL
        </div>
        <h2 style={{ fontSize: 52, fontWeight: 800, margin: '16px 0', color: '#FFFFFF' }}>
          Linear Decision Boundary
        </h2>
        <p style={{ fontSize: 24, color: '#94A3B8', lineHeight: 1.6 }}>
          A Perceptron separates two classes in N-dimensional feature space using a linear <strong>hyperplane</strong>.
        </p>

        <div style={{ marginTop: 30, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ padding: 20, backgroundColor: 'rgba(16, 185, 129, 0.1)', borderLeft: '4px solid #10B981', borderRadius: 8 }}>
            <div style={{ fontWeight: 700, color: '#34D399', fontSize: 20 }}>Positive Class (y = 1)</div>
            <div style={{ color: '#94A3B8', fontSize: 18 }}>w₁x₁ + w₂x₂ + b ≥ 0 (Above Hyperplane)</div>
          </div>
          <div style={{ padding: 20, backgroundColor: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #EF4444', borderRadius: 8 }}>
            <div style={{ fontWeight: 700, color: '#F87171', fontSize: 20 }}>Negative Class (y = 0)</div>
            <div style={{ color: '#94A3B8', fontSize: 18 }}>w₁x₁ + w₂x₂ + b &lt; 0 (Below Hyperplane)</div>
          </div>
        </div>
      </div>

      {/* Right Column: Interactive Animated Decision Boundary Plot */}
      <div
        style={{
          width: '50%',
          height: 600,
          backgroundColor: '#121620',
          border: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 24,
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
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

        {/* Animated Decision Line */}
        <div
          style={{
            position: 'absolute',
            width: '140%',
            height: 4,
            backgroundColor: '#38BDF8',
            boxShadow: '0 0 20px #38BDF8',
            transform: `rotate(${lineAngle}deg) translateY(${lineYIntercept}px)`,
            transformOrigin: 'center center',
          }}
        />

        {/* Data Points (Green Class 1) */}
        {[
          { x: 120, y: 140 },
          { x: 180, y: 100 },
          { x: 250, y: 160 },
          { x: 300, y: 90 },
        ].map((pt, i) => (
          <div
            key={`green-${i}`}
            style={{
              position: 'absolute',
              left: pt.x + 200,
              top: pt.y + 100,
              width: 24,
              height: 24,
              borderRadius: '50%',
              backgroundColor: '#10B981',
              boxShadow: '0 0 15px #10B981',
            }}
          />
        ))}

        {/* Data Points (Red Class 0) */}
        {[
          { x: 100, y: 350 },
          { x: 180, y: 420 },
          { x: 260, y: 380 },
          { x: 340, y: 440 },
        ].map((pt, i) => (
          <div
            key={`red-${i}`}
            style={{
              position: 'absolute',
              left: pt.x + 100,
              top: pt.y + 80,
              width: 24,
              height: 24,
              borderRadius: '50%',
              backgroundColor: '#EF4444',
              boxShadow: '0 0 15px #EF4444',
            }}
          />
        ))}

        <div style={{ position: 'absolute', bottom: 20, right: 25, fontFamily: 'monospace', color: '#64748B', fontSize: 16 }}>
          Decision Boundary Line: w₁x₁ + w₂x₂ + b = 0
        </div>
      </div>
    </div>
  );
};
