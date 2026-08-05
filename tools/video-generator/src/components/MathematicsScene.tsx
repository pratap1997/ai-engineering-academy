import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

export const MathematicsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const eq1Spring = spring({ frame: frame - 10, fps, config: { damping: 14 } });
  const eq2Spring = spring({ frame: frame - 30, fps, config: { damping: 14 } });
  const eq3Spring = spring({ frame: frame - 50, fps, config: { damping: 14 } });

  return (
    <div
      style={{
        width: 1920,
        height: 1080,
        backgroundColor: '#090C10',
        color: '#F3F5F7',
        fontFamily: 'Inter, system-ui, sans-serif',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '60px 100px',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{ fontFamily: 'monospace', color: '#A855F7', fontSize: 20, letterSpacing: '4px', marginBottom: 12 }}>
        CHAPTER 03 — FORMAL MATHEMATICAL DERIVATION
      </div>
      <h2 style={{ fontSize: 56, fontWeight: 800, margin: 0, color: '#FFFFFF', textAlign: 'center' }}>
        The Perceptron Learning Algorithm
      </h2>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 30, marginTop: 50, width: '100%', maxWidth: 1600 }}>
        {/* Card 1: Linear Combination */}
        <div
          style={{
            backgroundColor: '#121620',
            border: '1px solid rgba(168, 85, 247, 0.3)',
            borderRadius: 20,
            padding: 36,
            transform: `scale(${Math.max(0, eq1Spring)})`,
            boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
          }}
        >
          <div style={{ fontSize: 18, color: '#C084FC', fontFamily: 'monospace', marginBottom: 12 }}>01. PRE-ACTIVATION</div>
          <div style={{ fontSize: 32, fontWeight: 700, fontFamily: 'serif', color: '#FFFFFF', marginBottom: 16 }}>
            z = wᵀx + b = ∑ wᵢxᵢ + b
          </div>
          <div style={{ fontSize: 18, color: '#94A3B8', lineHeight: 1.5 }}>
            Computes the weighted sum of input features plus bias offset.
          </div>
        </div>

        {/* Card 2: Step Activation */}
        <div
          style={{
            backgroundColor: '#121620',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            borderRadius: 20,
            padding: 36,
            transform: `scale(${Math.max(0, eq2Spring)})`,
            boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
          }}
        >
          <div style={{ fontSize: 18, color: '#38BDF8', fontFamily: 'monospace', marginBottom: 12 }}>02. HEAVISIDE STEP FUNCTION</div>
          <div style={{ fontSize: 32, fontWeight: 700, fontFamily: 'serif', color: '#FFFFFF', marginBottom: 16 }}>
            ŷ = f(z) = 1 if z ≥ 0 else 0
          </div>
          <div style={{ fontSize: 18, color: '#94A3B8', lineHeight: 1.5 }}>
            Maps real-valued continuous score z to discrete binary output {0, 1}.
          </div>
        </div>

        {/* Card 3: Weight Update Rule */}
        <div
          style={{
            backgroundColor: '#121620',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            borderRadius: 20,
            padding: 36,
            transform: `scale(${Math.max(0, eq3Spring)})`,
            boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
          }}
        >
          <div style={{ fontSize: 18, color: '#34D399', fontFamily: 'monospace', marginBottom: 12 }}>03. WEIGHT UPDATE RULE</div>
          <div style={{ fontSize: 30, fontWeight: 700, fontFamily: 'serif', color: '#FFFFFF', marginBottom: 16 }}>
            w ← w + η(y - ŷ)x
          </div>
          <div style={{ fontSize: 18, color: '#94A3B8', lineHeight: 1.5 }}>
            Updates weight vector only when a misclassification occurs (y ≠ ŷ).
          </div>
        </div>
      </div>
    </div>
  );
};
