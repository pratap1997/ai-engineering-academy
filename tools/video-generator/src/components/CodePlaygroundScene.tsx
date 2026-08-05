import React from 'react';
import { interpolate, useCurrentFrame } from 'remotion';

interface CodePlaygroundSceneProps {
  codeSnippet: string;
}

export const CodePlaygroundScene: React.FC<CodePlaygroundSceneProps> = ({ codeSnippet }) => {
  const frame = useCurrentFrame();

  const codeProgress = interpolate(frame, [0, 180], [0, codeSnippet.length], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const displayedCode = codeSnippet.slice(0, Math.floor(codeProgress));

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
      <div style={{ fontFamily: 'monospace', color: '#10B981', fontSize: 18, letterSpacing: '3px', marginBottom: 12 }}>
        CHAPTER 04 — PURE PYTHON & NUMPY IMPLEMENTATION
      </div>
      <h2 style={{ fontSize: 48, fontWeight: 800, margin: '0 0 30px 0', color: '#FFFFFF' }}>
        04-implementation.py
      </h2>

      <div
        style={{
          width: '100%',
          maxWidth: 1300,
          backgroundColor: '#121620',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 20,
          padding: 36,
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
        }}
      >
        <div style={{ display: 'flex', gap: 10, marginBottom: 24, alignItems: 'center' }}>
          <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#EF4444' }} />
          <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#F59E0B' }} />
          <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#10B981' }} />
          <span style={{ fontSize: 16, fontFamily: 'monospace', color: '#64748B', marginLeft: 12 }}>
            modules/001-perceptron/04-implementation.py (Zero Framework Dependencies)
          </span>
        </div>

        <pre
          style={{
            fontFamily: 'JetBrains Mono, Fira Code, monospace',
            fontSize: 24,
            lineHeight: 1.6,
            color: '#A5B4FC',
            margin: 0,
            whiteSpace: 'pre-wrap',
          }}
        >
          {displayedCode}
          <span style={{ opacity: frame % 30 < 15 ? 1 : 0, color: '#38BDF8' }}>█</span>
        </pre>
      </div>
    </div>
  );
};
