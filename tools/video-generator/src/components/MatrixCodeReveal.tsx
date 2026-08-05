import React from 'react';
import { interpolate, useCurrentFrame } from 'remotion';

interface MatrixCodeRevealProps {
  codeSnippet: string;
}

/**
 * MatrixCodeReveal — Matrix digital rain that "solidifies" into clean Python code.
 * Phase 1 (0-60f): Random green matrix characters falling
 * Phase 2 (60-150f): Characters lock into final code positions one column at a time
 * Phase 3 (150f+): Clean final code glowing on screen
 */
export const MatrixCodeReveal: React.FC<MatrixCodeRevealProps> = ({ codeSnippet }) => {
  const frame = useCurrentFrame();

  const CHARS = 'アイウエオカキクケコABCDEF0123456789!@#$%^&*<>{}[]|01';
  const codeLines = codeSnippet.split('\n');
  const maxCols = Math.max(...codeLines.map(l => l.length));

  // How many columns are "resolved" to final code vs still raining
  const resolvedCols = interpolate(frame, [60, 200], [0, maxCols], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Overall opacity for code block
  const blockOpacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const getChar = (col: number, row: number): { char: string; color: string; glow: boolean } => {
    if (col < resolvedCols - 2) {
      // Column is resolved - show final code character
      const line = codeLines[row] || '';
      const char = line[col] || ' ';
      return { char, color: '#A5B4FC', glow: false };
    } else if (col < resolvedCols) {
      // Column is being resolved right now - bright white flash
      const line = codeLines[row] || '';
      const char = line[col] || ' ';
      return { char, color: '#FFFFFF', glow: true };
    } else {
      // Column still raining - random matrix char
      const seed = (frame * 7 + col * 13 + row * 31) % CHARS.length;
      const char = frame % 3 === 0 ? CHARS[Math.floor(seed)] : (CHARS[(seed + 7) % CHARS.length]);
      const brightness = row < (frame * 0.8) % codeLines.length ? 1 : 0.3;
      return {
        char,
        color: `rgba(0, ${Math.floor(180 + brightness * 75)}, ${Math.floor(60 + brightness * 20)}, ${brightness})`,
        glow: brightness > 0.8,
      };
    }
  };

  return (
    <div
      style={{
        width: 1920,
        height: 1080,
        backgroundColor: '#020804',
        fontFamily: 'JetBrains Mono, Fira Code, monospace',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden',
        opacity: blockOpacity,
      }}
    >
      {/* Background matrix rain columns */}
      <div style={{ position: 'absolute', inset: 0, display: 'flex', gap: 0 }}>
        {Array.from({ length: 48 }).map((_, col) => (
          <div
            key={col}
            style={{
              display: 'flex',
              flexDirection: 'column',
              width: '2.08%',
              fontSize: 14,
              fontFamily: 'monospace',
              color: '#003A12',
              overflow: 'hidden',
              lineHeight: 1.4,
            }}
          >
            {Array.from({ length: 32 }).map((_, row) => {
              const seed = (frame * 3 + col * 17 + row * 7) % CHARS.length;
              return (
                <span key={row} style={{ opacity: ((frame + row * 3 + col * 7) % 20) / 20 }}>
                  {CHARS[Math.floor(seed)]}
                </span>
              );
            })}
          </div>
        ))}
      </div>

      {/* Chapter header */}
      <div
        style={{
          position: 'absolute',
          top: 60,
          fontFamily: 'monospace',
          color: '#00FF41',
          fontSize: 18,
          letterSpacing: '4px',
          textShadow: '0 0 20px #00FF41',
          opacity: interpolate(frame, [0, 30], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }),
        }}
      >
        &gt;&gt; CHAPTER 04: PURE PYTHON IMPLEMENTATION — DECRYPTING CODE...
      </div>

      {/* Code block */}
      <div
        style={{
          position: 'relative',
          backgroundColor: 'rgba(2, 8, 4, 0.85)',
          border: '1px solid rgba(0, 255, 65, 0.3)',
          borderRadius: 16,
          padding: '32px 48px',
          boxShadow: '0 0 60px rgba(0, 255, 65, 0.15), inset 0 0 40px rgba(0, 0, 0, 0.8)',
          maxWidth: 1400,
          width: '100%',
        }}
      >
        {/* Window bar */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, alignItems: 'center' }}>
          <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#FF5F57' }} />
          <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#FEBC2E' }} />
          <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#28C840' }} />
          <span style={{ fontSize: 14, fontFamily: 'monospace', color: '#00FF41', marginLeft: 12 }}>
            04-implementation.py — AI Engineering Academy — Module 001
          </span>
        </div>

        {/* Matrix-to-code lines */}
        <pre style={{ margin: 0, fontSize: 22, lineHeight: 1.7, minHeight: 320 }}>
          {codeLines.map((line, row) => (
            <div key={row}>
              {/* Line number */}
              <span style={{ color: '#334155', marginRight: 16, userSelect: 'none', fontSize: 16 }}>
                {String(row + 1).padStart(2, '0')}
              </span>
              {/* Each character */}
              {Array.from({ length: Math.max(line.length, 1) }).map((_, col) => {
                const { char, color, glow } = getChar(col, row);
                return (
                  <span
                    key={col}
                    style={{
                      color,
                      textShadow: glow ? `0 0 12px ${color}` : 'none',
                      transition: 'color 0.1s',
                    }}
                  >
                    {char}
                  </span>
                );
              })}
            </div>
          ))}
          {/* Cursor */}
          <span
            style={{
              display: 'inline-block',
              width: 14,
              height: 26,
              backgroundColor: '#00FF41',
              opacity: frame % 30 < 15 ? 1 : 0,
              boxShadow: '0 0 10px #00FF41',
              verticalAlign: 'middle',
            }}
          />
        </pre>
      </div>
    </div>
  );
};
