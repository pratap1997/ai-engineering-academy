import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig, Audio, staticFile } from 'remotion';

interface MasterclassVideoProps {
  title: string;
  subtitle: string;
  codeSnippet: string;
  audioSrc: string;
}

export const MasterclassVideo: React.FC<MasterclassVideoProps> = ({
  title,
  subtitle,
  codeSnippet,
  audioSrc,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Animations driven by Remotion frame & spring physics
  const titleSpring = spring({
    frame,
    fps,
    config: { damping: 12, mass: 0.5 },
  });

  const subtitleOpacity = interpolate(frame, [20, 50], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const codeProgress = interpolate(frame, [60, 240], [0, codeSnippet.length], {
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
      {/* Audio Track Sync (NVIDIA Riva TTS Audio) */}
      <Audio src={staticFile(audioSrc)} />

      {/* Animated Glowing Background Grid */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundImage: 'radial-gradient(circle at 50% 30%, rgba(99, 102, 241, 0.15), transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      {/* Header Tag */}
      <div
        style={{
          fontFamily: 'monospace',
          fontSize: 20,
          letterSpacing: '4px',
          color: '#818CF8',
          textTransform: 'uppercase',
          marginBottom: 16,
          transform: `scale(${titleSpring})`,
        }}
      >
        ✦ MASTERCLASS ACADEMY ✦
      </div>

      {/* Main Title */}
      <h1
        style={{
          fontSize: 64,
          fontWeight: 800,
          textAlign: 'center',
          margin: 0,
          background: 'linear-gradient(135deg, #FFFFFF 0%, #A5B4FC 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          transform: `scale(${titleSpring})`,
        }}
      >
        {title}
      </h1>

      {/* Subtitle */}
      <p
        style={{
          fontSize: 28,
          color: '#94A3B8',
          marginTop: 16,
          marginBottom: 40,
          textAlign: 'center',
          opacity: subtitleOpacity,
          maxWidth: 1200,
        }}
      >
        {subtitle}
      </p>

      {/* Code Window */}
      <div
        style={{
          width: '100%',
          maxWidth: 1100,
          backgroundColor: '#121620',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 20,
          padding: 30,
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
          opacity: subtitleOpacity,
        }}
      >
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#EF4444' }} />
          <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#F59E0B' }} />
          <div style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#10B981' }} />
          <span style={{ fontSize: 14, fontFamily: 'monospace', color: '#64748B', marginLeft: 10 }}>
            04-implementation.py — Pure NumPy Architecture
          </span>
        </div>

        <pre
          style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: 22,
            lineHeight: 1.6,
            color: '#A5B4FC',
            margin: 0,
            whiteSpace: 'pre-wrap',
          }}
        >
          {displayedCode}
          <span style={{ opacity: frame % 30 < 15 ? 1 : 0 }}>|</span>
        </pre>
      </div>

      {/* Dynamic Audio Waveform Equalizer */}
      <div
        style={{
          position: 'absolute',
          bottom: 40,
          display: 'flex',
          gap: 6,
          alignItems: 'flex-end',
          height: 30,
        }}
      >
        {Array.from({ length: 24 }).map((_, i) => {
          const h = Math.abs(Math.sin((frame + i * 8) * 0.15)) * 25 + 5;
          return (
            <div
              key={i}
              style={{
                width: 6,
                height: `${h}px`,
                backgroundColor: '#6366F1',
                borderRadius: 3,
              }}
            />
          );
        })}
      </div>

    </div>
  );
};
