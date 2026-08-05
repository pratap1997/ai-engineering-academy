import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

interface IntroSceneProps {
  title: string;
  subtitle: string;
}

export const IntroScene: React.FC<IntroSceneProps> = ({ title, subtitle }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleSpring = spring({ frame, fps, config: { damping: 12 } });
  const badgeOpacity = interpolate(frame, [0, 20], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });

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
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Dynamic Background Glow */}
      <div
        style={{
          position: 'absolute',
          width: 800,
          height: 800,
          borderRadius: '50%',
          backgroundColor: '#10B981',
          filter: 'blur(180px)',
          opacity: 0.12,
          transform: `scale(${1 + Math.sin(frame * 0.05) * 0.1})`,
        }}
      />

      {/* Brand Header */}
      <div
        style={{
          opacity: badgeOpacity,
          fontFamily: 'monospace',
          color: '#10B981',
          fontSize: 20,
          letterSpacing: '4px',
          textTransform: 'uppercase',
          marginBottom: 24,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          padding: '8px 20px',
          borderRadius: 30,
          border: '1px solid rgba(16, 185, 129, 0.3)',
        }}
      >
        <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#10B981' }} />
        () AI Engineering Skool — Module 001
      </div>

      {/* Main Title */}
      <h1
        style={{
          fontSize: 72,
          fontWeight: 800,
          textAlign: 'center',
          maxWidth: 1400,
          margin: 0,
          lineHeight: 1.1,
          letterSpacing: '-0.02em',
          transform: `scale(${titleSpring})`,
          background: 'linear-gradient(180deg, #FFFFFF 0%, #A5B4FC 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        {title}
      </h1>

      {/* Subtitle */}
      <p
        style={{
          fontSize: 28,
          color: '#94A3B8',
          textAlign: 'center',
          maxWidth: 1000,
          marginTop: 24,
          lineHeight: 1.5,
          opacity: interpolate(frame, [20, 40], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }),
        }}
      >
        {subtitle}
      </p>
    </div>
  );
};
