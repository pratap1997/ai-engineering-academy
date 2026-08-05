import React from 'react';
import { interpolate, useCurrentFrame } from 'remotion';

/**
 * GlitchTransition — VHS-style glitch effect between chapters.
 * Horizontal scanlines + RGB channel split + static noise.
 * Duration: ~30 frames (0.5 seconds at 60fps)
 */
export const GlitchTransition: React.FC = () => {
  const frame = useCurrentFrame();

  // Peak glitch at frame 15, fade in/out
  const glitchIntensity = interpolate(frame, [0, 8, 15, 22, 30], [0, 1, 1, 1, 0], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const rgbOffsetX = (Math.sin(frame * 3.7) * 18 + Math.cos(frame * 7.3) * 12) * glitchIntensity;
  const rgbOffsetY = (Math.cos(frame * 5.1) * 8) * glitchIntensity;

  // Generate scanlines
  const scanlines = Array.from({ length: 16 }, (_, i) => ({
    y: Math.random() * 1080,
    height: Math.random() * 12 + 2,
    opacity: Math.random() * 0.6 + 0.2,
    offset: (Math.random() - 0.5) * 80 * glitchIntensity,
  }));

  // Pseudo-random based on frame (deterministic)
  const seed = frame * 137.508;
  const noiseLines = Array.from({ length: 8 }, (_, i) => {
    const y = ((seed * (i + 1) * 0.618) % 1) * 1080;
    const offset = ((seed * (i + 1) * 1.414) % 1 - 0.5) * 100 * glitchIntensity;
    const h = 4 + ((seed * (i + 1) * 0.333) % 1) * 20;
    return { y, offset, h };
  });

  return (
    <div
      style={{
        width: 1920,
        height: 1080,
        backgroundColor: '#000000',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Red channel layer */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: '#FF000018',
          transform: `translate(${rgbOffsetX}px, ${-rgbOffsetY}px)`,
          mixBlendMode: 'screen',
        }}
      />

      {/* Blue channel layer */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundColor: '#0000FF18',
          transform: `translate(${-rgbOffsetX}px, ${rgbOffsetY}px)`,
          mixBlendMode: 'screen',
        }}
      />

      {/* Horizontal glitch scan bars */}
      {noiseLines.map((line, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            top: line.y,
            left: 0,
            right: 0,
            height: line.h,
            backgroundColor: i % 3 === 0 ? '#FFFFFF' : i % 3 === 1 ? '#00FFFF' : '#FF00FF',
            opacity: 0.15 * glitchIntensity,
            transform: `translateX(${line.offset}px)`,
          }}
        />
      ))}

      {/* Scanlines overlay (always-on CRT effect) */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.4) 2px, rgba(0,0,0,0.4) 4px)',
          pointerEvents: 'none',
          opacity: 0.5 * glitchIntensity,
        }}
      />

      {/* Center text glitch */}
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: `translate(calc(-50% + ${rgbOffsetX * 0.5}px), -50%)`,
          fontFamily: 'monospace',
          fontSize: 32,
          color: '#00FF41',
          letterSpacing: '8px',
          opacity: glitchIntensity * 0.8,
          textShadow: `${rgbOffsetX}px 0 #FF0000, ${-rgbOffsetX}px 0 #0000FF`,
        }}
      >
        ▓▓▓ LOADING NEXT MODULE ▓▓▓
      </div>
    </div>
  );
};
