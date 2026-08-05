import React from 'react';
import { interpolate, useCurrentFrame } from 'remotion';

export interface SubtitleChunk {
  text: string;
  startFrame: number;
  endFrame: number;
}

interface SubtitlesOverlayProps {
  chunks?: SubtitleChunk[];
}

/**
 * SubtitlesOverlay — Frame-accurate burned-in subtitles with word-by-word highlighting.
 * Uses exact audio chunk timing calculated from Riva TTS PCM byte length.
 */
export const SubtitlesOverlay: React.FC<SubtitlesOverlayProps> = ({ chunks = [] }) => {
  const frame = useCurrentFrame();

  if (!chunks || chunks.length === 0) return null;

  const currentSub = chunks.find(s => frame >= s.startFrame && frame <= s.endFrame);

  if (!currentSub) return null;

  const words = currentSub.text.split(' ');
  const totalSubFrames = currentSub.endFrame - currentSub.startFrame;
  const progress = (frame - currentSub.startFrame) / Math.max(1, totalSubFrames);
  const activeWordIdx = Math.min(Math.floor(progress * words.length), words.length - 1);

  // Fade in / out opacity for smooth reading
  const opacity = interpolate(
    frame,
    [currentSub.startFrame, currentSub.startFrame + 8, currentSub.endFrame - 8, currentSub.endFrame],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 48,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 50,
        opacity,
        display: 'flex',
        justifyContent: 'center',
        maxWidth: 1400,
        width: '90%',
      }}
    >
      <div
        style={{
          backgroundColor: 'rgba(9, 12, 16, 0.92)',
          border: '1px solid rgba(16, 185, 129, 0.45)',
          borderRadius: 16,
          padding: '14px 32px',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.85), 0 0 24px rgba(16, 185, 129, 0.2)',
          backdropFilter: 'blur(8px)',
          textAlign: 'center',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: '8px 12px',
        }}
      >
        {words.map((word, i) => {
          const isActive = i === activeWordIdx;
          const isPast = i < activeWordIdx;
          return (
            <span
              key={i}
              style={{
                fontFamily: 'Inter, system-ui, sans-serif',
                fontSize: isActive ? 28 : 26,
                fontWeight: isActive ? 800 : 600,
                color: isActive ? '#10B981' : isPast ? '#FFFFFF' : '#94A3B8',
                textShadow: isActive ? '0 0 16px rgba(16, 185, 129, 0.9)' : 'none',
                transform: isActive ? 'scale(1.08)' : 'scale(1)',
                transition: 'all 0.08s ease',
              }}
            >
              {word}
            </span>
          );
        })}
      </div>
    </div>
  );
};
