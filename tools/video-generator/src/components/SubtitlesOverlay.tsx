import React from 'react';
import { interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion';

export interface SubtitleWord {
  text: string;
  startFrame: number;
  endFrame: number;
}

interface SubtitlesOverlayProps {
  subtitleChunks?: SubtitleWord[];
}

/**
 * SubtitlesOverlay — 60FPS Kinetic Word-by-Word Subtitle Overlay.
 * Renders glowing active spoken words aligned to NVIDIA Riva / Whisper audio timestamps.
 */
export const SubtitlesOverlay: React.FC<SubtitlesOverlayProps> = ({ subtitleChunks = [] }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  if (!subtitleChunks || subtitleChunks.length === 0) return null;

  // Find currently spoken word index
  const activeWordIdx = subtitleChunks.findIndex(
    (w) => frame >= w.startFrame && frame <= w.endFrame
  );

  if (activeWordIdx === -1) {
    // If between sentences within 10 frames, show last sentence snippet
    const prevWordIdx = subtitleChunks.findLastIndex((w) => frame >= w.endFrame);
    if (prevWordIdx === -1 || frame - subtitleChunks[prevWordIdx].endFrame > 15) {
      return null;
    }
  }

  const currentIdx = activeWordIdx >= 0 ? activeWordIdx : subtitleChunks.findLastIndex((w) => frame >= w.endFrame);
  
  // Show a rolling window of 6 words centered around current spoken word
  const windowSize = 6;
  const startIdx = Math.max(0, currentIdx - 2);
  const visibleWords = subtitleChunks.slice(startIdx, startIdx + windowSize);

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 54,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 100,
        display: 'flex',
        justifyContent: 'center',
        maxWidth: 1200,
        width: '85%',
      }}
    >
      <div
        style={{
          backgroundColor: 'rgba(9, 12, 16, 0.94)',
          border: '1.5px solid rgba(16, 185, 129, 0.5)',
          borderRadius: 20,
          padding: '16px 36px',
          boxShadow: '0 12px 36px rgba(0, 0, 0, 0.9), 0 0 30px rgba(16, 185, 129, 0.25)',
          backdropFilter: 'blur(12px)',
          textAlign: 'center',
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '10px 14px',
        }}
      >
        {visibleWords.map((wordObj, i) => {
          const isActive = frame >= wordObj.startFrame && frame <= wordObj.endFrame;
          const isPast = frame > wordObj.endFrame;

          // Pop animation spring for active spoken word
          const wordFrameOffset = Math.max(0, frame - wordObj.startFrame);
          const popScale = isActive
            ? spring({
                frame: wordFrameOffset,
                fps,
                config: { damping: 14, stiffness: 220 },
              })
            : 1;

          return (
            <span
              key={`${wordObj.startFrame}-${i}`}
              style={{
                fontFamily: 'Inter, system-ui, sans-serif',
                fontSize: isActive ? 32 : 28,
                fontWeight: isActive ? 900 : 600,
                color: isActive ? '#10B981' : isPast ? '#F3F4F6' : '#6B7280',
                textShadow: isActive
                  ? '0 0 20px rgba(16, 185, 129, 0.95), 0 0 10px rgba(16, 185, 129, 0.6)'
                  : 'none',
                transform: `scale(${isActive ? popScale * 1.12 : 1})`,
                transition: 'color 0.05s ease, transform 0.05s ease',
                letterSpacing: '-0.01em',
              }}
            >
              {wordObj.text}
            </span>
          );
        })}
      </div>
    </div>
  );
};
