import React from 'react';
import { Sequence, Audio, staticFile } from 'remotion';
import { IntroScene } from './components/IntroScene';
import { MentalModelScene } from './components/MentalModelScene';
import { MathematicsScene } from './components/MathematicsScene';
import { MatrixCodeReveal } from './components/MatrixCodeReveal';
import { WeightLearningAnimation } from './components/WeightLearningAnimation';
import { XorLimitScene } from './components/XorLimitScene';
import { GlitchTransition } from './components/GlitchTransition';
import { SubtitlesOverlay, SubtitleWord } from './components/SubtitlesOverlay';

interface MasterclassVideoProps {
  title: string;
  subtitle: string;
  codeSnippet: string;
  audioSrc: string;
  durationInFrames?: number;
  subtitleChunks?: SubtitleWord[];
}

/**
 * MasterclassVideo — Full 60fps cinematic AI Engineering Skool masterclass.
 * Features:
 *   - Frame-accurate dynamic TTS burned-in subtitles (<SubtitlesOverlay />)
 *   - VHS Glitch Transitions between chapters
 *   - Matrix rain code reveal
 *   - Live weight learning SVG scatter plot
 *   - 3Blue1Brown-style vector math cards
 */
export const MasterclassVideo: React.FC<MasterclassVideoProps> = ({
  title,
  subtitle,
  codeSnippet,
  audioSrc,
  subtitleChunks = [],
}) => {
  return (
    <div style={{ width: 1920, height: 1080, backgroundColor: '#090C10', position: 'relative' }}>
      {/* NVIDIA Riva TTS Audio — synchronized to full video */}
      <Audio src={staticFile(audioSrc)} />

      {/* Burned-in Subtitles Overlay (dynamically synchronized to Riva TTS word timestamps) */}
      <SubtitlesOverlay subtitleChunks={subtitleChunks} />

      {/* ── Chapter 1: Introduction ── */}
      <Sequence from={0} durationInFrames={600}>
        <IntroScene title={title} subtitle={subtitle} />
      </Sequence>

      {/* Glitch 1 */}
      <Sequence from={598} durationInFrames={32}>
        <GlitchTransition />
      </Sequence>

      {/* ── Chapter 2: Mental Model — Geometric Hyperplane ── */}
      <Sequence from={630} durationInFrames={650}>
        <MentalModelScene />
      </Sequence>

      {/* Glitch 2 */}
      <Sequence from={1278} durationInFrames={32}>
        <GlitchTransition />
      </Sequence>

      {/* ── Chapter 3: Mathematics — Equation Cards ── */}
      <Sequence from={1310} durationInFrames={750}>
        <MathematicsScene />
      </Sequence>

      {/* Glitch 3 */}
      <Sequence from={2058} durationInFrames={32}>
        <GlitchTransition />
      </Sequence>

      {/* ── Chapter 4: Matrix Code Reveal ── */}
      <Sequence from={2090} durationInFrames={730}>
        <MatrixCodeReveal codeSnippet={codeSnippet} />
      </Sequence>

      {/* Glitch 4 */}
      <Sequence from={2818} durationInFrames={32}>
        <GlitchTransition />
      </Sequence>

      {/* ── Chapter 5: Live Weight Learning Animation ── */}
      <Sequence from={2850} durationInFrames={600}>
        <WeightLearningAnimation />
      </Sequence>

      {/* Glitch 5 */}
      <Sequence from={3448} durationInFrames={32}>
        <GlitchTransition />
      </Sequence>

      {/* ── Chapter 6: XOR Limitation & Minsky Proof ── */}
      <Sequence from={3480} durationInFrames={370}>
        <XorLimitScene />
      </Sequence>
    </div>
  );
};
