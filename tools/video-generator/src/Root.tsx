import React from 'react';
import { Composition } from 'remotion';
import { MasterclassVideo } from './MasterclassVideo';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MasterclassVideo"
        component={MasterclassVideo}
        durationInFrames={360} // 6 seconds at 60fps
        fps={60}
        width={1920}
        height={1080}
        defaultProps={{
          title: "AI ENGINEERING ACADEMY",
          subtitle: "Masterclass: Building Autonomous AI Agents From First Principles",
          codeSnippet: "class Perceptron:\n    def __init__(self, d_in):\n        self.w = np.zeros(d_in)\n        self.b = 0.0\n    def forward(self, x):\n        return np.step(np.dot(self.w, x) + self.b)",
          audioSrc: "audio.wav",
        }}
      />
    </>
  );
};
