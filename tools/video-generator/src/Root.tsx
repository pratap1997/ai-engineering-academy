import React from 'react';
import { Composition } from 'remotion';
import { MasterclassVideo } from './MasterclassVideo';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="MasterclassVideo"
        component={MasterclassVideo}
        durationInFrames={3960} // Default 66 seconds @ 60fps
        fps={60}
        width={1920}
        height={1080}
        calculateMetadata={({ props }: { props: any }) => {
          return {
            durationInFrames: props.durationInFrames || 3960,
          };
        }}
        defaultProps={{
          title: "MODULE 001: THE PERCEPTRON",
          subtitle: "Complete Masterclass: Mathematical Derivation, NumPy Code & XOR Limits",
          codeSnippet: "class Perceptron:\n    def __init__(self, d_in, lr=0.01):\n        self.w = np.zeros(d_in)\n        self.b = 0.0\n        self.lr = lr\n\n    def predict(self, x):\n        z = np.dot(x, self.w) + self.b\n        return np.where(z >= 0, 1, 0)\n\n    def fit(self, X, y, epochs=100):\n        for _ in range(epochs):\n            for xi, target in zip(X, y):\n                update = self.lr * (target - self.predict(xi))\n                self.w += update * xi\n                self.b += update",
          audioSrc: "audio.wav",
          durationInFrames: 3960,
        }}
      />
    </>
  );
};
