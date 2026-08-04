import { getModuleArtifactContent } from './moduleContentLoader';

export interface AIMentorResponse {
  reply: string;
  isHintMode?: boolean;
}

/**
 * Generates an intelligent, module-contextual response using curriculum artifact knowledge.
 */
export function generateAIMentorResponse(
  moduleIdStr: string,
  moduleTitle: string,
  artifactKey: string,
  userPrompt: string
): AIMentorResponse {
  const query = userPrompt.toLowerCase().trim();
  const overviewContent = getModuleArtifactContent(moduleIdStr, 'overview') || '';

  // Challenge Mode Protection (07 Engineering Challenge)
  if (artifactKey === 'challenge') {
    return {
      reply: `[SOCRATIC HINT MODE] You are currently working on the Module ${moduleIdStr} Engineering Challenge (${moduleTitle}). To help you derive the solution yourself: Think about the input dimensions, the loss formulation, and how gradient steps update parameters. Try modifying the implementation code step-by-step!`,
      isHintMode: true,
    };
  }

  // Specific domain queries
  if (query.includes('xor') || query.includes('linear')) {
    return {
      reply: `Great question! In Module ${moduleIdStr} (${moduleTitle}), a single linear decision boundary (hyperplane \(w^T x + b = 0\)) can only separate linearly separable classes. XOR outputs (1,0) and (0,1) as 1, while (0,0) and (1,1) as 0, which requires at least a 2-layer Multilayer Perceptron (MLP) to form a non-linear convex boundary!`,
    };
  }

  if (query.includes('bias') || query.includes('weight')) {
    return {
      reply: `In neural architectures, weights \(w\) determine the orientation/slope of the hyper-plane, while the bias \(b\) shifts the decision boundary independently of the input vector \(x\). Without a bias term (\(b=0\)), the boundary is strictly anchored to pass through the origin \((0,0)\).`,
    };
  }

  if (query.includes('loss') || query.includes('gradient')) {
    return {
      reply: `In optimization, loss functions quantify prediction error \(e = y - \hat{y}\). Parameter updates adjust weights along the negative gradient vector \(-\nabla L(w)\) scaled by learning rate \(\eta\).`,
    };
  }

  // General contextual fallback using module overview metadata
  return {
    reply: `In Module ${moduleIdStr} (${moduleTitle}) [Section: ${artifactKey.toUpperCase()}]: The core principle is building an intuitive mental model, deriving the mathematics formally, and verifying it with zero-framework Python code. ${
      overviewContent ? 'Read the Overview for historical background and motivation!' : ''
    }`,
  };
}
