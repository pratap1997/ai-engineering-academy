// Vite raw glob imports for legacy modules (001-025) and modern modules (026-050)
const legacyFiles = import.meta.glob('../../../ai-engineering-academy/modules/*/*.{md,py}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const modernFiles = import.meta.glob('../../../modules/*/*.{md,py}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

// Merged file map
const allFiles: Record<string, string> = { ...legacyFiles, ...modernFiles };

export const artifactFileNames: Record<string, string> = {
  overview: '01-overview.md',
  mentalModel: '02-mental-model.md',
  math: '03-mathematics.md',
  code: '04-implementation.py',
  experiments: '05-experiments.py',
  applications: '06-real-applications.md',
  challenge: '07-engineering-challenge.md',
  assessment: '08-assessment.md',
  references: '09-references.md',
};

/**
 * Retrieves the raw content of a specific artifact file for a given module ID (1 to 50).
 */
export function getModuleArtifactContent(moduleIdStr: string, artifactKey: string): string | null {
  const num = parseInt(moduleIdStr, 10);
  const formattedId = num.toString().padStart(3, '0');
  const targetFileName = artifactFileNames[artifactKey];

  if (!targetFileName) return null;

  // Search through all loaded file paths for matching module folder and artifact file
  for (const filePath of Object.keys(allFiles)) {
    const isLegacyPath = filePath.includes(`/modules/${formattedId}-`);
    const isModernPath = filePath.includes(`/modules/${formattedId}-`);
    const isTargetFile = filePath.endsWith(targetFileName);

    if ((isLegacyPath || isModernPath) && isTargetFile) {
      return allFiles[filePath] || null;
    }
  }

  return null;
}
