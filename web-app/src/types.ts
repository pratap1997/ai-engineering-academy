export type PhaseId = 'phase1' | 'phase2' | 'phase3' | 'phase4';

export interface ModuleData {
  id: string; // e.g. "001", "026", "050"
  title: string;
  subtitle: string;
  phaseId: PhaseId;
  phaseName: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'capstone';
  estimatedMinutes: number;
  testCount: number;
  prerequisites: string[];
  topics: string[];
  overview: string;
  mentalModelSummary: string;
  mathHighlight: string;
  codeSnippet: string;
  engineeringChallengeTitle: string;
  engineeringChallengeGoal: string;
  diagramType?: 'architecture' | 'workflow' | 'sequence' | 'dataflow' | 'lifecycle';
  sources: { title: string; authors: string; year: number }[];
}

export interface PhaseInfo {
  id: PhaseId;
  name: string;
  range: string;
  description: string;
  badgeClass: string;
  accentColor: string;
}
