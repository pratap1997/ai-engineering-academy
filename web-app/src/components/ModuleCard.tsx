import React from 'react';
import type { ModuleData } from '../types';
import { Clock, ArrowRight, ShieldCheck } from 'lucide-react';

interface ModuleCardProps {
  module: ModuleData;
  onOpenDetail: (m: ModuleData) => void;
}

export const ModuleCard: React.FC<ModuleCardProps> = ({ module, onOpenDetail }) => {
  return (
    <div
      onClick={() => onOpenDetail(module)}
      className="panel-card p-6 cursor-pointer transition-all duration-200 group flex flex-col justify-between hover:shadow-md"
    >
      <div className="space-y-4">
        
        {/* Header Badges (AI Engineering Skool Style) */}
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="font-bold text-[var(--accent-primary)] bg-[var(--bg-elevated)] px-2.5 py-1 rounded border border-[var(--border-subtle)]">
            MODULE #{module.id}
          </span>
          <div className="flex items-center gap-1.5">
            <span className="badge-pill badge-emerald flex items-center gap-1">
              <ShieldCheck className="w-3 h-3 text-[var(--accent-primary)]" /> INDEXED
            </span>
            <span className="badge-pill">
              VERIFIED 16/16
            </span>
          </div>
        </div>

        {/* Title & Subtitle in Editorial Serif */}
        <div>
          <h4 className="font-heading text-lg font-bold text-[var(--text-primary)] group-hover:text-[var(--accent-primary)] transition-colors leading-snug">
            {module.title}
          </h4>
          <p className="text-xs text-[var(--text-secondary)] line-clamp-2 mt-2 leading-relaxed font-sans">
            {module.subtitle}
          </p>
        </div>

        {/* Metric Grid (OpenAgentSkill Decision Snapshot Style) */}
        <div className="grid grid-cols-3 gap-2 pt-2 text-[11px] font-mono border-t border-[var(--border-subtle)]">
          <div className="bg-[var(--bg-elevated)] p-2 rounded border border-[var(--border-subtle)]">
            <div className="text-[var(--text-muted)] text-[9px] uppercase">QUALITY</div>
            <div className="font-bold text-[var(--text-primary)] mt-0.5">100 / 100</div>
          </div>
          <div className="bg-[var(--bg-elevated)] p-2 rounded border border-[var(--border-subtle)]">
            <div className="text-[var(--text-muted)] text-[9px] uppercase">TRUST</div>
            <div className="font-bold text-[var(--accent-primary)] mt-0.5">PASSED ✓</div>
          </div>
          <div className="bg-[var(--bg-elevated)] p-2 rounded border border-[var(--border-subtle)]">
            <div className="text-[var(--text-muted)] text-[9px] uppercase">ESTIMATE</div>
            <div className="font-bold text-[var(--text-primary)] mt-0.5">{module.estimatedMinutes} mins</div>
          </div>
        </div>

      </div>

      {/* Footer Info & Action */}
      <div className="pt-4 mt-4 border-t border-[var(--border-subtle)] flex items-center justify-between text-xs">
        <div className="flex items-center gap-1.5 text-[var(--text-muted)] font-mono text-[11px]">
          <Clock className="w-3.5 h-3.5 text-[var(--text-muted)]" />
          <span>9 Artifacts Standard</span>
        </div>

        <button className="text-[var(--accent-primary)] font-mono font-semibold group-hover:text-[var(--accent-primary-hover)] flex items-center gap-1 transition-colors">
          Open Module <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
        </button>
      </div>

    </div>
  );
};
