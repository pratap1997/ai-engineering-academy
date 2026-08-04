import React, { useState } from 'react';
import { X, Play, Code, CheckCircle2 } from 'lucide-react';

interface YouTubePlayerModalProps {
  moduleTitle: string;
  youtubeId?: string;
  onClose: () => void;
  onOpenCode: () => void;
}

export const YouTubePlayerModal: React.FC<YouTubePlayerModalProps> = ({
  moduleTitle,
  youtubeId = 'dQw4w9WgXcQ', // Default video ID placeholder
  onClose,
  onOpenCode,
}) => {
  const [splitScreen, setSplitScreen] = useState(false);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 md:p-8 animate-fade-in">
      <div
        className={`bg-[#090C10] border border-white/10 rounded-2xl overflow-hidden shadow-2xl transition-all duration-300 flex flex-col ${
          splitScreen ? 'w-full h-full max-w-7xl max-h-[90vh]' : 'w-full max-w-4xl'
        }`}
      >
        {/* Header Bar */}
        <div className="px-6 py-4 bg-[#121620] border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400">
              <svg className="w-5 h-5 fill-current" viewBox="0 0 24 24">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100 font-heading">
                Masterclass Video: {moduleTitle}
              </h3>
              <p className="text-xs text-slate-400 font-mono">
                Official YouTube AI Engineering Academy Video
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setSplitScreen(!splitScreen)}
              className={`btn-subtle text-xs py-2 px-4 flex items-center gap-2 ${
                splitScreen ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300' : ''
              }`}
            >
              <Code className="w-4 h-4 text-indigo-400" />
              <span>{splitScreen ? 'Full Video Mode' : 'Split-Screen Watch & Code Mode'}</span>
            </button>

            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/5"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Video Player Container */}
        <div className={`grid gap-4 p-4 flex-1 ${splitScreen ? 'grid-cols-1 lg:grid-cols-12' : 'grid-cols-1'}`}>
          {/* YouTube IFrame Embed */}
          <div className={`${splitScreen ? 'lg:col-span-7' : 'w-full'} aspect-video bg-black rounded-xl overflow-hidden border border-white/5`}>
            <iframe
              className="w-full h-full"
              src={`https://www.youtube.com/embed/${youtubeId}?autoplay=1&enablejsapi=1`}
              title={`YouTube video for ${moduleTitle}`}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>

          {/* Split-Screen Code & Timestamps Sidebar */}
          {splitScreen && (
            <div className="lg:col-span-5 bg-[#121620] border border-white/10 rounded-xl p-5 space-y-4 overflow-y-auto font-mono text-xs">
              <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <span className="text-indigo-400 font-bold uppercase">Interactive Timestamps</span>
                <button onClick={onOpenCode} className="btn-indigo text-xs py-1.5 px-3">
                  Open Python Playground →
                </button>
              </div>

              <div className="space-y-2">
                <div className="p-3 bg-[#090C10] rounded-lg border border-white/5 hover:border-indigo-500/40 cursor-pointer flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Play className="w-3.5 h-3.5 text-indigo-400" />
                    <span>00:00 - Introduction & Motivation</span>
                  </div>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                </div>

                <div className="p-3 bg-[#090C10] rounded-lg border border-white/5 hover:border-indigo-500/40 cursor-pointer flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Play className="w-3.5 h-3.5 text-indigo-400" />
                    <span>01:15 - Intuitive Mental Model</span>
                  </div>
                </div>

                <div className="p-3 bg-[#090C10] rounded-lg border border-white/5 hover:border-indigo-500/40 cursor-pointer flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Play className="w-3.5 h-3.5 text-indigo-400" />
                    <span>03:45 - Formal Math Derivation</span>
                  </div>
                </div>

                <div className="p-3 bg-[#090C10] rounded-lg border border-white/5 hover:border-indigo-500/40 cursor-pointer flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Play className="w-3.5 h-3.5 text-indigo-400" />
                    <span>06:30 - Pure Python Code</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
