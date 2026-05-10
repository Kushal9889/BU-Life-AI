'use client';

import { useState, useEffect } from 'react';
import { sendQuery, streamQuery, QueryResponse } from '@/lib/api';

const EXAMPLE_PROMPTS = [
  'I have 25 mins before class near CDS, what should I do?',
  'Find me a quiet study spot near CAS with outlets',
  'How do I get resume help at BU?',
  'What events should I attend this week for AI?',
  'Where can I print something near GSU?',
  'What is OPT and how do I apply as an F-1 student?',
];

interface QueryBarProps {
  onResult: (response: QueryResponse) => void;
  onStreamToken: (token: string) => void;
  onStreamStart: () => void;
  onStreamEnd: () => void;
}

function getSessionId(): string {
  if (typeof window === 'undefined') return 'default';
  let id = localStorage.getItem('bulife_session_id');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('bulife_session_id', id);
  }
  return id;
}

export default function QueryBar({ onResult, onStreamToken, onStreamStart, onStreamEnd }: QueryBarProps) {
  const [query, setQuery] = useState('');
  const [location, setLocation] = useState('');
  const [timeAvailable, setTimeAvailable] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [loadingStage, setLoadingStage] = useState<'initial' | 'warmup'>('initial');

  const handleSubmit = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setLoadingStage('initial');
    onStreamStart();

    const sessionId = getSessionId();
    const startTime = Date.now();

    // Upgrade to "warmup" message after 8s (cold start detection)
    const warmupTimer = setTimeout(() => {
      setLoadingStage('warmup');
    }, 8000);

    try {
      await streamQuery(
        {
          message: query,
          location: location || undefined,
          time_available: timeAvailable ? parseInt(timeAvailable) : undefined,
          session_id: sessionId,
        },
        onStreamToken,
      );
      clearTimeout(warmupTimer);
      onStreamEnd();
    } catch (err) {
      clearTimeout(warmupTimer);
      try {
        const result = await sendQuery({
          message: query,
          location: location || undefined,
          time_available: timeAvailable ? parseInt(timeAvailable) : undefined,
          session_id: sessionId,
        });
        onResult(result);
        onStreamEnd();
      } catch (fallbackErr) {
        const elapsed = Date.now() - startTime;
        if (elapsed > 30000) {
          setError('Backend is taking too long to initialize. Please try again in 60 seconds.');
        } else {
          setError('Something went wrong. Please try again.');
        }
        onStreamEnd();
      }
    }
    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };

  return (
    <div className="rounded-xl border border-white/10 backdrop-blur-lg bg-white/5 p-6 mb-6">
      <textarea
        className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-lg text-[#f0f0f0] placeholder-[#666] resize-none focus:outline-none focus:ring-2 focus:ring-[#CC0000]/50 focus:border-[#CC0000]/50"
        rows={3}
        placeholder="What do you need help with today?"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
      />

      <div className="flex gap-3 mt-3">
        <input
          className="flex-1 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-[#f0f0f0] placeholder-[#666] focus:outline-none focus:ring-2 focus:ring-[#CC0000]/50"
          placeholder="Current location (e.g. CDS, CAS, GSU)"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
        />
        <input
          className="w-36 bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-[#f0f0f0] placeholder-[#666] focus:outline-none focus:ring-2 focus:ring-[#CC0000]/50"
          placeholder="Minutes free"
          type="number"
          min={1}
          value={timeAvailable}
          onChange={(e) => setTimeAvailable(e.target.value)}
        />
      </div>

      {error && <p className="mt-2 text-sm text-red-400">{error}</p>}

      <button
        className="mt-4 w-full bg-gradient-to-r from-[#CC0000] to-[#990000] text-white py-3 rounded-lg font-semibold hover:from-[#e60000] hover:to-[#CC0000] hover:shadow-lg hover:shadow-[#CC0000]/20 disabled:opacity-50 transition-all duration-200"
        onClick={handleSubmit}
        disabled={loading || !query.trim()}
      >
        {loading
          ? loadingStage === 'warmup'
            ? 'First-time setup (initializing AI models)…'
            : 'Loading campus data…'
          : 'Ask BU Life AI'}
      </button>

      <p className="text-xs text-[#666] mt-2 text-center">Cmd+Enter to submit</p>

      <div className="mt-5">
        <p className="text-sm text-[#a0a0a0] mb-2">Try asking:</p>
        <div className="flex flex-col gap-2">
          {EXAMPLE_PROMPTS.map((p, i) => (
            <button
              key={i}
              className="text-left text-sm bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-[#a0a0a0] hover:bg-[#CC0000]/10 hover:border-[#CC0000]/30 hover:text-[#f0f0f0] transition-all duration-200"
              onClick={() => setQuery(p)}
            >
              &ldquo;{p}&rdquo;
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
