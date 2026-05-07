'use client';

import { useState } from 'react';
import QueryBar from '@/components/QueryBar';
import RecommendationCard from '@/components/RecommendationCard';
import { QueryResponse } from '@/lib/api';

export default function Home() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [streamedText, setStreamedText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  const handleResult = (res: QueryResponse) => {
    setResult(res);
    setStreamedText('');
  };

  const handleStream = (token: string) => {
    setStreamedText((prev) => prev + token);
  };

  const displayText = streamedText || result?.response || '';

  return (
    <main className="min-h-screen">
      <header className="border-b border-white/10 backdrop-blur-lg bg-white/5 py-6 px-8">
        <h1 className="text-3xl font-bold text-[#f0f0f0]">
          BU <span className="text-[#CC0000]">Life</span> AI
        </h1>
        <p className="text-[#a0a0a0] mt-1">Your smart BU campus assistant</p>
      </header>

      <div className="max-w-2xl mx-auto py-12 px-4">
        <QueryBar
          onResult={handleResult}
          onStreamToken={handleStream}
          onStreamStart={() => { setIsStreaming(true); setStreamedText(''); setResult(null); }}
          onStreamEnd={() => setIsStreaming(false)}
        />

        {(displayText || isStreaming) && (
          <RecommendationCard
            response={displayText}
            type={result?.type || 'places'}
            sources={result?.sources}
            isStreaming={isStreaming}
          />
        )}
      </div>
    </main>
  );
}
