interface Props {
  response: string;
  type: 'places' | 'resource' | 'events' | 'time_assistant';
  sources?: string[];
  isStreaming?: boolean;
}

const TYPE_LABELS: Record<Props['type'], string> = {
  places: 'Place Finder',
  resource: 'BU Resource Copilot',
  events: 'Event Recommendations',
  time_assistant: 'Time Between Classes',
};

const TYPE_COLORS: Record<Props['type'], string> = {
  places: 'bg-blue-500/20 border-blue-500/30 text-blue-300',
  resource: 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300',
  events: 'bg-purple-500/20 border-purple-500/30 text-purple-300',
  time_assistant: 'bg-amber-500/20 border-amber-500/30 text-amber-300',
};

export default function RecommendationCard({ response, type, sources, isStreaming }: Props) {
  return (
    <div className="rounded-xl border border-white/10 backdrop-blur-lg bg-white/5 p-6 mb-6">
      <div className="flex items-center gap-2 mb-4">
        <span className={`text-xs font-semibold px-2 py-1 rounded-full border ${TYPE_COLORS[type]}`}>
          {TYPE_LABELS[type]}
        </span>
        {isStreaming && (
          <span className="text-xs text-[#CC0000] animate-pulse">Streaming...</span>
        )}
      </div>
      <div className="text-[#e0e0e0] whitespace-pre-wrap leading-relaxed">
        {response}
        {isStreaming && <span className="inline-block w-2 h-4 bg-[#CC0000] animate-pulse ml-0.5" />}
      </div>
      {sources && sources.length > 0 && (
        <div className="mt-4 pt-3 border-t border-white/10">
          <p className="text-xs text-[#a0a0a0] mb-1">Sources:</p>
          <ul className="text-xs text-[#888] space-y-0.5">
            {sources.map((src, i) => (
              <li key={i}>{src}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
