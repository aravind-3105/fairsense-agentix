import { useEffect, useMemo, useRef, useState } from "react";
import { clsx } from "clsx";
import { Loader2, Play, Upload, Sparkles, Activity } from "lucide-react";
import { analyze, analyzeFile, connectToStream, WorkflowID } from "./api";

type Mode = "text" | "image" | "csv";

const MODE_LABELS: Record<Mode, string> = {
  text: "Bias (Text)",
  image: "Bias (Image)",
  csv: "Risk"
};

interface TimelineEntry {
  timestamp: number;
  event: string;
  level: string;
  context: Record<string, unknown>;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("text");
  const [input, setInput] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const canSubmit =
    (!loading && mode !== "image" && input.trim().length > 0) ||
    (!loading && mode === "image" && file);

  async function handleAnalyze() {
    if (!canSubmit) {
      return;
    }
    setLoading(true);
    setTimeline([]);
    setResult(null);
    try {
      const payload =
        mode === "image"
          ? await analyzeFile(file as File, "bias_image")
          : await analyze({
              content: input,
              input_type: mode === "csv" ? "risk" : undefined
            });

      setResult(payload);
      if (payload?.run_id) {
        const ws = connectToStream(payload.run_id, (evt) => {
          try {
            const data = JSON.parse(evt.data);
            setTimeline((prev) => [...prev, data]);
          } catch {
            // ignore malformed payloads
          }
        });
        wsRef.current = ws;
      }
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Check logs for details.");
    } finally {
      setLoading(false);
    }
  }

  const highlightHtml = useMemo(() => {
    if (!result?.bias_result?.highlighted_html) {
      return null;
    }
    return { __html: result.bias_result.highlighted_html };
  }, [result]);

  return (
    <main className="min-h-screen px-6 py-10 bg-base">
      <header className="mb-8">
        <div className="flex items-center gap-3">
          <Sparkles className="text-accent-200" />
          <div>
            <h1 className="text-3xl font-semibold">FairSense AgentiX</h1>
            <p className="text-slate-400">
              Agentic fairness & AI-risk analysis visualized like a modern chat experience.
            </p>
          </div>
        </div>
      </header>

      <section className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-6">
          <ModeSelector mode={mode} onModeChange={setMode} />
          {mode === "image" ? (
            <ImageDropzone file={file} onFileChange={setFile} />
          ) : (
            <textarea
              className="w-full min-h-[220px] glass p-4 focus:outline-none focus:ring-2 focus:ring-accent-200"
              placeholder={
                mode === "csv"
                  ? "Describe your deployment scenario or paste CSV text..."
                  : "Paste your job posting or conversation snippet..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
          )}
          <div className="flex gap-4">
            <button
              disabled={!canSubmit}
              onClick={handleAnalyze}
              className={clsx(
                "flex items-center gap-2 rounded-xl px-5 py-3 font-medium transition",
                canSubmit
                  ? "bg-gradient-to-r from-accent-200 to-accent-100 text-black"
                  : "bg-panel text-slate-500 cursor-not-allowed"
              )}
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin" /> Analyzing…
                </>
              ) : (
                <>
                  <Play size={16} /> Run Analysis
                </>
              )}
            </button>
            <button
              className="rounded-xl border border-slate-700 px-4 py-3 text-sm text-slate-300 hover:border-slate-500"
              onClick={() => {
                setInput("");
                setFile(null);
                setResult(null);
                setTimeline([]);
              }}
            >
              Reset
            </button>
          </div>

          <TimelinePanel events={timeline} />
        </div>

        <div className="space-y-6">
          <ResultPanel result={result} />
          {highlightHtml && (
            <div className="glass p-4">
              <h3 className="mb-3 text-sm uppercase tracking-wide text-slate-400">
                Highlighted Text
              </h3>
              <div
                className="prose prose-invert max-w-none"
                dangerouslySetInnerHTML={highlightHtml}
              />
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function ModeSelector({
  mode,
  onModeChange
}: {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
}) {
  return (
    <div className="glass flex gap-2 p-2">
      {(Object.keys(MODE_LABELS) as Mode[]).map((key) => (
        <button
          key={key}
          onClick={() => onModeChange(key)}
          className={clsx(
            "flex-1 rounded-lg px-4 py-3 text-sm font-semibold transition",
            mode === key ? "bg-accent-200 text-black" : "text-slate-400"
          )}
        >
          {MODE_LABELS[key]}
        </button>
      ))}
    </div>
  );
}

function ImageDropzone({
  file,
  onFileChange
}: {
  file: File | null;
  onFileChange: (f: File | null) => void;
}) {
  return (
    <label className="glass flex cursor-pointer flex-col items-center justify-center gap-3 p-10 text-slate-400 hover:border-slate-500">
      <Upload />
      <span>{file ? file.name : "Drop an image or click to browse"}</span>
      <input
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
      />
    </label>
  );
}

function TimelinePanel({ events }: { events: TimelineEntry[] }) {
  if (!events.length) {
    return (
      <div className="glass px-4 py-6 text-center text-slate-400">
        Agentic flow will appear here.
      </div>
    );
  }
  return (
    <div className="glass p-4 space-y-3 max-h-[360px] overflow-auto">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-300">
        <Activity size={16} /> Agent Timeline
      </h3>
      <ol className="space-y-3 text-sm">
        {events.map((evt, idx) => (
          <li key={`${evt.timestamp}-${idx}`} className="rounded-lg bg-panel/60 p-3">
            <div className="flex items-center justify-between text-slate-400">
              <span className="uppercase tracking-wider text-xs">{evt.event}</span>
              <span className="text-xs">
                {new Date(evt.timestamp * 1000).toLocaleTimeString()}
              </span>
            </div>
            {evt.context?.message && (
              <p className="text-slate-200">{String(evt.context.message)}</p>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function ResultPanel({ result }: { result: any | null }) {
  if (!result) {
    return (
      <div className="glass p-6 text-slate-400">
        Run an analysis to view structured output.
      </div>
    );
  }

  if (result.workflow_id === "risk") {
    const risks = result.risk_result?.risks ?? [];
    return (
      <div className="glass p-5 space-y-4">
        <header>
          <p className="pill">Risk Insights</p>
          <h3 className="text-xl font-semibold">Top Risks</h3>
        </header>
        <div className="space-y-3">
          {risks.slice(0, 5).map((risk: any, idx: number) => (
            <div key={idx} className="rounded-lg border border-slate-800 p-3">
              <div className="flex justify-between text-sm text-slate-400">
                <span>{risk.name}</span>
                <span>Score: {risk.score?.toFixed(2)}</span>
              </div>
              <p className="text-slate-300">{risk.description}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const summary = result.bias_result?.summary;
  const instances = result.bias_result?.bias_instances ?? [];

  return (
    <div className="glass p-5 space-y-4">
      <header>
        <p className="pill">Bias Insights</p>
        <h3 className="text-xl font-semibold">
          {result.bias_result?.status === "success"
            ? "Analysis Complete"
            : "Analysis"}
        </h3>
      </header>
      {summary && <p className="text-slate-200">{summary}</p>}
      <div className="space-y-2">
        {instances.slice(0, 5).map((inst: any, idx: number) => (
          <div key={idx} className="rounded-lg border border-slate-800 p-3">
            <div className="flex items-center justify-between text-sm text-slate-400">
              <span className="font-semibold uppercase">{inst.type}</span>
              <span>{inst.severity}</span>
            </div>
            <p className="text-slate-300">
              <em>{inst.text_span}</em>
            </p>
            <p className="text-slate-400">{inst.explanation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
