import React, { useEffect, useMemo, useRef, useState } from "react";
import { clsx } from "clsx";
import { Loader2, Play, Upload, Sparkles, Activity, Power, FileText, Image, ShieldAlert } from "lucide-react";
import vectorLogo from "./assets/Vector Logo_Bilingual_White_Horizontal.png";
import { analyzeStart, analyzeFileStart, connectToStream, API_BASE } from "./api";

type Mode = "text" | "image" | "csv";

const MODE_LABELS: Record<Mode, string> = {
  text: "Bias (Text)",
  image: "Bias (Image)",
  csv: "Risk"
};

const MODE_ICONS: Record<Mode, React.ReactNode> = {
  text: <FileText size={14} />,
  image: <Image size={14} />,
  csv: <ShieldAlert size={14} />
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
  const [showLoadingBanner, setShowLoadingBanner] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  // Detect model download events from timeline and show loading banner
  useEffect(() => {
    const latestEvent = timeline[timeline.length - 1];
    if (!latestEvent) return;

    // Check for model download start event
    if (latestEvent.event === "model_download_start") {
      const modelName = latestEvent.context?.model_name || "AI model";
      setLoadingMessage(`⏳ Downloading ${modelName} (first use only, ~30-120s)...`);
      setShowLoadingBanner(true);
    }

    // Check for model download complete event
    if (latestEvent.event === "model_download_complete") {
      setShowLoadingBanner(false);
      setLoadingMessage("");
    }

    // Check for model download failure
    if (latestEvent.event === "model_download_failed") {
      setShowLoadingBanner(false);
      setLoadingMessage("");
      // Error is already displayed in timeline, no need for banner
    }
  }, [timeline]);

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
      // Step 1: Start analysis and get run_id immediately (fixes race condition!)
      const startResponse =
        mode === "image"
          ? await analyzeFileStart(file as File, "bias_image")
          : await analyzeStart({
            content: input,
            input_type: mode === "csv" ? "risk" : undefined
          });

      const runId = startResponse.run_id;

      // Step 2: Connect WebSocket BEFORE analysis completes (this is the fix!)
      const ws = connectToStream(runId, (evt) => {
        try {
          const data = JSON.parse(evt.data);

          // Check if this is the completion event
          if (data.event === "analysis_complete" && data.context?.result) {
            // Extract final result from completion event
            setResult(data.context.result);
            setLoading(false); // Stop loading spinner
          }

          // Check if this is an error event
          if (data.event === "analysis_error") {
            alert(`Analysis failed: ${data.context?.message || "Unknown error"}`);
            setLoading(false); // Stop loading spinner
          }

          // Add all events to timeline (including intermediate agent events)
          setTimeline((prev) => [...prev, data]);
        } catch {
          // ignore malformed payloads
        }
      });

      wsRef.current = ws;

      // Step 3: Analysis is now running in background, events will stream in!
    } catch (err) {
      console.error(err);
      alert("Analysis failed. Check logs for details.");
      setLoading(false);
    }
  }

  async function handleShutdown() {
    // Confirmation dialog to prevent accidental shutdowns
    const confirmed = window.confirm(
      "Are you sure you want to shutdown the server?\n\n" +
      "This will stop both the backend and frontend servers and clean up ports.\n" +
      "You'll need to restart the server manually to use the application again."
    );

    if (!confirmed) {
      return;
    }

    try {
      // Call the shutdown endpoint
      const response = await fetch(`${API_BASE}/v1/shutdown`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        // Show success message briefly before server shuts down
        alert("Server shutdown initiated. Both servers will stop in 1 second.");

        // Close any open WebSocket connections
        wsRef.current?.close();

        // UI will become unresponsive as backend shuts down - this is expected
      } else {
        alert("Failed to shutdown server. Check console for details.");
      }
    } catch (err) {
      console.error("Shutdown error:", err);
      alert("Error communicating with server. It may have already shut down.");
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
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Sparkles className="text-accent-200" />
            <div>
              <h1 className="text-3xl font-semibold">FairSense AgentiX</h1>
              <p className="text-slate-400">
                Agentic fairness & AI-risk analysis platform
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <img
              src={vectorLogo}
              alt="Vector Institute"
              className="h-9 opacity-70 hover:opacity-100 transition-opacity"
            />
            <button
              onClick={handleShutdown}
              className="flex items-center gap-2 rounded-xl border border-red-700/50 bg-red-900/20 px-4 py-2 text-sm text-red-300 hover:bg-red-900/40 hover:border-red-600 transition-colors"
              title="Shutdown both backend and frontend servers"
            >
              <Power size={16} />
              Shutdown
            </button>
          </div>
        </div>
      </header>

      {/* Model Download Loading Banner */}
      {showLoadingBanner && (
        <div className="mb-6 bg-yellow-500/10 border-2 border-yellow-500/50 rounded-xl p-6 animate-pulse">
          <div className="flex items-start gap-4">
            <Loader2 className="text-yellow-400 animate-spin flex-shrink-0 mt-1" size={24} />
            <div className="flex-1">
              <p className="text-lg font-semibold text-yellow-300">{loadingMessage}</p>
              <p className="text-sm text-slate-400 mt-2">
                This is a one-time download. Future requests will be instant (~100ms).
              </p>
              <p className="text-xs text-slate-500 mt-1">
                The AI model is being downloaded from HuggingFace and cached locally.
              </p>
            </div>
          </div>
        </div>
      )}

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
              className="rounded-xl border border-slate-700 px-4 py-3 text-sm text-slate-300 hover:border-slate-500 hover:bg-slate-800 transition-colors"
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
            <div className="glass p-5 space-y-5">
              <header>
                <h3 className="text-sm uppercase tracking-wide text-slate-400 flex items-center gap-2">
                  <span className="inline-block w-1 h-4 bg-accent-200 rounded-full" />
                  {result?.bias_result?.image_base64 ? "Visual Analysis" : "Highlighted Text"}
                </h3>
              </header>

              <BiasLegend />

              {/* Display image if available (for VLM image analysis) */}
              {result?.bias_result?.image_base64 && (
                <div className="analyzed-image-container">
                  <img
                    src={result.bias_result.image_base64}
                    alt="Analyzed image"
                    className="analyzed-image"
                  />
                </div>
              )}

              <div
                className="prose prose-invert max-w-none text-sm leading-relaxed custom-scrollbar max-h-[500px] overflow-y-auto pr-2"
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
            "flex-1 flex items-center justify-center gap-2 rounded-lg px-4 py-3 text-sm font-semibold transition",
            mode === key ? "bg-accent-200 text-black" : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/50"
          )}
        >
          {MODE_ICONS[key]}
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
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  return (
    <label className="glass flex cursor-pointer flex-col items-center justify-center gap-3 p-6 text-slate-400 hover:border-slate-500 transition-colors">
      {previewUrl ? (
        <>
          <img
            src={previewUrl}
            alt="Selected preview"
            className="max-h-48 max-w-full rounded-lg object-contain opacity-90"
          />
          <span className="text-xs text-slate-400 truncate max-w-full px-2">{file!.name}</span>
        </>
      ) : (
        <>
          <Upload />
          <span>Drop an image or click to browse</span>
        </>
      )}
      <input
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
      />
    </label>
  );
}

const EVENT_LABELS: Record<string, string> = {
  router_decision:        "Router Decision",
  llm_call:               "LLM Call",
  model_download_start:   "Downloading Model",
  model_download_complete:"Model Ready",
  model_download_failed:  "Download Failed",
  analysis_complete:      "Analysis Complete",
  analysis_error:         "Analysis Error",
  log_info:               "Info",
  log_warning:            "Warning",
};

const EVENT_DOT: Record<string, string> = {
  analysis_complete:      "bg-green-400",
  model_download_complete:"bg-green-400",
  analysis_error:         "bg-red-400",
  model_download_failed:  "bg-red-400",
  log_warning:            "bg-yellow-400",
  model_download_start:   "bg-yellow-400",
};

function eventDot(event: string) {
  return EVENT_DOT[event] ?? "bg-slate-500";
}

function eventLabel(event: string) {
  return EVENT_LABELS[event] ?? event.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function TimelinePanel({ events }: { events: TimelineEntry[] }) {
  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set());
  const bottomRef = useRef<HTMLLIElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [events]);

  if (!events.length) {
    return (
      <div className="glass px-4 py-6 text-center text-slate-400">
        Agentic flow will appear here.
      </div>
    );
  }

  const toggleEvent = (idx: number) => {
    setExpandedEvents((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  return (
    <div className="glass p-4 space-y-3 flex flex-col max-h-[calc(100vh-32rem)] min-h-[200px]">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-300 flex-shrink-0">
        <Activity size={16} /> Agent Timeline
      </h3>
      <ol className="space-y-3 text-sm overflow-y-auto custom-scrollbar flex-1 pr-2">
        {events.map((evt, idx) => {
          const isExpanded = expandedEvents.has(idx);
          const hasContext = evt.context && Object.keys(evt.context).length > 0;

          // Extract key context fields for display
          const contextEntries = evt.context ? Object.entries(evt.context) : [];
          const contextFieldsToShow = contextEntries.filter(
            ([key]) => !["run_id", "result"].includes(key)
          );

          // Extract message safely for type-checking
          const message = evt.context?.message;
          const messageText = message ? String(message) : null;

          return (
            <li ref={idx === events.length - 1 ? bottomRef : null} key={`${evt.timestamp}-${idx}`} className={clsx(
                "rounded-xl border p-3 transition-colors",
                evt.event === "analysis_error" || evt.event === "model_download_failed"
                  ? "bg-red-900/20 border-red-700/40 hover:bg-red-900/30"
                  : evt.event === "analysis_complete" || evt.event === "model_download_complete"
                  ? "bg-green-900/20 border-green-700/40 hover:bg-green-900/30"
                  : "bg-slate-900/40 border-slate-800/40 hover:bg-slate-900/60"
              )}>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-xs font-semibold text-slate-200">
                  <span className={clsx("inline-block h-2 w-2 rounded-full flex-shrink-0", eventDot(evt.event))} />
                  {eventLabel(evt.event)}
                </span>
                <span className="text-xs text-slate-500">
                  {new Date(evt.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>

              {messageText && (
                <p className="text-slate-300 mt-2 text-xs leading-relaxed">
                  {messageText}
                </p>
              )}

              {hasContext && contextFieldsToShow.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => toggleEvent(idx)}
                    className="flex items-center gap-1 text-xs text-accent-200 hover:text-accent-100 transition"
                  >
                    <Activity size={12} />
                    <span>{isExpanded ? "Hide" : "Show"} details</span>
                    <Play
                      size={10}
                      className={clsx(
                        "transition-transform",
                        isExpanded ? "rotate-90" : "rotate-0"
                      )}
                    />
                  </button>

                  {isExpanded && (
                    <div className="mt-2 space-y-1 pl-3 border-l-2 border-slate-700">
                      {contextFieldsToShow.map(([key, value]) => (
                        <div key={key} className="text-xs">
                          <span className="text-slate-500 font-mono">{key}:</span>{" "}
                          <span className="text-slate-300">
                            {typeof value === "object" && value !== null
                              ? JSON.stringify(value, null, 2)
                              : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function BiasLegend() {
  const biasTypes = [
    { type: "Gender", color: "#FFB3BA", description: "Gendered language or stereotypes" },
    { type: "Age", color: "#FFDFBA", description: "Age-related discrimination" },
    { type: "Racial", color: "#FFFFBA", description: "Racial or ethnic bias" },
    { type: "Disability", color: "#BAE1FF", description: "Ableist language" },
    { type: "Socioeconomic", color: "#E0BBE4", description: "Class-based assumptions" },
  ];

  return (
    <div className="rounded-xl border border-slate-700/30 bg-slate-800/20 p-4">
      <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
        Bias Type Legend
      </p>
      <div className="flex flex-wrap gap-3">
        {biasTypes.map(({ type, color, description }) => (
          <div
            key={type}
            className="flex items-center gap-2 text-sm text-slate-300 transition-colors hover:text-slate-100"
            title={description}
          >
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{
                backgroundColor: `${color}40`,
                border: `1.5px solid ${color}`,
              }}
            />
            <span className="text-xs">{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultPanel({ result }: { result: any | null }) {
  const [imageExpanded, setImageExpanded] = useState(false);

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
        <header className="space-y-2">
          <p className="pill">Risk Insights</p>
          <h3 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <span className="inline-block w-1 h-5 bg-accent-200 rounded-full" />
            Top Risks
          </h3>
        </header>
        <div className="space-y-3">
          {risks.slice(0, 5).map((risk: any, idx: number) => (
            <div key={idx} className="rounded-xl border border-slate-800/50 bg-slate-900/30 p-4 space-y-2 hover:border-slate-700/70 transition-colors">
              <div className="flex justify-between items-center">
                <span className="text-sm font-semibold text-slate-200">{risk.name}</span>
                <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-accent-200/20 text-accent-200">
                  {risk.score?.toFixed(2)}
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{risk.description}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const summary = result.bias_result?.summary;
  const instances = result.bias_result?.bias_instances ?? [];

  // Try multiple possible paths for image_base64
  const imageBase64 = result.bias_result?.image_base64 || result.image_base64 || result.workflow_result?.image_base64;

  return (
    <div className="glass p-5 space-y-4">
      <header className="space-y-2">
        <p className="pill">Bias Insights</p>
        <h3 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
          <span className="inline-block w-1 h-5 bg-accent-200 rounded-full" />
          {result.bias_result?.status === "success"
            ? "Analysis Complete"
            : "Analysis"}
        </h3>
      </header>

      {/* Collapsible image viewer for visual analysis */}
      {imageBase64 && (
        <div className="space-y-2">
          <button
            onClick={() => setImageExpanded(!imageExpanded)}
            className="flex items-center gap-2 text-xs text-accent-200 hover:text-accent-100 transition-colors"
          >
            <Upload size={14} />
            <span>{imageExpanded ? "Hide" : "View"} Analyzed Image</span>
            <Play
              size={10}
              className={clsx(
                "transition-transform",
                imageExpanded ? "rotate-90" : "rotate-0"
              )}
            />
          </button>

          {imageExpanded && (
            <div className="analyzed-image-container">
              <img
                src={imageBase64}
                alt="Analyzed content"
                className="analyzed-image"
              />
            </div>
          )}
        </div>
      )}

      {summary && <p className="text-slate-300 text-sm leading-relaxed">{summary}</p>}
      <div className="space-y-3">
        {instances.slice(0, 5).map((inst: any, idx: number) => (
          <div key={idx} className="rounded-xl border border-slate-800/50 bg-slate-900/30 p-4 space-y-2 hover:border-slate-700/70 transition-colors">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wide text-slate-300">{inst.type}</span>
              <span className={clsx(
                "px-2 py-0.5 rounded-full text-xs font-medium",
                inst.severity === "high" && "bg-red-500/20 text-red-300",
                inst.severity === "medium" && "bg-yellow-500/20 text-yellow-300",
                inst.severity === "low" && "bg-blue-500/20 text-blue-300"
              )}>{inst.severity}</span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              <em>"{inst.visual_element || inst.text_span}"</em>
            </p>
            <p className="text-xs text-slate-400 leading-relaxed">{inst.explanation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
