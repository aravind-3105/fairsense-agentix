import { useRef, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { FileText, Image, ShieldAlert, ChevronRight, ChevronDown, ScanSearch, Brain, BarChart2 } from "lucide-react";
import vectorLogo from "./assets/Vector Logo_Bilingual_White_Horizontal.png";
import fairsenseLogo from "./assets/fairsense-logo.png";
import aixpertLogo from "./assets/AIXPERT_logo_extended_white-2048x896.png";

const LinkedInIcon = ({ size = 16 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect x="2" y="9" width="4" height="12" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);

const XIcon = ({ size = 16 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
  </svg>
);

const BlueskyIcon = ({ size = 16 }: { size?: number }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 10.8c-1.087-2.114-4.046-6.053-6.798-7.995C2.566.944 1.561 1.266.902 1.565.139 1.908 0 3.08 0 3.768c0 .69.378 5.65.567 6.479.737 3.123 3.419 4.056 5.977 3.529-.005.039-.01.077-.014.116C2.535 14.315.733 16.37.58 18.409c-.206 2.763 2.43 5.765 5.77 4.578 2.668-.948 4.888-3.77 6.65-6.875 1.762 3.104 3.982 5.927 6.65 6.875 3.34 1.187 5.976-1.815 5.77-4.578-.153-2.038-1.955-4.094-5.95-4.515a7.422 7.422 0 0 1-.014-.116c2.558.527 5.24-.406 5.977-3.529C23.622 9.417 24 4.457 24 3.768c0-.688-.139-1.86-.902-2.203-.659-.299-1.664-.621-4.3 1.24C16.046 4.747 13.087 8.686 12 10.8Z" />
  </svg>
);

export default function Landing() {
  const navigate = useNavigate();
  const detailsRef = useRef<HTMLElement>(null);

  const scrollToDetails = () => {
    detailsRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const modes: { icon: ReactNode; title: string; mode: string; audience: string; description: string; example: string }[] = [
    {
      icon: <FileText size={22} />,
      mode: "text",
      title: "Text Analysis",
      audience: "HR teams, policy writers, content creators",
      description:
        "Paste a job posting, policy document, or conversation and instantly see which words or phrases carry hidden bias — with plain-language explanations of why each one matters.",
      example: "e.g. \"We need a young rockstar developer\" — flagged for age and gender bias.",
    },
    {
      icon: <Image size={22} />,
      mode: "image",
      title: "Image Analysis",
      audience: "Marketing teams, designers, researchers",
      description:
        "Upload an image and let the platform identify stereotypes, underrepresentation, or harmful visual associations — the kind of bias that's easy to miss but hard to defend.",
      example: "e.g. A stock photo showing only one demographic in a leadership role.",
    },
    {
      icon: <ShieldAlert size={22} />,
      mode: "csv",
      title: "Risk Assessment",
      audience: "AI teams, compliance officers, product managers",
      description:
        "Describe an AI system or paste a deployment scenario and receive a scored breakdown of fairness, security, and compliance risks — before they become real-world problems.",
      example: "e.g. A hiring algorithm trained on historical data with demographic imbalances.",
    },
  ];

  const steps = [
    {
      icon: <ScanSearch size={20} />,
      step: "1",
      title: "Choose your input",
      description: "Select text, image, or risk scenario — then paste or upload your content.",
    },
    {
      icon: <Brain size={20} />,
      step: "2",
      title: "Agent analyses",
      description: "An AI reasoning agent plans and runs its analysis, showing its thinking in real time.",
    },
    {
      icon: <BarChart2 size={20} />,
      step: "3",
      title: "Review results",
      description: "See highlighted bias instances, severity scores, and plain-language explanations.",
    },
  ];

  return (
    <main className="min-h-screen bg-base text-white relative overflow-x-hidden">

      {/* Decorative background: rose glow + dot grid */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background: "radial-gradient(ellipse 100% 50% at 50% 0%, rgba(235,8,138,0.13) 0%, transparent 60%)",
        }}
      />
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0)",
          backgroundSize: "28px 28px",
        }}
      />

      {/* Nav */}
      <nav className="flex items-center justify-between px-8 py-5 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <img src={fairsenseLogo} alt="FairSense AgentiX" className="h-12 mix-blend-screen" />
          <span className="text-lg font-semibold">FairSense AgentiX</span>
        </div>
        <div className="flex items-center gap-6">
          <a
            href="https://vectorinstitute.github.io/fairsense-agentix/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-slate-400 hover:text-accent-200 transition-colors"
          >
            Documentation
          </a>
          <button
            onClick={() => navigate("/analyze")}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-accent-200 to-accent-100 px-4 py-2 text-sm font-semibold text-black transition hover:opacity-90"
          >
            Launch App <ChevronRight size={15} />
          </button>
        </div>
      </nav>

      {/* Hero — above the fold */}
      <section className="min-h-[calc(100vh-73px)] flex flex-col items-center justify-center px-8 text-center space-y-6">
        <p className="pill inline-flex">Agentic Fairness Analysis</p>
        <h1 className="text-5xl font-bold leading-tight">
          Detect bias before it<br />becomes a problem.
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl leading-relaxed">
          FairSense AgentiX uses AI agents to analyze text, images, and deployment scenarios for hidden bias and fairness risks — giving you clear, actionable explanations, not just scores.
        </p>
        <div className="flex items-center justify-center gap-4 pt-2">
          <button
            onClick={() => navigate("/analyze")}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-accent-200 to-accent-100 px-6 py-3 font-semibold text-black transition hover:opacity-90"
          >
            Try it now <ChevronRight size={16} />
          </button>
          <a
            href="https://vectorinstitute.github.io/fairsense-agentix/"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-xl border border-slate-700 px-6 py-3 text-sm text-slate-300 hover:border-slate-500 hover:bg-slate-800 transition-colors"
          >
            Read the docs
          </a>
        </div>
        <button
          onClick={scrollToDetails}
          className="flex flex-col items-center gap-1 text-xs text-slate-500 hover:text-accent-200 transition-colors pt-6 animate-bounce"
        >
          Learn more
          <ChevronDown size={16} />
        </button>
      </section>

      {/* Details — below the fold */}
      <section ref={detailsRef} className="border-t border-slate-800">

        {/* Mode cards */}
        <div className="max-w-5xl mx-auto px-8 py-20">
          <h2 className="text-center text-xs font-semibold uppercase tracking-widest text-slate-500 mb-8">
            What can it analyse?
          </h2>
          <div className="grid gap-6 md:grid-cols-3">
            {modes.map((m) => (
              <div
                key={m.title}
                className="glass p-6 space-y-4 hover:border-accent-200/30 transition-colors cursor-pointer"
                onClick={() => navigate(`/analyze?mode=${m.mode}`)}
              >
                <div className="flex items-center gap-3">
                  <span className="text-accent-200">{m.icon}</span>
                  <h3 className="font-semibold text-slate-100">{m.title}</h3>
                </div>
                <p className="text-xs text-accent-200 uppercase tracking-wider">{m.audience}</p>
                <p className="text-sm text-slate-400 leading-relaxed">{m.description}</p>
                <p className="text-xs text-slate-600 italic leading-relaxed">{m.example}</p>
              </div>
            ))}
          </div>
        </div>

        {/* How it works */}
        <div className="border-t border-slate-800 py-20 px-8">
          <div className="max-w-3xl mx-auto space-y-10">
            <h2 className="text-center text-xs font-semibold uppercase tracking-widest text-slate-500">
              How it works
            </h2>
            <div className="grid gap-8 md:grid-cols-3">
              {steps.map((s) => (
                <div key={s.step} className="space-y-3 text-center">
                  <div className="mx-auto h-10 w-10 rounded-full bg-accent-200/10 border border-accent-200/30 flex items-center justify-center text-accent-200">
                    {s.icon}
                  </div>
                  <p className="text-xs text-slate-500 uppercase tracking-wider">Step {s.step}</p>
                  <p className="font-semibold text-slate-200">{s.title}</p>
                  <p className="text-sm text-slate-400 leading-relaxed">{s.description}</p>
                </div>
              ))}
            </div>
            <div className="text-center pt-4">
              <button
                onClick={() => navigate("/analyze")}
                className="flex items-center gap-2 mx-auto rounded-xl bg-gradient-to-r from-accent-200 to-accent-100 px-6 py-3 font-semibold text-black transition hover:opacity-90"
              >
                Get started <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </div>

      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800 px-8 py-6 flex items-center justify-between">
        <div className="flex items-center gap-6 text-xs text-slate-500">
          <span>
            Built as part of&nbsp;
            <a
              href="https://vectorinstitute.github.io/vector-aixpert/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-400 hover:text-accent-200 transition-colors"
            >
              Vector AIxpert
            </a>
          </span>
          <a href="https://aixpert-project.eu/" target="_blank" rel="noopener noreferrer">
            <img src={aixpertLogo} alt="AIxpert" className="h-8 opacity-75 hover:opacity-100 transition-opacity" />
          </a>
        </div>
        <div className="flex items-center gap-5">
          <a href="https://vectorinstitute.ai/" target="_blank" rel="noopener noreferrer">
            <img src={vectorLogo} alt="Vector Institute" className="h-7 opacity-60 hover:opacity-100 transition-opacity" />
          </a>
          <div className="w-px h-5 bg-slate-700" />
          <div className="flex items-center gap-3">
            <a href="https://www.linkedin.com/company/vector-institute/" target="_blank" rel="noopener noreferrer" className="text-slate-500 hover:text-accent-200 transition-colors"><LinkedInIcon size={15} /></a>
            <a href="https://x.com/vectorinst" target="_blank" rel="noopener noreferrer" className="text-slate-500 hover:text-accent-200 transition-colors"><XIcon size={15} /></a>
            <a href="https://bsky.app/profile/vectorinstitute.ai" target="_blank" rel="noopener noreferrer" className="text-slate-500 hover:text-accent-200 transition-colors"><BlueskyIcon size={15} /></a>
          </div>
        </div>
      </footer>

    </main>
  );
}
