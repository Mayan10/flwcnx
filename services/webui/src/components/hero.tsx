"use client";

import { useEffect, useState } from "react";

/* ── Exact TUI color palette from theme.rs ─────────────────────────────────── */
const C = {
  bg:          "#000000",
  surface:     "rgb(10,10,12)",
  surfaceUp:   "rgb(18,18,22)",
  fg:          "rgb(230,233,240)",
  fgSec:       "rgb(120,128,145)",
  fgDim:       "rgb(55,60,72)",
  fgGhost:     "rgb(35,38,46)",
  accent:      "rgb(99,140,255)",
  accentDim:   "rgb(60,85,160)",
  status:      "rgb(72,199,162)",
  statusDim:   "rgb(40,110,90)",
  border:      "rgb(30,32,38)",
  borderLight: "rgb(45,48,58)",
};

/* ── ASCII brand art from coming_soon.rs ───────────────────────────────────── */
const BRAND_ART = [
  "███████ ██       ██████  ██     ██  ██████  ██████  ███    ██ ██   ██",
  "██      ██      ██    ██ ██     ██ ██      ██    ██ ████   ██  ██ ██ ",
  "█████   ██      ██    ██ ██  █  ██ ██      ██    ██ ██ ██  ██   ███  ",
  "██      ██      ██    ██ ██ ███ ██ ██      ██    ██ ██  ██ ██  ██ ██ ",
  "██      ███████  ██████   ███ ███   ██████  ██████  ██   ████ ██   ██",
];

const LOADING_FRAMES = [
  "·        ", "· ·      ", "· · ·    ", "· · · ·  ", "· · · · ·",
  "  · · · ·", "    · · ·", "      · ·", "        ·", "         ",
];

/* ── TUI Screen Component ──────────────────────────────────────────────────── */

function TuiScreen() {
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 200);
    return () => clearInterval(id);
  }, []);

  const loadingIdx = Math.floor(tick / 2) % LOADING_FRAMES.length;
  const pulseBrand = Math.floor(tick / 5) % 2 === 0;
  const pulseStatus = Math.floor(tick / 3) % 2 === 0;

  const mono: React.CSSProperties = {
    fontFamily: "'Geist Mono', 'SF Mono', 'Fira Code', 'Cascadia Code', monospace",
  };

  return (
    <div
      style={{
        background: C.bg,
        color: C.fg,
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        ...mono,
      }}
    >
      {/* ── Header Bar ─────────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 16px 8px",
          borderBottom: `1px solid ${C.border}`,
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span style={{ color: C.accent, fontSize: "13px" }}>
            {pulseBrand ? "◆" : "◇"}
          </span>
          <span
            style={{
              color: C.accent,
              fontSize: "13px",
              fontWeight: 700,
              marginLeft: "4px",
            }}
          >
            FlowConX
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            style={{
              color: pulseStatus ? C.status : C.statusDim,
              fontSize: "11px",
              transition: "color 0.3s",
            }}
          >
            ●
          </span>
          <span style={{ color: C.fgDim, fontSize: "12px" }}>INITIALIZING</span>
          <span style={{ color: C.fgDim, fontSize: "12px", marginLeft: "4px" }}>
            v0.1.0
          </span>
        </div>
      </div>

      {/* ── Body ───────────────────────────────────────────────────── */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "0",
          padding: "24px 16px",
          minHeight: 0,
        }}
      >
        {/* Brand ASCII Art */}
        <div
          style={{
            textAlign: "center",
            lineHeight: 1.15,
            marginBottom: "20px",
            whiteSpace: "pre",
          }}
        >
          {BRAND_ART.map((line, i) => (
            <div
              key={i}
              style={{
                color: i < 2 ? C.fg : C.fgDim,
                fontWeight: i < 2 ? 700 : 700,
                fontSize: "clamp(6px, 1.3vw, 13px)",
                letterSpacing: "0.5px",
              }}
            >
              {line}
            </div>
          ))}
        </div>

        {/* Thin accent divider */}
        <div
          style={{
            width: "180px",
            height: "1px",
            background: C.borderLight,
            margin: "8px 0 20px",
          }}
        />

        {/* Descriptor */}
        <div
          style={{
            color: C.fgSec,
            fontSize: "12px",
            letterSpacing: "0.35em",
            marginBottom: "16px",
          }}
        >
          N E T W O R K &nbsp;&nbsp; I N T E L L I G E N C E
        </div>

        {/* Status Card */}
        <div
          style={{
            border: `1px solid ${C.borderLight}`,
            borderRadius: "0",
            padding: "12px 40px",
            marginBottom: "20px",
            textAlign: "center",
          }}
        >
          <span
            style={{
              color: C.status,
              fontSize: "12px",
              marginRight: "8px",
              transition: "opacity 0.3s",
            }}
          >
            {pulseStatus ? "●" : "◉"}
          </span>
          <span
            style={{
              color: C.status,
              fontSize: "12px",
              fontWeight: 700,
              letterSpacing: "0.1em",
            }}
          >
            COMING SOON
          </span>
        </div>

        {/* Supporting copy */}
        <div
          style={{
            textAlign: "center",
            color: C.fgDim,
            fontSize: "12.5px",
            lineHeight: 1.6,
            marginBottom: "12px",
          }}
        >
          Real-time network flow intelligence,
          <br />
          classification &amp; observability.
        </div>

        {/* Loading animation */}
        <div
          style={{
            color: C.accentDim,
            fontSize: "14px",
            letterSpacing: "2px",
            height: "18px",
          }}
        >
          {LOADING_FRAMES[loadingIdx]}
        </div>
      </div>

      {/* ── Footer Bar ─────────────────────────────────────────────── */}
      <div
        style={{
          borderTop: `1px solid ${C.border}`,
          padding: "8px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "20px",
          flexShrink: 0,
        }}
      >
        <span>
          <span style={{ color: C.fgSec, fontWeight: 700, fontSize: "11px" }}>
            ESC
          </span>
          <span style={{ color: C.fgDim, fontSize: "11px", marginLeft: "4px" }}>
            Exit
          </span>
        </span>
        <span>
          <span style={{ color: C.fgSec, fontWeight: 700, fontSize: "11px" }}>
            q
          </span>
          <span style={{ color: C.fgDim, fontSize: "11px", marginLeft: "4px" }}>
            Quit
          </span>
        </span>
        <span style={{ color: C.fgGhost, fontSize: "11px" }}>flowconx.dev</span>
      </div>
    </div>
  );
}

/* ── Hero Section ──────────────────────────────────────────────────────────── */

export function Hero() {
  return (
    <section
      className="relative flex flex-col items-center justify-start px-6 pt-36 sm:pt-48 pb-20 text-center overflow-hidden"
      style={{ background: "var(--bg)", minHeight: "100vh" }}
    >
      {/* Background glow */}
      <div
        className="absolute top-0 inset-x-0 h-[600px] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, var(--glow) 0%, transparent 70%)",
          opacity: 0.8,
        }}
      />

      <div className="relative max-w-[860px] mx-auto z-10 flex flex-col items-center">
        {/* Announcement Badge */}
        <a
          href="#features"
          className="anim-fade d-1 inline-flex items-center gap-2.5 px-1 py-1 pr-3 rounded-full mb-8 transition-transform duration-300 hover:scale-[1.02] cursor-pointer no-underline"
          style={{
            background: "var(--bg-subtle)",
            border: "1px solid var(--border)",
          }}
        >
          <span
            className="text-[10px] font-semibold tracking-widest uppercase px-2 py-1 rounded-full"
            style={{ background: "var(--fg)", color: "var(--bg)" }}
          >
            New
          </span>
          <span
            className="text-[12.5px] font-medium"
            style={{ color: "var(--fg-muted)" }}
          >
            Introducing FlowConX Core 1.0
          </span>
          <svg
            width="12"
            height="12"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            style={{ color: "var(--fg-muted)" }}
          >
            <path d="M6 12l4-4-4-4" />
          </svg>
        </a>

        {/* Main headline */}
        <h1
          className="anim-enter d-2"
          style={{
            fontSize: "clamp(2.8rem, 7vw, 5.5rem)",
            fontWeight: 500,
            letterSpacing: "-0.04em",
            lineHeight: 1.05,
            color: "var(--fg)",
            marginBottom: "1.5rem",
          }}
        >
          Network intelligence,{" "}
          <br className="hidden sm:block" />
          <span style={{ color: "var(--fg-subtle)" }}>without compromise.</span>
        </h1>

        {/* Subtitle */}
        <p
          className="anim-enter d-3"
          style={{
            fontSize: "clamp(1rem, 2vw, 1.25rem)",
            lineHeight: 1.6,
            color: "var(--fg-muted)",
            maxWidth: "580px",
            marginBottom: "3rem",
          }}
        >
          Unprecedented visibility into your network traffic. Real-time machine
          learning classification built for modern satellite and terrestrial
          infrastructure.
        </p>

        {/* CTAs */}
        <div className="anim-enter d-4 flex flex-col sm:flex-row items-center gap-4">
          <a
            href="#early-access"
            className="px-7 py-3.5 rounded-full text-[14px] font-medium transition-all duration-200 cursor-pointer no-underline"
            style={{
              background: "var(--fg)",
              color: "var(--bg)",
              boxShadow: "0 4px 14px var(--glow)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.transform = "translateY(-1px)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.transform = "translateY(0)")
            }
          >
            Start building
          </a>
          <a
            href="#architecture"
            className="px-7 py-3.5 rounded-full text-[14px] font-medium transition-colors duration-200 cursor-pointer flex items-center gap-2 no-underline"
            style={{
              background: "transparent",
              color: "var(--fg)",
              border: "1px solid var(--border)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "var(--bg-muted)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "transparent")
            }
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.379.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.138 20.161 22 16.418 22 12c0-5.523-4.477-10-10-10z" />
            </svg>
            View on GitHub
          </a>
        </div>
      </div>

      {/* ── 3D TUI Preview ─────────────────────────────────────────── */}
      <div
        className="relative w-full max-w-[900px] mx-auto mt-24 anim-enter d-6 z-0"
        style={{ perspective: "1200px" }}
      >
        <div
          className="relative w-full rounded-xl overflow-hidden"
          style={{
            aspectRatio: "16 / 9.5",
            border: `1px solid ${C.border}`,
            boxShadow: `
              0 0 0 1px ${C.border},
              0 20px 60px -15px rgba(0,0,0,0.7),
              0 0 100px -20px rgba(99,140,255,0.08)
            `,
            transform: "rotateX(6deg) scale(0.94)",
            transformOrigin: "bottom center",
          }}
        >
          {/* macOS window chrome */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "10px 14px",
              background: C.surface,
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "#ff5f57",
                opacity: 0.8,
              }}
            />
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "#febc2e",
                opacity: 0.8,
              }}
            />
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: "#28c840",
                opacity: 0.8,
              }}
            />
          </div>

          {/* The actual TUI replica */}
          <TuiScreen />
        </div>

        {/* Bottom fade mask */}
        <div
          className="absolute inset-x-0 bottom-0 h-32 pointer-events-none"
          style={{
            background:
              "linear-gradient(to bottom, transparent, var(--bg) 95%)",
          }}
        />
      </div>
    </section>
  );
}
