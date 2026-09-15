"use client";

import { useEffect, useRef, useState } from "react";

/* ── Feature data ──────────────────────────────────────────────────────────── */

const features = [
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
        <polyline points="7.5 4.21 12 6.81 16.5 4.21" />
        <polyline points="7.5 19.79 7.5 14.6 3 12" />
        <polyline points="21 12 16.5 14.6 16.5 19.79" />
        <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
        <line x1="12" y1="22.08" x2="12" y2="12" />
      </svg>
    ),
    label: "Classification",
    title: "ML-Powered Flow Analysis",
    description:
      "Transformer models identify applications from encrypted traffic patterns. No deep packet inspection. No decryption.",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" />
      </svg>
    ),
    label: "Infrastructure",
    title: "Satellite-Aware Intelligence",
    description:
      "Purpose-built for LEO, MEO, and GEO satellite links with orbital-aware latency modeling and handover prediction.",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
    label: "Observability",
    title: "Real-Time Telemetry",
    description:
      "Sub-second flow metrics, classification confidence scores, and anomaly detection streamed to your existing observability stack.",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
      </svg>
    ),
    label: "Performance",
    title: "Edge-Native, Zero-Copy",
    description:
      "Lightweight Rust agent classifies traffic at line rate on commodity hardware. No GPU required for inference.",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
      </svg>
    ),
    label: "Operations",
    title: "Terminal & Web Interfaces",
    description:
      "Operator-grade TUI and web console designed for NOC environments, on-call workflows, and engineering deep-dives.",
  },
  {
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    ),
    label: "Integration",
    title: "Open Protocol Support",
    description:
      "Export to Prometheus, Grafana, Kafka, or any OTLP-compatible backend. First-class gRPC and REST APIs.",
  },
];

/* ── Component ─────────────────────────────────────────────────────────────── */

export function Features() {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold: 0.08 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <section
      id="features"
      ref={ref}
      style={{
        background: "var(--bg)",
        borderTop: "1px solid var(--border)",
      }}
      className="py-28 sm:py-40 px-6"
    >
      <div className="max-w-[1080px] mx-auto">
        {/* Section header */}
        <div
          className="text-center mb-20"
          style={{
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(20px)",
            transition: "all 0.7s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          <p
            className="text-[11px] tracking-[0.2em] uppercase mb-4"
            style={{
              color: "rgb(99,140,255)",
              fontFamily: "var(--font-geist-mono)",
            }}
          >
            Capabilities
          </p>
          <h2
            style={{
              fontSize: "clamp(1.75rem, 4vw, 2.5rem)",
              fontWeight: 500,
              letterSpacing: "-0.03em",
              lineHeight: 1.15,
              color: "var(--fg)",
              marginBottom: "1rem",
            }}
          >
            Everything you need for
            <br />
            network intelligence.
          </h2>
          <p
            style={{
              fontSize: "1.05rem",
              lineHeight: 1.6,
              color: "var(--fg-muted)",
              maxWidth: "480px",
              margin: "0 auto",
            }}
          >
            Each component is purpose-built for high-throughput classification
            and real-time observability.
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-px"
             style={{ background: "var(--border)", borderRadius: "16px", overflow: "hidden" }}>
          {features.map((f, i) => (
            <div
              key={f.label}
              className="group"
              style={{
                background: "var(--bg)",
                padding: "32px 28px",
                opacity: visible ? 1 : 0,
                transform: visible ? "translateY(0)" : "translateY(16px)",
                transition: `all 0.6s cubic-bezier(0.16, 1, 0.3, 1) ${i * 0.07}s`,
                cursor: "default",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--bg-muted)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "var(--bg)";
              }}
            >
              {/* Icon */}
              <div
                className="mb-5 transition-colors duration-200"
                style={{ color: "var(--fg-subtle)" }}
              >
                {f.icon}
              </div>

              {/* Label */}
              <p
                className="text-[10px] tracking-[0.15em] uppercase mb-2.5"
                style={{
                  color: "rgb(99,140,255)",
                  fontFamily: "var(--font-geist-mono)",
                }}
              >
                {f.label}
              </p>

              {/* Title */}
              <h3
                className="text-[15px] font-semibold mb-2"
                style={{
                  color: "var(--fg)",
                  letterSpacing: "-0.01em",
                }}
              >
                {f.title}
              </h3>

              {/* Description */}
              <p
                className="text-[13.5px] leading-[1.65]"
                style={{ color: "var(--fg-muted)" }}
              >
                {f.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
