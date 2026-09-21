"use client";

import { useEffect, useRef, useState } from "react";

/* ── Architecture layers ───────────────────────────────────────────────────── */

const layers = [
  {
    name: "Capture Layer",
    tag: "L0",
    items: ["eBPF probes", "AF_PACKET sockets", "pcap integration"],
    color: "rgb(72,199,162)",
  },
  {
    name: "Processing Pipeline",
    tag: "L1",
    items: ["Flow assembly", "Feature extraction", "Protocol dissection"],
    color: "rgb(99,140,255)",
  },
  {
    name: "Classification Engine",
    tag: "L2",
    items: ["Transformer inference", "Ensemble voting", "Confidence scoring"],
    color: "rgb(99,140,255)",
  },
  {
    name: "Telemetry & Export",
    tag: "L3",
    items: ["Prometheus metrics", "OTLP export", "gRPC streaming"],
    color: "rgb(72,199,162)",
  },
];

const stats = [
  { value: "<1ms", label: "Classification latency" },
  { value: "10Gbps", label: "Wire-speed capture" },
  { value: "97.3%", label: "Accuracy on encrypted flows" },
  { value: "~4MB", label: "Agent memory footprint" },
];

/* ── Component ─────────────────────────────────────────────────────────────── */

export function Architecture() {
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
      id="architecture"
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
            System Design
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
            Layered for reliability.
            <br />
            Optimized for speed.
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
            Four composable layers, each independently deployable and
            horizontally scalable.
          </p>
        </div>

        {/* Architecture stack */}
        <div className="max-w-[720px] mx-auto mb-24">
          {layers.map((layer, i) => (
            <div
              key={layer.tag}
              className="relative group"
              style={{
                opacity: visible ? 1 : 0,
                transform: visible ? "translateY(0)" : "translateY(12px)",
                transition: `all 0.5s cubic-bezier(0.16, 1, 0.3, 1) ${0.15 + i * 0.08}s`,
              }}
            >
              {/* Connector line */}
              {i > 0 && (
                <div
                  className="mx-auto"
                  style={{
                    width: "1px",
                    height: "24px",
                    background: "var(--border)",
                  }}
                />
              )}

              {/* Layer card */}
              <div
                className="flex items-stretch rounded-xl overflow-hidden transition-all duration-200"
                style={{
                  border: "1px solid var(--border)",
                  background: "var(--bg)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "var(--border-strong)";
                  e.currentTarget.style.background = "var(--bg-muted)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.background = "var(--bg)";
                }}
              >
                {/* Tag */}
                <div
                  className="flex items-center justify-center px-5 shrink-0"
                  style={{
                    borderRight: "1px solid var(--border)",
                    fontFamily: "var(--font-geist-mono)",
                    fontSize: "12px",
                    fontWeight: 600,
                    color: layer.color,
                    minWidth: "60px",
                  }}
                >
                  {layer.tag}
                </div>

                {/* Content */}
                <div className="flex-1 px-5 py-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <h3
                      className="text-[14px] font-semibold"
                      style={{ color: "var(--fg)", letterSpacing: "-0.01em" }}
                    >
                      {layer.name}
                    </h3>
                    <div className="flex items-center gap-3 flex-wrap">
                      {layer.items.map((item) => (
                        <span
                          key={item}
                          className="text-[11px] px-2 py-0.5 rounded"
                          style={{
                            color: "var(--fg-muted)",
                            background: "var(--bg-subtle)",
                            border: "1px solid var(--border)",
                            fontFamily: "var(--font-geist-mono)",
                          }}
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Stats row */}
        <div
          className="grid grid-cols-2 lg:grid-cols-4 gap-px rounded-xl overflow-hidden"
          style={{
            background: "var(--border)",
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(16px)",
            transition: "all 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.5s",
          }}
        >
          {stats.map((s) => (
            <div
              key={s.label}
              className="text-center py-8 px-4"
              style={{ background: "var(--bg)" }}
            >
              <div
                className="text-[28px] sm:text-[32px] font-semibold mb-1"
                style={{
                  color: "var(--fg)",
                  letterSpacing: "-0.03em",
                  fontFamily: "var(--font-geist-mono)",
                }}
              >
                {s.value}
              </div>
              <div
                className="text-[12px]"
                style={{ color: "var(--fg-muted)" }}
              >
                {s.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
