"use client";

/**
 * Shared chart chrome. The live ML charts themselves are in
 * `@/components/ml/live-charts`.
 */

import type { ReactNode } from "react";

/* ── Shared chrome ──────────────────────────────────────────────────────── */

export function ChartFrame({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl p-5 ${className}`}
      style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        boxShadow: "var(--shadow-sm)",
      }}
    >
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <h3
            className="text-[14px] font-semibold tracking-tight m-0"
            style={{ color: "var(--fg)" }}
          >
            {title}
          </h3>
          {subtitle && (
            <p
              className="text-[12px] mt-1 m-0"
              style={{ color: "var(--fg-muted)" }}
            >
              {subtitle}
            </p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}
