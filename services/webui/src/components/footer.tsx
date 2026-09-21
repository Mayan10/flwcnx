"use client";

import { useState } from "react";

export function Footer() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email.trim()) {
      setSubmitted(true);
      setEmail("");
    }
  };

  return (
    <footer style={{ background: "var(--bg)", borderTop: "1px solid var(--border)" }}>
      {/* CTA Section */}
      <div id="early-access" className="py-28 sm:py-40 px-6">
        <div className="max-w-[520px] mx-auto text-center">
          <p
            className="text-[11px] tracking-[0.2em] uppercase mb-4"
            style={{
              color: "rgb(99,140,255)",
              fontFamily: "var(--font-geist-mono)",
            }}
          >
            Early Access
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
            Be the first to deploy FlowConX.
          </h2>
          <p
            className="mb-10"
            style={{
              fontSize: "1.05rem",
              lineHeight: 1.6,
              color: "var(--fg-muted)",
            }}
          >
            We&apos;re onboarding design partners for private beta.
            Drop your email — no spam, only meaningful updates.
          </p>

          {submitted ? (
            <div
              className="inline-flex items-center gap-3 px-6 py-3.5 rounded-full"
              style={{
                border: "1px solid var(--border)",
                fontFamily: "var(--font-geist-mono)",
                fontSize: "13px",
                color: "var(--fg-muted)",
              }}
            >
              <span style={{ color: "rgb(72,199,162)" }}>✓</span>
              You&apos;re on the list. We&apos;ll be in touch.
            </div>
          ) : (
            <form
              onSubmit={handleSubmit}
              className="flex flex-col sm:flex-row items-stretch gap-3 max-w-[420px] mx-auto"
            >
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                required
                className="flex-1 px-4 py-3 rounded-full text-[13px] outline-none transition-all duration-200"
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  color: "var(--fg)",
                  fontFamily: "var(--font-geist-mono)",
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = "rgb(99,140,255)";
                  e.currentTarget.style.boxShadow = "0 0 0 3px rgba(99,140,255,0.1)";
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.boxShadow = "none";
                }}
              />
              <button
                type="submit"
                className="px-6 py-3 rounded-full text-[13px] font-medium transition-all duration-200 cursor-pointer whitespace-nowrap"
                style={{
                  background: "var(--fg)",
                  color: "var(--bg)",
                  border: "none",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.opacity = "0.85";
                  e.currentTarget.style.transform = "translateY(-1px)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.opacity = "1";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                Request Access
              </button>
            </form>
          )}
        </div>
      </div>

      {/* Bottom bar */}
      <div
        className="px-6 py-6"
        style={{ borderTop: "1px solid var(--border)" }}
      >
        <div className="max-w-[1080px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Left: brand */}
          <div className="flex items-center gap-2.5">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--fg-subtle)" }}>
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            <span
              className="text-[13px] font-semibold"
              style={{ color: "var(--fg-subtle)", fontFamily: "var(--font-geist-mono)" }}
            >
              FlowConX
            </span>
          </div>

          {/* Center: links */}
          <div className="flex items-center gap-6">
            {["Documentation", "GitHub", "Twitter"].map((link) => (
              <a
                key={link}
                href="#"
                className="text-[12px] no-underline transition-colors duration-200"
                style={{ color: "var(--fg-subtle)" }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "var(--fg)")}
                onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-subtle)")}
              >
                {link}
              </a>
            ))}
          </div>

          {/* Right: copyright */}
          <span className="text-[12px]" style={{ color: "var(--fg-ghost)" }}>
            © {new Date().getFullYear()} FlowConX
          </span>
        </div>
      </div>
    </footer>
  );
}
