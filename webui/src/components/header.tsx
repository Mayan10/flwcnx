"use client";

import { useTheme } from "./theme-provider";

export function Header() {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 py-5 px-6 pointer-events-none">
      <div 
        className="max-w-[1000px] mx-auto flex items-center justify-between px-5 h-14 rounded-full pointer-events-auto transition-all duration-300"
        style={{ 
          background: "color-mix(in srgb, var(--bg) 85%, transparent)",
          border: "1px solid var(--border)",
          boxShadow: "var(--shadow-sm)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)"
        }}
      >
        <div className="flex items-center gap-3">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--fg)" }}>
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
          <span className="text-[15px] font-semibold tracking-tight" style={{ color: "var(--fg)" }}>
            FlowConX
          </span>
        </div>
        
        <nav className="hidden md:flex items-center gap-8">
          {["Platform", "Solutions", "Developers"].map((item) => (
            <a key={item} href="#" className="text-[13.5px] font-medium transition-colors duration-200 no-underline" style={{ color: "var(--fg-muted)" }}
               onMouseEnter={e => e.currentTarget.style.color = "var(--fg)"}
               onMouseLeave={e => e.currentTarget.style.color = "var(--fg-muted)"}
            >
              {item}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-5">
          <button onClick={toggleTheme} className="text-[13.5px] font-medium transition-colors duration-200 cursor-pointer" style={{ color: "var(--fg-muted)", background: "none", border: "none" }}
               onMouseEnter={e => e.currentTarget.style.color = "var(--fg)"}
               onMouseLeave={e => e.currentTarget.style.color = "var(--fg-muted)"}>
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <div className="w-px h-4" style={{ background: "var(--border)" }} />
          <a href="/login" className="text-[13.5px] font-medium px-4 py-1.5 rounded-full transition-all duration-200 no-underline"
             style={{ background: "transparent", color: "var(--fg)", border: "1px solid var(--border)" }}
             onMouseEnter={e => { e.currentTarget.style.background = "var(--fg)"; e.currentTarget.style.color = "var(--bg)"; }}
             onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--fg)"; }}>
            Sign in
          </a>
          <a href="/register" className="text-[13.5px] font-medium px-4 py-1.5 rounded-full transition-all duration-200 no-underline"
             style={{ background: "var(--fg)", color: "var(--bg)" }}
             onMouseEnter={e => e.currentTarget.style.opacity = "0.85"}
             onMouseLeave={e => e.currentTarget.style.opacity = "1"}>
            Register
          </a>
        </div>
      </div>
    </header>
  );
}
