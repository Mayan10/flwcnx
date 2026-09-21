'use client';

/**
 * Live charts for the ML monitor. Plain SVG, sized to the container in real
 * pixels (not a stretched viewBox) so text and 2px strokes stay crisp at any
 * width. Colours come from the page's CSS tokens and follow the theme toggle.
 */

import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ML_WINDOW, type DecisionRecord } from '@/lib/ml-stream';

const ACTUAL = 'var(--chart-muted)';
const FORECAST = 'var(--series-1)';
const BOUND = 'var(--series-3)';
const RISK = 'var(--series-2)';

/** Seconds of history on the x axis: grows with the data from one minute up to
 *  the full window, so a fresh stream is not squeezed against the right edge. */
function domainFor(n: number) {
  return Math.min(Math.max(n, 60), ML_WINDOW);
}

/** Measures an element's content width and keeps it current. */
function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Round tick values: 0 / 50 / 100 rather than 37.4 / 81.9. */
function niceTicks(lo: number, hi: number, count = 5): number[] {
  const span = Math.max(hi - lo, 1);
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Math.round(v * 1000) / 1000);
  return ticks;
}

/** "Seconds ago" ticks on round marks (15 s, 30 s or 1 min apart), oldest first. */
function timeTicks(domain: number): number[] {
  const step = domain <= 90 ? 15 : domain <= 180 ? 30 : 60;
  const ticks: number[] = [];
  // Tick k marks the decision (k * step) seconds before the newest one.
  for (let ago = 0; ago <= domain - 1; ago += step) ticks.push(ago === 0 ? 0 : ago - 1);
  return ticks.reverse();
}

function fmtAgo(seconds: number) {
  if (seconds < 90) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s ? `${m}m${String(s).padStart(2, '0')}` : `${m}m`;
}

function fmt(v: number, digits = 1) {
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/* ── Throughput, forecast and safe bound ──────────────────────────────────── */

export function ThroughputChart({ records, height = 360 }: { records: DecisionRecord[]; height?: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const pad = { top: 22, right: 16, bottom: 26, left: 52 };
  const riskStrip = 10; // the over-promise ticks sit in their own band under the plot
  const plotW = Math.max(width - pad.left - pad.right, 0);
  const plotH = height - pad.top - pad.bottom - riskStrip - 6;
  const n = records.length;

  // Newest at the right edge. Once the window is full the scale stops changing,
  // so a point does not slide around under the reader.
  const domain = domainFor(n);
  const x = (i: number) => pad.left + ((domain - n + i) / (domain - 1)) * plotW;

  const values = records.flatMap((r) => [r.actual_mbps, r.predicted_mbps, r.bound_mbps]);
  const ticks = niceTicks(Math.min(...values) - 5, Math.max(...values) + 5);
  const yLo = ticks[0];
  const yHi = ticks[ticks.length - 1];
  const y = (v: number) => pad.top + plotH - ((v - yLo) / (yHi - yLo)) * plotH;

  const path = (key: 'actual_mbps' | 'predicted_mbps' | 'bound_mbps') =>
    records.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(r[key]).toFixed(1)}`).join('');
  const boundWash = n
    ? `${path('bound_mbps')}L${x(n - 1).toFixed(1)},${pad.top + plotH}L${x(0).toFixed(1)},${pad.top + plotH}Z`
    : '';

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const bounds = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - bounds.left;
    const s = Math.round((px / plotW) * (domain - 1));
    const i = s - (domain - n);
    setHover(i >= 0 && i < n ? i : null);
  };

  const latest = records.at(-1);
  const hovered = hover !== null ? records[hover] : null;
  const stripY = pad.top + plotH + 6;

  return (
    <div ref={ref} className="relative">
      <LiveLegend
        items={[
          { label: 'Actual throughput', color: ACTUAL, value: latest?.actual_mbps, width: 2 },
          { label: 'Point forecast', color: FORECAST, value: latest?.predicted_mbps, width: 2 },
          { label: 'Safe bound', color: BOUND, value: latest?.bound_mbps, width: 3 },
          { label: 'Over-promise (bound > actual)', color: RISK, tick: true },
        ]}
      />

      {width > 0 && n >= 2 ? (
        <svg
          width={width}
          height={height}
          className="block mt-3"
          role="img"
          aria-label="Actual throughput, point forecast and safe bound over the last five minutes"
        >
          {ticks.map((v) => (
            <g key={v}>
              <line x1={pad.left} x2={pad.left + plotW} y1={y(v)} y2={y(v)} stroke="var(--chart-grid)" strokeWidth={1} />
              <text
                x={pad.left - 8}
                y={y(v) + 4}
                textAnchor="end"
                fontSize={11}
                fill="var(--chart-muted)"
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {Math.round(v)}
              </text>
            </g>
          ))}
          <text x={pad.left - 8} y={pad.top - 12} textAnchor="end" fontSize={10.5} fill="var(--chart-muted)">
            Mbps
          </text>

          <path d={boundWash} fill={BOUND} opacity={0.1} />
          <path d={path('actual_mbps')} fill="none" stroke={ACTUAL} strokeWidth={1.5} strokeLinejoin="round" />
          <path d={path('predicted_mbps')} fill="none" stroke={FORECAST} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          <path d={path('bound_mbps')} fill="none" stroke={BOUND} strokeWidth={2.5} strokeLinejoin="round" strokeLinecap="round" />

          {/* End dots with a surface ring mark "now" on each series. */}
          {latest &&
            (['actual_mbps', 'predicted_mbps', 'bound_mbps'] as const).map((key, k) => (
              <circle
                key={key}
                cx={x(n - 1)}
                cy={y(latest[key])}
                r={4}
                fill={[ACTUAL, FORECAST, BOUND][k]}
                stroke="var(--bg-elevated)"
                strokeWidth={2}
              />
            ))}

          {/* Over-promise band: one tick per decision where the bound exceeded reality. */}
          <line x1={pad.left} x2={pad.left + plotW} y1={stripY + riskStrip} y2={stripY + riskStrip} stroke="var(--chart-axis)" strokeWidth={1} />
          {records.map((r, i) =>
            r.risk_event ? (
              <rect key={r.step} x={x(i) - 1} y={stripY} width={2} height={riskStrip} rx={1} fill={RISK} />
            ) : null,
          )}

          {timeTicks(domain).map((ago) => {
            const tx = pad.left + ((domain - 1 - ago) / (domain - 1)) * plotW;
            return (
              <text
                key={ago}
                x={tx}
                y={height - 6}
                textAnchor={ago === 0 ? 'end' : tx - pad.left < 24 ? 'start' : 'middle'}
                fontSize={11}
                fill="var(--chart-muted)"
                style={{ fontVariantNumeric: 'tabular-nums' }}
              >
                {ago === 0 ? 'now' : `-${fmtAgo(ago + 1)}`}
              </text>
            );
          })}

          {hovered && hover !== null && (
            <>
              <line x1={x(hover)} x2={x(hover)} y1={pad.top} y2={stripY + riskStrip} stroke="var(--fg-muted)" strokeWidth={1} />
              {(['actual_mbps', 'predicted_mbps', 'bound_mbps'] as const).map((key, k) => (
                <circle
                  key={key}
                  cx={x(hover)}
                  cy={y(hovered[key])}
                  r={4.5}
                  fill={[ACTUAL, FORECAST, BOUND][k]}
                  stroke="var(--bg-elevated)"
                  strokeWidth={2}
                />
              ))}
            </>
          )}

          <rect
            x={pad.left}
            y={pad.top}
            width={plotW}
            height={plotH + riskStrip + 6}
            fill="transparent"
            onPointerMove={onMove}
            onPointerLeave={() => setHover(null)}
          />
        </svg>
      ) : (
        <div
          className="flex items-center justify-center text-[13px] mt-3"
          style={{ height, color: 'var(--fg-muted)' }}
        >
          Waiting for the ML data stream…
        </div>
      )}

      {hovered && hover !== null && (
        <ChartTooltip x={x(hover)} width={width}>
          <div className="mb-1" style={{ color: 'var(--fg-muted)' }}>
            Step {hovered.step} · {new Date(hovered.timestamp).toLocaleTimeString()}
          </div>
          <TooltipRow color={ACTUAL} label="Actual" value={`${fmt(hovered.actual_mbps)} Mbps`} />
          <TooltipRow color={FORECAST} label="Forecast" value={`${fmt(hovered.predicted_mbps)} Mbps`} />
          <TooltipRow color={BOUND} label="Safe bound" value={`${fmt(hovered.bound_mbps)} Mbps`} />
          <div className="mt-1 pt-1" style={{ borderTop: '1px solid var(--border)', color: 'var(--fg-muted)' }}>
            {hovered.regime} · {hovered.admitted} admitted
            {hovered.risk_event ? ' · over-promise' : ''}
          </div>
        </ChartTooltip>
      )}
    </div>
  );
}

/* ── Session counts (admitted / dropped / unused) ─────────────────────────── */

export function SessionSeries({
  records,
  field,
  color,
  height = 72,
}: {
  records: DecisionRecord[];
  field: 'admitted' | 'dropped' | 'unused';
  color: string;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const n = records.length;
  const pad = { top: 6, right: 6, bottom: 6, left: 0 };
  const plotW = Math.max(width - pad.left - pad.right, 0);
  const plotH = height - pad.top - pad.bottom;
  const max = Math.max(1, ...records.map((r) => r[field]));
  const domain = domainFor(n);
  const x = (i: number) => pad.left + ((domain - n + i) / (domain - 1)) * plotW;
  const y = (v: number) => pad.top + plotH - (v / max) * plotH;

  const line = records.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(r[field]).toFixed(1)}`).join('');
  const wash = n ? `${line}L${x(n - 1).toFixed(1)},${pad.top + plotH}L${x(0).toFixed(1)},${pad.top + plotH}Z` : '';

  return (
    <div ref={ref}>
      {width > 0 && n >= 2 ? (
        <svg width={width} height={height} className="block" aria-hidden>
          <line x1={0} x2={width} y1={pad.top + plotH} y2={pad.top + plotH} stroke="var(--chart-grid)" strokeWidth={1} />
          <path d={wash} fill={color} opacity={0.1} />
          <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
          <circle cx={x(n - 1)} cy={y(records[n - 1][field])} r={4} fill={color} stroke="var(--bg-elevated)" strokeWidth={2} />
        </svg>
      ) : (
        <div style={{ height }} />
      )}
    </div>
  );
}

/* ── Small shared pieces ──────────────────────────────────────────────────── */

function LiveLegend({
  items,
}: {
  items: { label: string; color: string; value?: number; width?: number; tick?: boolean }[];
}) {
  return (
    <ul className="flex flex-wrap gap-x-5 gap-y-2 list-none p-0 m-0">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-2 text-[12px]">
          {item.tick ? (
            <span aria-hidden className="inline-block rounded-[1px]" style={{ width: 2, height: 10, background: item.color }} />
          ) : (
            <span aria-hidden className="inline-block rounded-full" style={{ width: 14, height: item.width ?? 2, background: item.color }} />
          )}
          <span style={{ color: 'var(--fg-muted)' }}>{item.label}</span>
          {item.value !== undefined && (
            <span className="font-medium" style={{ color: 'var(--fg)', fontVariantNumeric: 'tabular-nums' }}>
              {fmt(item.value)}
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}

function ChartTooltip({ x, width, children }: { x: number; width: number; children: ReactNode }) {
  const flip = x > width * 0.62;
  return (
    <div
      className="pointer-events-none absolute z-20 rounded-lg px-3 py-2 text-[11.5px] leading-relaxed whitespace-nowrap"
      style={{
        left: flip ? undefined : x + 12,
        right: flip ? width - x + 12 : undefined,
        top: 48,
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border-strong)',
        boxShadow: 'var(--shadow-md)',
      }}
    >
      {children}
    </div>
  );
}

function TooltipRow({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span aria-hidden className="inline-block rounded-full" style={{ width: 10, height: 2, background: color }} />
      <span className="font-semibold" style={{ color: 'var(--fg)', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
      <span style={{ color: 'var(--fg-muted)' }}>{label}</span>
    </div>
  );
}
