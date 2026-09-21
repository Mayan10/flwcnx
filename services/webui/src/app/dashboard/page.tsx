'use client';

/**
 * The web console: the account's API key and the live ML monitor.
 *
 * The monitor mirrors the terminal UI's ML screen panel for panel (connection,
 * risk rate, safe bound, dropped, congestion, the throughput chart, covariates
 * and session counts), fed by the same WebSocket from the flwcnx ML API, so a
 * number read here matches the number in the terminal.
 */

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { fetchWithAuth } from '@/lib/api';
import { useMlStream, ML_WINDOW, type DecisionRecord, type MlConnection } from '@/lib/ml-stream';
import { useTheme } from '@/components/theme-provider';
import { ChartFrame } from '@/components/charts/primitives';
import { SessionSeries, ThroughputChart } from '@/components/ml/live-charts';

interface Company {
  id: string;
  name: string;
  domain: string;
  city: string;
  apiKey: string;
}

export default function Dashboard() {
  const router = useRouter();
  const { theme, toggleTheme } = useTheme();

  const [company, setCompany] = useState<Company | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchWithAuth('/company/me')
      .then(setCompany)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load account');
        router.push('/login');
      });
  }, [router]);

  if (error) {
    return (
      <Centered>
        <span style={{ color: 'var(--status-critical)' }}>{error}</span>
      </Centered>
    );
  }

  if (!company) {
    return (
      <Centered>
        <span style={{ color: 'var(--fg-muted)' }}>Loading…</span>
      </Centered>
    );
  }

  return (
    <div className="min-h-screen noise" style={{ background: 'var(--bg-subtle)' }}>
      {/* ── Top bar ───────────────────────────────────────────────────── */}
      <header
        className="sticky top-0 z-30"
        style={{
          background: 'color-mix(in srgb, var(--bg) 88%, transparent)',
          borderBottom: '1px solid var(--border)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
        }}
      >
        <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2.5 no-underline shrink-0">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--fg)' }}>
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
            <span className="text-[15px] font-semibold tracking-tight" style={{ color: 'var(--fg)' }}>
              FlowConX
            </span>
          </Link>

          <span className="w-px h-5 shrink-0" style={{ background: 'var(--border)' }} />

          <div className="min-w-0">
            <div className="text-[13.5px] font-medium truncate" style={{ color: 'var(--fg)' }}>
              {company.name}
            </div>
            <div className="text-[11.5px] truncate" style={{ color: 'var(--fg-subtle)' }}>
              {company.domain} · {company.city}
            </div>
          </div>

          <div className="ml-auto flex items-center gap-3">
            <button
              onClick={toggleTheme}
              className="text-[12.5px] font-medium cursor-pointer px-3 py-1.5 rounded-full"
              style={{ color: 'var(--fg-muted)', background: 'none', border: '1px solid var(--border)' }}
            >
              {theme === 'dark' ? 'Light' : 'Dark'}
            </button>
            <button
              onClick={() => {
                localStorage.removeItem('token');
                router.push('/login');
              }}
              className="text-[12.5px] font-medium px-3.5 py-1.5 rounded-full cursor-pointer"
              style={{ background: 'var(--fg)', color: 'var(--bg)', border: 'none' }}
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-6 py-8 relative z-10 grid gap-6">
        <ApiKeyCard apiKey={company.apiKey} />
        <MlMonitor />
      </main>
    </div>
  );
}

/* ── API key ─────────────────────────────────────────────────────────────── */

function ApiKeyCard({ apiKey }: { apiKey: string }) {
  const [visible, setVisible] = useState(false);
  const [copied, setCopied] = useState(false);

  return (
    <ChartFrame title="API key">
      <div className="flex flex-wrap items-center gap-2">
        <code
          className="flex-1 min-w-[220px] px-3 py-2.5 rounded-lg text-[12.5px] break-all"
          style={{ background: 'var(--bg-muted)', border: '1px solid var(--border)', color: 'var(--fg)', fontFamily: 'var(--font-geist-mono), monospace' }}
        >
          {visible ? apiKey : '•'.repeat(Math.min(44, apiKey.length))}
        </code>
        <button
          onClick={() => setVisible((v) => !v)}
          className="text-[12.5px] font-medium px-3.5 py-2 rounded-lg cursor-pointer"
          style={{ background: 'transparent', color: 'var(--fg)', border: '1px solid var(--border)' }}
        >
          {visible ? 'Hide' : 'Reveal'}
        </button>
        <button
          onClick={() => {
            navigator.clipboard.writeText(apiKey);
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
          }}
          className="text-[12.5px] font-medium px-3.5 py-2 rounded-lg cursor-pointer"
          style={{ background: 'var(--fg)', color: 'var(--bg)', border: 'none' }}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </ChartFrame>
  );
}

/* ── Live ML monitor ─────────────────────────────────────────────────────── */

function MlMonitor() {
  const { connection, records, latest, totals } = useMlStream();
  const [showTable, setShowTable] = useState(false);

  const conn = connectionView(connection);
  const epsilon = connection.kind === 'connected' ? connection.epsilon : 0.35;
  const riskRate = latest?.realised_risk_rate ?? 0;
  const overBudget = riskRate > epsilon;

  return (
    <section className="grid gap-6">
      {/* ── KPI tiles, same five as the terminal ─────────────────────── */}
      <div className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        <Tile label="Connection" value={conn.label} note={`${totals.decisions.toLocaleString()} decisions`} status={conn.status} />
        <Tile
          label="Risk rate"
          value={riskRate.toFixed(3)}
          note={`${totals.riskEvents.toLocaleString()} violations · budget ${epsilon.toFixed(2)}`}
          status={latest ? (overBudget ? 'critical' : 'good') : undefined}
          statusText={latest ? (overBudget ? 'Over budget' : 'Within budget') : undefined}
        />
        <Tile
          label="Safe bound"
          value={latest ? `${latest.bound_mbps.toFixed(1)} Mbps` : '—'}
          note={latest ? `${latest.admitted}/${latest.oracle} sessions` : '—'}
        />
        <Tile
          label="Dropped"
          value={latest ? String(latest.dropped) : '—'}
          note={`${totals.dropped.toLocaleString()} total`}
          status={latest ? (latest.dropped > 0 ? 'critical' : 'good') : undefined}
          statusText={latest ? (latest.dropped > 0 ? 'Sessions dropped' : 'None dropped') : undefined}
        />
        <Tile
          label="Congestion"
          value={latest ? (latest.congested ? 'Detected' : 'Healthy') : '—'}
          note={`${totals.congestedSlots.toLocaleString()} slots`}
          status={latest ? (latest.congested ? 'critical' : 'good') : undefined}
        />
      </div>

      {/* ── Chart + covariates ───────────────────────────────────────── */}
      <div className="grid gap-6 grid-cols-1 xl:grid-cols-[1fr_300px]">
        <ChartFrame
          title="Throughput, forecast and safe bound"
          actions={
            <button
              onClick={() => setShowTable((v) => !v)}
              className="text-[12px] font-medium px-3 py-1.5 rounded-full cursor-pointer shrink-0"
              style={{ background: 'transparent', color: 'var(--fg-muted)', border: '1px solid var(--border)' }}
            >
              {showTable ? 'Chart view' : 'Table view'}
            </button>
          }
        >
          {showTable ? <DecisionTable records={records} /> : <ThroughputChart records={records} />}
        </ChartFrame>

        <ChartFrame title="Covariates">
          <dl className="m-0 grid gap-3">
            <Covariate label="Regime" value={latest?.regime ?? '—'} />
            <Covariate label="Alpha" value={latest ? latest.alpha.toFixed(4) : '—'} />
            <Covariate label="Offset" value={latest ? `${latest.offset_mbps.toFixed(1)} Mbps` : '—'} />
            <Covariate label="Step" value={latest ? `#${latest.step}` : '—'} />
            <Covariate label="Risk events" value={totals.riskEvents.toLocaleString()} />
            <Covariate label="Total decisions" value={totals.decisions.toLocaleString()} />
          </dl>
        </ChartFrame>
      </div>

      {/* ── Session counts ───────────────────────────────────────────── */}
      <div className="grid gap-6 grid-cols-1 md:grid-cols-3">
        {(
          [
            { field: 'admitted', title: 'Admitted', color: 'var(--series-3)' },
            { field: 'dropped', title: 'Dropped', color: 'var(--series-2)' },
            { field: 'unused', title: 'Unused', color: 'var(--series-4)' },
          ] as const
        ).map(({ field, title, color }) => (
          <ChartFrame key={field} title={title}>
            <div className="text-[26px] font-medium leading-none tracking-tight mb-3" style={{ color: 'var(--fg)' }}>
              {latest ? latest[field] : '—'}
              <span className="text-[13px] font-normal ml-1.5" style={{ color: 'var(--fg-muted)' }}>
                sessions
              </span>
            </div>
            <SessionSeries records={records} field={field} color={color} />
          </ChartFrame>
        ))}
      </div>
    </section>
  );
}

function connectionView(c: MlConnection): { label: string; status: Status } {
  switch (c.kind) {
    case 'connected':
      return { label: `${c.location.toUpperCase()} (${c.mode})`, status: 'good' };
    case 'training':
      return { label: 'Training model…', status: 'warning' };
    case 'connecting':
      return { label: 'Connecting…', status: 'warning' };
    case 'disconnected':
      return { label: 'Disconnected', status: 'critical' };
  }
}

/* ── Small local building blocks ────────────────────────────────────────── */

type Status = 'good' | 'warning' | 'critical';

const STATUS: Record<Status, { color: string; glyph: string }> = {
  good: { color: 'var(--status-good)', glyph: '✓' },
  warning: { color: 'var(--status-warning)', glyph: '…' },
  critical: { color: 'var(--status-critical)', glyph: '!' },
};

function Tile({
  label,
  value,
  note,
  status,
  statusText,
}: {
  label: string;
  value: string;
  note: string;
  status?: Status;
  statusText?: string;
}) {
  const s = status ? STATUS[status] : null;
  return (
    <div
      className="rounded-xl p-4"
      style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', boxShadow: 'var(--shadow-sm)' }}
    >
      <div className="flex items-center justify-between gap-2 mb-2">
        <span className="text-[11.5px] font-medium" style={{ color: 'var(--fg-muted)' }}>
          {label}
        </span>
        {s && (
          // Status never rides on colour alone: glyph + text accompany the dot.
          <span className="inline-flex items-center gap-1 text-[11px] font-medium" style={{ color: 'var(--fg-muted)' }}>
            <span
              aria-hidden
              className="inline-flex items-center justify-center rounded-full text-[9px] font-bold"
              style={{ width: 14, height: 14, background: s.color, color: 'var(--bg)' }}
            >
              {s.glyph}
            </span>
            {statusText}
          </span>
        )}
      </div>
      <div className="text-[24px] font-medium leading-tight tracking-tight truncate" style={{ color: 'var(--fg)' }}>
        {value}
      </div>
      <div className="mt-1.5 text-[11.5px] truncate" style={{ color: 'var(--fg-subtle)' }}>
        {note}
      </div>
    </div>
  );
}

function Covariate({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 pb-3" style={{ borderBottom: '1px solid var(--border)' }}>
      <dt className="text-[12px]" style={{ color: 'var(--fg-muted)' }}>
        {label}
      </dt>
      <dd className="m-0 text-[13.5px] font-medium tabular-nums" style={{ color: 'var(--fg)' }}>
        {value}
      </dd>
    </div>
  );
}

function DecisionTable({ records }: { records: DecisionRecord[] }) {
  const rows = records.slice(-60).reverse();
  return (
    <div className="overflow-auto -mx-1 px-1" style={{ maxHeight: 400 }}>
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr>
            <Th>Step</Th>
            <Th>Time</Th>
            <Th>Regime</Th>
            <Th align="right">Actual</Th>
            <Th align="right">Forecast</Th>
            <Th align="right">Safe bound</Th>
            <Th align="right">Admitted</Th>
            <Th align="right">Dropped</Th>
            <Th>Over-promise</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.step}>
              <Td>{r.step}</Td>
              <Td muted>{new Date(r.timestamp).toLocaleTimeString()}</Td>
              <Td muted>{r.regime}</Td>
              <Td align="right">{r.actual_mbps.toFixed(1)}</Td>
              <Td align="right">{r.predicted_mbps.toFixed(1)}</Td>
              <Td align="right">{r.bound_mbps.toFixed(1)}</Td>
              <Td align="right">{r.admitted}</Td>
              <Td align="right">{r.dropped}</Td>
              <Td muted>{r.risk_event ? 'Yes' : 'No'}</Td>
            </tr>
          ))}
          {!rows.length && (
            <tr>
              <Td muted>Waiting for the ML data stream… (window holds {ML_WINDOW} decisions)</Td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex items-center justify-center text-[13.5px]" style={{ background: 'var(--bg)' }}>
      {children}
    </div>
  );
}

function Th({ children, align = 'left' }: { children: React.ReactNode; align?: 'left' | 'right' }) {
  return (
    <th
      className="text-[11px] font-medium uppercase tracking-wide pb-2 px-2 whitespace-nowrap sticky top-0"
      style={{ color: 'var(--fg-subtle)', textAlign: align, borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
    >
      {children}
    </th>
  );
}

function Td({ children, align = 'left', muted }: { children: React.ReactNode; align?: 'left' | 'right'; muted?: boolean }) {
  return (
    <td
      className="py-2 px-2 whitespace-nowrap tabular-nums"
      style={{ textAlign: align, color: muted ? 'var(--fg-muted)' : 'var(--fg)', borderBottom: '1px solid var(--border)' }}
    >
      {children}
    </td>
  );
}
