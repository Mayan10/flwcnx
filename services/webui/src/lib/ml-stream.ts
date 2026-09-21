'use client';

/**
 * Live decision stream from the flwcnx ML API (`flwcnx/api`).
 *
 * The same WebSocket the terminal UI reads: an optional `status` while the
 * forecaster trains, one `init`, then a `decision` per second. The hook keeps a
 * rolling window plus running totals, and reconnects with exponential backoff
 * exactly like `tui/src/ml_client.rs`, so the two views always agree.
 */

import { useEffect, useRef, useState } from 'react';

export const ML_WS_URL =
  process.env.NEXT_PUBLIC_ML_WS_URL || 'ws://localhost:8010/ws/ml/stream';

/** Decisions kept for the charts (~5 minutes at 1 Hz), same as the TUI. */
export const ML_WINDOW = 300;

const MAX_BACKOFF_MS = 30_000;

export interface DecisionRecord {
  step: number;
  timestamp: string;
  regime: string;
  predicted_mbps: number;
  bound_mbps: number;
  actual_mbps: number;
  offset_mbps: number;
  alpha: number;
  admitted: number;
  oracle: number;
  dropped: number;
  unused: number;
  risk_event: boolean;
  congested: boolean;
  realised_risk_rate: number;
}

export type MlConnection =
  | { kind: 'connecting' }
  | { kind: 'training'; location: string; mode: string }
  | { kind: 'connected'; location: string; mode: string; epsilon: number }
  | { kind: 'disconnected'; reason: string };

export interface MlTotals {
  decisions: number;
  riskEvents: number;
  dropped: number;
  congestedSlots: number;
}

export interface MlStream {
  connection: MlConnection;
  records: DecisionRecord[];
  latest: DecisionRecord | null;
  totals: MlTotals;
}

const EMPTY_TOTALS: MlTotals = { decisions: 0, riskEvents: 0, dropped: 0, congestedSlots: 0 };

export function useMlStream(url: string = ML_WS_URL): MlStream {
  const [connection, setConnection] = useState<MlConnection>({ kind: 'connecting' });
  const [records, setRecords] = useState<DecisionRecord[]>([]);
  const [totals, setTotals] = useState<MlTotals>(EMPTY_TOTALS);
  const backoff = useRef(1_000);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      setConnection({ kind: 'connecting' });
      socket = new WebSocket(url);

      socket.onmessage = (event) => {
        let msg: Record<string, unknown>;
        try {
          msg = JSON.parse(event.data as string);
        } catch {
          return;
        }
        switch (msg.type) {
          case 'status':
            setConnection({ kind: 'training', location: String(msg.location), mode: String(msg.mode) });
            break;
          case 'init':
            backoff.current = 1_000;
            // A new session restarts the step counter, so the window restarts too.
            setRecords([]);
            setTotals(EMPTY_TOTALS);
            setConnection({
              kind: 'connected',
              location: String(msg.location),
              mode: String(msg.mode),
              epsilon: Number(msg.epsilon),
            });
            break;
          case 'decision': {
            const rec = msg as unknown as DecisionRecord;
            setRecords((prev) => {
              const next = prev.length >= ML_WINDOW ? prev.slice(prev.length - ML_WINDOW + 1) : prev.slice();
              next.push(rec);
              return next;
            });
            setTotals((t) => ({
              decisions: t.decisions + 1,
              riskEvents: t.riskEvents + (rec.risk_event ? 1 : 0),
              dropped: t.dropped + rec.dropped,
              congestedSlots: t.congestedSlots + (rec.congested ? 1 : 0),
            }));
            break;
          }
          case 'error':
            setConnection({ kind: 'disconnected', reason: String(msg.message ?? 'server error') });
            break;
        }
      };

      socket.onclose = () => {
        if (closed) return;
        setConnection({ kind: 'disconnected', reason: 'connection lost' });
        retry = setTimeout(connect, backoff.current);
        backoff.current = Math.min(backoff.current * 2, MAX_BACKOFF_MS);
      };
    };

    connect();
    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      socket?.close();
    };
  }, [url]);

  return { connection, records, latest: records.at(-1) ?? null, totals };
}
