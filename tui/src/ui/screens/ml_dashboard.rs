//! ML Monitor — real-time throughput forecast and risk-control dashboard.
//!
//! This screen visualises the live output of the `flwcnx` ML pipeline:
//! the actual throughput, the point forecast, the risk-calibrated safe bound,
//! the admission control decisions, and congestion alerts.

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Chart, Dataset, GraphType, Paragraph, Sparkline,
};
use ratatui::Frame;

use crate::app::App;
use crate::ml_client::MlConnectionState;
use crate::ui::components::{chrome, widgets};
use crate::ui::theme;

pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    frame.render_widget(Block::default().style(theme::bg()), area);

    let rows = Layout::vertical([
        Constraint::Length(3),  // chrome
        Constraint::Length(5),  // KPI tiles
        Constraint::Min(0),    // body (chart + sidebar)
    ])
    .split(area);

    chrome::render(frame, app, rows[0]);
    render_kpis(frame, app, rows[1]);

    // Below the KPIs: throughput chart on the left, covariate panel on the right.
    let sidebar_width: u16 = if area.width >= 120 { 30 } else if area.width >= 90 { 24 } else { 0 };

    if sidebar_width == 0 {
        // Narrow terminal — chart takes everything, sparklines below.
        let body = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(rows[2]);
        render_throughput_chart(frame, app, body[0]);
        render_admission_sparklines(frame, app, body[1]);
        return;
    }

    let body = Layout::vertical([Constraint::Min(0), Constraint::Length(7)]).split(rows[2]);
    let cols = Layout::horizontal([
        Constraint::Min(0),
        Constraint::Length(sidebar_width),
    ])
    .split(body[0]);

    render_throughput_chart(frame, app, cols[0]);
    render_sidebar(frame, app, cols[1]);
    render_admission_sparklines(frame, app, body[1]);
}

// ── KPI Tiles ────────────────────────────────────────────────────────────────

fn render_kpis(frame: &mut Frame, app: &App, area: Rect) {
    let (risk_rate, bound_val, admitted, oracle, dropped, congested) =
        if let Some(ref rec) = app.ml_latest {
            (
                rec.realised_risk_rate,
                rec.bound_mbps,
                rec.admitted,
                rec.oracle,
                rec.dropped,
                rec.congested,
            )
        } else {
            (0.0, 0.0, 0, 0, 0, false)
        };

    let conn_label = match &app.ml_connection_state {
        MlConnectionState::Connected { location, mode } => {
            format!("● {} ({})", location.to_uppercase(), mode)
        }
        MlConnectionState::Connecting => "CONNECTING...".into(),
        MlConnectionState::Disconnected => "DISCONNECTED".into(),
        MlConnectionState::Error(e) => format!("ERR: {}", &e[..e.len().min(16)]),
    };
    let conn_color = match &app.ml_connection_state {
        MlConnectionState::Connected { .. } => theme::GREEN,
        MlConnectionState::Connecting => theme::AMBER,
        _ => theme::RED,
    };

    let tiles = vec![
        widgets::Kpi {
            label: "CONNECTION",
            value: conn_label,
            note: format!("{} decisions", app.ml_total_decisions),
            color: conn_color,
        },
        widgets::Kpi {
            label: "RISK RATE",
            value: format!("{:.3}", risk_rate),
            note: format!("{} violations", app.ml_total_risk_events),
            color: if risk_rate > 0.35 { theme::RED } else { theme::GREEN },
        },
        widgets::Kpi {
            label: "SAFE BOUND",
            value: format!("{:.1} Mbps", bound_val),
            note: format!("{}/{} sessions", admitted, oracle),
            color: theme::CYAN,
        },
        widgets::Kpi {
            label: "DROPPED",
            value: format!("{}", dropped),
            note: format!("{} total", app.ml_total_dropped),
            color: if dropped > 0 { theme::RED } else { theme::GREEN },
        },
        widgets::Kpi {
            label: "CONGESTION",
            value: if congested {
                "DETECTED".into()
            } else {
                "HEALTHY".into()
            },
            note: format!("{} slots", app.ml_total_congested_slots),
            color: if congested { theme::RED } else { theme::GREEN },
        },
    ];

    widgets::kpi_row(frame, area, &tiles);
}

// ── Throughput Chart ─────────────────────────────────────────────────────────

fn render_throughput_chart(frame: &mut Frame, app: &App, area: Rect) {
    let block = widgets::padded_panel("THROUGHPUT · FORECAST · SAFE BOUND", theme::ACCENT);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.width < 12 || inner.height < 5 {
        return;
    }

    let n = app.ml_actual.len();

    // Legend row with the live values, above the plot so it never covers data.
    let rows = Layout::vertical([
        Constraint::Length(1), // legend
        Constraint::Length(1), // gap
        Constraint::Min(3),    // chart
        Constraint::Length(1), // over-promise strip
    ])
    .split(inner);

    let latest = app.ml_latest.as_ref();
    let value = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into());
    let legend = Line::from(vec![
        Span::styled("── ", theme::fg(COLOR_ACTUAL)),
        Span::styled("actual ", theme::text_dim()),
        Span::styled(value(latest.map(|r| r.actual_mbps)), theme::fg_bold(theme::FG_PRIMARY)),
        Span::styled("    ── ", theme::fg(COLOR_FORECAST)),
        Span::styled("forecast ", theme::text_dim()),
        Span::styled(value(latest.map(|r| r.predicted_mbps)), theme::fg_bold(theme::FG_PRIMARY)),
        Span::styled("    ━━ ", theme::fg(COLOR_BOUND)),
        Span::styled("safe bound ", theme::text_dim()),
        Span::styled(value(latest.map(|r| r.bound_mbps)), theme::fg_bold(theme::FG_PRIMARY)),
        Span::styled("    ▮ ", theme::fg(COLOR_RISK)),
        Span::styled("over-promise", theme::text_dim()),
        Span::styled("   Mbps", theme::text_ghost()),
    ]);
    frame.render_widget(Paragraph::new(legend).style(theme::bg()), rows[0]);

    if n < 2 {
        let msg = Paragraph::new(Line::from(Span::styled(
            "Waiting for ML data stream...",
            theme::text_dim(),
        )))
        .alignment(Alignment::Center)
        .style(theme::bg());
        frame.render_widget(msg, rows[2]);
        return;
    }

    // One decision per terminal cell. Squeezing all 300 buffered points into
    // ~100 cells made the three lines smear into one spiky band; at one point
    // per cell the 15 s scheduling waves and the forecast/bound gap stay legible.
    let y_label_width: u16 = 5;
    let plot_cols = rows[2].width.saturating_sub(y_label_width + 1).max(8) as usize;
    let window = n.min(plot_cols).max(2);
    let start = n - window;

    let series = |buf: &std::collections::VecDeque<f64>| -> Vec<(f64, f64)> {
        buf.iter()
            .skip(start)
            .enumerate()
            .map(|(i, v)| (i as f64 - (window as f64 - 1.0), *v))
            .collect()
    };
    let actual_data = series(&app.ml_actual);
    let predicted_data = series(&app.ml_predicted);
    let bound_data = series(&app.ml_bound);

    // Round y ticks on a clean step, padded so no line rides the frame.
    let visible = actual_data.iter().chain(&predicted_data).chain(&bound_data).map(|p| p.1);
    let lo = visible.clone().fold(f64::INFINITY, f64::min);
    let hi = visible.fold(f64::NEG_INFINITY, f64::max);
    let (y_min, y_max, step) = nice_bounds(lo, hi, 4);
    let tick_count = ((y_max - y_min) / step).round() as usize;
    let y_labels: Vec<Span> = (0..=tick_count)
        .map(|k| Span::styled(format!("{:>4.0}", y_min + step * k as f64), theme::text_dim()))
        .collect();

    // Drawn back to front: the safe bound is what admission acts on, so it
    // goes last and stays on top wherever the lines cross.
    let datasets = vec![
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(theme::fg(COLOR_ACTUAL))
            .data(&actual_data),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(theme::fg(COLOR_FORECAST))
            .data(&predicted_data),
        Dataset::default()
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(theme::fg_bold(COLOR_BOUND))
            .data(&bound_data),
    ];

    let x_lo = -(window as f64 - 1.0);
    let chart = Chart::new(datasets)
        .style(theme::bg())
        .legend_position(None)
        .x_axis(
            Axis::default()
                .style(theme::rule())
                .bounds([x_lo, 0.0])
                .labels([
                    Span::styled(format!("-{}s", window - 1), theme::text_dim()),
                    Span::styled(format!("-{}s", (window - 1) / 2), theme::text_dim()),
                    Span::styled("now", theme::text_dim()),
                ])
                .labels_alignment(Alignment::Right),
        )
        .y_axis(
            Axis::default()
                .style(theme::rule())
                .bounds([y_min, y_max])
                .labels(y_labels),
        );
    frame.render_widget(chart, rows[2]);

    // Over-promise strip: one mark per cell in which any decision's bound
    // exceeded the throughput that actually arrived.
    let strip_area = Rect {
        x: rows[3].x + y_label_width + 1,
        width: rows[3].width.saturating_sub(y_label_width + 1),
        ..rows[3]
    };
    let risks: Vec<bool> = app
        .ml_actual
        .iter()
        .zip(&app.ml_bound)
        .skip(start)
        .map(|(a, b)| b > a)
        .collect();
    let cells = strip_area.width as usize;
    let strip: String = (0..cells)
        .map(|c| {
            // Map the cell to its slice of the window, right-aligned to "now".
            let from = c * window / cells.max(1);
            let to = ((c + 1) * window / cells.max(1)).max(from + 1).min(window);
            if risks[from..to].iter().any(|r| *r) { '▮' } else { ' ' }
        })
        .collect();
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(strip, theme::fg(COLOR_RISK)))).style(theme::bg()),
        strip_area,
    );
}

const COLOR_ACTUAL: ratatui::style::Color = theme::FG_SECONDARY;
const COLOR_FORECAST: ratatui::style::Color = theme::ACCENT;
const COLOR_BOUND: ratatui::style::Color = theme::GREEN;
const COLOR_RISK: ratatui::style::Color = theme::ORANGE;

/// Pads `[lo, hi]` out to round multiples of a 1/2/2.5/5 x 10^k step, giving
/// about `count` intervals. Returns `(min, max, step)`.
fn nice_bounds(lo: f64, hi: f64, count: usize) -> (f64, f64, f64) {
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 100.0, 25.0);
    }
    let span = (hi - lo).max(1.0);
    let raw = span / count as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 2.5, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(raw);
    let min = (lo / step).floor() * step;
    let mut max = (hi / step).ceil() * step;
    if max <= min {
        max = min + step;
    }
    (min, max, step)
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

fn render_sidebar(frame: &mut Frame, app: &App, area: Rect) {
    let block = widgets::padded_panel("COVARIATES", theme::TEAL);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 4 {
        return;
    }

    let (regime, alpha, offset, step) = if let Some(ref rec) = app.ml_latest {
        (
            rec.regime.clone(),
            format!("{:.4}", rec.alpha),
            format!("{:.1} Mbps", rec.offset_mbps),
            format!("#{}", rec.step),
        )
    } else {
        ("—".into(), "—".into(), "—".into(), "—".into())
    };

    let rows: Vec<Line> = vec![
        kv_line("REGIME", &regime, theme::AMBER),
        Line::from(""),
        kv_line("ALPHA", &alpha, theme::CYAN),
        Line::from(""),
        kv_line("OFFSET", &offset, theme::ORANGE),
        Line::from(""),
        kv_line("STEP", &step, theme::VIOLET),
        Line::from(""),
        kv_line("RISK EVTS", &format!("{}", app.ml_total_risk_events), theme::RED),
        Line::from(""),
        kv_line("TOTAL DEC", &format!("{}", app.ml_total_decisions), theme::FG_SECONDARY),
    ];

    frame.render_widget(Paragraph::new(rows).style(theme::bg()), inner);
}

fn kv_line(key: &str, value: &str, color: ratatui::style::Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<10}", key), theme::text_dim()),
        Span::styled(value.to_string(), theme::fg_bold(color)),
    ])
}

// ── Admission Sparklines ─────────────────────────────────────────────────────

fn render_admission_sparklines(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Ratio(1, 3),
        Constraint::Ratio(1, 3),
        Constraint::Ratio(1, 3),
    ])
    .split(area);

    spark_u64(
        frame,
        cols[0],
        "ADMITTED",
        &app.ml_admitted,
        theme::GREEN,
        app.ml_latest
            .as_ref()
            .map(|r| format!("{} sessions", r.admitted))
            .unwrap_or_else(|| "—".into()),
    );

    let drop_color = if app.ml_latest.as_ref().is_some_and(|r| r.dropped > 0) {
        theme::RED
    } else {
        theme::GREEN
    };

    spark_u64(
        frame,
        cols[1],
        "DROPPED",
        &app.ml_dropped,
        drop_color,
        app.ml_latest
            .as_ref()
            .map(|r| format!("{} sessions", r.dropped))
            .unwrap_or_else(|| "—".into()),
    );

    spark_u64(
        frame,
        cols[2],
        "UNUSED",
        &app.ml_unused,
        theme::AMBER,
        app.ml_latest
            .as_ref()
            .map(|r| format!("{} sessions", r.unused))
            .unwrap_or_else(|| "—".into()),
    );
}

fn spark_u64(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    data: &std::collections::VecDeque<u64>,
    color: ratatui::style::Color,
    current: String,
) {
    let block = widgets::padded_panel(title, color);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if inner.height < 2 {
        return;
    }

    let rows = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).split(inner);

    // Sparkline needs a contiguous slice — use the last `width` values.
    let width = inner.width.saturating_sub(2).max(4) as usize;
    let slice: Vec<u64> = data.iter().rev().take(width).rev().copied().collect();

    // Rebase for visibility — add a floor so zero values still show a baseline.
    let max = slice.iter().copied().max().unwrap_or(1).max(1);
    let shaped: Vec<u64> = slice
        .iter()
        .map(|v| 10 + (*v as f64 / max as f64 * 90.0) as u64)
        .collect();

    frame.render_widget(
        Sparkline::default().data(&shaped).style(theme::fg(color)),
        rows[0],
    );

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(current, theme::fg_bold(color)),
        ]))
        .style(theme::bg()),
        rows[1],
    );
}
