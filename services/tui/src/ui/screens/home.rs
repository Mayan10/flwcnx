//! Landing screen: the brand mark and a menu of destinations.

use ratatui::layout::{Alignment, Constraint, Flex, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::ui::theme;

// ── Brand Mark ───────────────────────────────────────────────────────────────

const BRAND_ART: &[&str] = &[
    "███████ ██       ██████  ██     ██  ██████  ██████  ███    ██ ██   ██",
    "██      ██      ██    ██ ██     ██ ██      ██    ██ ████   ██  ██ ██ ",
    "█████   ██      ██    ██ ██  █  ██ ██      ██    ██ ██ ██  ██   ███  ",
    "██      ██      ██    ██ ██ ███ ██ ██      ██    ██ ██  ██ ██  ██ ██ ",
    "██      ███████  ██████   ███ ███   ██████  ██████  ██   ████ ██   ██",
];

const BRAND_ART_SMALL: &str = "F L O W C O N X";

/// Widest line of [`BRAND_ART`], below which the compact wordmark is used.
const BRAND_ART_WIDTH: u16 = 70;

/// A destination on the landing menu.
struct NavItem {
    title: &'static str,
    command: &'static str,
    key: &'static str,
}

const NAV: [NavItem; 2] = [
    NavItem { title: "ML Monitor", command: "/ml", key: "m" },
    NavItem { title: "Authenticate", command: "/login", key: "k" },
];

/// Inner width of the menu: marker, title, command, key badge.
const NAV_WIDTH: u16 = 40;

pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    frame.render_widget(Block::default().style(theme::bg()), area);

    let layout = Layout::vertical([
        Constraint::Length(3), // header bar
        Constraint::Min(0),    // body
    ])
    .split(area);

    render_header(frame, app, layout[0]);
    render_body(frame, layout[1]);
}

// ── Header Bar ───────────────────────────────────────────────────────────────

fn render_header(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(34)]).split(rows[1]);

    let brand = Line::from(vec![Span::raw(" "), Span::styled("Thalweg", theme::brand())]);
    frame.render_widget(Paragraph::new(brand).style(theme::bg()), cols[0]);

    let (label, color) = if app.is_authenticated() {
        ("SESSION ACTIVE", theme::GREEN)
    } else {
        ("NOT AUTHENTICATED", theme::AMBER)
    };
    let right = Line::from(vec![
        Span::styled("● ", theme::fg(color)),
        Span::styled(label, theme::fg(color)),
        Span::styled("  v0.1.0 ", theme::text_dim()),
    ]);
    frame.render_widget(
        Paragraph::new(right)
            .alignment(Alignment::Right)
            .style(theme::bg()),
        cols[1],
    );

    let border = Line::from(Span::styled("─".repeat(area.width as usize), theme::rule()));
    frame.render_widget(Paragraph::new(border).style(theme::bg()), rows[2]);
}

// ── Body ─────────────────────────────────────────────────────────────────────

fn render_body(frame: &mut Frame, area: Rect) {
    let full_art = area.width >= BRAND_ART_WIDTH && area.height >= 14;
    let art_height: u16 = if full_art { BRAND_ART.len() as u16 } else { 1 };
    let nav_height = NAV.len() as u16 + 2; // one row per item plus borders

    let rows = Layout::vertical([
        Constraint::Length(art_height),
        Constraint::Length(2), // gap
        Constraint::Length(nav_height),
    ])
    .flex(Flex::Center)
    .split(area);

    if full_art {
        render_brand_art(frame, rows[0]);
    } else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(BRAND_ART_SMALL, theme::brand_hero())))
                .alignment(Alignment::Center)
                .style(theme::bg()),
            rows[0],
        );
    }

    let nav = Layout::horizontal([Constraint::Length((NAV_WIDTH + 4).min(area.width))])
        .flex(Flex::Center)
        .split(rows[2]);
    render_nav(frame, nav[0]);
}

fn render_brand_art(frame: &mut Frame, area: Rect) {
    let lines: Vec<Line> = BRAND_ART
        .iter()
        .enumerate()
        .map(|(i, line)| {
            // Bright at the top, fading to dim — a lit-from-above look.
            let style = match i {
                0 | 1 => theme::brand_hero(),
                2 => theme::fg_bold(theme::FG_SECONDARY),
                _ => theme::text_dim().add_modifier(Modifier::BOLD),
            };
            Line::from(Span::styled(*line, style)).alignment(Alignment::Center)
        })
        .collect();

    frame.render_widget(Paragraph::new(lines).style(theme::bg()), area);
}

// ── Navigation Menu ──────────────────────────────────────────────────────────

fn render_nav(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::rule())
        .style(theme::bg())
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let lines: Vec<Line> = NAV
        .iter()
        .map(|item| {
            Line::from(vec![
                Span::styled("▸ ", theme::fg(theme::GREEN)),
                Span::styled(format!("{:<16}", item.title), theme::fg_bold(theme::FG_PRIMARY)),
                Span::styled(format!("{:<14}", item.command), theme::fg(theme::ACCENT)),
                Span::styled(format!(" {} ", item.key), theme::key_badge()),
            ])
        })
        .collect();

    frame.render_widget(Paragraph::new(lines).style(theme::bg()), inner);
}
