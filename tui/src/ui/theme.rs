#![allow(dead_code)]

use ratatui::style::{Color, Modifier, Style};

// ── Color Palette ────────────────────────────────────────────────────────────

/// Pure black background.
pub const BG: Color = Color::Rgb(0, 0, 0);

/// Slightly elevated surface — for subtle panels.
pub const SURFACE: Color = Color::Rgb(10, 10, 12);

/// Panel border / elevated card.
pub const SURFACE_RAISED: Color = Color::Rgb(18, 18, 22);

/// Primary text — crisp white.
pub const FG_PRIMARY: Color = Color::Rgb(230, 233, 240);

/// Secondary text — cool gray.
pub const FG_SECONDARY: Color = Color::Rgb(120, 128, 145);

/// Tertiary / dim text — very muted.
pub const FG_DIM: Color = Color::Rgb(55, 60, 72);

/// Ghost text — barely visible.
pub const FG_GHOST: Color = Color::Rgb(35, 38, 46);

/// Primary brand accent — electric indigo-blue.
pub const ACCENT: Color = Color::Rgb(99, 140, 255);

/// Brand accent dimmed.
pub const ACCENT_DIM: Color = Color::Rgb(60, 85, 160);

/// Status indicator — soft teal.
pub const STATUS_ACTIVE: Color = Color::Rgb(72, 199, 162);

/// Status indicator dimmed.
pub const STATUS_DIM: Color = Color::Rgb(40, 110, 90);

/// Warning / attention.
pub const WARN: Color = Color::Rgb(230, 180, 80);

/// Subtle border / rule color.
pub const BORDER: Color = Color::Rgb(30, 32, 38);

/// Slightly brighter border for emphasis.
pub const BORDER_LIGHT: Color = Color::Rgb(45, 48, 58);

// ── Styles ───────────────────────────────────────────────────────────────────

/// Background fill style.
pub fn bg() -> Style {
    Style::default().bg(BG)
}

/// Surface fill style.
pub fn surface() -> Style {
    Style::default().bg(SURFACE)
}

/// Primary text.
pub fn text_primary() -> Style {
    Style::default().fg(FG_PRIMARY)
}

/// Secondary / muted text.
pub fn text_secondary() -> Style {
    Style::default().fg(FG_SECONDARY)
}

/// Dim / tertiary text.
pub fn text_dim() -> Style {
    Style::default().fg(FG_DIM)
}

/// Ghost text — barely visible structural elements.
pub fn text_ghost() -> Style {
    Style::default().fg(FG_GHOST)
}

/// Brand name — bold accent.
pub fn brand() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// Brand name hero — large, bright, commanding.
pub fn brand_hero() -> Style {
    Style::default()
        .fg(FG_PRIMARY)
        .add_modifier(Modifier::BOLD)
}

/// Subtitle / descriptor label — uppercase tracking feel.
pub fn label() -> Style {
    Style::default().fg(FG_SECONDARY)
}

/// Label with accent color.
pub fn label_accent() -> Style {
    Style::default().fg(ACCENT_DIM)
}

/// Status indicator style.
pub fn status() -> Style {
    Style::default().fg(STATUS_ACTIVE)
}

/// Status indicator dimmed.
pub fn status_dim() -> Style {
    Style::default().fg(STATUS_DIM)
}

/// Keyboard shortcut key badge.
pub fn key_badge() -> Style {
    Style::default()
        .fg(FG_SECONDARY)
        .add_modifier(Modifier::BOLD)
}

/// Key description text.
pub fn key_desc() -> Style {
    Style::default().fg(FG_DIM)
}

/// Horizontal rule / border style.
pub fn rule() -> Style {
    Style::default().fg(BORDER)
}

/// Slightly brighter rule.
pub fn rule_light() -> Style {
    Style::default().fg(BORDER_LIGHT)
}

// ── Data / Classification Palette ────────────────────────────────────────────
// Saturated hues used for traffic classes, gauges and insight tiles. Chosen to
// stay legible against the pure-black background.

pub const VIOLET: Color = Color::Rgb(167, 139, 250);
pub const CYAN: Color = Color::Rgb(34, 211, 238);
pub const GREEN: Color = Color::Rgb(52, 211, 153);
pub const AMBER: Color = Color::Rgb(251, 191, 36);
pub const RED: Color = Color::Rgb(248, 113, 113);
pub const PINK: Color = Color::Rgb(244, 114, 182);
pub const BLUE: Color = Color::Rgb(96, 165, 250);
pub const ORANGE: Color = Color::Rgb(251, 146, 60);
pub const TEAL: Color = Color::Rgb(45, 212, 191);
pub const LIME: Color = Color::Rgb(163, 230, 53);
pub const SKY: Color = Color::Rgb(56, 189, 248);
pub const MAGENTA: Color = Color::Rgb(232, 121, 249);

/// Track color for gauges — the unfilled portion.
pub const TRACK: Color = Color::Rgb(26, 28, 34);

// ── Style Helpers ────────────────────────────────────────────────────────────

/// Colored text in one of the palette hues.
pub fn fg(color: Color) -> Style {
    Style::default().fg(color)
}

/// Bold colored text.
pub fn fg_bold(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

/// Inverted chip — colored background, black text. Used for action badges.
pub fn chip(color: Color) -> Style {
    Style::default()
        .fg(BG)
        .bg(color)
        .add_modifier(Modifier::BOLD)
}

/// Panel title styling — bold accent on black.
pub fn panel_title() -> Style {
    Style::default().fg(FG_PRIMARY).add_modifier(Modifier::BOLD)
}

/// Table header row.
pub fn table_header() -> Style {
    Style::default()
        .fg(FG_SECONDARY)
        .add_modifier(Modifier::BOLD)
}

/// Severity color for a 0.0–1.0 load ratio: green → amber → red.
pub fn load_color(ratio: f64) -> Color {
    if ratio < 0.55 {
        GREEN
    } else if ratio < 0.75 {
        LIME
    } else if ratio < 0.88 {
        AMBER
    } else {
        RED
    }
}
