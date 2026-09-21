//! API-key authentication. Reached explicitly from Home, or automatically when
//! the operator tries to open a gated screen while logged out.

use ratatui::layout::{Alignment, Constraint, Flex, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::app::App;
use crate::ui::components::chrome;
use crate::ui::theme;

/// Spinner shown while the key is being verified against the backend.
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    frame.render_widget(Block::default().style(theme::bg()), area);

    let layout = Layout::vertical([
        Constraint::Length(3), // chrome
        Constraint::Min(0),    // body
        Constraint::Length(3), // footer
    ])
    .split(area);

    chrome::render(frame, app, layout[0]);
    render_card(frame, app, layout[1]);
    render_footer(frame, app, layout[2]);
}

fn render_card(frame: &mut Frame, app: &App, area: Rect) {
    // The card grows by one row when there is an error to show.
    let card_height: u16 = if app.auth_error.is_some() { 15 } else { 13 };
    let card_width = area.width.min(66);

    let v = Layout::vertical([Constraint::Length(card_height.min(area.height))])
        .flex(Flex::Center)
        .split(area);
    let h = Layout::horizontal([Constraint::Length(card_width)])
        .flex(Flex::Center)
        .split(v[0]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(theme::rule_light())
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled("AUTHENTICATE", theme::fg_bold(theme::AMBER)),
            Span::raw(" "),
        ]))
        .style(theme::bg())
        .padding(Padding::new(2, 2, 1, 0));
    let inner = block.inner(h[0]);
    frame.render_widget(block, h[0]);

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            "Verify your Thalweg API key to unlock the console.",
            theme::text_secondary(),
        )),
        Line::from(Span::styled(
            "Find it in the web console under Dashboard → API Key.",
            theme::text_dim(),
        )),
        Line::from(""),
    ];

    // Input row — the key is masked, with a live character count.
    if app.auth_in_progress {
        let spin = SPINNER[(app.tick as usize) % SPINNER.len()];
        lines.push(Line::from(vec![
            Span::styled(format!("{spin} "), theme::fg(theme::CYAN)),
            Span::styled("Verifying key with gateway…", theme::fg(theme::CYAN)),
        ]));
    } else {
        let masked = "•".repeat(app.api_key_input.chars().count());
        let cursor = if app.tick % 6 < 3 { "▌" } else { " " };
        lines.push(Line::from(vec![
            Span::styled("❯ ", theme::fg(theme::ACCENT)),
            Span::styled("API KEY  ", theme::text_secondary()),
            Span::styled(masked, theme::fg_bold(theme::FG_PRIMARY)),
            Span::styled(cursor, theme::fg(theme::ACCENT)),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("  ", theme::text_dim()),
        Span::styled(
            format!("{} characters", app.api_key_input.chars().count()),
            theme::text_dim(),
        ),
    ]));

    if let Some(err) = app.auth_error.as_deref() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(err.to_string(), theme::fg(theme::RED)),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(" Enter ", theme::chip(theme::ACCENT)),
        Span::styled("  verify    ", theme::text_dim()),
        Span::styled(" Esc ", theme::key_badge()),
        Span::styled("  back to home", theme::text_dim()),
    ]));

    frame.render_widget(Paragraph::new(lines).style(theme::bg()), inner);
}

fn render_footer(frame: &mut Frame, _app: &App, area: Rect) {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    let border = Line::from(Span::styled("─".repeat(area.width as usize), theme::rule()));
    frame.render_widget(Paragraph::new(border).style(theme::bg()), rows[0]);

    // Tell the operator where they will land once the key checks out.
    let note = Line::from(vec![
        Span::styled("continues to ", theme::text_dim()),
        Span::styled(crate::app::Screen::MlDashboard.title(), theme::fg(theme::ACCENT)),
    ])
    .alignment(Alignment::Center);
    frame.render_widget(Paragraph::new(note).style(theme::bg()), rows[1]);
}
