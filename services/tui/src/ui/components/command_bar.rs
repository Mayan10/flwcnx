//! Persistent bottom bar: slash-command prompt on the left, key hints right.

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::app::{App, Screen};
use crate::ui::theme;

pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(theme::rule())
        .style(theme::bg());
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(46)]).split(inner);

    // ── Prompt ──
    let prompt = if app.is_typing_command {
        let cursor = if app.tick % 6 < 3 { "▌" } else { " " };
        Line::from(vec![
            Span::styled(" ❯ ", theme::fg(theme::ACCENT)),
            Span::styled(app.command_input.clone(), theme::fg_bold(theme::FG_PRIMARY)),
            Span::styled(cursor, theme::fg(theme::ACCENT)),
            Span::styled(
                format!("   {}", suggestion(&app.command_input)),
                theme::text_ghost(),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled(" / ", theme::chip(theme::ACCENT_DIM)),
            Span::styled("  commands  ", theme::text_dim()),
            Span::styled(
                "/home  /ml  /login  /logout  /quit",
                theme::text_ghost(),
            ),
        ])
    };
    frame.render_widget(Paragraph::new(prompt).style(theme::bg()), cols[0]);

    // ── Contextual key hints ──
    let hints: Vec<Span> = match app.current_screen {
        Screen::Login => vec![
            Span::styled(" Enter ", theme::key_badge()),
            Span::styled("verify  ", theme::key_desc()),
            Span::styled(" Esc ", theme::key_badge()),
            Span::styled("home ", theme::key_desc()),
        ],
        _ => vec![
            Span::styled(" h ", theme::key_badge()),
            Span::styled("home  ", theme::key_desc()),
            Span::styled(" m ", theme::key_badge()),
            Span::styled("ml monitor  ", theme::key_desc()),
            Span::styled(" k ", theme::key_badge()),
            Span::styled("login  ", theme::key_desc()),
            Span::styled(" q ", theme::key_badge()),
            Span::styled("quit ", theme::key_desc()),
        ],
    };

    frame.render_widget(
        Paragraph::new(Line::from(hints))
            .alignment(Alignment::Right)
            .style(theme::bg()),
        cols[1],
    );
}

/// Inline ghost completion for a partially typed command.
fn suggestion(input: &str) -> &'static str {
    const COMMANDS: [&str; 5] = ["/home", "/ml", "/login", "/logout", "/quit"];
    if input.len() < 2 {
        return "";
    }
    COMMANDS
        .iter()
        .find(|c| c.starts_with(input) && **c != input)
        .copied()
        .unwrap_or("")
}
