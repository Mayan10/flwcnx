//! Shared window chrome: the top brand/status bar drawn by every data screen.

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::app::App;
use crate::ml_client::MlConnectionState;
use crate::ui::theme;

/// Draws the three-row header: padding, content, hairline rule.
/// Expects an area of exactly 3 rows.
pub fn render(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .split(area);

    let cols = Layout::horizontal([Constraint::Min(0), Constraint::Length(52)]).split(rows[1]);

    // Left — brand mark plus the active screen breadcrumb.
    let left = Line::from(vec![
        Span::raw(" "),
        Span::styled("FlowConX", theme::brand()),
        Span::styled("  ▸  ", theme::text_ghost()),
        Span::styled(app.current_screen.title(), theme::fg_bold(theme::FG_PRIMARY)),
    ]);
    frame.render_widget(Paragraph::new(left).style(theme::bg()), cols[0]);

    // Right — ML stream state, tenant, version.
    let (link_label, link_color) = match &app.ml_connection_state {
        MlConnectionState::Connected { .. } => ("ML LINK UP", theme::GREEN),
        MlConnectionState::Connecting => ("ML CONNECTING", theme::AMBER),
        MlConnectionState::Disconnected => ("ML LINK DOWN", theme::RED),
        MlConnectionState::Error(_) => ("ML LINK ERROR", theme::RED),
    };
    let mut right = vec![
        Span::styled("● ", theme::fg(link_color)),
        Span::styled(link_label, theme::fg_bold(link_color)),
    ];
    if let Some(name) = app.company_name.as_deref() {
        right.push(Span::styled("  ·  ", theme::text_ghost()));
        right.push(Span::styled(name.to_uppercase(), theme::text_secondary()));
    } else {
        right.push(Span::styled("  ·  ", theme::text_ghost()));
        right.push(Span::styled("GUEST", theme::text_dim()));
    }
    right.push(Span::styled("  v0.1.0 ", theme::text_dim()));

    frame.render_widget(
        Paragraph::new(Line::from(right))
            .alignment(Alignment::Right)
            .style(theme::bg()),
        cols[1],
    );

    // Hairline rule under the header.
    let rule = Line::from(Span::styled(
        "─".repeat(area.width as usize),
        theme::rule(),
    ));
    frame.render_widget(Paragraph::new(rule).style(theme::bg()), rows[2]);
}
