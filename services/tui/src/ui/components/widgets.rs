//! Small reusable building blocks shared by the data screens.

#![allow(dead_code)]

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::Color;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Padding, Paragraph};
use ratatui::Frame;

use crate::ui::theme;

/// A bordered panel with a colored title.
pub fn panel<'a>(title: &'a str, accent: Color) -> Block<'a> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(theme::rule())
        .title(Line::from(vec![
            Span::raw(" "),
            Span::styled(title, theme::fg_bold(accent)),
            Span::raw(" "),
        ]))
        .style(theme::bg())
}

/// Same as [`panel`] but with one column of inner breathing room.
pub fn padded_panel<'a>(title: &'a str, accent: Color) -> Block<'a> {
    panel(title, accent).padding(Padding::new(1, 1, 0, 0))
}

/// A headline metric tile: label on the first row, big value on the second,
/// a delta/context note on the third.
pub struct Kpi<'a> {
    pub label: &'a str,
    pub value: String,
    pub note: String,
    pub color: Color,
}

/// Renders a row of KPI tiles spread evenly across `area`.
pub fn kpi_row(frame: &mut Frame, area: Rect, tiles: &[Kpi]) {
    if tiles.is_empty() {
        return;
    }
    let constraints: Vec<Constraint> = tiles
        .iter()
        .map(|_| Constraint::Ratio(1, tiles.len() as u32))
        .collect();
    let cols = Layout::horizontal(constraints).split(area);

    for (tile, col) in tiles.iter().zip(cols.iter()) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(theme::rule())
            .style(theme::bg())
            .padding(Padding::new(1, 1, 0, 0));
        let inner = block.inner(*col);
        frame.render_widget(block, *col);

        let body = vec![
            Line::from(Span::styled(tile.label, theme::text_secondary())),
            Line::from(Span::styled(
                tile.value.clone(),
                theme::fg_bold(tile.color),
            )),
            Line::from(Span::styled(tile.note.clone(), theme::text_dim())),
        ];
        frame.render_widget(Paragraph::new(body).style(theme::bg()), inner);
    }
}

/// Centered dim text — used for empty states.
pub fn hint(frame: &mut Frame, area: Rect, text: &str) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(text, theme::text_dim())))
            .alignment(Alignment::Center)
            .style(theme::bg()),
        area,
    );
}
