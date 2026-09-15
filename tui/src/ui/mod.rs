pub mod components;
pub mod screens;
pub mod theme;

use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::Frame;

use crate::app::{App, Screen};

/// Top-level render dispatcher. Routes rendering to the active screen and
/// draws the persistent command bar beneath it.
pub fn render(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(2)].as_ref())
        .split(frame.area());

    match app.current_screen {
        Screen::Home => screens::home::render(frame, app, chunks[0]),
        Screen::Login => screens::login::render(frame, app, chunks[0]),
        Screen::MlDashboard => screens::ml_dashboard::render(frame, app, chunks[0]),
    }

    components::command_bar::render(frame, app, chunks[1]);
}
