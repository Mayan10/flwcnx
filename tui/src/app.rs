use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use crate::ml_client::{self, DecisionRecord, MlConnectionState, MlMessage};
use crate::ui;

/// Maximum number of data points kept in the rolling buffers (~5 min at 1 Hz).
const ML_BUFFER_SIZE: usize = 300;

/// The currently active screen in the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Home,
    Login,
    MlDashboard,
}

impl Screen {
    /// Label shown in the header breadcrumb.
    pub fn title(self) -> &'static str {
        match self {
            Screen::Home => "HOME",
            Screen::Login => "AUTHENTICATE",
            Screen::MlDashboard => "ML MONITOR",
        }
    }
}

/// Core application state.
#[allow(dead_code)]
pub struct App {
    /// Whether the application should exit.
    pub should_quit: bool,
    /// The currently active screen.
    pub current_screen: Screen,
    /// Monotonic tick counter, used for animations.
    pub tick: u64,
    /// Timestamp of when the app was created, used for elapsed-based animations.
    pub start_time: Instant,
    pub api_key_input: String,
    pub auth_error: Option<String>,
    pub auth_in_progress: bool,
    pub company_name: Option<String>,
    pub command_input: String,
    pub is_typing_command: bool,
    pub tx: tokio::sync::mpsc::Sender<AuthResult>,
    pub rx: tokio::sync::mpsc::Receiver<AuthResult>,

    // ── ML real-time data ────────────────────────────────────────────────
    /// Rolling buffer of actual throughput values (Mbps).
    pub ml_actual: VecDeque<f64>,
    /// Rolling buffer of point forecast values (Mbps).
    pub ml_predicted: VecDeque<f64>,
    /// Rolling buffer of safe bound values (Mbps).
    pub ml_bound: VecDeque<f64>,
    /// Rolling buffer of alpha (operating point) values.
    pub ml_alpha: VecDeque<f64>,
    /// Rolling buffer of admitted session counts.
    pub ml_admitted: VecDeque<u64>,
    /// Rolling buffer of dropped session counts.
    pub ml_dropped: VecDeque<u64>,
    /// Rolling buffer of unused capacity counts.
    pub ml_unused: VecDeque<u64>,
    /// The most recent decision record from the ML engine.
    pub ml_latest: Option<DecisionRecord>,
    /// Connection state of the ML WebSocket client.
    pub ml_connection_state: MlConnectionState,
    /// Cumulative statistics.
    pub ml_total_decisions: u64,
    pub ml_total_risk_events: u64,
    pub ml_total_dropped: u64,
    pub ml_total_congested_slots: u64,
    /// Channel for receiving ML data from the background WebSocket task.
    pub ml_rx: tokio::sync::mpsc::Receiver<MlMessage>,
}

pub enum AuthResult {
    Success { company_name: String },
    Error(String),
}

impl App {
    pub fn new() -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        // ML client channel — buffer up to 256 messages so the WS reader
        // never blocks even if the UI loop is briefly behind.
        let (ml_tx, ml_rx) = tokio::sync::mpsc::channel(256);

        // Spawn the ML WebSocket client in the background.
        ml_client::spawn_ml_client(ml_tx);

        Self {
            should_quit: false,
            // The TUI opens on the landing screen; login is reached explicitly.
            current_screen: Screen::Home,
            tick: 0,
            start_time: Instant::now(),
            api_key_input: String::new(),
            auth_error: None,
            auth_in_progress: false,
            company_name: None,
            command_input: String::new(),
            is_typing_command: false,
            tx,
            rx,
            // ML state — empty until the WebSocket connects.
            ml_actual: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_predicted: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_bound: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_alpha: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_admitted: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_dropped: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_unused: VecDeque::with_capacity(ML_BUFFER_SIZE),
            ml_latest: None,
            ml_connection_state: MlConnectionState::Connecting,
            ml_total_decisions: 0,
            ml_total_risk_events: 0,
            ml_total_dropped: 0,
            ml_total_congested_slots: 0,
            ml_rx,
        }
    }

    /// Whether a valid API key has been verified this session.
    pub fn is_authenticated(&self) -> bool {
        self.company_name.is_some()
    }

    /// Navigates to `screen`.
    fn navigate(&mut self, screen: Screen) {
        self.auth_error = None;
        self.current_screen = screen;
    }

    /// Push a value into a fixed-size deque, evicting the oldest if full.
    fn push_rolling(buf: &mut VecDeque<f64>, value: f64) {
        if buf.len() >= ML_BUFFER_SIZE {
            buf.pop_front();
        }
        buf.push_back(value);
    }

    fn push_rolling_u64(buf: &mut VecDeque<u64>, value: u64) {
        if buf.len() >= ML_BUFFER_SIZE {
            buf.pop_front();
        }
        buf.push_back(value);
    }

    /// Process a single ML decision record into the rolling buffers.
    fn ingest_decision(&mut self, record: DecisionRecord) {
        Self::push_rolling(&mut self.ml_actual, record.actual_mbps);
        Self::push_rolling(&mut self.ml_predicted, record.predicted_mbps);
        Self::push_rolling(&mut self.ml_bound, record.bound_mbps);
        Self::push_rolling(&mut self.ml_alpha, record.alpha);
        Self::push_rolling_u64(&mut self.ml_admitted, record.admitted as u64);
        Self::push_rolling_u64(&mut self.ml_dropped, record.dropped as u64);
        Self::push_rolling_u64(&mut self.ml_unused, record.unused as u64);

        self.ml_total_decisions += 1;
        if record.risk_event {
            self.ml_total_risk_events += 1;
        }
        self.ml_total_dropped += record.dropped as u64;
        if record.congested {
            self.ml_total_congested_slots += 1;
        }

        self.ml_latest = Some(record);
    }

    /// Drain all pending ML messages from the channel.
    fn poll_ml_messages(&mut self) {
        while let Ok(msg) = self.ml_rx.try_recv() {
            match msg {
                MlMessage::Connected(init) => {
                    self.ml_connection_state = MlConnectionState::Connected {
                        location: init.location,
                        mode: init.mode,
                    };
                }
                MlMessage::Decision(record) => {
                    self.ingest_decision(record);
                }
                MlMessage::Disconnected(reason) => {
                    self.ml_connection_state = MlConnectionState::Disconnected;
                    let _ = reason; // logged by the client task
                }
                MlMessage::Error(err) => {
                    self.ml_connection_state = MlConnectionState::Error(err);
                }
            }
        }
    }

    /// Main event loop.
    pub async fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
        let tick_rate = Duration::from_millis(150);
        let mut last_tick = Instant::now();

        loop {
            // Draw
            terminal.draw(|frame| {
                ui::render(frame, self);
            })?;

            // Event handling with tick timeout
            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                    }
                }
            }

            // Check for auth results
            if let Ok(result) = self.rx.try_recv() {
                self.auth_in_progress = false;
                match result {
                    AuthResult::Success { company_name } => {
                        self.company_name = Some(company_name);
                        self.current_screen = Screen::MlDashboard;
                        self.auth_error = None;
                        self.api_key_input.clear();
                    }
                    AuthResult::Error(err) => {
                        self.auth_error = Some(err);
                    }
                }
            }

            // Drain ML messages
            self.poll_ml_messages();

            // Tick
            if last_tick.elapsed() >= tick_rate {
                self.tick = self.tick.wrapping_add(1);
                last_tick = Instant::now();
            }

            if self.should_quit {
                return Ok(());
            }
        }
    }

    fn handle_key(&mut self, key: KeyCode) {
        if self.auth_in_progress { return; }

        if self.is_typing_command {
            match key {
                KeyCode::Esc => {
                    self.is_typing_command = false;
                    self.command_input.clear();
                }
                KeyCode::Backspace => {
                    self.command_input.pop();
                    if self.command_input.is_empty() {
                        self.is_typing_command = false;
                    }
                }
                KeyCode::Enter => {
                    let cmd = self.command_input.trim().to_string();
                    self.run_command(&cmd);
                    self.is_typing_command = false;
                    self.command_input.clear();
                }
                KeyCode::Char(c) => {
                    self.command_input.push(c);
                }
                _ => {}
            }
            return;
        }

        // The login screen owns every printable key, so the command prompt and
        // the global quit keys must not steal characters from the key field.
        if self.current_screen == Screen::Login {
            match key {
                KeyCode::Esc => self.current_screen = Screen::Home,
                KeyCode::Char(c) => self.api_key_input.push(c),
                KeyCode::Backspace => { self.api_key_input.pop(); }
                KeyCode::Enter => self.submit_api_key(),
                _ => {}
            }
            return;
        }

        match key {
            KeyCode::Char('/') => {
                self.is_typing_command = true;
                self.command_input.push('/');
            }
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Esc => {
                // Esc steps back to Home, and only quits from Home itself.
                if self.current_screen == Screen::Home {
                    self.should_quit = true;
                } else {
                    self.current_screen = Screen::Home;
                }
            }
            // Single-key navigation shortcuts.
            KeyCode::Char('h') => self.navigate(Screen::Home),
            KeyCode::Char('k') => self.navigate(Screen::Login),
            KeyCode::Char('m') => self.navigate(Screen::MlDashboard),
            _ => {}
        }
    }

    /// Executes a slash command typed into the command bar.
    fn run_command(&mut self, cmd: &str) {
        match cmd {
            "/home" => self.navigate(Screen::Home),
            "/login" | "/auth" => {
                self.auth_error = None;
                self.current_screen = Screen::Login;
            }
            "/logout" => {
                self.company_name = None;
                self.api_key_input.clear();
                self.auth_error = None;
                self.current_screen = Screen::Home;
            }
            "/ml" | "/forecast" | "/monitor" => self.navigate(Screen::MlDashboard),
            "/quit" | "/exit" => self.should_quit = true,
            _ => {}
        }
    }

    /// Fires the API-key verification request against the backend.
    fn submit_api_key(&mut self) {
        if self.api_key_input.is_empty() {
            return;
        }
        self.auth_in_progress = true;
        self.auth_error = None;
        let api_key = self.api_key_input.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            let res = client
                .post("http://localhost:3001/api/auth/verify-key")
                .json(&serde_json::json!({ "apiKey": api_key }))
                .send()
                .await;

            match res {
                Ok(response) => {
                    if response.status().is_success() {
                        if let Ok(data) = response.json::<serde_json::Value>().await {
                            if let Some(name) = data["company"]["name"].as_str() {
                                let _ = tx
                                    .send(AuthResult::Success { company_name: name.to_string() })
                                    .await;
                                return;
                            }
                        }
                    }
                    let _ = tx.send(AuthResult::Error("Invalid API key".to_string())).await;
                }
                Err(e) => {
                    let _ = tx
                        .send(AuthResult::Error(format!("Connection error: {e}")))
                        .await;
                }
            }
        });
    }

    /// Returns elapsed seconds since the app started.
    #[allow(dead_code)]
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}
