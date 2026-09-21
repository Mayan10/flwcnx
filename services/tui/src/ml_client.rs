//! WebSocket client for the thalweg ML API server.
//!
//! Connects to the FastAPI WebSocket at `/ws/ml/stream`, deserializes
//! `DecisionRecord` JSON messages, and pushes them into an mpsc channel
//! that the TUI event loop reads from. Handles reconnection with
//! exponential backoff.

use futures_util::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;

/// A single decision from the ML engine — mirrors the Python
/// `DecisionRecord` dataclass exactly.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionRecord {
    pub step: u64,
    pub timestamp: String,
    pub regime: String,
    pub predicted_mbps: f64,
    pub bound_mbps: f64,
    pub actual_mbps: f64,
    pub offset_mbps: f64,
    pub alpha: f64,
    pub admitted: i64,
    pub oracle: i64,
    pub dropped: i64,
    pub unused: i64,
    pub risk_event: bool,
    pub congested: bool,
    pub realised_risk_rate: f64,
}

/// Handshake message sent by the server on connection.
#[derive(Debug, Clone, Deserialize)]
pub struct MlInitMessage {
    pub mode: String,
    pub location: String,
    pub epsilon: f64,
    pub direction: String,
    #[serde(default)]
    pub feature_set: String,
    #[serde(default)]
    pub n_test_samples: Option<u64>,
}

/// Messages sent from the ML client task to the app.
#[derive(Debug)]
pub enum MlMessage {
    Connected(MlInitMessage),
    Decision(DecisionRecord),
    Disconnected(String),
    Error(String),
}

/// Connection state shown in the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MlConnectionState {
    Disconnected,
    Connecting,
    Connected {
        location: String,
        mode: String,
    },
    Error(String),
}

impl std::fmt::Display for MlConnectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MlConnectionState::Disconnected => write!(f, "DISCONNECTED"),
            MlConnectionState::Connecting => write!(f, "CONNECTING..."),
            MlConnectionState::Connected { location, mode } => {
                write!(f, "● {} ({})", location.to_uppercase(), mode)
            }
            MlConnectionState::Error(e) => write!(f, "ERROR: {}", e),
        }
    }
}

/// Default API URL for the ML server.
const DEFAULT_URL: &str = "ws://127.0.0.1:8010/ws/ml/stream";

/// Maximum backoff between reconnection attempts.
const MAX_BACKOFF_SECS: u64 = 30;

/// Spawn the WebSocket client as a background tokio task. Returns the
/// receiving end of the channel; the task runs until the sender is dropped.
pub fn spawn_ml_client(tx: mpsc::Sender<MlMessage>) {
    let url = std::env::var("THALWEG_API_URL").unwrap_or_else(|_| DEFAULT_URL.to_string());

    tokio::spawn(async move {
        let mut backoff: u64 = 1;

        loop {
            match connect_and_stream(&url, &tx).await {
                Ok(()) => {
                    // Clean end of stream — reconnect immediately for a new
                    // session (the server may have more data).
                    let _ = tx
                        .send(MlMessage::Disconnected("stream ended".into()))
                        .await;
                    backoff = 1;
                }
                Err(e) => {
                    let msg = format!("{e}");
                    let _ = tx.send(MlMessage::Error(msg)).await;
                    tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                    backoff = (backoff * 2).min(MAX_BACKOFF_SECS);
                }
            }
        }
    });
}

async fn connect_and_stream(
    url: &str,
    tx: &mpsc::Sender<MlMessage>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (ws, _response) = tokio_tungstenite::connect_async(url).await?;
    let (_write, mut read) = ws.split();

    while let Some(msg) = read.next().await {
        let msg = msg?;
        if msg.is_text() {
            let text = msg.to_text()?;
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(text) {
                match value.get("type").and_then(|t| t.as_str()) {
                    Some("init") => {
                        if let Ok(init) = serde_json::from_value::<MlInitMessage>(value) {
                            let _ = tx.send(MlMessage::Connected(init)).await;
                        }
                    }
                    Some("decision") => {
                        if let Ok(record) = serde_json::from_value::<DecisionRecord>(value) {
                            let _ = tx.send(MlMessage::Decision(record)).await;
                        }
                    }
                    Some("done") => {
                        break;
                    }
                    Some("error") => {
                        let msg = value
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error")
                            .to_string();
                        let _ = tx.send(MlMessage::Error(msg)).await;
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
