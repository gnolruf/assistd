//! JSON-RPC 2.0 frames and the request/response [`Correlator`] shared
//! by both transports.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;

use crate::error::McpError;

/// Outbound JSON-RPC 2.0 request frame.
#[derive(Debug, Serialize)]
pub struct Request<'a> {
    pub jsonrpc: &'a str,
    pub id: u64,
    pub method: &'a str,
    pub params: Value,
}

/// Outbound JSON-RPC 2.0 notification frame (no `id`, no response).
#[derive(Debug, Serialize)]
pub struct Notification<'a> {
    pub jsonrpc: &'a str,
    pub method: &'a str,
    pub params: Value,
}

/// Inbound JSON-RPC 2.0 response. Either `result` or `error` is set;
/// notifications from the server have no `id`.
#[derive(Debug, Deserialize)]
pub struct Response {
    pub id: Option<u64>,
    pub result: Option<Value>,
    pub error: Option<RpcError>,
}

/// JSON-RPC 2.0 error object carried in a [`Response`] when the server reports failure.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

/// Hard cap on outstanding requests; prevents a misbehaving server
/// from leaking memory by never replying.
pub const MAX_IN_FLIGHT: usize = 256;

/// Outcome of one JSON-RPC round trip.
pub type Reply = Result<Value, RpcError>;

/// Matches outbound JSON-RPC request ids to their waiting [`oneshot`] receivers.
#[derive(Debug)]
pub struct Correlator {
    next_id: AtomicU64,
    pending: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
}

impl Correlator {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Reserve an id and a receiver for its reply. The slot is held
    /// until the returned [`Pending`] is dropped. Errors with
    /// `TooManyInFlight` once [`MAX_IN_FLIGHT`] requests are pending.
    pub fn next_request(
        &self,
        method: &'static str,
        params: Value,
    ) -> Result<Pending<'_>, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            let mut guard = self.pending.lock();
            if guard.len() >= MAX_IN_FLIGHT {
                return Err(McpError::TooManyInFlight);
            }
            guard.insert(id, tx);
        }
        Ok(Pending {
            correlator: self,
            id,
            method,
            params,
            rx,
        })
    }

    /// Wake the caller waiting on `response.id`. Unknown ids are logged
    /// and dropped; they occur when a reply lands after a reconnect.
    pub fn deliver(&self, response: Response) {
        let Some(id) = response.id else {
            return;
        };
        let tx = {
            let mut guard = self.pending.lock();
            guard.remove(&id)
        };
        let Some(tx) = tx else {
            tracing::warn!(target: "assistd::mcp", id, "received response for unknown request id");
            return;
        };
        let reply = match (response.result, response.error) {
            (_, Some(err)) => Err(err),
            (Some(value), None) => Ok(value),
            (None, None) => Ok(Value::Null),
        };
        let _ = tx.send(reply);
    }

    /// Drop every pending reply sender, so each waiting receiver
    /// observes a closed channel.
    pub fn fail_all(&self) {
        self.pending.lock().clear();
    }

    pub fn in_flight(&self) -> usize {
        self.pending.lock().len()
    }
}

impl Default for Correlator {
    fn default() -> Self {
        Self::new()
    }
}

/// A reserved request: frame it, send it, then await `rx`. Dropping it
/// releases the id's slot, so a timed-out or abandoned request never
/// counts against [`MAX_IN_FLIGHT`].
#[derive(Debug)]
pub struct Pending<'a> {
    correlator: &'a Correlator,
    pub id: u64,
    method: &'static str,
    params: Value,
    pub rx: oneshot::Receiver<Reply>,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        self.correlator.pending.lock().remove(&self.id);
    }
}

impl Pending<'_> {
    /// [`Self::frame_json`] plus a trailing newline.
    pub fn frame_line(&self) -> Result<Vec<u8>, McpError> {
        let mut bytes = self.frame_json()?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn frame_json(&self) -> Result<Vec<u8>, McpError> {
        let req = Request {
            jsonrpc: "2.0",
            id: self.id,
            method: self.method,
            params: self.params.clone(),
        };
        Ok(serde_json::to_vec(&req)?)
    }
}

/// Encode a notification as a single newline-terminated line.
pub fn notification_line(method: &'static str, params: Value) -> Result<Vec<u8>, McpError> {
    let n = Notification {
        jsonrpc: "2.0",
        method,
        params,
    };
    let mut bytes = serde_json::to_vec(&n)?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::sync::oneshot::error::TryRecvError;

    #[tokio::test]
    async fn deliver_wakes_the_matching_request() {
        let c = Correlator::new();
        let mut pending = c.next_request("ping", json!({})).unwrap();

        c.deliver(Response {
            id: Some(pending.id),
            result: Some(json!({"ok": true})),
            error: None,
        });

        let value = (&mut pending.rx).await.unwrap().unwrap();
        assert_eq!(value, json!({"ok": true}));
        assert_eq!(c.in_flight(), 0);
    }

    #[tokio::test]
    async fn rpc_error_surfaces() {
        let c = Correlator::new();
        let mut pending = c.next_request("bad", json!({})).unwrap();
        c.deliver(Response {
            id: Some(pending.id),
            result: None,
            error: Some(RpcError {
                code: -32601,
                message: "method not found".into(),
                data: None,
            }),
        });
        let err = (&mut pending.rx).await.unwrap().unwrap_err();
        assert_eq!(
            (err.code, err.message.as_str()),
            (-32601, "method not found")
        );
    }

    #[test]
    fn unknown_id_leaves_pending_requests_untouched() {
        let c = Correlator::new();
        let mut pending = c.next_request("ping", json!({})).unwrap();
        c.deliver(Response {
            id: Some(pending.id + 1),
            result: Some(Value::Null),
            error: None,
        });
        assert_eq!(c.in_flight(), 1);
        assert!(matches!(pending.rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[tokio::test]
    async fn fail_all_closes_every_pending_reply_channel() {
        let c = Correlator::new();
        let mut p1 = c.next_request("a", json!({})).unwrap();
        let mut p2 = c.next_request("b", json!({})).unwrap();
        assert_eq!(c.in_flight(), 2);

        c.fail_all();
        assert_eq!(c.in_flight(), 0);

        (&mut p1.rx).await.expect_err("sender dropped");
        (&mut p2.rx).await.expect_err("sender dropped");
    }

    #[test]
    fn in_flight_cap_rejects_until_a_pending_request_is_dropped() {
        let c = Correlator::new();
        let held: Vec<_> = (0..MAX_IN_FLIGHT)
            .map(|_| c.next_request("x", json!({})).unwrap())
            .collect();
        assert!(matches!(
            c.next_request("y", json!({})),
            Err(McpError::TooManyInFlight)
        ));
        drop(held);
        assert_eq!(c.in_flight(), 0);
        c.next_request("y", json!({})).unwrap();
    }

    #[test]
    fn frame_line_encodes_newline_terminated_request() {
        let c = Correlator::new();
        let pending = c.next_request("ping", json!({"echo": 1})).unwrap();
        let line = pending.frame_line().unwrap();
        let (json_bytes, newline) = line.split_at(line.len() - 1);
        assert_eq!(newline, b"\n");
        let parsed: Value = serde_json::from_slice(json_bytes).unwrap();
        assert_eq!(
            parsed,
            json!({"jsonrpc": "2.0", "id": pending.id, "method": "ping", "params": {"echo": 1}})
        );
    }

    #[test]
    fn notification_line_has_no_id() {
        let line = notification_line("notifications/initialized", json!({})).unwrap();
        let (json_bytes, newline) = line.split_at(line.len() - 1);
        assert_eq!(newline, b"\n");
        let parsed: Value = serde_json::from_slice(json_bytes).unwrap();
        assert_eq!(
            parsed,
            json!({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        );
    }
}
