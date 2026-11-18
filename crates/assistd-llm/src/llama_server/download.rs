//! The router's model event stream, read for the download of one model.

use std::collections::HashMap;
use std::time::Duration;

use assistd_ipc::DownloadProgress;
use serde::Deserialize;
use serde_json::Value;
use tokio::time::{Instant, timeout_at};

use super::error::LlamaServerError;
use crate::completion::sse::{SseEvent, SseLineReader};

const DOWNLOAD_PROGRESS: &str = "download_progress";
const DOWNLOAD_FINISHED: &str = "download_finished";
const DOWNLOAD_FAILED: &str = "download_failed";

/// `GET /models/sse` narrowed to the download events of one model.
pub(super) struct DownloadEvents {
    response: reqwest::Response,
    reader: SseLineReader,
    tally: DownloadTally,
}

impl DownloadEvents {
    pub(super) fn new(response: reqwest::Response, model: &str) -> Self {
        Self {
            response,
            reader: SseLineReader::new(),
            tally: DownloadTally::new(model),
        }
    }

    /// Read until the router reports the download finished, passing each
    /// byte count to `on_progress`. Fails once `stall` passes without the
    /// count growing.
    pub(super) async fn wait_until_finished(
        mut self,
        stall: Duration,
        mut on_progress: impl FnMut(DownloadProgress),
    ) -> Result<(), LlamaServerError> {
        let mut most_done = 0;
        let mut stalled_at = Instant::now() + stall;
        loop {
            let event = timeout_at(stalled_at, self.next_event())
                .await
                .map_err(|_| LlamaServerError::DownloadStalled {
                    model: self.tally.model.clone(),
                    idle: stall,
                })??;
            match event {
                Some(DownloadEvent::Progress(progress)) => {
                    if progress.done_bytes > most_done {
                        most_done = progress.done_bytes;
                        stalled_at = Instant::now() + stall;
                    }
                    on_progress(progress);
                }
                Some(DownloadEvent::Finished) => return Ok(()),
                Some(DownloadEvent::Failed) => {
                    return Err(LlamaServerError::DownloadFailed {
                        model: self.tally.model,
                    });
                }
                None => {
                    return Err(LlamaServerError::DownloadInterrupted {
                        model: self.tally.model,
                    });
                }
            }
        }
    }

    /// `None` once the router closes the stream.
    async fn next_event(&mut self) -> Result<Option<DownloadEvent>, LlamaServerError> {
        loop {
            while let Some(event) = self
                .reader
                .next_event()
                .map_err(LlamaServerError::EventStream)?
            {
                let SseEvent::Data(payload) = event else {
                    continue;
                };
                if let Some(event) = self.tally.record(&payload)? {
                    return Ok(Some(event));
                }
            }
            match self.response.chunk().await? {
                Some(chunk) => self.reader.feed(&chunk),
                None => return Ok(None),
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DownloadEvent {
    Progress(DownloadProgress),
    Finished,
    Failed,
}

struct DownloadTally {
    model: String,
    files: HashMap<String, FileProgress>,
}

impl DownloadTally {
    fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            files: HashMap::new(),
        }
    }

    /// Progress is summed over every file seen so far; an event need not
    /// name them all.
    fn record(&mut self, payload: &str) -> Result<Option<DownloadEvent>, LlamaServerError> {
        let event: ModelEvent = serde_json::from_str(payload)?;
        if event.model != self.model {
            return Ok(None);
        }
        match event.event.as_str() {
            DOWNLOAD_PROGRESS => {
                let report = DownloadReport::deserialize(event.data)?;
                self.files.extend(report.progress);
                Ok(Some(DownloadEvent::Progress(self.progress())))
            }
            DOWNLOAD_FINISHED => Ok(Some(DownloadEvent::Finished)),
            DOWNLOAD_FAILED => Ok(Some(DownloadEvent::Failed)),
            _ => Ok(None),
        }
    }

    fn progress(&self) -> DownloadProgress {
        self.files
            .values()
            .fold(DownloadProgress::default(), |sum, file| DownloadProgress {
                done_bytes: sum.done_bytes.saturating_add(file.done),
                total_bytes: sum.total_bytes.saturating_add(file.total),
            })
    }
}

/// The shape of `data` depends on `event`, so it is decoded per kind.
#[derive(Deserialize)]
struct ModelEvent {
    model: String,
    event: String,
    #[serde(default)]
    data: Value,
}

#[derive(Deserialize)]
struct DownloadReport {
    /// By file URL.
    progress: HashMap<String, FileProgress>,
}

#[derive(Deserialize)]
struct FileProgress {
    done: u64,
    total: u64,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const MODEL: &str = "owner/repo-GGUF:Q4_K_M";

    fn file_progress(url: &str, done: u64, total: u64) -> Value {
        json!({
            "model": MODEL,
            "event": "download_progress",
            "data": {"progress": {url: {"done": done, "total": total}}},
        })
    }

    #[test]
    fn events_are_read_for_one_model_with_bytes_summed_over_its_files() {
        let mut tally = DownloadTally::new(MODEL);
        let mut read = |payload: Value| tally.record(&payload.to_string()).unwrap();
        let progress = |done_bytes, total_bytes| {
            Some(DownloadEvent::Progress(DownloadProgress {
                done_bytes,
                total_bytes,
            }))
        };
        let loading = json!({
            "model": MODEL,
            "event": "model_status",
            "data": {"status": "loading", "progress": {"stages": ["text_model"], "value": 0.5}},
        });

        assert_eq!(
            read(file_progress("https://hf/model.gguf", 0, 100)),
            progress(0, 100)
        );
        assert_eq!(
            read(file_progress("https://hf/mmproj.gguf", 10, 50)),
            progress(10, 150)
        );
        assert_eq!(
            read(file_progress("https://hf/model.gguf", 100, 100)),
            progress(110, 150)
        );
        assert_eq!(
            read(json!({"model": "other/model", "event": "download_finished"})),
            None
        );
        assert_eq!(read(loading), None);
        assert_eq!(
            read(json!({"model": MODEL, "event": "download_failed"})),
            Some(DownloadEvent::Failed)
        );
        assert_eq!(
            read(json!({"model": MODEL, "event": "download_finished"})),
            Some(DownloadEvent::Finished)
        );
    }
}
