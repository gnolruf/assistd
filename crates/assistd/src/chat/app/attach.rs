//! `/attach`: loading images for the next query and tab-completing their
//! paths.

use std::fs;
use std::path::{Path, PathBuf};

use assistd_tools::{Attachment, load_image_attachment};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;

use super::{App, AttachLoadedPayload, ChatEvent, PendingAttachment};

impl App {
    /// Read and decode the image at the shell-quoted `raw` path in the
    /// background; the result arrives as a [`ChatEvent`].
    pub(super) fn load_attachment(&mut self, raw: &str) {
        let Some(args) = shlex::split(raw) else {
            self.output
                .push_error("/attach: unterminated quote in path");
            self.set_notice("/attach: bad quoting");
            return;
        };
        let path = match <[String; 1]>::try_from(args) {
            Ok([path]) => path,
            Err(args) => {
                self.output.push_error(&format!(
                    "/attach: expected exactly one path, got {}",
                    args.len()
                ));
                self.set_notice("/attach: need one path");
                return;
            }
        };
        let name = Path::new(&path)
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        self.set_notice(&format!("📎 reading {name}…"));
        let chat_tx = self.chat_tx.clone();
        let picker = self.picker.clone();
        self.tasks.spawn(async move {
            let event = read_attachment(path, name, picker).await;
            let _ = chat_tx.send(event).await;
        });
    }

    pub(super) fn on_attach_loaded(&mut self, payload: AttachLoadedPayload) {
        let AttachLoadedPayload {
            name,
            mime,
            size,
            bytes,
            protocol,
        } = payload;
        let label = format!("📎 attached: {name} ({mime}, {})", human_size_short(size));
        self.output.push_info(&label);
        self.set_notice(&format!("📎 {name} attached"));
        self.pending_attachments.push(PendingAttachment {
            name,
            mime,
            bytes,
            protocol,
        });
    }

    pub(super) fn on_attach_failed(&mut self, path: &str, message: &str) {
        self.output
            .push_error(&format!("/attach {path}: {message}"));
        self.set_notice(&format!("📎 {path}: {message}"));
    }

    /// Tab-complete the path after `/attach `. Returns whether the buffer
    /// holds an `/attach` command, whether or not anything completed.
    pub(super) fn try_complete_attach_path(&mut self) -> bool {
        let Some(partial) = self.input.buffer().strip_prefix("/attach ") else {
            return false;
        };
        if let Some(completed) = complete_path(partial) {
            self.input.set_buffer(format!("/attach {completed}"));
        }
        true
    }
}

async fn read_attachment(path: String, name: String, picker: Option<Picker>) -> ChatEvent {
    match load_image_attachment(Path::new(&path)).await {
        Ok((Attachment::Image { mime, bytes }, size)) => {
            let protocol = picker.and_then(|picker| thumbnail_protocol(&picker, &bytes, &path));
            ChatEvent::AttachLoaded(Box::new(AttachLoadedPayload {
                name,
                mime,
                size,
                bytes,
                protocol,
            }))
        }
        Err(e) => ChatEvent::AttachFailed {
            path,
            message: e.user_message(),
        },
    }
}

fn thumbnail_protocol(picker: &Picker, bytes: &[u8], path: &str) -> Option<StatefulProtocol> {
    match image::load_from_memory(bytes) {
        Ok(img) => Some(picker.new_resize_protocol(img)),
        Err(e) => {
            tracing::warn!("/attach: thumbnail decode failed for {path}: {e}");
            None
        }
    }
}

/// `partial` extended as far as the matching directory entries agree, or
/// `None` when that adds nothing.
fn complete_path(partial: &str) -> Option<String> {
    let (search_dir, dir_part, file_prefix) = match partial.rsplit_once('/') {
        Some((dir, file)) => {
            let search_dir = if dir.is_empty() {
                PathBuf::from("/")
            } else {
                expand_tilde(dir)
            };
            (search_dir, format!("{dir}/"), file)
        }
        None => (PathBuf::from("."), String::new(), partial),
    };
    let entries = matching_entries(&search_dir, file_prefix);
    let completed = completed_name(&entries, file_prefix)?;
    Some(format!("{dir_part}{completed}"))
}

/// `(name, is_dir)` of each entry in `dir` starting with `prefix`; empty
/// when `dir` cannot be read.
fn matching_entries(dir: &Path, prefix: &str) -> Vec<(String, bool)> {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return Vec::new();
    };
    read_dir
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.starts_with(prefix).then(|| {
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                (name, is_dir)
            })
        })
        .collect()
}

/// A lone match completes in full, with a trailing `/` for a directory;
/// several complete to their common prefix when it extends `prefix`.
fn completed_name(entries: &[(String, bool)], prefix: &str) -> Option<String> {
    if let [(name, is_dir)] = entries {
        return Some(if *is_dir {
            format!("{name}/")
        } else {
            name.clone()
        });
    }
    let names: Vec<&str> = entries.iter().map(|(name, _)| name.as_str()).collect();
    let common = longest_common_prefix(&names);
    (common.len() > prefix.len()).then(|| common.to_string())
}

fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    if p == "~"
        && let Ok(home) = std::env::var("HOME")
    {
        return PathBuf::from(home);
    }
    PathBuf::from(p)
}

pub(super) fn longest_common_prefix<'a>(xs: &[&'a str]) -> &'a str {
    let Some((first, rest)) = xs.split_first() else {
        return "";
    };
    let end = rest.iter().fold(first.len(), |end, s| {
        first[..end]
            .char_indices()
            .zip(s.chars())
            .find(|((_, a), b)| a != b)
            .map_or(end.min(s.len()), |((i, _), _)| i)
    });
    &first[..end]
}

fn human_size_short(n: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    if n >= GB {
        format!("{:.1}GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.1}MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{}KB", n / KB)
    } else {
        format!("{n}B")
    }
}
