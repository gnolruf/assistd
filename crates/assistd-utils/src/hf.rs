//! Hugging Face file identifiers of the form `<owner>/<repo>[@<revision>]:<file>`.

use std::fmt;

/// A validated Hugging Face file reference. Every component is a plain
/// name: no `.`/`..` segments, no absolute paths, and only URL-safe chars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfFileId {
    owner: String,
    name: String,
    revision: Option<String>,
    file: String,
}

/// Why an identifier failed [`HfFileId::parse`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct InvalidHfId(&'static str);

impl HfFileId {
    /// Parse `<owner>/<repo>[@<revision>]:<file>`, where `<file>` is a
    /// relative `/`-separated path inside the repo.
    pub fn parse(id: &str) -> Result<Self, InvalidHfId> {
        let (repo, file) = id.split_once(':').ok_or(InvalidHfId("missing ':'"))?;
        let (repo, revision) = match repo.split_once('@') {
            Some((repo, revision)) => (repo, Some(revision)),
            None => (repo, None),
        };
        let (owner, name) = repo
            .split_once('/')
            .ok_or(InvalidHfId("repo must be 'owner/name'"))?;
        if !is_plain_name(owner) || !is_plain_name(name) {
            return Err(InvalidHfId(
                "owner and repo name must be non-empty and use only [A-Za-z0-9._-]",
            ));
        }
        if revision.is_some_and(|revision| !is_plain_name(revision)) {
            return Err(InvalidHfId(
                "revision must be non-empty and use only [A-Za-z0-9._-]",
            ));
        }
        if !file.split('/').all(is_plain_name) {
            return Err(InvalidHfId(
                "file must be a relative path of non-empty segments using only \
                 [A-Za-z0-9._+-], with no '.' or '..' segments",
            ));
        }
        Ok(Self {
            owner: owner.to_string(),
            name: name.to_string(),
            revision: revision.map(str::to_string),
            file: file.to_string(),
        })
    }

    /// `<owner>/<repo>`.
    pub fn repo(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }

    /// The pinned revision, if any.
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    /// The file's path inside the repo.
    pub fn file(&self) -> &str {
        &self.file
    }
}

impl fmt::Display for HfFileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.owner, self.name)?;
        if let Some(revision) = &self.revision {
            write!(formatter, "@{revision}")?;
        }
        write!(formatter, ":{}", self.file)
    }
}

fn is_plain_name(segment: &str) -> bool {
    !segment.is_empty()
        && segment != "."
        && segment != ".."
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_file_with_revision() {
        let id =
            HfFileId::parse("rhasspy/piper-voices@3d796cc:en/en_US/lessac/medium/a.onnx").unwrap();
        assert_eq!(id.repo(), "rhasspy/piper-voices");
        assert_eq!(id.revision(), Some("3d796cc"));
        assert_eq!(id.file(), "en/en_US/lessac/medium/a.onnx");
        assert_eq!(
            id.to_string(),
            "rhasspy/piper-voices@3d796cc:en/en_US/lessac/medium/a.onnx"
        );
    }

    #[test]
    fn rejects_malformed_and_escaping_ids() {
        for id in [
            "ggerganov/whisper.cpp",
            "whisper:file.bin",
            "owner/repo:",
            "owner/repo:a:b",
            "/repo:file.bin",
            "owner/:file.bin",
            "owner/repo@:file.bin",
            "owner/repo/../..:file.bin",
            "../repo:file.bin",
            "owner/..:file.bin",
            "owner/repo:/etc/passwd",
            "owner/repo:../../.local/bin/x",
            "owner/repo:a/../../b",
            "owner/repo:a//b",
            "owner/repo:./a",
            "owner/repo:a\\..\\b",
            "owner/repo@../x:file.bin",
            "owner/repo:file.bin?download=1",
        ] {
            assert!(HfFileId::parse(id).is_err(), "{id} should be rejected");
        }
    }
}
