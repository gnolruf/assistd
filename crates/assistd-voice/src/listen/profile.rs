//! The enrolled wake-phrase profile on disk: `profile.json` plus the
//! enrollment recordings as `samples/NN.wav`, all owner-only.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use assistd_config::UnitInterval;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::enrollment::{Enrollment, Recording};
use super::vad::SAMPLE_RATE_HZ;
use super::wake_phrase::WakePhrase;

/// Profile format this build reads and writes.
pub const PROFILE_VERSION: u32 = 1;

const PROFILE_FILE: &str = "profile.json";
const SAMPLES_DIR: &str = "samples";
const DIR_MODE: u32 = 0o700;
const FILE_MODE: u32 = 0o600;

/// Errors reading, writing or deleting a [`ProfileStore`].
#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("wake-word profile I/O failed at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("wake-word profile at {path} is malformed: {source}")]
    Malformed {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "wake-word profile is format version {found}, but this build reads version \
         {PROFILE_VERSION}; re-enroll the wake phrase"
    )]
    Version { found: u32 },
    #[error("failed to write enrollment recording {path}: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("wake-word profile worker panicked: {0}")]
    Worker(#[from] tokio::task::JoinError),
}

/// An enrolled wake phrase and the spellings Whisper was heard to produce for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WakeWordProfile {
    pub phrase: String,
    pub variants: Vec<String>,
    /// What Whisper heard in each recording, in `samples/` file order.
    pub samples: Vec<String>,
}

impl WakeWordProfile {
    pub fn wake_phrase(&self, min_similarity: UnitInterval) -> WakePhrase {
        WakePhrase::new(&self.phrase, &self.variants, min_similarity)
    }
}

#[derive(Serialize)]
struct ProfileFileRef<'a> {
    version: u32,
    #[serde(flatten)]
    profile: &'a WakeWordProfile,
}

#[derive(Deserialize)]
struct ProfileFile {
    version: u32,
    #[serde(flatten)]
    profile: serde_json::Value,
}

/// The directory holding the enrolled profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileStore {
    dir: PathBuf,
}

impl ProfileStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// `$XDG_DATA_HOME/assistd/wakeword`, or `$HOME/.local/share/assistd/wakeword`.
    pub fn default_location() -> Self {
        let data_dir = dirs::data_dir().unwrap_or_else(|| PathBuf::from(".local/share"));
        Self::new(data_dir.join("assistd").join("wakeword"))
    }

    /// The saved profile, `None` when nothing is enrolled. A profile of
    /// another format version is [`ProfileError::Version`].
    pub async fn load(&self) -> Result<Option<WakeWordProfile>, ProfileError> {
        let path = self.dir.join(PROFILE_FILE);
        tokio::task::spawn_blocking(move || read_profile(&path)).await?
    }

    /// Write `enrollment` as the profile, replacing any saved one: the
    /// recordings and the profile each land under a temporary name first.
    pub async fn save(&self, enrollment: Enrollment) -> Result<WakeWordProfile, ProfileError> {
        let dir = self.dir.clone();
        tokio::task::spawn_blocking(move || write_profile(&dir, enrollment)).await?
    }

    /// Delete the saved profile and its recordings; a no-op when none exists.
    pub async fn reset(&self) -> Result<(), ProfileError> {
        let dir = self.dir.clone();
        tokio::task::spawn_blocking(move || remove_dir_if_present(&dir)).await?
    }
}

fn read_profile(path: &Path) -> Result<Option<WakeWordProfile>, ProfileError> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(io_error(path, source)),
    };
    let malformed = |source| ProfileError::Malformed {
        path: path.to_path_buf(),
        source,
    };
    let file: ProfileFile = serde_json::from_slice(&bytes).map_err(malformed)?;
    if file.version != PROFILE_VERSION {
        return Err(ProfileError::Version {
            found: file.version,
        });
    }
    serde_json::from_value(file.profile)
        .map(Some)
        .map_err(malformed)
}

fn write_profile(dir: &Path, enrollment: Enrollment) -> Result<WakeWordProfile, ProfileError> {
    let (phrase, variants, recordings) = enrollment.into_parts();
    create_private_dir(dir)?;
    let staged_samples = dir.join(format!("{SAMPLES_DIR}.new"));
    remove_dir_if_present(&staged_samples)?;
    create_private_dir(&staged_samples)?;
    for (index, recording) in recordings.iter().enumerate() {
        write_wav(&staged_samples.join(sample_file_name(index)), recording)?;
    }
    let profile = WakeWordProfile {
        phrase,
        variants,
        samples: recordings
            .into_iter()
            .map(|recording| recording.heard)
            .collect(),
    };
    let staged_profile = dir.join(format!("{PROFILE_FILE}.new"));
    write_profile_file(&staged_profile, &profile)?;

    let samples = dir.join(SAMPLES_DIR);
    remove_dir_if_present(&samples)?;
    fs::rename(&staged_samples, &samples).map_err(|source| io_error(&samples, source))?;
    let path = dir.join(PROFILE_FILE);
    fs::rename(&staged_profile, &path).map_err(|source| io_error(&path, source))?;
    Ok(profile)
}

fn sample_file_name(index: usize) -> String {
    format!("{:02}.wav", index + 1)
}

fn write_profile_file(path: &Path, profile: &WakeWordProfile) -> Result<(), ProfileError> {
    let contents = ProfileFileRef {
        version: PROFILE_VERSION,
        profile,
    };
    let json = serde_json::to_vec_pretty(&contents).map_err(|source| ProfileError::Malformed {
        path: path.to_path_buf(),
        source,
    })?;
    let mut file = create_private_file(path)?;
    file.write_all(&json)
        .and_then(|()| file.sync_all())
        .map_err(|source| io_error(path, source))
}

fn write_wav(path: &Path, recording: &Recording) -> Result<(), ProfileError> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE_HZ,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let wav_error = |source| ProfileError::Wav {
        path: path.to_path_buf(),
        source,
    };
    let file = BufWriter::new(create_private_file(path)?);
    let mut writer = hound::WavWriter::new(file, spec).map_err(wav_error)?;
    recording
        .pcm
        .iter()
        .try_for_each(|&sample| writer.write_sample(sample))
        .map_err(wav_error)?;
    writer.finalize().map_err(wav_error)
}

fn create_private_dir(dir: &Path) -> Result<(), ProfileError> {
    DirBuilder::new()
        .recursive(true)
        .mode(DIR_MODE)
        .create(dir)
        .map_err(|source| io_error(dir, source))
}

fn create_private_file(path: &Path) -> Result<File, ProfileError> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(FILE_MODE)
        .open(path)
        .map_err(|source| io_error(path, source))
}

fn remove_dir_if_present(dir: &Path) -> Result<(), ProfileError> {
    match fs::remove_dir_all(dir) {
        Err(source) if source.kind() != io::ErrorKind::NotFound => Err(io_error(dir, source)),
        _ => Ok(()),
    }
}

fn io_error(path: &Path, source: io::Error) -> ProfileError {
    ProfileError::Io {
        path: path.to_path_buf(),
        source,
    }
}
