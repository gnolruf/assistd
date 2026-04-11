//! Minimal GGUF v2/v3 header parser.
//!
//! Just enough of the format to pull `general.architecture` and
//! `{arch}.block_count` out of the metadata section, plus a helper that turns
//! a VRAM budget into a `-ngl` value. Everything is read little-endian, per
//! spec. Pathologically large fields are rejected by the safety caps below.

use super::error::LlamaServerError;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

// Safety caps. GGUF files can declare huge sizes; we refuse anything that
// would require more than a handful of MiB of scanning to locate `block_count`.
const MAX_KEY_LEN: u64 = 1024;
const MAX_STRING_VALUE_LEN: u64 = 64 * 1024;
const MAX_ARRAY_LEN: u64 = 1_000_000;
const MAX_BYTES_SCANNED: u64 = 16 * 1024 * 1024;

// GGUF value type tags. Spec: github.com/ggml-org/ggml/blob/master/docs/gguf.md
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// Just the two metadata fields the supervisor cares about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub architecture: String,
    pub block_count: u32,
}

/// Inputs to [`compute_ngl`]: total layer count and the model file size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerBudget {
    pub file_size_mb: u64,
    pub block_count: u32,
}

pub fn parse_header(path: &Path) -> Result<GgufHeader, LlamaServerError> {
    let file = File::open(path).map_err(LlamaServerError::Io)?;
    let reader = BufReader::new(file);
    let mut r = CountingReader::new(reader, path.to_path_buf());

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(LlamaServerError::GgufInvalid {
            path: r.path.clone(),
            reason: format!("bad magic {magic:?}"),
        });
    }

    let version = r.read_u32_le()?;
    if version != 2 && version != 3 {
        return Err(LlamaServerError::GgufUnsupportedVersion {
            path: r.path.clone(),
            version,
        });
    }

    let _tensor_count = r.read_u64_le()?;
    let metadata_kv_count = r.read_u64_le()?;

    let mut architecture: Option<String> = None;
    let mut block_count: Option<u32> = None;

    for _ in 0..metadata_kv_count {
        let key = r.read_string()?;
        let value_type = r.read_u32_le()?;

        // Capture or skip the value depending on whether we care about this key.
        if key == "general.architecture" && value_type == GGUF_TYPE_STRING {
            architecture = Some(r.read_string()?);
        } else if let Some(arch) = architecture.as_ref() {
            let expected = format!("{arch}.block_count");
            if key == expected {
                // block_count is conventionally u32, but some models use u64.
                match value_type {
                    GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 => {
                        block_count = Some(r.read_u32_le()?);
                    }
                    GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 => {
                        let v = r.read_u64_le()?;
                        block_count = Some(v.min(u32::MAX as u64) as u32);
                    }
                    _ => {
                        return Err(LlamaServerError::GgufInvalid {
                            path: r.path.clone(),
                            reason: format!("block_count has unexpected type tag {value_type}"),
                        });
                    }
                }
            } else {
                r.skip_value(value_type)?;
            }
        } else {
            r.skip_value(value_type)?;
        }

        if architecture.is_some() && block_count.is_some() {
            break;
        }

        if r.bytes_read > MAX_BYTES_SCANNED {
            return Err(LlamaServerError::GgufMetadataMissing {
                path: r.path.clone(),
                key: "block_count",
            });
        }
    }

    let architecture = architecture.ok_or_else(|| LlamaServerError::GgufMetadataMissing {
        path: r.path.clone(),
        key: "general.architecture",
    })?;
    let block_count = block_count.ok_or_else(|| LlamaServerError::GgufMetadataMissing {
        path: r.path.clone(),
        key: "block_count",
    })?;

    Ok(GgufHeader {
        architecture,
        block_count,
    })
}

/// Derive `-ngl` from a VRAM budget, the configured context length, and the
/// model's layer footprint. Returns the number of layers to offload to GPU,
/// clamped into `[0, block_count]`.
///
/// Overhead accounting is intentionally conservative: a fixed 512 MiB is
/// reserved for CUDA runtime/workspace, plus ~0.5 MiB per context token as a
/// rough KV-cache approximation. These are `const`s, easy to tune.
pub fn compute_ngl(budget_mb: u64, context_length: u32, layers: &LayerBudget) -> u32 {
    const FIXED_OVERHEAD_MB: u64 = 512;

    if layers.block_count == 0 || budget_mb == 0 {
        return 0;
    }
    let kv_overhead = (context_length as u64) / 2;
    let overhead = FIXED_OVERHEAD_MB + kv_overhead;
    if budget_mb <= overhead {
        return 0;
    }
    let usable = budget_mb - overhead;
    // `div_ceil` over-estimates per-layer cost, which errs on the safe side
    // (better to offload one fewer layer than to OOM the GPU).
    let per_layer = layers
        .file_size_mb
        .div_ceil(layers.block_count as u64)
        .max(1);
    let ngl = usable / per_layer;
    ngl.min(layers.block_count as u64) as u32
}

// ---------------------------------------------------------------------------
// Internal reader that tracks bytes consumed so the safety caps can fire.
// ---------------------------------------------------------------------------

struct CountingReader<R: Read> {
    inner: R,
    bytes_read: u64,
    path: PathBuf,
}

impl<R: Read> CountingReader<R> {
    fn new(inner: R, path: PathBuf) -> Self {
        Self {
            inner,
            bytes_read: 0,
            path,
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), LlamaServerError> {
        self.inner.read_exact(buf)?;
        self.bytes_read = self.bytes_read.saturating_add(buf.len() as u64);
        Ok(())
    }

    fn read_u32_le(&mut self) -> Result<u32, LlamaServerError> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64_le(&mut self) -> Result<u64, LlamaServerError> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String, LlamaServerError> {
        let len = self.read_u64_le()?;
        if len > MAX_KEY_LEN.max(MAX_STRING_VALUE_LEN) {
            return Err(LlamaServerError::GgufInvalid {
                path: self.path.clone(),
                reason: format!("string length {len} exceeds cap"),
            });
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|e| LlamaServerError::GgufInvalid {
            path: self.path.clone(),
            reason: format!("non-UTF-8 string: {e}"),
        })
    }

    fn skip_bytes(&mut self, n: u64) -> Result<(), LlamaServerError> {
        // Read in chunks to avoid allocating a giant buffer for benign cases.
        let mut remaining = n;
        let mut buf = [0u8; 4096];
        while remaining > 0 {
            let take = remaining.min(buf.len() as u64) as usize;
            self.read_exact(&mut buf[..take])?;
            remaining -= take as u64;
        }
        Ok(())
    }

    fn skip_value(&mut self, value_type: u32) -> Result<(), LlamaServerError> {
        match value_type {
            GGUF_TYPE_UINT8 | GGUF_TYPE_INT8 | GGUF_TYPE_BOOL => self.skip_bytes(1),
            GGUF_TYPE_UINT16 | GGUF_TYPE_INT16 => self.skip_bytes(2),
            GGUF_TYPE_UINT32 | GGUF_TYPE_INT32 | GGUF_TYPE_FLOAT32 => self.skip_bytes(4),
            GGUF_TYPE_UINT64 | GGUF_TYPE_INT64 | GGUF_TYPE_FLOAT64 => self.skip_bytes(8),
            GGUF_TYPE_STRING => {
                let len = self.read_u64_le()?;
                if len > MAX_STRING_VALUE_LEN {
                    return Err(LlamaServerError::GgufInvalid {
                        path: self.path.clone(),
                        reason: format!("string length {len} exceeds cap"),
                    });
                }
                self.skip_bytes(len)
            }
            GGUF_TYPE_ARRAY => {
                let elem_type = self.read_u32_le()?;
                let count = self.read_u64_le()?;
                if count > MAX_ARRAY_LEN {
                    return Err(LlamaServerError::GgufInvalid {
                        path: self.path.clone(),
                        reason: format!("array length {count} exceeds cap"),
                    });
                }
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
                Ok(())
            }
            other => Err(LlamaServerError::GgufInvalid {
                path: self.path.clone(),
                reason: format!("unknown value type tag {other}"),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Builder {
        buf: Vec<u8>,
    }

    impl Builder {
        fn new() -> Self {
            Self { buf: Vec::new() }
        }
        fn magic(mut self) -> Self {
            self.buf.extend_from_slice(GGUF_MAGIC);
            self
        }
        fn bad_magic(mut self) -> Self {
            self.buf.extend_from_slice(b"XXXX");
            self
        }
        fn u32(mut self, v: u32) -> Self {
            self.buf.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(mut self, v: u64) -> Self {
            self.buf.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn string(mut self, s: &str) -> Self {
            self.buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(s.as_bytes());
            self
        }
        fn metadata_string(self, key: &str, value: &str) -> Self {
            self.string(key).u32(GGUF_TYPE_STRING).string(value)
        }
        fn metadata_u32(self, key: &str, value: u32) -> Self {
            self.string(key).u32(GGUF_TYPE_UINT32).u32(value)
        }
        fn metadata_u64(self, key: &str, value: u64) -> Self {
            self.string(key).u32(GGUF_TYPE_UINT64).u64(value)
        }
        fn metadata_array_u32(self, key: &str, values: &[u32]) -> Self {
            let mut s = self.string(key).u32(GGUF_TYPE_ARRAY);
            s = s.u32(GGUF_TYPE_UINT32);
            s = s.u64(values.len() as u64);
            for v in values {
                s = s.u32(*v);
            }
            s
        }
    }

    fn write_tmp(bytes: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(bytes).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn parse_header_rejects_bad_magic() {
        let blob = Builder::new().bad_magic().u32(3).u64(0).u64(0).buf;
        let tmp = write_tmp(&blob);
        let err = parse_header(tmp.path()).unwrap_err();
        assert!(matches!(err, LlamaServerError::GgufInvalid { .. }));
    }

    #[test]
    fn parse_header_rejects_unsupported_version() {
        let blob = Builder::new().magic().u32(1).u64(0).u64(0).buf;
        let tmp = write_tmp(&blob);
        let err = parse_header(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            LlamaServerError::GgufUnsupportedVersion { version: 1, .. }
        ));
    }

    #[test]
    fn parse_header_reads_minimal_valid_blob() {
        let blob = Builder::new()
            .magic()
            .u32(3)
            .u64(0) // tensor_count
            .u64(2) // metadata_kv_count
            .metadata_string("general.architecture", "llama")
            .metadata_u32("llama.block_count", 32)
            .buf;
        let tmp = write_tmp(&blob);
        let header = parse_header(tmp.path()).unwrap();
        assert_eq!(header.architecture, "llama");
        assert_eq!(header.block_count, 32);
    }

    #[test]
    fn parse_header_skips_unrelated_keys() {
        let blob = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(4)
            .metadata_string("general.name", "Mystery")
            .metadata_u64("general.parameter_count", 7_000_000_000)
            .metadata_string("general.architecture", "qwen2")
            .metadata_u32("qwen2.block_count", 28)
            .buf;
        let tmp = write_tmp(&blob);
        let header = parse_header(tmp.path()).unwrap();
        assert_eq!(header.architecture, "qwen2");
        assert_eq!(header.block_count, 28);
    }

    #[test]
    fn parse_header_handles_array_metadata() {
        let blob = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(3)
            .metadata_string("general.architecture", "llama")
            .metadata_array_u32("tokenizer.ggml.token_type", &[1, 2, 3, 4, 5])
            .metadata_u32("llama.block_count", 80)
            .buf;
        let tmp = write_tmp(&blob);
        let header = parse_header(tmp.path()).unwrap();
        assert_eq!(header.architecture, "llama");
        assert_eq!(header.block_count, 80);
    }

    #[test]
    fn parse_header_errors_on_missing_block_count() {
        let blob = Builder::new()
            .magic()
            .u32(3)
            .u64(0)
            .u64(1)
            .metadata_string("general.architecture", "llama")
            .buf;
        let tmp = write_tmp(&blob);
        let err = parse_header(tmp.path()).unwrap_err();
        assert!(matches!(
            err,
            LlamaServerError::GgufMetadataMissing {
                key: "block_count",
                ..
            }
        ));
    }

    #[test]
    fn compute_ngl_zero_budget_is_zero() {
        let layers = LayerBudget {
            file_size_mb: 4000,
            block_count: 32,
        };
        assert_eq!(compute_ngl(0, 4096, &layers), 0);
    }

    #[test]
    fn compute_ngl_zero_layers_is_zero() {
        let layers = LayerBudget {
            file_size_mb: 4000,
            block_count: 0,
        };
        assert_eq!(compute_ngl(16_000, 4096, &layers), 0);
    }

    #[test]
    fn compute_ngl_budget_below_overhead_is_zero() {
        let layers = LayerBudget {
            file_size_mb: 4000,
            block_count: 32,
        };
        // 512 MiB fixed + 2048 MiB kv = 2560 MiB of overhead
        assert_eq!(compute_ngl(2560, 4096, &layers), 0);
    }

    #[test]
    fn compute_ngl_budget_exceeds_model_clamps_to_block_count() {
        let layers = LayerBudget {
            file_size_mb: 4000,
            block_count: 32,
        };
        // Massive budget; ngl must not exceed block_count.
        assert_eq!(compute_ngl(1_000_000, 4096, &layers), 32);
    }

    #[test]
    fn compute_ngl_typical_case() {
        // 7B-class: ~4 GiB file, 32 layers.
        // Budget 16 GiB, context 4096 → overhead = 512 + 2048 = 2560.
        // usable = 16384 - 2560 = 13824; per_layer ≈ 125; ngl = 110 → clamped to 32.
        let layers = LayerBudget {
            file_size_mb: 4000,
            block_count: 32,
        };
        assert_eq!(compute_ngl(16_000, 4096, &layers), 32);
    }

    #[test]
    fn compute_ngl_tight_budget_partial_offload() {
        // 14 GiB file, 32 layers → per_layer ≈ 448 MiB.
        // Budget 6 GiB (6144 MiB), context 4096 → overhead 2560, usable 3584.
        // 3584 / 448 = 8 layers.
        let layers = LayerBudget {
            file_size_mb: 14_000,
            block_count: 32,
        };
        assert_eq!(compute_ngl(6144, 4096, &layers), 8);
    }
}
