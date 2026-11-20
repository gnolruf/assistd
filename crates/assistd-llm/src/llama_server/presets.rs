//! The presets file the router reads at launch: the context every model loads
//! with, and the OCR model under a name and a context of its own.

use std::fs;
use std::io::{self, ErrorKind};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::page_transcriber::MAX_TRANSCRIPTION_TOKENS;

/// The name the router serves the OCR model under. A preset named after the model
/// itself would stop the router downloading it.
pub const OCR_MODEL_ALIAS: &str = "assistd-ocr";

/// Most tokens the image of one rendered page takes; 4,094 were measured.
const PAGE_IMAGE_TOKENS: u32 = 4_100;

/// Tokens kept for the prompt and the chat template around a page image.
const PAGE_PROMPT_TOKENS: u32 = 512;

/// The context the OCR model loads with: one page image, its prompt and its
/// transcription.
const OCR_CONTEXT_TOKENS: u32 = PAGE_IMAGE_TOKENS + PAGE_PROMPT_TOKENS + MAX_TRANSCRIPTION_TOKENS;

/// Where the router's presets file is written, and the OCR model it defines.
#[derive(Debug, Clone)]
pub struct ModelPresets {
    path: Arc<Path>,
    ocr_model: Option<Arc<str>>,
}

impl ModelPresets {
    /// Presets written to `path` that serve `ocr_model`, a Hugging Face name, as
    /// [`OCR_MODEL_ALIAS`] when there is one.
    pub fn new(path: PathBuf, ocr_model: Option<&str>) -> Self {
        Self {
            path: path.into(),
            ocr_model: ocr_model.map(Arc::from),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the file for a router whose other models load with `context_length`
    /// tokens. Errors on an OCR model name that holds a control character.
    pub(crate) fn write(&self, context_length: NonZeroU32) -> io::Result<()> {
        let presets = render(context_length, self.ocr_model.as_deref())?;
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(&self.path, presets)
    }
}

/// The file's text: a key is a server flag without its dashes, `[*]` holds what
/// every model loads with, and a named section overrides it for that model.
fn render(context_length: NonZeroU32, ocr_model: Option<&str>) -> io::Result<String> {
    let ocr_section = ocr_model.map(ocr_section).transpose()?.unwrap_or_default();
    Ok(format!(
        "version = 1\n\n[*]\nctx-size = {context_length}\n{ocr_section}"
    ))
}

fn ocr_section(model: &str) -> io::Result<String> {
    if model.chars().any(char::is_control) {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "documents.ocr.model holds a control character",
        ));
    }
    Ok(format!(
        "\n[{OCR_MODEL_ALIAS}]\nhf-repo = {model}\nctx-size = {OCR_CONTEXT_TOKENS}\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ocr_model_gets_a_section_of_its_own_and_cannot_add_another() {
        let context = NonZeroU32::new(65_536).expect("non-zero");
        assert_eq!(
            render(context, None).unwrap(),
            "version = 1\n\n[*]\nctx-size = 65536\n"
        );
        assert_eq!(
            render(context, Some("ggml-org/GLM-OCR-GGUF:Q8_0")).unwrap(),
            "version = 1\n\n[*]\nctx-size = 65536\n\n[assistd-ocr]\n\
             hf-repo = ggml-org/GLM-OCR-GGUF:Q8_0\nctx-size = 8708\n"
        );
        let injected = render(context, Some("a/b:Q8_0\n[*]\ntools = all"));
        assert_eq!(injected.unwrap_err().kind(), ErrorKind::InvalidInput);
    }
}
