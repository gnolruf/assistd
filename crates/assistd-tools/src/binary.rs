//! Telling binary bytes from text, so that only text reaches the model.

use assistd_utils::binary::sniff_binary;

/// Why `raw` must not reach the model: a sniffed MIME type (or
/// `application/octet-stream`) for NUL bytes, `invalid-utf8`, or
/// `control-chars` above 10% controls. `None` for plain text.
pub(crate) fn binary_label(raw: &[u8]) -> Option<String> {
    if raw.is_empty() {
        return None;
    }

    if raw.contains(&0u8) {
        let mime = sniff_binary(raw).unwrap_or("application/octet-stream");
        return Some(mime.to_owned());
    }

    let Ok(s) = std::str::from_utf8(raw) else {
        return Some("invalid-utf8".into());
    };

    let total = s.chars().count();
    if total == 0 {
        return None;
    }
    let controls = s.chars().filter(|c| is_suspicious_control(*c)).count();
    if controls * 10 > total {
        return Some("control-chars".into());
    }

    None
}

fn is_suspicious_control(c: char) -> bool {
    let cp = c as u32;
    if cp == 0x09 || cp == 0x0A || cp == 0x0D {
        return false;
    }
    cp <= 0x1F || cp == 0x7F
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::PNG_BYTES;

    #[test]
    fn binary_label_flags_bytes_the_model_should_not_see() {
        let mut nul_text = b"plain text".to_vec();
        nul_text.push(0);
        let cases: [(&str, &[u8], Option<&str>); 8] = [
            ("png magic", PNG_BYTES, Some("image/png")),
            (
                "NUL without magic",
                &nul_text[..],
                Some("application/octet-stream"),
            ),
            (
                "invalid utf-8",
                &[0xC3, 0x28, b' ', b'h', b'i'][..],
                Some("invalid-utf8"),
            ),
            (
                "15% controls",
                b"abcdef\x01\x02\x03ghijklmnopq",
                Some("control-chars"),
            ),
            (
                "exactly 10% controls is still text",
                b"abcdefgh\x01\x02ijklmnopqr",
                None,
            ),
            ("tabs and newlines", b"a\tb\nc\td\ne\tf\n", None),
            ("empty", b"", None),
            ("multibyte utf-8", "héllo wörld ñ 日本語\n".as_bytes(), None),
        ];
        for (case, bytes, expected) in cases {
            assert_eq!(binary_label(bytes).as_deref(), expected, "{case}");
        }
    }
}
