//! Recovers the id of a JSON-RPC response too large to buffer by
//! scanning its bytes as they stream past.

/// Longest top-level key worth remembering; `method` is the longest
/// one the scanner looks for.
const KEY_PREFIX_BYTES: usize = "method".len() + 1;

/// Longest top-level `id` value worth remembering; `u64::MAX` has 20 digits.
const ID_TEXT_BYTES: usize = 21;

/// Streaming scan of one JSON object for its top-level `id` and
/// `method` keys, holding a few bytes of state however long the frame.
#[derive(Debug, Default)]
pub(crate) struct ResponseIdScanner {
    depth: usize,
    in_string: bool,
    escaped: bool,
    last_top_level_string: Vec<u8>,
    in_id_value: bool,
    id_text: Vec<u8>,
    id: Option<u64>,
    has_method: bool,
}

impl ResponseIdScanner {
    /// Scan the next run of bytes of the frame.
    pub(crate) fn feed(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            if self.in_string {
                self.scan_string_byte(byte);
            } else {
                self.scan_structural_byte(byte);
            }
        }
    }

    /// The numeric id of the frame if it is a response; `None` for a
    /// request, a notification, or a frame without a numeric id.
    pub(crate) fn response_id(&self) -> Option<u64> {
        self.id.filter(|_| !self.has_method)
    }

    fn scan_string_byte(&mut self, byte: u8) {
        match (self.escaped, byte) {
            (true, _) => self.escaped = false,
            (false, b'\\') => self.escaped = true,
            (false, b'"') => {
                self.in_string = false;
                return;
            }
            _ => {}
        }
        if self.depth == 1 && self.last_top_level_string.len() < KEY_PREFIX_BYTES {
            self.last_top_level_string.push(byte);
        }
    }

    fn scan_structural_byte(&mut self, byte: u8) {
        match byte {
            b'"' => {
                self.in_string = true;
                self.last_top_level_string.clear();
            }
            b'{' | b'[' => self.depth += 1,
            b'}' | b']' => {
                if self.depth == 1 {
                    self.end_top_level_value();
                }
                self.depth = self.depth.saturating_sub(1);
            }
            b':' if self.depth == 1 => self.start_top_level_value(),
            b',' if self.depth == 1 => self.end_top_level_value(),
            _ if self.depth == 1
                && self.in_id_value
                && !byte.is_ascii_whitespace()
                && self.id_text.len() < ID_TEXT_BYTES =>
            {
                self.id_text.push(byte);
            }
            _ => {}
        }
    }

    fn start_top_level_value(&mut self) {
        let key = self.last_top_level_string.as_slice();
        self.has_method |= key == b"method";
        self.in_id_value = key == b"id";
        self.id_text.clear();
    }

    fn end_top_level_value(&mut self) {
        if self.in_id_value {
            self.id = std::str::from_utf8(&self.id_text)
                .ok()
                .and_then(|text| text.parse().ok());
            self.in_id_value = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response_id_of(frame: &str) -> Option<u64> {
        let mut scanner = ResponseIdScanner::default();
        scanner.feed(frame.as_bytes());
        scanner.response_id()
    }

    #[test]
    fn finds_the_id_wherever_it_sits_among_the_top_level_keys() {
        let cases = [
            r#"{"jsonrpc":"2.0","id":7,"result":{"content":[]}}"#,
            r#"{"result":{"content":[]},"jsonrpc":"2.0","id":7}"#,
            r#"{ "jsonrpc" : "2.0" , "id" : 7 , "error" : {"code":-1,"message":"x"} }"#,
            "{\"result\":{},\n\"id\":7}\n",
        ];
        for frame in cases {
            assert_eq!(response_id_of(frame), Some(7), "{frame}");
        }
    }

    #[test]
    fn ignores_id_lookalikes_nested_or_inside_strings() {
        let cases = [
            r#"{"jsonrpc":"2.0","result":{"id":7}}"#,
            r#"{"jsonrpc":"2.0","result":[{"id":7}]}"#,
            r#"{"jsonrpc":"2.0","result":"\"id\":7"}"#,
            r#"{"jsonrpc":"2.0","result":"id","x":7}"#,
            r#"{"jsonrpc":"2.0","identifier":7,"result":{}}"#,
        ];
        for frame in cases {
            assert_eq!(response_id_of(frame), None, "{frame}");
        }
    }

    #[test]
    fn non_numeric_ids_and_server_requests_have_no_response_id() {
        let cases = [
            r#"{"jsonrpc":"2.0","id":"7","result":{}}"#,
            r#"{"jsonrpc":"2.0","id":null,"result":{}}"#,
            r#"{"jsonrpc":"2.0","id":-7,"result":{}}"#,
            r#"{"jsonrpc":"2.0","id":7.5,"result":{}}"#,
            r#"{"jsonrpc":"2.0","id":184467440737095516150,"result":{}}"#,
            r#"{"jsonrpc":"2.0","id":7,"method":"sampling/createMessage","params":{}}"#,
            r#"{"method":"ping","id":7}"#,
        ];
        for frame in cases {
            assert_eq!(response_id_of(frame), None, "{frame}");
        }
    }

    #[test]
    fn a_frame_fed_in_pieces_scans_like_one_fed_whole() {
        let frame = r#"{"result":{"text":"a \"quoted\" \\ id"},"id":42}"#.as_bytes();
        let mut scanner = ResponseIdScanner::default();
        for byte in frame.chunks(1) {
            scanner.feed(byte);
        }
        assert_eq!(scanner.response_id(), Some(42));
    }
}
