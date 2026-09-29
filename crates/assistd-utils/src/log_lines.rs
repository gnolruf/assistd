//! Forwards a child's line-oriented output to tracing: one call to
//! `emit` per line, decoded lossily and capped so no line is buffered whole.

use std::io;

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};

/// Bytes of one line kept before the rest of it is discarded.
const MAX_LINE_BYTES: usize = 8 * 1024;

/// Read `stream` to EOF, calling `emit` once per line without its line
/// ending. Invalid UTF-8 is replaced, a line past 8 KiB is cut
/// with a marker, and only a read error ends the loop early.
pub async fn forward_lines<R, F>(stream: R, mut emit: F) -> io::Result<()>
where
    R: AsyncRead + Unpin,
    F: FnMut(&str),
{
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    loop {
        line.clear();
        match read_capped_line(&mut reader, &mut line).await? {
            LineRead::Eof => return Ok(()),
            LineRead::Whole => emit(String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n'])),
            LineRead::Cut => {
                let head = String::from_utf8_lossy(&line);
                emit(&format!("{head} [line cut at {MAX_LINE_BYTES} bytes]"));
            }
        }
    }
}

enum LineRead {
    Eof,
    Whole,
    Cut,
}

async fn read_capped_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    line: &mut Vec<u8>,
) -> io::Result<LineRead> {
    let read = (&mut *reader)
        .take(MAX_LINE_BYTES as u64 + 1)
        .read_until(b'\n', line)
        .await?;
    if read == 0 {
        return Ok(LineRead::Eof);
    }
    if line.len() <= MAX_LINE_BYTES || line.ends_with(b"\n") {
        return Ok(LineRead::Whole);
    }
    line.truncate(MAX_LINE_BYTES);
    discard_rest_of_line(reader).await?;
    Ok(LineRead::Cut)
}

async fn discard_rest_of_line<R: AsyncBufRead + Unpin>(reader: &mut R) -> io::Result<()> {
    loop {
        let buffered = reader.fill_buf().await?;
        if buffered.is_empty() {
            return Ok(());
        }
        let newline_at = buffered.iter().position(|byte| *byte == b'\n');
        let consumed = newline_at.map_or(buffered.len(), |at| at + 1);
        reader.consume(consumed);
        if newline_at.is_some() {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn collect(input: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        forward_lines(input, |line| lines.push(line.to_string()))
            .await
            .unwrap();
        lines
    }

    #[tokio::test]
    async fn invalid_utf8_is_replaced_and_later_lines_still_arrive() {
        let lines = collect(b"ok\n\xff\xfe bad\nafter").await;
        assert_eq!(lines, ["ok", "\u{FFFD}\u{FFFD} bad", "after"]);
    }

    #[tokio::test]
    async fn oversized_line_is_cut_without_buffering_the_rest() {
        let mut input = vec![b'x'; MAX_LINE_BYTES * 3];
        input.extend_from_slice(b"\ntail\r\n");
        let lines = collect(&input).await;
        let [long, tail] = lines.as_slice() else {
            panic!("expected two lines, got {}", lines.len());
        };
        assert!(long.starts_with(&"x".repeat(MAX_LINE_BYTES)));
        assert!(long.ends_with(&format!(" [line cut at {MAX_LINE_BYTES} bytes]")));
        assert_eq!(
            long.len(),
            MAX_LINE_BYTES + " [line cut at 8192 bytes]".len()
        );
        assert_eq!(tail, "tail");
    }

    #[tokio::test]
    async fn a_line_of_exactly_the_cap_is_whole() {
        let mut input = vec![b'y'; MAX_LINE_BYTES];
        input.push(b'\n');
        let lines = collect(&input).await;
        assert_eq!(lines, ["y".repeat(MAX_LINE_BYTES)]);
    }
}
