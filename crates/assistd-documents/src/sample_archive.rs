//! Builds zip archives for tests.

use std::io::{Cursor, Write};

use zip::ZipWriter;
use zip::write::SimpleFileOptions;

/// A zip archive of `entries`, each a name and its contents, deflated.
pub(crate) fn archive_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
    for (name, contents) in entries {
        archive
            .start_file(*name, SimpleFileOptions::default())
            .expect("entry");
        archive.write_all(contents).expect("contents");
    }
    archive.finish().expect("archive").into_inner()
}
