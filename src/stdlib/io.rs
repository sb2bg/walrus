use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::span::Span;
use crate::value::Value;

use super::FILE_TABLE;
pub fn file_open(path: &str, mode: &str, _span: Span) -> WalrusResult<Value> {
    let file = match mode {
        "r" => OpenOptions::new().read(true).open(path),
        "w" => OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path),
        "a" => OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(path),
        "rw" => OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path),
        _ => {
            return Err(WalrusError::InvalidFileMode {
                mode: mode.to_string(),
            });
        }
    };

    match file {
        Ok(f) => {
            let handle = FILE_TABLE.with(|table| table.borrow_mut().insert(f));
            Ok(Value::Int(handle))
        }
        Err(e) => Err(WalrusError::FileOpenFailed {
            path: path.to_string(),
            reason: e.to_string(),
        }),
    }
}

/// Read entire file contents as a string
pub fn file_read(handle: i64, _span: Span) -> WalrusResult<String> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(entry) = table.get_mut(handle) {
            let mut contents = String::new();
            match entry.file.read_to_string(&mut contents) {
                Ok(_) => Ok(contents),
                Err(e) => Err(WalrusError::FileReadFailed {
                    handle,
                    reason: e.to_string(),
                }),
            }
        } else {
            Err(WalrusError::InvalidFileHandle { handle })
        }
    })
}

/// Read a single line from a file
pub fn file_read_line(handle: i64, _span: Span) -> WalrusResult<Option<String>> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(entry) = table.get_mut(handle) {
            // Read byte-by-byte so repeated calls preserve exact file cursor semantics.
            let mut bytes = Vec::new();
            let mut buf = [0u8; 1];

            loop {
                match entry.file.read(&mut buf) {
                    Ok(0) => {
                        if bytes.is_empty() {
                            return Ok(None); // EOF with no data
                        }
                        break;
                    }
                    Ok(_) => {
                        if buf[0] == b'\n' {
                            break;
                        }
                        bytes.push(buf[0]);
                    }
                    Err(e) => {
                        return Err(WalrusError::FileReadLineFailed {
                            handle,
                            reason: e.to_string(),
                        });
                    }
                }
            }

            if bytes.last().copied() == Some(b'\r') {
                bytes.pop();
            }

            match String::from_utf8(bytes) {
                Ok(line) => Ok(Some(line)),
                Err(e) => Err(WalrusError::FileReadLineFailed {
                    handle,
                    reason: e.to_string(),
                }),
            }
        } else {
            Err(WalrusError::InvalidFileHandle { handle })
        }
    })
}

/// Write string to file, returns bytes written
pub fn file_write(handle: i64, content: &str, _span: Span) -> WalrusResult<i64> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(entry) = table.get_mut(handle) {
            match entry.file.write_all(content.as_bytes()) {
                Ok(_) => Ok(content.len() as i64),
                Err(e) => Err(WalrusError::FileWriteFailed {
                    handle,
                    reason: e.to_string(),
                }),
            }
        } else {
            Err(WalrusError::InvalidFileHandle { handle })
        }
    })
}

/// Close a file handle
pub fn file_close(handle: i64, _span: Span) -> WalrusResult<()> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if table.remove(handle).is_some() {
            Ok(())
        } else {
            Err(WalrusError::InvalidFileHandle { handle })
        }
    })
}

/// Check if a file exists
pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Read entire file as string (convenience function - opens, reads, closes)
pub fn read_file(path: &str, _span: Span) -> WalrusResult<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(content),
        Err(e) => Err(WalrusError::ReadFileFailed {
            path: path.to_string(),
            reason: e.to_string(),
        }),
    }
}

/// Write string to file (convenience function - opens, writes, closes)
pub fn write_file(path: &str, content: &str, _span: Span) -> WalrusResult<()> {
    match std::fs::write(path, content) {
        Ok(_) => Ok(()),
        Err(e) => Err(WalrusError::WriteFileFailed {
            path: path.to_string(),
            reason: e.to_string(),
        }),
    }
}
