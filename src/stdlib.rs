// Standard Library - Native Functions for Walrus
//
// This module provides the `import()` system and native functions.
// Usage: let io = import("std/io")
//        let content = io.read_file("test.txt")

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;

// File handle table - maps integer handles to open files
thread_local! {
    static FILE_TABLE: RefCell<FileTable> = RefCell::new(FileTable::new());
}

struct FileEntry {
    file: File,
    path: String,
    mode: String,
}

struct FileTable {
    files: HashMap<i64, FileEntry>,
    next_handle: i64,
}

impl FileTable {
    fn new() -> Self {
        Self {
            files: HashMap::new(),
            next_handle: 1, // Start at 1 so 0 can mean "invalid"
        }
    }

    fn insert(&mut self, file: File, path: String, mode: String) -> i64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.files.insert(handle, FileEntry { file, path, mode });
        handle
    }

    fn get_mut(&mut self, handle: i64) -> Option<&mut FileEntry> {
        self.files.get_mut(&handle)
    }

    fn remove(&mut self, handle: i64) -> Option<FileEntry> {
        self.files.remove(&handle)
    }
}

/// Get the list of native functions for a module
pub fn get_module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    match module {
        "std/io" => Some(vec![
            NativeFunction::FileOpen,
            NativeFunction::FileRead,
            NativeFunction::FileReadLine,
            NativeFunction::FileWrite,
            NativeFunction::FileClose,
            NativeFunction::FileExists,
            NativeFunction::ReadFile,
            NativeFunction::WriteFile,
        ]),
        "std/sys" => Some(vec![
            NativeFunction::EnvGet,
            NativeFunction::Args,
            NativeFunction::Cwd,
            NativeFunction::Exit,
        ]),
        _ => None,
    }
}

/// Open a file and return a handle
/// Modes: "r" (read), "w" (write/create), "a" (append), "rw" (read+write)
pub fn file_open(path: &str, mode: &str, span: Span) -> WalrusResult<Value> {
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
            return Err(WalrusError::Exception {
                message: format!("Invalid file mode '{}'. Use 'r', 'w', 'a', or 'rw'", mode),
                span,
                src: String::new(),
                filename: String::new(),
            });
        }
    };

    match file {
        Ok(f) => {
            let handle = FILE_TABLE.with(|table| {
                table
                    .borrow_mut()
                    .insert(f, path.to_string(), mode.to_string())
            });
            Ok(Value::Int(handle))
        }
        Err(e) => Err(WalrusError::Exception {
            message: format!("Failed to open '{}': {}", path, e),
            span,
            src: String::new(),
            filename: String::new(),
        }),
    }
}

/// Read entire file contents as a string
pub fn file_read(handle: i64, span: Span) -> WalrusResult<String> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(entry) = table.get_mut(handle) {
            let mut contents = String::new();
            match entry.file.read_to_string(&mut contents) {
                Ok(_) => Ok(contents),
                Err(e) => Err(WalrusError::Exception {
                    message: format!("Failed to read file: {}", e),
                    span,
                    src: String::new(),
                    filename: String::new(),
                }),
            }
        } else {
            Err(WalrusError::Exception {
                message: format!("Invalid file handle: {}", handle),
                span,
                src: String::new(),
                filename: String::new(),
            })
        }
    })
}

/// Read a single line from a file
pub fn file_read_line(handle: i64, span: Span) -> WalrusResult<Option<String>> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(entry) = table.get_mut(handle) {
            use std::io::BufRead;
            let mut reader = std::io::BufReader::new(&entry.file);
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => Ok(None), // EOF
                Ok(_) => {
                    // Remove trailing newline
                    if line.ends_with('\n') {
                        line.pop();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                    }
                    Ok(Some(line))
                }
                Err(e) => Err(WalrusError::Exception {
                    message: format!("Failed to read line: {}", e),
                    span,
                    src: String::new(),
                    filename: String::new(),
                }),
            }
        } else {
            Err(WalrusError::Exception {
                message: format!("Invalid file handle: {}", handle),
                span,
                src: String::new(),
                filename: String::new(),
            })
        }
    })
}

/// Write string to file, returns bytes written
pub fn file_write(handle: i64, content: &str, span: Span) -> WalrusResult<i64> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(entry) = table.get_mut(handle) {
            match entry.file.write_all(content.as_bytes()) {
                Ok(_) => Ok(content.len() as i64),
                Err(e) => Err(WalrusError::Exception {
                    message: format!("Failed to write to file: {}", e),
                    span,
                    src: String::new(),
                    filename: String::new(),
                }),
            }
        } else {
            Err(WalrusError::Exception {
                message: format!("Invalid file handle: {}", handle),
                span,
                src: String::new(),
                filename: String::new(),
            })
        }
    })
}

/// Close a file handle
pub fn file_close(handle: i64, span: Span) -> WalrusResult<()> {
    FILE_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if table.remove(handle).is_some() {
            Ok(())
        } else {
            Err(WalrusError::Exception {
                message: format!("Invalid file handle: {}", handle),
                span,
                src: String::new(),
                filename: String::new(),
            })
        }
    })
}

/// Check if a file exists
pub fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Check if a path is a directory
pub fn is_dir(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// Check if a path is a file
pub fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
}

/// Read entire file as string (convenience function - opens, reads, closes)
pub fn read_file(path: &str, span: Span) -> WalrusResult<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(content),
        Err(e) => Err(WalrusError::Exception {
            message: format!("Failed to read '{}': {}", path, e),
            span,
            src: String::new(),
            filename: String::new(),
        }),
    }
}

/// Write string to file (convenience function - opens, writes, closes)
pub fn write_file(path: &str, content: &str, span: Span) -> WalrusResult<()> {
    match std::fs::write(path, content) {
        Ok(_) => Ok(()),
        Err(e) => Err(WalrusError::Exception {
            message: format!("Failed to write '{}': {}", path, e),
            span,
            src: String::new(),
            filename: String::new(),
        }),
    }
}

/// Get environment variable
pub fn env_get(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Get all command line arguments
pub fn args() -> Vec<String> {
    std::env::args().collect()
}

/// Get current working directory
pub fn cwd() -> Option<String> {
    std::env::current_dir()
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}
