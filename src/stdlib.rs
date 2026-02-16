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

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::WalrusResult;

// File handle table - maps integer handles to open files
thread_local! {
    static FILE_TABLE: RefCell<FileTable> = RefCell::new(FileTable::new());
    static RNG_STATE: RefCell<StdRng> = RefCell::new(new_rng());
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

fn new_rng() -> StdRng {
    let mut seeder = rand::thread_rng();
    let mut seed = [0u8; 32];
    seeder.fill(&mut seed);
    StdRng::from_seed(seed)
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
    crate::native_registry::module_functions(module)
}

/// Open a file and return a handle
/// Modes: "r" (read), "w" (write/create), "a" (append), "rw" (read+write)
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
            let handle = FILE_TABLE.with(|table| {
                table
                    .borrow_mut()
                    .insert(f, path.to_string(), mode.to_string())
            });
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

/// Check if a path is a directory
pub fn is_dir(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// Check if a path is a file
pub fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
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

pub fn math_seed(seed: i64) {
    RNG_STATE.with(|rng| {
        *rng.borrow_mut() = StdRng::seed_from_u64(seed as u64);
    });
}

pub fn math_rand_float() -> f64 {
    RNG_STATE.with(|rng| rng.borrow_mut().gen_range(0.0..1.0))
}

pub fn math_rand_bool() -> bool {
    RNG_STATE.with(|rng| rng.borrow_mut().gen_bool(0.5))
}

pub fn math_rand_int(min: i64, max: i64, _span: Span) -> WalrusResult<i64> {
    if min > max {
        return Err(WalrusError::GenericError {
            message: format!("math.rand_int: invalid range [{min}, {max}]"),
        });
    }

    Ok(RNG_STATE.with(|rng| rng.borrow_mut().gen_range(min..=max)))
}

pub fn math_rand_range(min: f64, max: f64, _span: Span) -> WalrusResult<f64> {
    if !min.is_finite() || !max.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.rand_range: range bounds must be finite numbers".to_string(),
        });
    }

    if min > max {
        return Err(WalrusError::GenericError {
            message: format!("math.rand_range: invalid range [{min}, {max}]"),
        });
    }

    if (min - max).abs() < f64::EPSILON {
        return Ok(min);
    }

    Ok(RNG_STATE.with(|rng| rng.borrow_mut().gen_range(min..max)))
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
