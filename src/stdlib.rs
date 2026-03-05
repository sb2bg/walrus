// Standard Library - Native Functions for Walrus
//
// This module provides the `import()` system and native functions.
// Usage: let io = import("std/io")
//        let content = io.read_file("test.txt")

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;

use hyper::http::header::{CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HeaderName, HeaderValue};
use hyper::http::{HeaderMap, Method, Response, StatusCode, Uri, Version};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::{HeapValue, Resolve};
use crate::error::WalrusError;
use crate::function::{NativeFunction, RustFunction, WalrusFunction};
use crate::source_ref::SourceRef;
use crate::span::Span;
use crate::value::{IoHttpOutcome, IoHttpRequest, Value};

// File handle table - maps integer handles to open files
thread_local! {
    static FILE_TABLE: RefCell<FileTable> = RefCell::new(FileTable::new());
    static NET_TABLE: RefCell<NetState> = RefCell::new(NetState::new());
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

struct NetState {
    listeners: HashMap<i64, TcpListener>,
    streams: HashMap<i64, TcpStream>,
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

impl NetState {
    fn new() -> Self {
        Self {
            listeners: HashMap::new(),
            streams: HashMap::new(),
            next_handle: 1,
        }
    }

    fn next(&mut self) -> i64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        handle
    }

    fn insert_listener(&mut self, listener: TcpListener) -> i64 {
        let handle = self.next();
        self.listeners.insert(handle, listener);
        handle
    }

    fn insert_stream(&mut self, stream: TcpStream) -> i64 {
        let handle = self.next();
        self.streams.insert(handle, stream);
        handle
    }

    fn get_mut_listener(&mut self, handle: i64) -> Option<&mut TcpListener> {
        self.listeners.get_mut(&handle)
    }

    fn get_mut_stream(&mut self, handle: i64) -> Option<&mut TcpStream> {
        self.streams.get_mut(&handle)
    }

    fn remove_listener(&mut self, handle: i64) -> Option<TcpListener> {
        self.listeners.remove(&handle)
    }

    fn remove_stream(&mut self, handle: i64) -> Option<TcpStream> {
        self.streams.remove(&handle)
    }
}

/// Get the list of native functions for a module
pub fn get_module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    crate::native_registry::module_functions(module)
}

pub fn interpreter_module(module: &str) -> Option<Value> {
    match module {
        "std/core" => Some(interpreter_core_module()),
        _ => None,
    }
}

fn interpreter_core_module() -> Value {
    fn add(
        dict: &mut FxHashMap<Value, Value>,
        name: &'static str,
        arity: usize,
        func: fn(Vec<Value>, SourceRef, Span) -> WalrusResult<Value>,
    ) {
        let key = HeapValue::String(name).alloc();
        let val = HeapValue::Function(WalrusFunction::Rust(RustFunction::new(
            name.to_string(),
            arity,
            func,
        )))
        .alloc();
        dict.insert(key, val);
    }

    fn core_len(args: Vec<Value>, source_ref: SourceRef, span: Span) -> WalrusResult<Value> {
        match args[0] {
            Value::String(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            Value::List(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            Value::Dict(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            Value::Module(key) => Ok(Value::Int(key.resolve()?.len() as i64)),
            other => Err(WalrusError::NoLength {
                type_name: other.get_type().to_string(),
                span,
                src: source_ref.source().to_string(),
                filename: source_ref.filename().to_string(),
            }),
        }
    }

    fn core_str(args: Vec<Value>, _source_ref: SourceRef, _span: Span) -> WalrusResult<Value> {
        let rendered = args[0].stringify()?;
        Ok(HeapValue::String(&rendered).alloc())
    }

    fn core_type(args: Vec<Value>, _source_ref: SourceRef, _span: Span) -> WalrusResult<Value> {
        Ok(HeapValue::String(args[0].get_type()).alloc())
    }

    fn core_input(args: Vec<Value>, _source_ref: SourceRef, _span: Span) -> WalrusResult<Value> {
        print!("{}", args[0].stringify()?);
        std::io::stdout()
            .flush()
            .map_err(|source| WalrusError::IOError { source })?;

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|source| WalrusError::IOError { source })?;
        Ok(HeapValue::String(&input).alloc())
    }

    fn core_gc_unavailable(
        _args: Vec<Value>,
        _source_ref: SourceRef,
        _span: Span,
    ) -> WalrusResult<Value> {
        Err(WalrusError::GenericError {
            message: "std/core.gc is only available in VM mode".to_string(),
        })
    }

    fn core_heap_stats_unavailable(
        _args: Vec<Value>,
        _source_ref: SourceRef,
        _span: Span,
    ) -> WalrusResult<Value> {
        Err(WalrusError::GenericError {
            message: "std/core.heap_stats is only available in VM mode".to_string(),
        })
    }

    fn core_gc_threshold_unavailable(
        _args: Vec<Value>,
        _source_ref: SourceRef,
        _span: Span,
    ) -> WalrusResult<Value> {
        Err(WalrusError::GenericError {
            message: "std/core.gc_threshold is only available in VM mode".to_string(),
        })
    }

    let mut dict = FxHashMap::default();
    add(&mut dict, "len", 1, core_len);
    add(&mut dict, "str", 1, core_str);
    add(&mut dict, "type", 1, core_type);
    add(&mut dict, "input", 1, core_input);
    add(&mut dict, "gc", 0, core_gc_unavailable);
    add(&mut dict, "heap_stats", 0, core_heap_stats_unavailable);
    add(&mut dict, "gc_threshold", 1, core_gc_threshold_unavailable);

    HeapValue::Module(dict).alloc()
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

pub fn ensure_valid_port(port: i64, op: &str) -> WalrusResult<u16> {
    if !(0..=65535).contains(&port) {
        return Err(WalrusError::GenericError {
            message: format!("net.{op}: invalid port {port} (expected 0..65535)"),
        });
    }

    Ok(port as u16)
}

/// Bind a TCP listener and return a listener handle.
pub fn tcp_bind(host: &str, port: i64, _span: Span) -> WalrusResult<Value> {
    let port = ensure_valid_port(port, "tcp_bind")?;
    let addr = format!("{host}:{port}");

    match TcpListener::bind(&addr) {
        Ok(listener) => {
            let handle = NET_TABLE.with(|table| table.borrow_mut().insert_listener(listener));
            Ok(Value::Int(handle))
        }
        Err(err) => Err(WalrusError::GenericError {
            message: format!("net.tcp_bind: failed to bind '{addr}': {err}"),
        }),
    }
}

/// Accept one incoming connection from a listener handle and return a stream handle.
pub fn tcp_accept(listener_handle: i64, _span: Span) -> WalrusResult<Value> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let listener =
            table
                .get_mut_listener(listener_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_accept: invalid listener handle {listener_handle}"),
                })?;

        let (stream, _addr) = listener.accept().map_err(|err| WalrusError::GenericError {
            message: format!("net.tcp_accept: accept failed: {err}"),
        })?;

        let stream_handle = table.insert_stream(stream);
        Ok(Value::Int(stream_handle))
    })
}

/// Connect to a TCP host/port and return a stream handle.
pub fn tcp_connect(host: &str, port: i64, _span: Span) -> WalrusResult<Value> {
    let port = ensure_valid_port(port, "tcp_connect")?;
    let addr = format!("{host}:{port}");

    match TcpStream::connect(&addr) {
        Ok(stream) => {
            let handle = NET_TABLE.with(|table| table.borrow_mut().insert_stream(stream));
            Ok(Value::Int(handle))
        }
        Err(err) => Err(WalrusError::GenericError {
            message: format!("net.tcp_connect: failed to connect to '{addr}': {err}"),
        }),
    }
}

/// Return the bound local port for a listener handle.
pub fn tcp_local_port(listener_handle: i64, _span: Span) -> WalrusResult<i64> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let listener =
            table
                .get_mut_listener(listener_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!(
                        "net.tcp_local_port: invalid listener handle {listener_handle}"
                    ),
                })?;

        let port = listener
            .local_addr()
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_local_port: failed to read local addr: {err}"),
            })?
            .port() as i64;

        Ok(port)
    })
}

/// Read up to max_bytes from a stream. Returns None on EOF.
pub fn tcp_read(stream_handle: i64, max_bytes: i64, _span: Span) -> WalrusResult<Option<String>> {
    if max_bytes <= 0 {
        return Err(WalrusError::GenericError {
            message: format!("net.tcp_read: max_bytes must be > 0, got {max_bytes}"),
        });
    }

    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let stream =
            table
                .get_mut_stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_read: invalid stream handle {stream_handle}"),
                })?;

        let mut buf = vec![0u8; max_bytes as usize];
        let read = stream
            .read(&mut buf)
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_read: read failed: {err}"),
            })?;

        if read == 0 {
            return Ok(None);
        }

        buf.truncate(read);
        let text = String::from_utf8(buf).map_err(|err| WalrusError::GenericError {
            message: format!("net.tcp_read: received non-utf8 data: {err}"),
        })?;

        Ok(Some(text))
    })
}

/// Read one line from a stream. Returns None on EOF before data.
pub fn tcp_read_line(stream_handle: i64, _span: Span) -> WalrusResult<Option<String>> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let stream =
            table
                .get_mut_stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_read_line: invalid stream handle {stream_handle}"),
                })?;

        let mut bytes = Vec::new();
        let mut buf = [0u8; 1];

        loop {
            match stream.read(&mut buf) {
                Ok(0) => {
                    if bytes.is_empty() {
                        return Ok(None);
                    }
                    break;
                }
                Ok(_) => {
                    if buf[0] == b'\n' {
                        break;
                    }
                    bytes.push(buf[0]);
                }
                Err(err) => {
                    return Err(WalrusError::GenericError {
                        message: format!("net.tcp_read_line: read failed: {err}"),
                    });
                }
            }
        }

        if bytes.last().copied() == Some(b'\r') {
            bytes.pop();
        }

        let line = String::from_utf8(bytes).map_err(|err| WalrusError::GenericError {
            message: format!("net.tcp_read_line: received non-utf8 data: {err}"),
        })?;

        Ok(Some(line))
    })
}

/// Write utf8 data to a stream and return bytes written.
pub fn tcp_write(stream_handle: i64, content: &str, _span: Span) -> WalrusResult<i64> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let stream =
            table
                .get_mut_stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_write: invalid stream handle {stream_handle}"),
                })?;

        stream
            .write_all(content.as_bytes())
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_write: write failed: {err}"),
            })?;

        Ok(content.len() as i64)
    })
}

/// Close a stream handle.
pub fn tcp_close(stream_handle: i64, _span: Span) -> WalrusResult<()> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if table.remove_stream(stream_handle).is_some() {
            Ok(())
        } else {
            Err(WalrusError::GenericError {
                message: format!("net.tcp_close: invalid stream handle {stream_handle}"),
            })
        }
    })
}

/// Close a listener handle.
pub fn tcp_close_listener(listener_handle: i64, _span: Span) -> WalrusResult<()> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if table.remove_listener(listener_handle).is_some() {
            Ok(())
        } else {
            Err(WalrusError::GenericError {
                message: format!(
                    "net.tcp_close_listener: invalid listener handle {listener_handle}"
                ),
            })
        }
    })
}

/// Store a TcpStream in the thread-local NET_TABLE and return its handle.
/// Used by the VM to convert IoResult::Stream back into a handle on the VM thread.
pub fn store_tcp_stream(stream: TcpStream) -> i64 {
    NET_TABLE.with(|table| table.borrow_mut().insert_stream(stream))
}

/// Store a TcpListener in the thread-local NET_TABLE and return its handle.
/// Used by the VM to convert IoResult::Listener back into a handle on the VM thread.
pub fn store_tcp_listener(listener: TcpListener) -> i64 {
    NET_TABLE.with(|table| table.borrow_mut().insert_listener(listener))
}

/// Clone a TcpListener for use in a background thread.
/// Returns None if the handle is invalid.
pub fn clone_tcp_listener(handle: i64) -> Option<TcpListener> {
    NET_TABLE.with(|table| {
        let table = table.borrow();
        table
            .listeners
            .get(&handle)
            .and_then(|l| l.try_clone().ok())
    })
}

/// Clone a TcpStream for use in a background thread.
/// Returns None if the handle is invalid.
pub fn clone_tcp_stream(handle: i64) -> Option<TcpStream> {
    NET_TABLE.with(|table| {
        let table = table.borrow();
        table.streams.get(&handle).and_then(|s| s.try_clone().ok())
    })
}

#[derive(Debug, Clone)]
pub struct HttpRequestLine {
    pub method: String,
    pub target: String,
    pub path: String,
    pub query: String,
    pub version: String,
}

#[derive(Debug, Clone)]
pub struct HttpRouteMatch {
    pub found: bool,
    pub pattern: String,
    pub path: String,
    pub params: Vec<(String, String)>,
    pub wildcard: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub target: String,
    pub path: String,
    pub query: String,
    pub version: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
    pub content_length: i64,
}

#[derive(Debug, Clone)]
pub enum HttpReadOutcome {
    Eof,
    Request(HttpRequest),
    BadRequest(String),
}

const HTTP_MAX_HEADER_BYTES: usize = 64 * 1024;
const HTTP_READ_CHUNK_BYTES: usize = 4096;
const HTTP_MAX_HEADER_COUNT: usize = 128;

pub fn http_parse_request_line(line: &str) -> Result<HttpRequestLine, String> {
    let mut parts = line.split_whitespace();
    let method_token = parts
        .next()
        .ok_or_else(|| "expected HTTP method".to_string())?;
    let target_token = parts
        .next()
        .ok_or_else(|| "expected request target".to_string())?;
    let version_token = parts
        .next()
        .ok_or_else(|| "expected HTTP version".to_string())?;

    if parts.next().is_some() {
        return Err("too many tokens in request line".to_string());
    }

    let method = Method::from_bytes(method_token.as_bytes())
        .map_err(|_| format!("invalid HTTP method '{method_token}'"))?;
    let uri: Uri = target_token
        .parse()
        .map_err(|_| format!("invalid request target '{target_token}'"))?;
    let version = parse_http_version(version_token)?;

    let path = if uri.path().is_empty() {
        "/".to_string()
    } else {
        uri.path().to_string()
    };
    let query = uri.query().unwrap_or("").to_string();

    Ok(HttpRequestLine {
        method: method.as_str().to_string(),
        target: target_token.to_string(),
        path,
        query,
        version: http_version_token(version).to_string(),
    })
}

pub fn http_parse_query(query: &str) -> Vec<(String, String)> {
    let query = query.strip_prefix('?').unwrap_or(query);
    if query.is_empty() {
        return Vec::new();
    }

    query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            if let Some((k, v)) = pair.split_once('=') {
                (k.to_string(), v.to_string())
            } else {
                (pair.to_string(), String::new())
            }
        })
        .collect()
}

pub fn http_normalize_path(path: &str) -> String {
    let path = path.split_once('?').map_or(path, |(p, _)| p);

    let mut out = String::with_capacity(path.len() + 1);
    if !path.starts_with('/') {
        out.push('/');
    }

    let mut prev_slash = false;
    for ch in path.chars() {
        if ch == '/' {
            if !prev_slash {
                out.push('/');
                prev_slash = true;
            }
        } else {
            out.push(ch);
            prev_slash = false;
        }
    }

    if out.is_empty() {
        out.push('/');
    }

    if out.len() > 1 && out.ends_with('/') {
        out.pop();
    }

    out
}

pub fn http_match_route(pattern: &str, path: &str) -> HttpRouteMatch {
    let normalized_pattern = http_normalize_path(pattern);
    let normalized_path = http_normalize_path(path);
    let pattern_segments = split_segments(&normalized_pattern);
    let path_segments = split_segments(&normalized_path);

    let mut params = Vec::new();
    let mut index = 0usize;

    while index < pattern_segments.len() {
        let part = pattern_segments[index];

        if part == "*" {
            if index + 1 != pattern_segments.len() {
                return HttpRouteMatch {
                    found: false,
                    pattern: normalized_pattern,
                    path: normalized_path,
                    params: Vec::new(),
                    wildcard: None,
                };
            }

            let remainder = if index >= path_segments.len() {
                String::new()
            } else {
                path_segments[index..].join("/")
            };

            return HttpRouteMatch {
                found: true,
                pattern: normalized_pattern,
                path: normalized_path,
                params,
                wildcard: Some(remainder),
            };
        }

        if index >= path_segments.len() {
            return HttpRouteMatch {
                found: false,
                pattern: normalized_pattern,
                path: normalized_path,
                params: Vec::new(),
                wildcard: None,
            };
        }

        let actual = path_segments[index];
        if let Some(param_name) = part.strip_prefix(':') {
            if param_name.is_empty() {
                return HttpRouteMatch {
                    found: false,
                    pattern: normalized_pattern,
                    path: normalized_path,
                    params: Vec::new(),
                    wildcard: None,
                };
            }
            params.push((param_name.to_string(), actual.to_string()));
        } else if part != actual {
            return HttpRouteMatch {
                found: false,
                pattern: normalized_pattern,
                path: normalized_path,
                params: Vec::new(),
                wildcard: None,
            };
        }

        index += 1;
    }

    if index != path_segments.len() {
        return HttpRouteMatch {
            found: false,
            pattern: normalized_pattern,
            path: normalized_path,
            params: Vec::new(),
            wildcard: None,
        };
    }

    HttpRouteMatch {
        found: true,
        pattern: normalized_pattern,
        path: normalized_path,
        params,
        wildcard: None,
    }
}

pub fn http_status_text(code: i64) -> &'static str {
    u16::try_from(code)
        .ok()
        .and_then(|n| StatusCode::from_u16(n).ok())
        .and_then(|status| status.canonical_reason())
        .unwrap_or("Status")
}

pub fn http_build_response(
    status_code: i64,
    body: &str,
    headers: &[(String, String)],
    _span: Span,
) -> WalrusResult<String> {
    let status = status_code_from_i64(status_code, "http.response")?;
    let mut header_map = HeaderMap::new();

    for (name, value) in headers {
        if name.contains('\r')
            || name.contains('\n')
            || value.contains('\r')
            || value.contains('\n')
        {
            return Err(WalrusError::GenericError {
                message: "http.response: header names/values must not contain newlines".to_string(),
            });
        }

        let parsed_name =
            HeaderName::from_bytes(name.as_bytes()).map_err(|_| WalrusError::GenericError {
                message: format!("http.response: invalid header name '{name}'"),
            })?;
        let parsed_value = HeaderValue::from_str(value).map_err(|_| WalrusError::GenericError {
            message: format!("http.response: invalid header value for '{name}'"),
        })?;
        header_map.append(parsed_name, parsed_value);
    }

    if !header_map.contains_key(CONTENT_LENGTH) {
        let len = body.as_bytes().len().to_string();
        let len_value = HeaderValue::from_str(&len).expect("content-length is always valid");
        header_map.insert(CONTENT_LENGTH, len_value);
    }

    if !header_map.contains_key(CONTENT_TYPE) {
        header_map.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("text/plain; charset=utf-8"),
        );
    }

    if !header_map.contains_key(CONNECTION) {
        header_map.insert(CONNECTION, HeaderValue::from_static("close"));
    }

    let mut builder = Response::builder().version(Version::HTTP_11).status(status);
    for (name, value) in &header_map {
        builder = builder.header(name, value);
    }

    let response = builder
        .body(body.to_string())
        .map_err(|err| WalrusError::GenericError {
            message: format!("http.response: failed to build response: {err}"),
        })?;

    Ok(serialize_http_response(&response))
}

pub fn http_read_request(
    stream_handle: i64,
    max_body_bytes: i64,
    _span: Span,
) -> WalrusResult<HttpReadOutcome> {
    if max_body_bytes < 0 {
        return Err(WalrusError::GenericError {
            message: format!(
                "http.read_request: max_body_bytes must be >= 0, got {max_body_bytes}"
            ),
        });
    }
    let max_body_bytes =
        usize::try_from(max_body_bytes).map_err(|_| WalrusError::GenericError {
            message: "http.read_request: max_body_bytes cannot be represented on this platform"
                .to_string(),
        })?;

    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        let stream =
            table
                .get_mut_stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("http.read_request: invalid stream handle {stream_handle}"),
                })?;

        http_read_request_from_stream(stream, max_body_bytes)
            .map_err(|msg| WalrusError::GenericError { message: msg })
    })
}

/// Read and parse an HTTP request directly from a TcpStream.
/// This is the thread-safe version that can be called from worker threads.
pub fn http_read_request_from_stream(
    stream: &mut TcpStream,
    max_body_bytes: usize,
) -> Result<HttpReadOutcome, String> {
    use crate::value::{IoHttpOutcome, IoHttpRequest};

    let mut buf = Vec::with_capacity(HTTP_READ_CHUNK_BYTES);
    let mut temp = [0u8; HTTP_READ_CHUNK_BYTES];

    let parsed_head = loop {
        match parse_http_head(&buf) {
            Ok(Some(head)) => break head,
            Ok(None) => {
                if buf.len() >= HTTP_MAX_HEADER_BYTES {
                    return Ok(HttpReadOutcome::BadRequest(
                        "request headers too large".to_string(),
                    ));
                }

                let read = stream
                    .read(&mut temp)
                    .map_err(|err| format!("http.read_request: read failed: {err}"))?;

                if read == 0 {
                    if buf.is_empty() {
                        return Ok(HttpReadOutcome::Eof);
                    }
                    return Ok(HttpReadOutcome::BadRequest(
                        "unexpected EOF while reading request headers".to_string(),
                    ));
                }

                buf.extend_from_slice(&temp[..read]);
            }
            Err(message) => return Ok(HttpReadOutcome::BadRequest(message)),
        }
    };

    if parsed_head.content_length > max_body_bytes {
        return Ok(HttpReadOutcome::BadRequest(format!(
            "request body too large: {} > {}",
            parsed_head.content_length, max_body_bytes
        )));
    }

    let mut body_bytes = if parsed_head.bytes_consumed < buf.len() {
        buf[parsed_head.bytes_consumed..].to_vec()
    } else {
        Vec::new()
    };

    if body_bytes.len() > parsed_head.content_length {
        body_bytes.truncate(parsed_head.content_length);
    }

    while body_bytes.len() < parsed_head.content_length {
        let remaining = parsed_head.content_length - body_bytes.len();
        let to_read = remaining.min(HTTP_READ_CHUNK_BYTES);

        let read = stream
            .read(&mut temp[..to_read])
            .map_err(|err| format!("http.read_request: read failed: {err}"))?;

        if read == 0 {
            return Ok(HttpReadOutcome::BadRequest(
                "unexpected EOF while reading request body".to_string(),
            ));
        }

        body_bytes.extend_from_slice(&temp[..read]);
    }

    let body = match String::from_utf8(body_bytes) {
        Ok(body) => body,
        Err(_) => {
            return Ok(HttpReadOutcome::BadRequest(
                "request body is not valid UTF-8".to_string(),
            ));
        }
    };

    let content_length = i64::try_from(parsed_head.content_length)
        .map_err(|_| "http.read_request: content-length exceeds supported size".to_string())?;

    let path = if parsed_head.uri.path().is_empty() {
        "/".to_string()
    } else {
        parsed_head.uri.path().to_string()
    };

    Ok(HttpReadOutcome::Request(HttpRequest {
        method: parsed_head.method.as_str().to_string(),
        target: parsed_head.target,
        path,
        query: parsed_head.uri.query().unwrap_or("").to_string(),
        version: http_version_token(parsed_head.version).to_string(),
        headers: parsed_head.header_pairs,
        body,
        content_length,
    }))
}

/// Convert an HttpReadOutcome to IoHttpOutcome for use with IoResult.
pub fn http_outcome_to_io(outcome: HttpReadOutcome) -> IoHttpOutcome {
    match outcome {
        HttpReadOutcome::Eof => IoHttpOutcome::Eof,
        HttpReadOutcome::BadRequest(msg) => IoHttpOutcome::BadRequest(msg),
        HttpReadOutcome::Request(req) => IoHttpOutcome::Request(IoHttpRequest {
            method: req.method,
            target: req.target,
            path: req.path,
            query: req.query,
            version: req.version,
            headers: req.headers,
            body: req.body,
            content_length: req.content_length,
        }),
    }
}

fn split_segments(path: &str) -> Vec<&str> {
    path.split('/')
        .filter(|segment| !segment.is_empty())
        .collect()
}

#[derive(Debug)]
struct ParsedHttpHead {
    method: Method,
    target: String,
    uri: Uri,
    version: Version,
    header_pairs: Vec<(String, String)>,
    content_length: usize,
    bytes_consumed: usize,
}

fn parse_http_head(buf: &[u8]) -> Result<Option<ParsedHttpHead>, String> {
    let mut headers = [httparse::EMPTY_HEADER; HTTP_MAX_HEADER_COUNT];
    let mut parsed = httparse::Request::new(&mut headers);

    let bytes_consumed = match parsed.parse(buf) {
        Ok(httparse::Status::Complete(n)) => n,
        Ok(httparse::Status::Partial) => return Ok(None),
        Err(err) => return Err(format!("malformed HTTP request: {err}")),
    };

    let method_token = parsed
        .method
        .ok_or_else(|| "missing HTTP method".to_string())?;
    let target = parsed
        .path
        .ok_or_else(|| "missing request target".to_string())?
        .to_string();
    let method = Method::from_bytes(method_token.as_bytes())
        .map_err(|_| format!("invalid HTTP method '{method_token}'"))?;
    let uri: Uri = target
        .parse()
        .map_err(|_| "invalid request target URI".to_string())?;

    let version = match parsed.version {
        Some(0) => Version::HTTP_10,
        Some(1) => Version::HTTP_11,
        Some(other) => return Err(format!("unsupported HTTP version 1.{other}")),
        None => return Err("missing HTTP version".to_string()),
    };

    let mut header_map = HeaderMap::new();
    let mut header_pairs = Vec::with_capacity(parsed.headers.len());
    for header in parsed.headers.iter() {
        let name = HeaderName::from_bytes(header.name.as_bytes())
            .map_err(|_| format!("invalid header name '{}'", header.name))?;
        let value = HeaderValue::from_bytes(header.value)
            .map_err(|_| format!("invalid header value for '{}'", header.name))?;

        let value_text = value
            .to_str()
            .map_err(|_| format!("header '{}' is not valid UTF-8", header.name))?
            .to_string();

        header_pairs.push((name.as_str().to_ascii_lowercase(), value_text));
        header_map.append(name, value);
    }

    let content_length = match header_map.get(CONTENT_LENGTH) {
        Some(value) => {
            let text = value
                .to_str()
                .map_err(|_| "invalid content-length header".to_string())?;
            text.parse::<usize>()
                .map_err(|_| "invalid content-length header".to_string())?
        }
        None => 0,
    };

    Ok(Some(ParsedHttpHead {
        method,
        target,
        uri,
        version,
        header_pairs,
        content_length,
        bytes_consumed,
    }))
}

fn parse_http_version(token: &str) -> Result<Version, String> {
    match token {
        "HTTP/0.9" => Ok(Version::HTTP_09),
        "HTTP/1.0" => Ok(Version::HTTP_10),
        "HTTP/1.1" => Ok(Version::HTTP_11),
        "HTTP/2" | "HTTP/2.0" => Ok(Version::HTTP_2),
        "HTTP/3" | "HTTP/3.0" => Ok(Version::HTTP_3),
        _ => Err(format!("invalid HTTP version '{token}'")),
    }
}

fn http_version_token(version: Version) -> &'static str {
    match version {
        Version::HTTP_09 => "HTTP/0.9",
        Version::HTTP_10 => "HTTP/1.0",
        Version::HTTP_11 => "HTTP/1.1",
        Version::HTTP_2 => "HTTP/2.0",
        Version::HTTP_3 => "HTTP/3.0",
        _ => "HTTP/1.1",
    }
}

fn status_code_from_i64(code: i64, context: &str) -> WalrusResult<StatusCode> {
    let code_u16 = u16::try_from(code).map_err(|_| WalrusError::GenericError {
        message: format!("{context}: invalid status code {code}"),
    })?;

    StatusCode::from_u16(code_u16).map_err(|_| WalrusError::GenericError {
        message: format!("{context}: invalid status code {code}"),
    })
}

fn serialize_http_response(response: &Response<String>) -> String {
    let mut out = String::new();
    let _ = write!(
        &mut out,
        "{} {} {}\r\n",
        http_version_token(response.version()),
        response.status().as_u16(),
        response.status().canonical_reason().unwrap_or("Status")
    );

    for (name, value) in response.headers() {
        out.push_str(name.as_str());
        out.push_str(": ");
        if let Ok(text) = value.to_str() {
            out.push_str(text);
        }
        out.push_str("\r\n");
    }

    out.push_str("\r\n");
    out.push_str(response.body());
    out
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
