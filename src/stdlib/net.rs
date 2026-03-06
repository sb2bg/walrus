use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::span::Span;
use crate::value::Value;

use super::NET_TABLE;
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
        let listener =
            table
                .borrow()
                .listener(listener_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_accept: invalid listener handle {listener_handle}"),
                })?;

        let (stream, _addr) = listener.accept().map_err(|err| WalrusError::GenericError {
            message: format!("net.tcp_accept: accept failed: {err}"),
        })?;

        let stream_handle = table.borrow_mut().insert_stream(stream);
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
        let listener =
            table
                .borrow()
                .listener(listener_handle)
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

pub fn tcp_peer_addr(stream_handle: i64, _span: Span) -> WalrusResult<String> {
    NET_TABLE.with(|table| {
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_peer_addr: invalid stream handle {stream_handle}"),
                })?;
        let stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_peer_addr: stream lock poisoned".to_string(),
        })?;
        stream
            .peer_addr()
            .map(|addr| addr.to_string())
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_peer_addr: failed to read peer addr: {err}"),
            })
    })
}

pub fn tcp_stream_local_addr(stream_handle: i64, _span: Span) -> WalrusResult<String> {
    NET_TABLE.with(|table| {
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!(
                        "net.tcp_stream_local_addr: invalid stream handle {stream_handle}"
                    ),
                })?;
        let stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_stream_local_addr: stream lock poisoned".to_string(),
        })?;
        stream
            .local_addr()
            .map(|addr| addr.to_string())
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_stream_local_addr: failed to read local addr: {err}"),
            })
    })
}

pub fn tcp_set_read_timeout(
    stream_handle: i64,
    timeout_ms: Option<u64>,
    _span: Span,
) -> WalrusResult<()> {
    NET_TABLE.with(|table| {
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!(
                        "net.tcp_set_read_timeout: invalid stream handle {stream_handle}"
                    ),
                })?;
        let stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_set_read_timeout: stream lock poisoned".to_string(),
        })?;
        stream
            .set_read_timeout(timeout_ms.map(Duration::from_millis))
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_set_read_timeout: failed: {err}"),
            })
    })
}

pub fn tcp_set_write_timeout(
    stream_handle: i64,
    timeout_ms: Option<u64>,
    _span: Span,
) -> WalrusResult<()> {
    NET_TABLE.with(|table| {
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!(
                        "net.tcp_set_write_timeout: invalid stream handle {stream_handle}"
                    ),
                })?;
        let stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_set_write_timeout: stream lock poisoned".to_string(),
        })?;
        stream
            .set_write_timeout(timeout_ms.map(Duration::from_millis))
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_set_write_timeout: failed: {err}"),
            })
    })
}

pub fn tcp_set_nodelay(stream_handle: i64, enabled: bool, _span: Span) -> WalrusResult<()> {
    NET_TABLE.with(|table| {
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_set_nodelay: invalid stream handle {stream_handle}"),
                })?;
        let stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_set_nodelay: stream lock poisoned".to_string(),
        })?;
        stream
            .set_nodelay(enabled)
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_set_nodelay: failed: {err}"),
            })
    })
}

pub fn tcp_shutdown(stream_handle: i64, how: &str, _span: Span) -> WalrusResult<()> {
    let shutdown = match how {
        "read" => Shutdown::Read,
        "write" => Shutdown::Write,
        "both" => Shutdown::Both,
        _ => {
            return Err(WalrusError::GenericError {
                message: format!(
                    "net.tcp_shutdown: invalid mode '{how}', expected 'read', 'write', or 'both'"
                ),
            });
        }
    };

    NET_TABLE.with(|table| {
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_shutdown: invalid stream handle {stream_handle}"),
                })?;
        let stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_shutdown: stream lock poisoned".to_string(),
        })?;
        stream
            .shutdown(shutdown)
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_shutdown: failed: {err}"),
            })
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
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_read: invalid stream handle {stream_handle}"),
                })?;
        let mut stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_read: stream lock poisoned".to_string(),
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
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_read_line: invalid stream handle {stream_handle}"),
                })?;
        let mut stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_read_line: stream lock poisoned".to_string(),
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
        let stream =
            table
                .borrow()
                .stream(stream_handle)
                .ok_or_else(|| WalrusError::GenericError {
                    message: format!("net.tcp_write: invalid stream handle {stream_handle}"),
                })?;
        let mut stream = stream.lock().map_err(|_| WalrusError::GenericError {
            message: "net.tcp_write: stream lock poisoned".to_string(),
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
        if let Some(stream) = table.remove_stream(stream_handle) {
            if let Ok(stream) = stream.lock() {
                let _ = stream.shutdown(Shutdown::Both);
            }
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

/// Share a TcpListener for use in background I/O.
/// Returns None if the handle is invalid.
pub fn shared_tcp_listener(handle: i64) -> Option<Arc<TcpListener>> {
    NET_TABLE.with(|table| table.borrow().listener(handle))
}

/// Share a TcpStream for use in background I/O.
/// Returns None if the handle is invalid.
pub fn shared_tcp_stream(handle: i64) -> Option<Arc<Mutex<TcpStream>>> {
    NET_TABLE.with(|table| table.borrow().stream(handle))
}
