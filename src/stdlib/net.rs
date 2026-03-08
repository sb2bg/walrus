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
            .stream
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
            .stream
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
            .stream
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
            .stream
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
            .stream
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
            .stream
            .shutdown(shutdown)
            .map_err(|err| WalrusError::GenericError {
                message: format!("net.tcp_shutdown: failed: {err}"),
            })
    })
}

/// Close a stream handle.
pub fn tcp_close(stream_handle: i64, _span: Span) -> WalrusResult<()> {
    NET_TABLE.with(|table| {
        let mut table = table.borrow_mut();
        if let Some(stream) = table.remove_stream(stream_handle) {
            if let Ok(stream) = stream.lock() {
                let _ = stream.stream.shutdown(Shutdown::Both);
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
pub fn shared_tcp_stream(handle: i64) -> Option<Arc<Mutex<super::SharedTcpStream>>> {
    NET_TABLE.with(|table| table.borrow().stream(handle))
}
