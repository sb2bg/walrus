use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::NetTcpBind,
        "std/net",
        "tcp_bind",
        &["host", "port"],
        "Bind a TCP listener and return its handle.",
        native_net_tcp_bind,
    ),
    define_native_spec(
        NativeFunction::NetTcpAccept,
        "std/net",
        "tcp_accept",
        &["listener"],
        "Accept one incoming connection and return a stream handle.",
        native_net_tcp_accept,
    ),
    define_native_spec(
        NativeFunction::NetTcpConnect,
        "std/net",
        "tcp_connect",
        &["host", "port"],
        "Connect to a TCP host/port and return a stream handle.",
        native_net_tcp_connect,
    ),
    define_native_spec(
        NativeFunction::NetTcpLocalPort,
        "std/net",
        "tcp_local_port",
        &["listener"],
        "Return the listener's local bound port.",
        native_net_tcp_local_port,
    ),
    define_native_spec(
        NativeFunction::NetTcpPeerAddr,
        "std/net",
        "tcp_peer_addr",
        &["stream"],
        "Return the remote peer address for a connected stream.",
        native_net_tcp_peer_addr,
    ),
    define_native_spec(
        NativeFunction::NetTcpStreamLocalAddr,
        "std/net",
        "tcp_stream_local_addr",
        &["stream"],
        "Return the local socket address for a connected stream.",
        native_net_tcp_stream_local_addr,
    ),
    define_native_spec(
        NativeFunction::NetTcpSetReadTimeout,
        "std/net",
        "tcp_set_read_timeout",
        &["stream", "timeout_ms"],
        "Set the stream read timeout in milliseconds, or pass void to clear it.",
        native_net_tcp_set_read_timeout,
    ),
    define_native_spec(
        NativeFunction::NetTcpSetWriteTimeout,
        "std/net",
        "tcp_set_write_timeout",
        &["stream", "timeout_ms"],
        "Set the stream write timeout in milliseconds, or pass void to clear it.",
        native_net_tcp_set_write_timeout,
    ),
    define_native_spec(
        NativeFunction::NetTcpSetNodelay,
        "std/net",
        "tcp_set_nodelay",
        &["stream", "enabled"],
        "Enable or disable TCP_NODELAY on a connected stream.",
        native_net_tcp_set_nodelay,
    ),
    define_native_spec(
        NativeFunction::NetTcpShutdown,
        "std/net",
        "tcp_shutdown",
        &["stream", "how"],
        "Shutdown the read side, write side, or both sides of a TCP stream.",
        native_net_tcp_shutdown,
    ),
    define_native_spec(
        NativeFunction::NetTcpRead,
        "std/net",
        "tcp_read",
        &["stream", "max_bytes"],
        "Read up to max_bytes from a stream; returns void on EOF.",
        native_net_tcp_read,
    ),
    define_native_spec(
        NativeFunction::NetTcpReadLine,
        "std/net",
        "tcp_read_line",
        &["stream"],
        "Read one line from a stream; returns void on EOF.",
        native_net_tcp_read_line,
    ),
    define_native_spec(
        NativeFunction::NetTcpWrite,
        "std/net",
        "tcp_write",
        &["stream", "content"],
        "Write utf8 content to a stream and return bytes written.",
        native_net_tcp_write,
    ),
    define_native_spec(
        NativeFunction::NetTcpClose,
        "std/net",
        "tcp_close",
        &["stream"],
        "Close a TCP stream handle.",
        native_net_tcp_close,
    ),
    define_native_spec(
        NativeFunction::NetTcpCloseListener,
        "std/net",
        "tcp_close_listener",
        &["listener"],
        "Close a TCP listener handle.",
        native_net_tcp_close_listener,
    ),
];

fn native_net_tcp_bind(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let host = vm.value_to_string(args[0], span)?;
    let port = vm.value_to_int(args[1], span)?;
    crate::stdlib::tcp_bind(&host, port, span)
}

fn native_net_tcp_accept(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let listener_handle = vm.value_to_int(args[0], span)?;
    let listener = crate::stdlib::shared_tcp_listener(listener_handle).ok_or_else(|| {
        WalrusError::GenericError {
            message: format!("net.tcp_accept: invalid listener handle {listener_handle}"),
        }
    })?;
    Ok(vm.spawn_io(move || match listener.accept() {
        Ok((stream, _addr)) => Ok(crate::value::IoResult::Stream(stream)),
        Err(err) => Err(format!("net.tcp_accept: accept failed: {err}")),
    }))
}

fn native_net_tcp_connect(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let host = vm.value_to_string(args[0], span)?;
    let port = vm.value_to_int(args[1], span)?;
    crate::stdlib::ensure_valid_port(port, "tcp_connect")?;
    let addr = format!("{host}:{port}");
    Ok(
        vm.spawn_io(move || match std::net::TcpStream::connect(&addr) {
            Ok(stream) => Ok(crate::value::IoResult::Stream(stream)),
            Err(err) => Err(format!(
                "net.tcp_connect: failed to connect to '{addr}': {err}"
            )),
        }),
    )
}

fn native_net_tcp_local_port(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let listener = vm.value_to_int(args[0], span)?;
    Ok(Value::Int(crate::stdlib::tcp_local_port(listener, span)?))
}

fn native_net_tcp_peer_addr(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let addr = crate::stdlib::tcp_peer_addr(stream, span)?;
    Ok(vm
        .get_heap_mut()
        .push(crate::arenas::HeapValue::String(&addr)))
}

fn native_net_tcp_stream_local_addr(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let addr = crate::stdlib::tcp_stream_local_addr(stream, span)?;
    Ok(vm
        .get_heap_mut()
        .push(crate::arenas::HeapValue::String(&addr)))
}

fn timeout_arg(vm: &VM<'_>, value: Value, span: Span, name: &str) -> WalrusResult<Option<u64>> {
    if matches!(value, Value::Void) {
        return Ok(None);
    }
    let timeout = vm.value_to_int(value, span)?;
    if timeout < 0 {
        return Err(WalrusError::GenericError {
            message: format!("{name} must be >= 0 or void"),
        });
    }
    Ok(Some(timeout as u64))
}

fn native_net_tcp_set_read_timeout(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let timeout_ms = timeout_arg(vm, args[1], span, "net.tcp_set_read_timeout")?;
    crate::stdlib::tcp_set_read_timeout(stream, timeout_ms, span)?;
    Ok(Value::Void)
}

fn native_net_tcp_set_write_timeout(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let timeout_ms = timeout_arg(vm, args[1], span, "net.tcp_set_write_timeout")?;
    crate::stdlib::tcp_set_write_timeout(stream, timeout_ms, span)?;
    Ok(Value::Void)
}

fn native_net_tcp_set_nodelay(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let enabled = match args[1] {
        Value::Bool(enabled) => enabled,
        other => {
            return Err(WalrusError::TypeMismatch {
                expected: "bool".to_string(),
                found: other.get_type().to_string(),
                span,
                src: vm.source_ref().source().into(),
                filename: vm.source_ref().filename().into(),
            });
        }
    };
    crate::stdlib::tcp_set_nodelay(stream, enabled, span)?;
    Ok(Value::Void)
}

fn native_net_tcp_shutdown(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let how = vm.value_to_string(args[1], span)?;
    crate::stdlib::tcp_shutdown(stream, &how, span)?;
    Ok(Value::Void)
}

fn native_net_tcp_read(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream_handle = vm.value_to_int(args[0], span)?;
    let max_bytes = vm.value_to_int(args[1], span)?;
    if max_bytes <= 0 {
        return Err(WalrusError::GenericError {
            message: format!("net.tcp_read: max_bytes must be > 0, got {max_bytes}"),
        });
    }
    let stream = crate::stdlib::shared_tcp_stream(stream_handle).ok_or_else(|| {
        WalrusError::GenericError {
            message: format!("net.tcp_read: invalid stream handle {stream_handle}"),
        }
    })?;
    Ok(vm.spawn_io(move || {
        use std::io::Read;
        let mut stream = stream
            .lock()
            .map_err(|_| "net.tcp_read: stream lock poisoned".to_string())?;
        let mut buf = vec![0u8; max_bytes as usize];
        match stream.read(&mut buf) {
            Ok(0) => Ok(crate::value::IoResult::Void),
            Ok(n) => {
                buf.truncate(n);
                Ok(crate::value::IoResult::Bytes(buf))
            }
            Err(err) => Err(format!("net.tcp_read: read failed: {err}")),
        }
    }))
}

fn native_net_tcp_read_line(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream_handle = vm.value_to_int(args[0], span)?;
    let stream = crate::stdlib::shared_tcp_stream(stream_handle).ok_or_else(|| {
        WalrusError::GenericError {
            message: format!("net.tcp_read_line: invalid stream handle {stream_handle}"),
        }
    })?;
    Ok(vm.spawn_io(move || {
        use std::io::Read;
        let mut stream = stream
            .lock()
            .map_err(|_| "net.tcp_read_line: stream lock poisoned".to_string())?;
        let mut bytes = Vec::new();
        let mut buf = [0u8; 1];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => {
                    if bytes.is_empty() {
                        return Ok(crate::value::IoResult::Void);
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
                    return Err(format!("net.tcp_read_line: read failed: {err}"));
                }
            }
        }
        if bytes.last().copied() == Some(b'\r') {
            bytes.pop();
        }
        Ok(crate::value::IoResult::Bytes(bytes))
    }))
}

fn native_net_tcp_write(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream_handle = vm.value_to_int(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    let stream = crate::stdlib::shared_tcp_stream(stream_handle).ok_or_else(|| {
        WalrusError::GenericError {
            message: format!("net.tcp_write: invalid stream handle {stream_handle}"),
        }
    })?;
    Ok(vm.spawn_io(move || {
        use std::io::Write;
        let mut stream = stream
            .lock()
            .map_err(|_| "net.tcp_write: stream lock poisoned".to_string())?;
        let len = content.len();
        match stream.write_all(content.as_bytes()) {
            Ok(()) => Ok(crate::value::IoResult::ByteCount(len)),
            Err(err) => Err(format!("net.tcp_write: write failed: {err}")),
        }
    }))
}

fn native_net_tcp_close(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    crate::stdlib::tcp_close(stream, span)?;
    Ok(Value::Void)
}

fn native_net_tcp_close_listener(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let listener = vm.value_to_int(args[0], span)?;
    crate::stdlib::tcp_close_listener(listener, span)?;
    Ok(Value::Void)
}
