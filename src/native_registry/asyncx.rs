use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::helpers::{
    non_negative_millis, task_key_from_value, value_sequence, value_sequence_or_void,
};
use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::AsyncSpawn,
        "std/async",
        "spawn",
        &["callable", "args"],
        "Schedule a task from a function+args (or pass through an existing task).",
        native_async_spawn,
    ),
    define_native_spec(
        NativeFunction::AsyncSleep,
        "std/async",
        "sleep",
        &["ms"],
        "Return a task that resolves after the requested delay in milliseconds.",
        native_async_sleep,
    ),
    define_native_spec(
        NativeFunction::AsyncTimeout,
        "std/async",
        "timeout",
        &["task", "ms"],
        "Wrap a task and fail it if it does not complete before the timeout.",
        native_async_timeout,
    ),
    define_native_spec(
        NativeFunction::AsyncGather,
        "std/async",
        "gather",
        &["tasks"],
        "Return a task that resolves to a list of results once all tasks complete.",
        native_async_gather,
    ),
    define_native_spec(
        NativeFunction::AsyncRace,
        "std/async",
        "race",
        &["tasks"],
        "Return a task that resolves to the first completed result from a list of tasks.",
        native_async_race,
    ),
    define_native_spec(
        NativeFunction::AsyncAllSettled,
        "std/async",
        "all_settled",
        &["tasks"],
        "Return a task that resolves to a list of {status, value/error} dicts once all tasks settle.",
        native_async_all_settled,
    ),
    define_native_spec(
        NativeFunction::AsyncChannel,
        "std/async",
        "channel",
        &[],
        "Create a channel and return [sender, receiver] for task-to-task communication.",
        native_async_channel,
    ),
    define_native_spec(
        NativeFunction::AsyncSend,
        "std/async",
        "send",
        &["sender", "value"],
        "Send a value into a channel. Returns true on success, false if receiver is closed.",
        native_async_send,
    ),
    define_native_spec(
        NativeFunction::AsyncRecv,
        "std/async",
        "recv",
        &["receiver"],
        "Return a task that resolves to the next value from a channel.",
        native_async_recv,
    ),
    define_native_spec(
        NativeFunction::AsyncClose,
        "std/async",
        "close",
        &["endpoint"],
        "Close a channel sender or receiver. Pending receivers resolve to void and future sends return false.",
        native_async_close,
    ),
    define_native_spec(
        NativeFunction::AsyncStatus,
        "std/async",
        "status",
        &["task"],
        "Return the status of a task as a string: pending, ready, failed, or cancelled.",
        native_async_status,
    ),
    define_native_spec(
        NativeFunction::AsyncCancel,
        "std/async",
        "cancel",
        &["task"],
        "Request cancellation for a task. Returns true when cancellation was applied.",
        native_async_cancel,
    ),
    define_native_spec(
        NativeFunction::AsyncCancelled,
        "std/async",
        "cancelled",
        &["task"],
        "Return true if a task is in the cancelled state.",
        native_async_cancelled,
    ),
    define_native_spec(
        NativeFunction::AsyncYield,
        "std/async",
        "yield",
        &[],
        "Give other queued tasks a turn to run, then resume.",
        native_async_yield,
    ),
];

fn native_async_spawn(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let call_args = value_sequence_or_void(vm, args[1], span)?;
    vm.spawn_task_from_callable(args[0], call_args, span)
}

fn native_async_sleep(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let delay_ms = non_negative_millis(vm, args[0], span, "sleep milliseconds")?;
    Ok(vm.create_sleep_task(delay_ms))
}

fn native_async_timeout(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let task_key = task_key_from_value(vm, args[0], span)?;
    let timeout_ms = non_negative_millis(vm, args[1], span, "timeout milliseconds")?;
    Ok(vm.create_timeout_task(task_key, timeout_ms))
}

fn native_async_gather(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let items = value_sequence(vm, args[0], span)?;
    let mut tasks = Vec::with_capacity(items.len());
    for value in items {
        match value {
            Value::Task(task_key) => tasks.push(task_key),
            other => {
                return Err(WalrusError::TypeMismatch {
                    expected: "task".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: vm.source_ref().source().into(),
                    filename: vm.source_ref().filename().into(),
                });
            }
        }
    }
    Ok(vm.create_gather_task(tasks))
}

fn native_async_race(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let items = value_sequence(vm, args[0], span)?;
    let mut tasks = Vec::with_capacity(items.len());
    for value in items {
        match value {
            Value::Task(task_key) => tasks.push(task_key),
            other => {
                return Err(WalrusError::TypeMismatch {
                    expected: "task".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: vm.source_ref().source().into(),
                    filename: vm.source_ref().filename().into(),
                });
            }
        }
    }
    Ok(vm.create_race_task(tasks))
}

fn native_async_all_settled(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let items = value_sequence(vm, args[0], span)?;
    let mut tasks = Vec::with_capacity(items.len());
    for value in items {
        match value {
            Value::Task(task_key) => tasks.push(task_key),
            other => {
                return Err(WalrusError::TypeMismatch {
                    expected: "task".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: vm.source_ref().source().into(),
                    filename: vm.source_ref().filename().into(),
                });
            }
        }
    }
    Ok(vm.create_all_settled_task(tasks))
}

fn native_async_channel(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let (sender, receiver) = vm.create_user_channel();
    let list = vm
        .get_heap_mut()
        .push(HeapValue::List(vec![sender, receiver]));
    Ok(list)
}

fn native_async_send(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let sender_key = match args[0] {
        Value::Dict(key) => key,
        other => {
            return Err(WalrusError::TypeMismatch {
                expected: "channel sender".to_string(),
                found: other.get_type().to_string(),
                span,
                src: vm.source_ref().source().into(),
                filename: vm.source_ref().filename().into(),
            });
        }
    };
    let value = args[1];
    Ok(Value::Bool(vm.channel_send(sender_key, value)?))
}

fn native_async_recv(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let receiver_key = match args[0] {
        Value::Dict(key) => key,
        other => {
            return Err(WalrusError::TypeMismatch {
                expected: "channel receiver".to_string(),
                found: other.get_type().to_string(),
                span,
                src: vm.source_ref().source().into(),
                filename: vm.source_ref().filename().into(),
            });
        }
    };
    Ok(vm.channel_recv(receiver_key)?)
}

fn native_async_close(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let endpoint_key = match args[0] {
        Value::Dict(key) => key,
        other => {
            return Err(WalrusError::TypeMismatch {
                expected: "channel endpoint".to_string(),
                found: other.get_type().to_string(),
                span,
                src: vm.source_ref().source().into(),
                filename: vm.source_ref().filename().into(),
            });
        }
    };
    Ok(Value::Bool(vm.channel_close(endpoint_key)?))
}

fn native_async_status(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let task_key = task_key_from_value(vm, args[0], span)?;
    let status = vm.task_status_string(task_key)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(status)))
}

fn native_async_cancel(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let task_key = task_key_from_value(vm, args[0], span)?;
    Ok(Value::Bool(vm.cancel_task(task_key)?))
}

fn native_async_cancelled(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let task_key = task_key_from_value(vm, args[0], span)?;
    Ok(Value::Bool(vm.task_is_cancelled(task_key)?))
}

fn native_async_yield(vm: &mut VM<'_>, _args: &[Value], span: Span) -> WalrusResult<Value> {
    vm.run_queued_tasks(span)?;
    Ok(Value::Void)
}
