use float_ord::FloatOrd;
use std::io::Write;

use crate::WalrusResult;
use crate::arenas::HeapValue;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

pub type NativeHandler = for<'a> fn(&mut VM<'a>, &[Value], Span) -> WalrusResult<Value>;

#[derive(Clone, Copy)]
pub struct NativeSpec {
    pub id: NativeFunction,
    pub module: &'static str,
    pub name: &'static str,
    pub arity: usize,
    pub params: &'static [&'static str],
    pub docs: &'static str,
    pub handler: NativeHandler,
}

macro_rules! count_params {
    () => {
        0usize
    };
    ($head:literal $(, $tail:literal)*) => {
        1usize + count_params!($($tail),*)
    };
}

macro_rules! native_spec {
    (
        $id:ident,
        $module:literal,
        $name:literal,
        [$($param:literal),* $(,)?],
        $docs:literal,
        $handler:path
    ) => {
        NativeSpec {
            id: NativeFunction::$id,
            module: $module,
            name: $name,
            arity: count_params!($($param),*),
            params: &[$($param),*],
            docs: $docs,
            handler: $handler,
        }
    };
}

pub static NATIVE_SPECS: &[NativeSpec] = &[
    native_spec!(
        CoreLen,
        "std/core",
        "len",
        ["value"],
        "Return length of a string, list, dict, or module.",
        native_core_len
    ),
    native_spec!(
        CoreStr,
        "std/core",
        "str",
        ["value"],
        "Convert any value to a string.",
        native_core_str
    ),
    native_spec!(
        CoreType,
        "std/core",
        "type",
        ["value"],
        "Return the runtime type name for a value.",
        native_core_type
    ),
    native_spec!(
        CoreInput,
        "std/core",
        "input",
        ["prompt"],
        "Print a prompt and read one line from stdin.",
        native_core_input
    ),
    native_spec!(
        CoreGc,
        "std/core",
        "gc",
        [],
        "Trigger garbage collection and return collection stats.",
        native_core_gc
    ),
    native_spec!(
        CoreHeapStats,
        "std/core",
        "heap_stats",
        [],
        "Return heap and GC statistics.",
        native_core_heap_stats
    ),
    native_spec!(
        CoreGcThreshold,
        "std/core",
        "gc_threshold",
        ["threshold"],
        "Set GC allocation threshold and return previous value.",
        native_core_gc_threshold
    ),
    native_spec!(
        AsyncSpawn,
        "std/async",
        "spawn",
        ["callable", "args"],
        "Schedule a task from a function+args (or pass through an existing task).",
        native_async_spawn
    ),
    native_spec!(
        AsyncSleep,
        "std/async",
        "sleep",
        ["ms"],
        "Return a task that resolves after the requested delay in milliseconds.",
        native_async_sleep
    ),
    native_spec!(
        AsyncTimeout,
        "std/async",
        "timeout",
        ["task", "ms"],
        "Wrap a task and fail it if it does not complete before the timeout.",
        native_async_timeout
    ),
    native_spec!(
        AsyncGather,
        "std/async",
        "gather",
        ["tasks"],
        "Return a task that resolves to a list of results once all tasks complete.",
        native_async_gather
    ),
    native_spec!(
        FileOpen,
        "std/io",
        "file_open",
        ["path", "mode"],
        "Open a file and return a handle.",
        native_file_open
    ),
    native_spec!(
        FileRead,
        "std/io",
        "file_read",
        ["handle"],
        "Read entire contents from an open file handle.",
        native_file_read
    ),
    native_spec!(
        FileReadLine,
        "std/io",
        "file_read_line",
        ["handle"],
        "Read one line from an open file handle.",
        native_file_read_line
    ),
    native_spec!(
        FileWrite,
        "std/io",
        "file_write",
        ["handle", "content"],
        "Write string content to an open file handle.",
        native_file_write
    ),
    native_spec!(
        FileClose,
        "std/io",
        "file_close",
        ["handle"],
        "Close an open file handle.",
        native_file_close
    ),
    native_spec!(
        FileExists,
        "std/io",
        "file_exists",
        ["path"],
        "Return true if the path exists.",
        native_file_exists
    ),
    native_spec!(
        ReadFile,
        "std/io",
        "read_file",
        ["path"],
        "Read a file into a string.",
        native_read_file
    ),
    native_spec!(
        WriteFile,
        "std/io",
        "write_file",
        ["path", "content"],
        "Write a string to a file.",
        native_write_file
    ),
    native_spec!(
        EnvGet,
        "std/sys",
        "env_get",
        ["name"],
        "Get an environment variable by name.",
        native_env_get
    ),
    native_spec!(
        Args,
        "std/sys",
        "args",
        [],
        "Get command-line arguments.",
        native_args
    ),
    native_spec!(
        Cwd,
        "std/sys",
        "cwd",
        [],
        "Get the current working directory.",
        native_cwd
    ),
    native_spec!(
        Exit,
        "std/sys",
        "exit",
        ["code"],
        "Exit the process with a status code.",
        native_exit
    ),
    native_spec!(
        NetTcpBind,
        "std/net",
        "tcp_bind",
        ["host", "port"],
        "Bind a TCP listener and return its handle.",
        native_net_tcp_bind
    ),
    native_spec!(
        NetTcpAccept,
        "std/net",
        "tcp_accept",
        ["listener"],
        "Accept one incoming connection and return a stream handle.",
        native_net_tcp_accept
    ),
    native_spec!(
        NetTcpConnect,
        "std/net",
        "tcp_connect",
        ["host", "port"],
        "Connect to a TCP host/port and return a stream handle.",
        native_net_tcp_connect
    ),
    native_spec!(
        NetTcpLocalPort,
        "std/net",
        "tcp_local_port",
        ["listener"],
        "Return the listener's local bound port.",
        native_net_tcp_local_port
    ),
    native_spec!(
        NetTcpRead,
        "std/net",
        "tcp_read",
        ["stream", "max_bytes"],
        "Read up to max_bytes from a stream; returns void on EOF.",
        native_net_tcp_read
    ),
    native_spec!(
        NetTcpReadLine,
        "std/net",
        "tcp_read_line",
        ["stream"],
        "Read one line from a stream; returns void on EOF.",
        native_net_tcp_read_line
    ),
    native_spec!(
        NetTcpWrite,
        "std/net",
        "tcp_write",
        ["stream", "content"],
        "Write utf8 content to a stream and return bytes written.",
        native_net_tcp_write
    ),
    native_spec!(
        NetTcpClose,
        "std/net",
        "tcp_close",
        ["stream"],
        "Close a TCP stream handle.",
        native_net_tcp_close
    ),
    native_spec!(
        NetTcpCloseListener,
        "std/net",
        "tcp_close_listener",
        ["listener"],
        "Close a TCP listener handle.",
        native_net_tcp_close_listener
    ),
    native_spec!(
        HttpParseRequestLine,
        "std/http",
        "parse_request_line",
        ["line"],
        "Parse an HTTP request line into method/path/query/version fields.",
        native_http_parse_request_line
    ),
    native_spec!(
        HttpParseQuery,
        "std/http",
        "parse_query",
        ["query"],
        "Parse a query string into a dict of key/value pairs.",
        native_http_parse_query
    ),
    native_spec!(
        HttpNormalizePath,
        "std/http",
        "normalize_path",
        ["path"],
        "Normalize an HTTP path (collapse duplicate slashes, trim trailing slash).",
        native_http_normalize_path
    ),
    native_spec!(
        HttpMatchRoute,
        "std/http",
        "match_route",
        ["pattern", "path"],
        "Match a route pattern against a path with :params and trailing * wildcard.",
        native_http_match_route
    ),
    native_spec!(
        HttpStatusText,
        "std/http",
        "status_text",
        ["status"],
        "Return the canonical reason phrase for an HTTP status code.",
        native_http_status_text
    ),
    native_spec!(
        HttpResponse,
        "std/http",
        "response",
        ["status", "body"],
        "Build an HTTP/1.1 response string with default headers.",
        native_http_response
    ),
    native_spec!(
        HttpResponseWithHeaders,
        "std/http",
        "response_with_headers",
        ["status", "body", "headers"],
        "Build an HTTP/1.1 response string with caller-provided headers.",
        native_http_response_with_headers
    ),
    native_spec!(
        HttpReadRequest,
        "std/http",
        "read_request",
        ["stream", "max_body_bytes"],
        "Read and parse one HTTP request from a TCP stream; returns void on EOF.",
        native_http_read_request
    ),
    native_spec!(MathPi, "std/math", "pi", [], "Return pi.", native_math_pi),
    native_spec!(
        MathE,
        "std/math",
        "e",
        [],
        "Return Euler's number.",
        native_math_e
    ),
    native_spec!(
        MathTau,
        "std/math",
        "tau",
        [],
        "Return tau (2*pi).",
        native_math_tau
    ),
    native_spec!(
        MathInf,
        "std/math",
        "inf",
        [],
        "Return positive infinity.",
        native_math_inf
    ),
    native_spec!(
        MathNaN,
        "std/math",
        "nan",
        [],
        "Return NaN.",
        native_math_nan
    ),
    native_spec!(
        MathAbs,
        "std/math",
        "abs",
        ["x"],
        "Absolute value.",
        native_math_abs
    ),
    native_spec!(
        MathSign,
        "std/math",
        "sign",
        ["x"],
        "Sign as -1, 0, or 1.",
        native_math_sign
    ),
    native_spec!(
        MathMin,
        "std/math",
        "min",
        ["a", "b"],
        "Minimum of two numbers.",
        native_math_min
    ),
    native_spec!(
        MathMax,
        "std/math",
        "max",
        ["a", "b"],
        "Maximum of two numbers.",
        native_math_max
    ),
    native_spec!(
        MathClamp,
        "std/math",
        "clamp",
        ["x", "lo", "hi"],
        "Clamp to [lo, hi].",
        native_math_clamp
    ),
    native_spec!(
        MathFloor,
        "std/math",
        "floor",
        ["x"],
        "Round down to integer.",
        native_math_floor
    ),
    native_spec!(
        MathCeil,
        "std/math",
        "ceil",
        ["x"],
        "Round up to integer.",
        native_math_ceil
    ),
    native_spec!(
        MathRound,
        "std/math",
        "round",
        ["x"],
        "Round to nearest integer.",
        native_math_round
    ),
    native_spec!(
        MathTrunc,
        "std/math",
        "trunc",
        ["x"],
        "Truncate fractional component.",
        native_math_trunc
    ),
    native_spec!(
        MathFract,
        "std/math",
        "fract",
        ["x"],
        "Fractional component.",
        native_math_fract
    ),
    native_spec!(
        MathSqrt,
        "std/math",
        "sqrt",
        ["x"],
        "Square root (x >= 0).",
        native_math_sqrt
    ),
    native_spec!(
        MathCbrt,
        "std/math",
        "cbrt",
        ["x"],
        "Cube root.",
        native_math_cbrt
    ),
    native_spec!(
        MathPow,
        "std/math",
        "pow",
        ["x", "y"],
        "Raise x to power y.",
        native_math_pow
    ),
    native_spec!(
        MathHypot,
        "std/math",
        "hypot",
        ["x", "y"],
        "Euclidean norm sqrt(x*x+y*y).",
        native_math_hypot
    ),
    native_spec!(
        MathSin,
        "std/math",
        "sin",
        ["x"],
        "Sine in radians.",
        native_math_sin
    ),
    native_spec!(
        MathCos,
        "std/math",
        "cos",
        ["x"],
        "Cosine in radians.",
        native_math_cos
    ),
    native_spec!(
        MathTan,
        "std/math",
        "tan",
        ["x"],
        "Tangent in radians.",
        native_math_tan
    ),
    native_spec!(
        MathAsin,
        "std/math",
        "asin",
        ["x"],
        "Inverse sine for x in [-1,1].",
        native_math_asin
    ),
    native_spec!(
        MathAcos,
        "std/math",
        "acos",
        ["x"],
        "Inverse cosine for x in [-1,1].",
        native_math_acos
    ),
    native_spec!(
        MathAtan,
        "std/math",
        "atan",
        ["x"],
        "Inverse tangent.",
        native_math_atan
    ),
    native_spec!(
        MathAtan2,
        "std/math",
        "atan2",
        ["y", "x"],
        "Quadrant-aware inverse tangent.",
        native_math_atan2
    ),
    native_spec!(MathExp, "std/math", "exp", ["x"], "e^x.", native_math_exp),
    native_spec!(
        MathLn,
        "std/math",
        "ln",
        ["x"],
        "Natural log for x > 0.",
        native_math_ln
    ),
    native_spec!(
        MathLog2,
        "std/math",
        "log2",
        ["x"],
        "Base-2 log for x > 0.",
        native_math_log2
    ),
    native_spec!(
        MathLog10,
        "std/math",
        "log10",
        ["x"],
        "Base-10 log for x > 0.",
        native_math_log10
    ),
    native_spec!(
        MathLog,
        "std/math",
        "log",
        ["x", "base"],
        "Log in a custom base.",
        native_math_log
    ),
    native_spec!(
        MathLerp,
        "std/math",
        "lerp",
        ["a", "b", "t"],
        "Linear interpolation between a and b.",
        native_math_lerp
    ),
    native_spec!(
        MathDegrees,
        "std/math",
        "degrees",
        ["r"],
        "Radians to degrees.",
        native_math_degrees
    ),
    native_spec!(
        MathRadians,
        "std/math",
        "radians",
        ["d"],
        "Degrees to radians.",
        native_math_radians
    ),
    native_spec!(
        MathIsFinite,
        "std/math",
        "is_finite",
        ["x"],
        "True if finite.",
        native_math_is_finite
    ),
    native_spec!(
        MathIsNaN,
        "std/math",
        "is_nan",
        ["x"],
        "True if NaN.",
        native_math_is_nan
    ),
    native_spec!(
        MathIsInf,
        "std/math",
        "is_inf",
        ["x"],
        "True if infinite.",
        native_math_is_inf
    ),
    native_spec!(
        MathSeed,
        "std/math",
        "seed",
        ["n"],
        "Seed RNG state.",
        native_math_seed
    ),
    native_spec!(
        MathRandFloat,
        "std/math",
        "rand_float",
        [],
        "Random float in [0.0,1.0).",
        native_math_rand_float
    ),
    native_spec!(
        MathRandBool,
        "std/math",
        "rand_bool",
        [],
        "Random boolean.",
        native_math_rand_bool
    ),
    native_spec!(
        MathRandInt,
        "std/math",
        "rand_int",
        ["a", "b"],
        "Random integer in [a,b].",
        native_math_rand_int
    ),
    native_spec!(
        MathRandRange,
        "std/math",
        "rand_range",
        ["a", "b"],
        "Random float in [a,b).",
        native_math_rand_range
    ),
];

pub fn native_spec(function: NativeFunction) -> &'static NativeSpec {
    NATIVE_SPECS
        .iter()
        .find(|spec| spec.id == function)
        .expect("native function not registered")
}

pub fn module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    let functions: Vec<NativeFunction> = NATIVE_SPECS
        .iter()
        .filter(|spec| spec.module == module)
        .map(|spec| spec.id)
        .collect();

    if functions.is_empty() {
        None
    } else {
        Some(functions)
    }
}

pub fn dispatch_native(
    vm: &mut VM<'_>,
    native_fn: NativeFunction,
    args: Vec<Value>,
    span: Span,
) -> WalrusResult<Value> {
    let spec = native_spec(native_fn);

    if args.len() != spec.arity {
        return Err(WalrusError::InvalidArgCount {
            name: spec.name.to_string(),
            expected: spec.arity,
            got: args.len(),
            span,
            src: vm.source_ref().source().into(),
            filename: vm.source_ref().filename().into(),
        });
    }

    (spec.handler)(vm, &args, span)
}

fn native_core_len(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::String(key) => {
            let s = vm.get_heap().get_string(key)?;
            Ok(Value::Int(s.len() as i64))
        }
        Value::List(key) => {
            let list = vm.get_heap().get_list(key)?;
            Ok(Value::Int(list.len() as i64))
        }
        Value::Dict(key) => {
            let dict = vm.get_heap().get_dict(key)?;
            Ok(Value::Int(dict.len() as i64))
        }
        Value::Module(key) => {
            let module = vm.get_heap().get_module(key)?;
            Ok(Value::Int(module.len() as i64))
        }
        other => Err(WalrusError::NoLength {
            type_name: other.get_type().to_string(),
            span,
            src: vm.source_ref().source().to_string(),
            filename: vm.source_ref().filename().to_string(),
        }),
    }
}

fn native_core_str(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    let rendered = vm.get_heap().stringify(args[0])?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&rendered)))
}

fn native_core_type(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(vm
        .get_heap_mut()
        .push(HeapValue::String(args[0].get_type())))
}

fn native_core_input(vm: &mut VM<'_>, args: &[Value], _span: Span) -> WalrusResult<Value> {
    let prompt = vm.get_heap().stringify(args[0])?;
    print!("{prompt}");
    std::io::stdout()
        .flush()
        .map_err(|source| WalrusError::IOError { source })?;

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .map_err(|source| WalrusError::IOError { source })?;

    Ok(vm.get_heap_mut().push(HeapValue::String(&input)))
}

fn native_core_gc(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let roots = vm.collect_roots();
    let result = vm.get_heap_mut().force_collect(&roots);

    let mut dict = rustc_hash::FxHashMap::default();
    let heap = vm.get_heap_mut();

    let key_freed = heap.push(HeapValue::String("objects_freed"));
    let key_before = heap.push(HeapValue::String("objects_before"));
    let key_after = heap.push(HeapValue::String("objects_after"));
    let key_collections = heap.push(HeapValue::String("total_collections"));

    dict.insert(key_freed, Value::Int(result.objects_freed as i64));
    dict.insert(key_before, Value::Int(result.objects_before as i64));
    dict.insert(key_after, Value::Int(result.objects_after as i64));
    dict.insert(key_collections, Value::Int(result.collections_total as i64));

    Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
}

fn native_core_heap_stats(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let stats = vm.get_heap().heap_stats();
    let gc_info = vm.get_heap().gc_stats();

    let mut dict = rustc_hash::FxHashMap::default();
    let heap = vm.get_heap_mut();

    let key_lists = heap.push(HeapValue::String("lists"));
    let key_tuples = heap.push(HeapValue::String("tuples"));
    let key_dicts = heap.push(HeapValue::String("dicts"));
    let key_functions = heap.push(HeapValue::String("functions"));
    let key_iterators = heap.push(HeapValue::String("iterators"));
    let key_struct_defs = heap.push(HeapValue::String("struct_defs"));
    let key_struct_insts = heap.push(HeapValue::String("struct_instances"));
    let key_total = heap.push(HeapValue::String("total_objects"));

    let key_alloc_count = heap.push(HeapValue::String("allocation_count"));
    let key_bytes = heap.push(HeapValue::String("bytes_allocated"));
    let key_bytes_freed = heap.push(HeapValue::String("total_bytes_freed"));
    let key_collections = heap.push(HeapValue::String("total_collections"));
    let key_threshold = heap.push(HeapValue::String("allocation_threshold"));
    let key_mem_threshold = heap.push(HeapValue::String("memory_threshold"));

    dict.insert(key_lists, Value::Int(stats.lists as i64));
    dict.insert(key_tuples, Value::Int(stats.tuples as i64));
    dict.insert(key_dicts, Value::Int(stats.dicts as i64));
    dict.insert(key_functions, Value::Int(stats.functions as i64));
    dict.insert(key_iterators, Value::Int(stats.iterators as i64));
    dict.insert(key_struct_defs, Value::Int(stats.struct_defs as i64));
    dict.insert(key_struct_insts, Value::Int(stats.struct_instances as i64));
    dict.insert(key_total, Value::Int(stats.total_objects() as i64));

    dict.insert(key_alloc_count, Value::Int(gc_info.allocation_count as i64));
    dict.insert(key_bytes, Value::Int(gc_info.bytes_allocated as i64));
    dict.insert(
        key_bytes_freed,
        Value::Int(gc_info.total_bytes_freed as i64),
    );
    dict.insert(
        key_collections,
        Value::Int(gc_info.total_collections as i64),
    );
    dict.insert(
        key_threshold,
        Value::Int(gc_info.allocation_threshold as i64),
    );
    dict.insert(
        key_mem_threshold,
        Value::Int(gc_info.memory_threshold as i64),
    );

    Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
}

fn native_core_gc_threshold(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let n = vm.value_to_int(args[0], span)?;
    if n <= 0 {
        return Err(WalrusError::InvalidGcThresholdArg {
            span,
            src: vm.source_ref().source().to_string(),
            filename: vm.source_ref().filename().to_string(),
        });
    }

    let old = crate::gc::set_allocation_threshold(n as usize);
    Ok(Value::Int(old as i64))
}

fn value_sequence(vm: &VM<'_>, value: Value, span: Span) -> WalrusResult<Vec<Value>> {
    match value {
        Value::List(key) => Ok(vm.get_heap().get_list(key)?.to_vec()),
        Value::Tuple(key) => Ok(vm.get_heap().get_tuple(key)?.to_vec()),
        other => Err(WalrusError::TypeMismatch {
            expected: "list or tuple".to_string(),
            found: other.get_type().to_string(),
            span,
            src: vm.source_ref().source().into(),
            filename: vm.source_ref().filename().into(),
        }),
    }
}

fn value_sequence_or_void(vm: &VM<'_>, value: Value, span: Span) -> WalrusResult<Vec<Value>> {
    if matches!(value, Value::Void) {
        return Ok(Vec::new());
    }
    value_sequence(vm, value, span)
}

fn non_negative_millis(vm: &VM<'_>, value: Value, span: Span, name: &str) -> WalrusResult<u64> {
    let ms = vm.value_to_int(value, span)?;
    if ms < 0 {
        return Err(WalrusError::GenericError {
            message: format!("{name} must be >= 0"),
        });
    }
    Ok(ms as u64)
}

fn native_async_spawn(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let call_args = value_sequence_or_void(vm, args[1], span)?;
    vm.spawn_task_from_callable(args[0], call_args, span)
}

fn native_async_sleep(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let delay_ms = non_negative_millis(vm, args[0], span, "sleep milliseconds")?;
    Ok(vm.create_sleep_task(delay_ms))
}

fn native_async_timeout(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let task_key = match args[0] {
        Value::Task(task_key) => task_key,
        other => {
            return Err(WalrusError::TypeMismatch {
                expected: "task".to_string(),
                found: other.get_type().to_string(),
                span,
                src: vm.source_ref().source().into(),
                filename: vm.source_ref().filename().into(),
            });
        }
    };
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

fn native_file_open(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let mode = vm.value_to_string(args[1], span)?;
    crate::stdlib::file_open(&path, &mode, span)
}

fn native_file_read(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    let content = crate::stdlib::file_read(handle, span)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&content)))
}

fn native_file_read_line(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    match crate::stdlib::file_read_line(handle, span)? {
        Some(line) => Ok(vm.get_heap_mut().push(HeapValue::String(&line))),
        None => Ok(Value::Void),
    }
}

fn native_file_write(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    let bytes = crate::stdlib::file_write(handle, &content, span)?;
    Ok(Value::Int(bytes))
}

fn native_file_close(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let handle = vm.value_to_int(args[0], span)?;
    crate::stdlib::file_close(handle, span)?;
    Ok(Value::Void)
}

fn native_file_exists(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    Ok(Value::Bool(crate::stdlib::file_exists(&path)))
}

fn native_read_file(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let content = crate::stdlib::read_file(&path, span)?;
    Ok(vm.get_heap_mut().push(HeapValue::String(&content)))
}

fn native_write_file(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    crate::stdlib::write_file(&path, &content, span)?;
    Ok(Value::Void)
}

fn native_env_get(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let name = vm.value_to_string(args[0], span)?;
    match crate::stdlib::env_get(&name) {
        Some(value) => Ok(vm.get_heap_mut().push(HeapValue::String(&value))),
        None => Ok(Value::Void),
    }
}

fn native_args(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    let cli_args = crate::stdlib::args();
    let mut list = Vec::with_capacity(cli_args.len());
    for arg in cli_args {
        let s = vm.get_heap_mut().push(HeapValue::String(&arg));
        list.push(s);
    }
    Ok(vm.get_heap_mut().push(HeapValue::List(list)))
}

fn native_cwd(vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    match crate::stdlib::cwd() {
        Some(path) => Ok(vm.get_heap_mut().push(HeapValue::String(&path))),
        None => Ok(Value::Void),
    }
}

fn native_exit(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let code = vm.value_to_int(args[0], span)?;
    std::process::exit(code as i32);
}

fn native_net_tcp_bind(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let host = vm.value_to_string(args[0], span)?;
    let port = vm.value_to_int(args[1], span)?;
    crate::stdlib::tcp_bind(&host, port, span)
}

fn native_net_tcp_accept(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let listener = vm.value_to_int(args[0], span)?;
    crate::stdlib::tcp_accept(listener, span)
}

fn native_net_tcp_connect(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let host = vm.value_to_string(args[0], span)?;
    let port = vm.value_to_int(args[1], span)?;
    crate::stdlib::tcp_connect(&host, port, span)
}

fn native_net_tcp_local_port(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let listener = vm.value_to_int(args[0], span)?;
    Ok(Value::Int(crate::stdlib::tcp_local_port(listener, span)?))
}

fn native_net_tcp_read(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let max_bytes = vm.value_to_int(args[1], span)?;
    match crate::stdlib::tcp_read(stream, max_bytes, span)? {
        Some(text) => Ok(vm.get_heap_mut().push(HeapValue::String(&text))),
        None => Ok(Value::Void),
    }
}

fn native_net_tcp_read_line(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    match crate::stdlib::tcp_read_line(stream, span)? {
        Some(line) => Ok(vm.get_heap_mut().push(HeapValue::String(&line))),
        None => Ok(Value::Void),
    }
}

fn native_net_tcp_write(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let content = vm.value_to_string(args[1], span)?;
    Ok(Value::Int(crate::stdlib::tcp_write(
        stream, &content, span,
    )?))
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

fn insert_key_value(
    vm: &mut VM<'_>,
    dict: &mut rustc_hash::FxHashMap<Value, Value>,
    key: &str,
    value: Value,
) {
    let key_value = vm.get_heap_mut().push(HeapValue::String(key));
    dict.insert(key_value, value);
}

fn heap_string(vm: &mut VM<'_>, text: &str) -> Value {
    vm.get_heap_mut().push(HeapValue::String(text))
}

fn type_mismatch_error(vm: &VM<'_>, expected: &str, found: Value, span: Span) -> WalrusError {
    WalrusError::TypeMismatch {
        expected: expected.to_string(),
        found: found.get_type().to_string(),
        span,
        src: vm.source_ref().source().to_string(),
        filename: vm.source_ref().filename().to_string(),
    }
}

fn http_error_dict(vm: &mut VM<'_>, message: &str) -> Value {
    let mut dict = rustc_hash::FxHashMap::default();
    insert_key_value(vm, &mut dict, "ok", Value::Bool(false));
    let message_value = heap_string(vm, message);
    insert_key_value(vm, &mut dict, "error", message_value);
    vm.get_heap_mut().push(HeapValue::Dict(dict))
}

fn native_http_parse_request_line(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let line = vm.value_to_string(args[0], span)?;

    match crate::stdlib::http_parse_request_line(&line) {
        Ok(parsed) => {
            let mut dict = rustc_hash::FxHashMap::default();
            insert_key_value(vm, &mut dict, "ok", Value::Bool(true));
            let method_value = heap_string(vm, &parsed.method);
            insert_key_value(vm, &mut dict, "method", method_value);
            let target_value = heap_string(vm, &parsed.target);
            insert_key_value(vm, &mut dict, "target", target_value);
            let path_value = heap_string(vm, &parsed.path);
            insert_key_value(vm, &mut dict, "path", path_value);
            let query_value = heap_string(vm, &parsed.query);
            insert_key_value(vm, &mut dict, "query", query_value);
            let version_value = heap_string(vm, &parsed.version);
            insert_key_value(vm, &mut dict, "version", version_value);
            Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
        }
        Err(message) => Ok(http_error_dict(vm, &message)),
    }
}

fn native_http_parse_query(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let query = vm.value_to_string(args[0], span)?;
    let pairs = crate::stdlib::http_parse_query(&query);

    let mut dict = rustc_hash::FxHashMap::default();
    for (key, value) in pairs {
        let key_value = heap_string(vm, &key);
        let val_value = heap_string(vm, &value);
        dict.insert(key_value, val_value);
    }

    Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
}

fn native_http_normalize_path(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let path = vm.value_to_string(args[0], span)?;
    let normalized = crate::stdlib::http_normalize_path(&path);
    Ok(heap_string(vm, &normalized))
}

fn native_http_match_route(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let pattern = vm.value_to_string(args[0], span)?;
    let path = vm.value_to_string(args[1], span)?;
    let matched = crate::stdlib::http_match_route(&pattern, &path);

    let mut params = rustc_hash::FxHashMap::default();
    for (name, value) in matched.params {
        let key = heap_string(vm, &name);
        let val = heap_string(vm, &value);
        params.insert(key, val);
    }
    let params_dict = vm.get_heap_mut().push(HeapValue::Dict(params));

    let wildcard_value = match matched.wildcard {
        Some(wildcard) => heap_string(vm, &wildcard),
        None => Value::Void,
    };

    let mut result = rustc_hash::FxHashMap::default();
    insert_key_value(vm, &mut result, "found", Value::Bool(matched.found));
    let pattern_value = heap_string(vm, &matched.pattern);
    insert_key_value(vm, &mut result, "pattern", pattern_value);
    let path_value = heap_string(vm, &matched.path);
    insert_key_value(vm, &mut result, "path", path_value);
    insert_key_value(vm, &mut result, "params", params_dict);
    insert_key_value(vm, &mut result, "wildcard", wildcard_value);

    Ok(vm.get_heap_mut().push(HeapValue::Dict(result)))
}

fn native_http_status_text(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let status = vm.value_to_int(args[0], span)?;
    let reason = crate::stdlib::http_status_text(status);
    Ok(heap_string(vm, reason))
}

fn native_http_response(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let status = vm.value_to_int(args[0], span)?;
    let body = vm.value_to_string(args[1], span)?;
    let response = crate::stdlib::http_build_response(status, &body, &[], span)?;
    Ok(heap_string(vm, &response))
}

fn native_http_response_with_headers(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let status = vm.value_to_int(args[0], span)?;
    let body = vm.value_to_string(args[1], span)?;

    let headers_key = match args[2] {
        Value::Dict(key) => key,
        other => return Err(type_mismatch_error(vm, "dict", other, span)),
    };

    let header_entries = vm
        .get_heap()
        .get_dict(headers_key)?
        .iter()
        .map(|(&k, &v)| (k, v))
        .collect::<Vec<_>>();

    let mut headers = Vec::with_capacity(header_entries.len());
    for (key_value, value_value) in header_entries {
        let name = vm.value_to_string(key_value, span)?;
        let value = vm.value_to_string(value_value, span)?;
        headers.push((name, value));
    }

    let response = crate::stdlib::http_build_response(status, &body, &headers, span)?;
    Ok(heap_string(vm, &response))
}

fn native_http_read_request(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream = vm.value_to_int(args[0], span)?;
    let max_body_bytes = vm.value_to_int(args[1], span)?;

    match crate::stdlib::http_read_request(stream, max_body_bytes, span)? {
        crate::stdlib::HttpReadOutcome::Eof => Ok(Value::Void),
        crate::stdlib::HttpReadOutcome::BadRequest(message) => Ok(http_error_dict(vm, &message)),
        crate::stdlib::HttpReadOutcome::Request(request) => {
            let mut headers = rustc_hash::FxHashMap::default();
            for (name, value) in request.headers {
                let key = heap_string(vm, &name);
                let val = heap_string(vm, &value);
                headers.insert(key, val);
            }
            let headers_value = vm.get_heap_mut().push(HeapValue::Dict(headers));

            let query_pairs = crate::stdlib::http_parse_query(&request.query);
            let mut query = rustc_hash::FxHashMap::default();
            for (name, value) in query_pairs {
                let key = heap_string(vm, &name);
                let val = heap_string(vm, &value);
                query.insert(key, val);
            }
            let query_value = vm.get_heap_mut().push(HeapValue::Dict(query));

            let mut dict = rustc_hash::FxHashMap::default();
            insert_key_value(vm, &mut dict, "ok", Value::Bool(true));
            let method_value = heap_string(vm, &request.method);
            insert_key_value(vm, &mut dict, "method", method_value);
            let target_value = heap_string(vm, &request.target);
            insert_key_value(vm, &mut dict, "target", target_value);
            let path_value = heap_string(vm, &request.path);
            insert_key_value(vm, &mut dict, "path", path_value);
            let query_text_value = heap_string(vm, &request.query);
            insert_key_value(vm, &mut dict, "query", query_text_value);
            insert_key_value(vm, &mut dict, "query_params", query_value);
            let version_value = heap_string(vm, &request.version);
            insert_key_value(vm, &mut dict, "version", version_value);
            insert_key_value(vm, &mut dict, "headers", headers_value);
            let body_value = heap_string(vm, &request.body);
            insert_key_value(vm, &mut dict, "body", body_value);
            insert_key_value(
                vm,
                &mut dict,
                "content_length",
                Value::Int(request.content_length),
            );

            Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
        }
    }
}

fn native_math_pi(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(std::f64::consts::PI)))
}

fn native_math_e(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(std::f64::consts::E)))
}

fn native_math_tau(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(std::f64::consts::TAU)))
}

fn native_math_inf(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(f64::INFINITY)))
}

fn native_math_nan(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(f64::NAN)))
}

fn native_math_abs(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => {
            let abs = n.checked_abs().ok_or_else(|| WalrusError::GenericError {
                message: "math.abs: overflow for i64::MIN".to_string(),
            })?;
            Ok(Value::Int(abs))
        }
        _ => {
            let n = vm.value_to_number(args[0], span)?;
            Ok(Value::Float(FloatOrd(n.abs())))
        }
    }
}

fn native_math_sign(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n.signum())),
        _ => {
            let n = vm.value_to_number(args[0], span)?;
            if n.is_nan() {
                return Err(WalrusError::GenericError {
                    message: "math.sign: cannot determine sign of NaN".to_string(),
                });
            }
            let sign = if n > 0.0 {
                1
            } else if n < 0.0 {
                -1
            } else {
                0
            };
            Ok(Value::Int(sign))
        }
    }
}

fn native_math_min(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match (args[0], args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.min(b))),
        _ => {
            let a = vm.value_to_number(args[0], span)?;
            let b = vm.value_to_number(args[1], span)?;
            Ok(Value::Float(FloatOrd(a.min(b))))
        }
    }
}

fn native_math_max(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match (args[0], args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.max(b))),
        _ => {
            let a = vm.value_to_number(args[0], span)?;
            let b = vm.value_to_number(args[1], span)?;
            Ok(Value::Float(FloatOrd(a.max(b))))
        }
    }
}

fn native_math_clamp(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match (args[0], args[1], args[2]) {
        (Value::Int(v), Value::Int(min), Value::Int(max)) => {
            if min > max {
                return Err(WalrusError::GenericError {
                    message: format!("math.clamp: min ({min}) cannot be greater than max ({max})"),
                });
            }
            Ok(Value::Int(v.clamp(min, max)))
        }
        _ => {
            let v = vm.value_to_number(args[0], span)?;
            let min = vm.value_to_number(args[1], span)?;
            let max = vm.value_to_number(args[2], span)?;
            if v.is_nan() || min.is_nan() || max.is_nan() {
                return Err(WalrusError::GenericError {
                    message: "math.clamp: NaN is not supported".to_string(),
                });
            }
            if min > max {
                return Err(WalrusError::GenericError {
                    message: format!("math.clamp: min ({min}) cannot be greater than max ({max})"),
                });
            }
            let clamped = if v < min {
                min
            } else if v > max {
                max
            } else {
                v
            };
            Ok(Value::Float(FloatOrd(clamped)))
        }
    }
}

fn native_math_floor(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.floor(),
        ))),
    }
}

fn native_math_ceil(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.ceil(),
        ))),
    }
}

fn native_math_round(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.round(),
        ))),
    }
}

fn native_math_trunc(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    match args[0] {
        Value::Int(n) => Ok(Value::Int(n)),
        _ => Ok(Value::Float(FloatOrd(
            vm.value_to_number(args[0], span)?.trunc(),
        ))),
    }
}

fn native_math_fract(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.fract())))
}

fn native_math_sqrt(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value < 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.sqrt: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.sqrt())))
}

fn native_math_cbrt(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.cbrt())))
}

fn native_math_pow(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let base = vm.value_to_number(args[0], span)?;
    let exponent = vm.value_to_number(args[1], span)?;
    let result = base.powf(exponent);
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.pow: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_hypot(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let x = vm.value_to_number(args[0], span)?;
    let y = vm.value_to_number(args[1], span)?;
    let result = x.hypot(y);
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.hypot: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_sin(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.sin())))
}

fn native_math_cos(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.cos())))
}

fn native_math_tan(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.tan())))
}

fn native_math_asin(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if !(-1.0..=1.0).contains(&value) {
        return Err(WalrusError::GenericError {
            message: format!("math.asin: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.asin())))
}

fn native_math_acos(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if !(-1.0..=1.0).contains(&value) {
        return Err(WalrusError::GenericError {
            message: format!("math.acos: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.acos())))
}

fn native_math_atan(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.atan())))
}

fn native_math_atan2(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let y = vm.value_to_number(args[0], span)?;
    let x = vm.value_to_number(args[1], span)?;
    Ok(Value::Float(FloatOrd(y.atan2(x))))
}

fn native_math_exp(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    let result = value.exp();
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.exp: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_ln(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.ln: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.ln())))
}

fn native_math_log2(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.log2: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.log2())))
}

fn native_math_log10(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.log10: domain error for value {value}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.log10())))
}

fn native_math_log(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    let base = vm.value_to_number(args[1], span)?;
    if value <= 0.0 {
        return Err(WalrusError::GenericError {
            message: format!("math.log: domain error for value {value}"),
        });
    }
    if base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
        return Err(WalrusError::GenericError {
            message: format!("math.log: invalid base {base}"),
        });
    }
    Ok(Value::Float(FloatOrd(value.log(base))))
}

fn native_math_lerp(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let a = vm.value_to_number(args[0], span)?;
    let b = vm.value_to_number(args[1], span)?;
    let t = vm.value_to_number(args[2], span)?;
    let result = a + (b - a) * t;
    if !result.is_finite() {
        return Err(WalrusError::GenericError {
            message: "math.lerp: result is not finite".to_string(),
        });
    }
    Ok(Value::Float(FloatOrd(result)))
}

fn native_math_degrees(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.to_degrees())))
}

fn native_math_radians(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Float(FloatOrd(value.to_radians())))
}

fn native_math_is_finite(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Bool(value.is_finite()))
}

fn native_math_is_nan(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Bool(value.is_nan()))
}

fn native_math_is_inf(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let value = vm.value_to_number(args[0], span)?;
    Ok(Value::Bool(value.is_infinite()))
}

fn native_math_seed(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let seed = vm.value_to_int(args[0], span)?;
    crate::stdlib::math_seed(seed);
    Ok(Value::Void)
}

fn native_math_rand_float(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Float(FloatOrd(crate::stdlib::math_rand_float())))
}

fn native_math_rand_bool(_vm: &mut VM<'_>, _args: &[Value], _span: Span) -> WalrusResult<Value> {
    Ok(Value::Bool(crate::stdlib::math_rand_bool()))
}

fn native_math_rand_int(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let min = vm.value_to_int(args[0], span)?;
    let max = vm.value_to_int(args[1], span)?;
    Ok(Value::Int(crate::stdlib::math_rand_int(min, max, span)?))
}

fn native_math_rand_range(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let min = vm.value_to_number(args[0], span)?;
    let max = vm.value_to_number(args[1], span)?;
    Ok(Value::Float(FloatOrd(crate::stdlib::math_rand_range(
        min, max, span,
    )?)))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn native_specs_are_unique_and_well_formed() {
        let mut ids = HashSet::new();
        let mut module_names = HashSet::new();

        for spec in NATIVE_SPECS {
            assert!(
                ids.insert(spec.id),
                "duplicate native function id detected: {:?}",
                spec.id
            );
            assert!(
                module_names.insert((spec.module, spec.name)),
                "duplicate native function name detected: {}/{}",
                spec.module,
                spec.name
            );
            assert_eq!(
                spec.arity,
                spec.params.len(),
                "arity mismatch for {}/{}",
                spec.module,
                spec.name
            );
            assert!(
                !spec.docs.trim().is_empty(),
                "missing docs for {}/{}",
                spec.module,
                spec.name
            );
            assert!(
                spec.module.starts_with("std/"),
                "unexpected module prefix for {}/{}",
                spec.module,
                spec.name
            );
        }
    }

    #[test]
    fn module_lookup_matches_registered_specs() {
        let mut expected_by_module: HashMap<&str, Vec<NativeFunction>> = HashMap::new();
        for spec in NATIVE_SPECS {
            expected_by_module
                .entry(spec.module)
                .or_default()
                .push(spec.id);
        }

        for (module, expected_fns) in expected_by_module {
            let actual = module_functions(module)
                .unwrap_or_else(|| panic!("module '{module}' should have registered functions"));
            assert_eq!(
                actual.len(),
                expected_fns.len(),
                "function count mismatch for module '{module}'"
            );

            for expected in expected_fns {
                assert!(
                    actual.contains(&expected),
                    "module '{module}' missing function {:?}",
                    expected
                );
            }
        }

        assert!(
            module_functions("std/unknown").is_none(),
            "unknown module unexpectedly returned functions"
        );
    }
}
