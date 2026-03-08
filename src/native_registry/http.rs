use rustc_hash::FxHashMap;

use crate::WalrusResult;
use crate::arenas::{DictKey, HeapValue};
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

use super::helpers::{heap_string, http_error_dict, insert_key_value, type_mismatch_error};
use super::{NativeSpec, define_native_spec};

pub static SPECS: &[NativeSpec] = &[
    define_native_spec(
        NativeFunction::HttpParseRequestLine,
        "std/http",
        "parse_request_line",
        &["line"],
        "Parse an HTTP request line into method/path/query/version fields.",
        native_http_parse_request_line,
    ),
    define_native_spec(
        NativeFunction::HttpParseQuery,
        "std/http",
        "parse_query",
        &["query"],
        "Parse a query string into a dict of decoded key/value pairs.",
        native_http_parse_query,
    ),
    define_native_spec(
        NativeFunction::HttpParseQueryPairs,
        "std/http",
        "parse_query_pairs",
        &["query"],
        "Parse a query string into a decoded list of [key, value] pairs.",
        native_http_parse_query_pairs,
    ),
    define_native_spec(
        NativeFunction::HttpNormalizePath,
        "std/http",
        "normalize_path",
        &["path"],
        "Normalize an HTTP path (collapse duplicate slashes, trim trailing slash).",
        native_http_normalize_path,
    ),
    define_native_spec(
        NativeFunction::HttpMatchRoute,
        "std/http",
        "match_route",
        &["pattern", "path"],
        "Match a route pattern against a path with :params and trailing * wildcard.",
        native_http_match_route,
    ),
    define_native_spec(
        NativeFunction::HttpStatusText,
        "std/http",
        "status_text",
        &["status"],
        "Return the canonical reason phrase for an HTTP status code.",
        native_http_status_text,
    ),
    define_native_spec(
        NativeFunction::HttpResponse,
        "std/http",
        "response",
        &["status", "body"],
        "Build an HTTP/1.1 response string with default headers.",
        native_http_response,
    ),
    define_native_spec(
        NativeFunction::HttpResponseWithHeaders,
        "std/http",
        "response_with_headers",
        &["status", "body", "headers"],
        "Build an HTTP/1.1 response string with caller-provided headers.",
        native_http_response_with_headers,
    ),
    define_native_spec(
        NativeFunction::HttpMakeResponse,
        "std/http",
        "make_response",
        &["status", "body"],
        "Build a response object with status/body/default header fields.",
        native_http_make_response,
    ),
    define_native_spec(
        NativeFunction::HttpMakeResponseWithHeaders,
        "std/http",
        "make_response_with_headers",
        &["status", "body", "headers"],
        "Build a response object with status/body/headers for later serialization.",
        native_http_make_response_with_headers,
    ),
    define_native_spec(
        NativeFunction::HttpSerializeResponse,
        "std/http",
        "serialize_response",
        &["response"],
        "Serialize a response object into an HTTP/1.1 response string.",
        native_http_serialize_response,
    ),
    define_native_spec(
        NativeFunction::HttpReadRequest,
        "std/http",
        "read_request",
        &["stream", "max_body_bytes"],
        "Read and parse one HTTP request from a TCP stream; returns void on EOF.",
        native_http_read_request,
    ),
];

fn pair_list_value(vm: &mut VM<'_>, pairs: &[(String, String)]) -> Value {
    let mut list = Vec::with_capacity(pairs.len());
    for (name, value) in pairs {
        let key = heap_string(vm, name);
        let val = heap_string(vm, value);
        let pair = vm.get_heap_mut().push(HeapValue::List(vec![key, val]));
        list.push(pair);
    }
    vm.get_heap_mut().push(HeapValue::List(list))
}

fn header_dict_value(vm: &mut VM<'_>, pairs: &[(String, String)]) -> Value {
    let mut dict = FxHashMap::default();
    for (name, value) in pairs {
        let key = heap_string(vm, name);
        let val = heap_string(vm, value);
        dict.insert(key, val);
    }
    vm.get_heap_mut().push(HeapValue::Dict(dict))
}

fn build_response_value(
    vm: &mut VM<'_>,
    status: i64,
    body: &str,
    headers: &[(String, String)],
) -> Value {
    let headers_value = header_dict_value(vm, headers);
    let header_pairs_value = pair_list_value(vm, headers);
    let body_value = heap_string(vm, body);

    let mut dict = FxHashMap::default();
    insert_key_value(vm, &mut dict, "status", Value::Int(status));
    insert_key_value(vm, &mut dict, "body", body_value);
    insert_key_value(vm, &mut dict, "headers", headers_value);
    insert_key_value(vm, &mut dict, "header_pairs", header_pairs_value);
    vm.get_heap_mut().push(HeapValue::Dict(dict))
}

fn dict_get_string_key(vm: &VM<'_>, dict_key: DictKey, name: &str) -> WalrusResult<Option<Value>> {
    for (&key, &value) in vm.get_heap().get_dict(dict_key)? {
        if let Value::String(str_key) = key {
            if vm.get_heap().get_string(str_key)? == name {
                return Ok(Some(value));
            }
        }
    }
    Ok(None)
}

fn dict_to_header_pairs(
    vm: &VM<'_>,
    headers_key: DictKey,
    span: Span,
) -> WalrusResult<Vec<(String, String)>> {
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
    Ok(headers)
}

fn pair_sequence_to_strings(
    vm: &VM<'_>,
    value: Value,
    span: Span,
) -> WalrusResult<Vec<(String, String)>> {
    let items = match value {
        Value::List(key) => vm.get_heap().get_list(key)?.to_vec(),
        Value::Tuple(key) => vm.get_heap().get_tuple(key)?.to_vec(),
        other => {
            return Err(WalrusError::TypeMismatch {
                expected: "list or tuple".to_string(),
                found: other.get_type().to_string(),
                span,
                src: vm.source_ref().source().into(),
                filename: vm.source_ref().filename().into(),
            });
        }
    };

    let mut pairs = Vec::with_capacity(items.len());
    for item in items {
        let pair = match item {
            Value::Tuple(key) => vm.get_heap().get_tuple(key)?.to_vec(),
            Value::List(key) => vm.get_heap().get_list(key)?.to_vec(),
            other => {
                return Err(WalrusError::TypeMismatch {
                    expected: "tuple or list".to_string(),
                    found: other.get_type().to_string(),
                    span,
                    src: vm.source_ref().source().into(),
                    filename: vm.source_ref().filename().into(),
                });
            }
        };

        if pair.len() != 2 {
            return Err(WalrusError::GenericError {
                message: "http.serialize_response: header_pairs entries must have exactly 2 items"
                    .to_string(),
            });
        }

        let name = vm.value_to_string(pair[0], span)?;
        let value = vm.value_to_string(pair[1], span)?;
        pairs.push((name, value));
    }

    Ok(pairs)
}

fn response_from_value(
    vm: &VM<'_>,
    value: Value,
    span: Span,
) -> WalrusResult<(i64, String, Vec<(String, String)>)> {
    let dict_key = match value {
        Value::Dict(key) => key,
        other => return Err(type_mismatch_error(vm, "dict", other, span)),
    };

    let status = dict_get_string_key(vm, dict_key, "status")?
        .ok_or_else(|| WalrusError::GenericError {
            message: "http.serialize_response: response is missing 'status'".to_string(),
        })
        .and_then(|value| vm.value_to_int(value, span))?;
    let body = dict_get_string_key(vm, dict_key, "body")?
        .ok_or_else(|| WalrusError::GenericError {
            message: "http.serialize_response: response is missing 'body'".to_string(),
        })
        .and_then(|value| vm.value_to_string(value, span))?;

    let headers = if let Some(header_pairs) = dict_get_string_key(vm, dict_key, "header_pairs")? {
        pair_sequence_to_strings(vm, header_pairs, span)?
    } else if let Some(headers) = dict_get_string_key(vm, dict_key, "headers")? {
        match headers {
            Value::Dict(headers_key) => dict_to_header_pairs(vm, headers_key, span)?,
            other => return Err(type_mismatch_error(vm, "dict", other, span)),
        }
    } else {
        Vec::new()
    };

    Ok((status, body, headers))
}

fn native_http_parse_request_line(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let line = vm.value_to_string(args[0], span)?;

    match crate::stdlib::http_parse_request_line(&line) {
        Ok(parsed) => {
            let mut dict = FxHashMap::default();
            let method_value = heap_string(vm, &parsed.method);
            let target_value = heap_string(vm, &parsed.target);
            let path_value = heap_string(vm, &parsed.path);
            let query_value = heap_string(vm, &parsed.query);
            let version_value = heap_string(vm, &parsed.version);
            insert_key_value(vm, &mut dict, "ok", Value::Bool(true));
            insert_key_value(vm, &mut dict, "method", method_value);
            insert_key_value(vm, &mut dict, "target", target_value);
            insert_key_value(vm, &mut dict, "path", path_value);
            insert_key_value(vm, &mut dict, "query", query_value);
            insert_key_value(vm, &mut dict, "version", version_value);
            Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
        }
        Err(message) => Ok(http_error_dict(vm, &message)),
    }
}

fn native_http_parse_query(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let query = vm.value_to_string(args[0], span)?;
    let pairs = crate::stdlib::http_parse_query(&query);

    let mut dict = FxHashMap::default();
    for (key, value) in pairs {
        let key_value = heap_string(vm, &key);
        let val_value = heap_string(vm, &value);
        dict.insert(key_value, val_value);
    }

    Ok(vm.get_heap_mut().push(HeapValue::Dict(dict)))
}

fn native_http_parse_query_pairs(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let query = vm.value_to_string(args[0], span)?;
    let pairs = crate::stdlib::http_parse_query(&query);
    Ok(pair_list_value(vm, &pairs))
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

    let mut params = FxHashMap::default();
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

    let mut result = FxHashMap::default();
    let pattern_value = heap_string(vm, &matched.pattern);
    let path_value = heap_string(vm, &matched.path);
    insert_key_value(vm, &mut result, "found", Value::Bool(matched.found));
    insert_key_value(vm, &mut result, "pattern", pattern_value);
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

    let headers = dict_to_header_pairs(vm, headers_key, span)?;
    let response = crate::stdlib::http_build_response(status, &body, &headers, span)?;
    Ok(heap_string(vm, &response))
}

fn native_http_make_response(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let status = vm.value_to_int(args[0], span)?;
    let body = vm.value_to_string(args[1], span)?;
    Ok(build_response_value(vm, status, &body, &[]))
}

fn native_http_make_response_with_headers(
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
    let headers = dict_to_header_pairs(vm, headers_key, span)?;
    Ok(build_response_value(vm, status, &body, &headers))
}

fn native_http_serialize_response(
    vm: &mut VM<'_>,
    args: &[Value],
    span: Span,
) -> WalrusResult<Value> {
    let (status, body, headers) = response_from_value(vm, args[0], span)?;
    let response = crate::stdlib::http_build_response(status, &body, &headers, span)?;
    Ok(heap_string(vm, &response))
}

fn native_http_read_request(vm: &mut VM<'_>, args: &[Value], span: Span) -> WalrusResult<Value> {
    let stream_handle = vm.value_to_int(args[0], span)?;
    let max_body_bytes = vm.value_to_int(args[1], span)?;
    if max_body_bytes < 0 {
        return Err(WalrusError::GenericError {
            message: format!(
                "http.read_request: max_body_bytes must be >= 0, got {max_body_bytes}"
            ),
        });
    }
    let max_body_bytes = max_body_bytes as usize;

    let stream = crate::stdlib::shared_tcp_stream(stream_handle).ok_or_else(|| {
        WalrusError::GenericError {
            message: format!("http.read_request: invalid stream handle {stream_handle}"),
        }
    })?;

    Ok(vm.spawn_io(move || {
        let outcome = crate::stdlib::http_read_request_from_shared_stream(&stream, max_body_bytes)?;
        Ok(crate::value::IoResult::HttpOutcome(
            crate::stdlib::http_outcome_to_io(outcome),
        ))
    }))
}
