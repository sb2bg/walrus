use std::fmt::Write as FmtWrite;
use std::io::Read;
use std::sync::{Arc, Mutex};

use hyper::http::header::{CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HeaderName, HeaderValue};
use hyper::http::{HeaderMap, Method, Response, StatusCode, Uri, Version};

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::span::Span;
use crate::value::{IoHttpOutcome, IoHttpRequest};

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

    url::form_urlencoded::parse(query.as_bytes())
        .into_owned()
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
        header_map.insert(CONNECTION, HeaderValue::from_static("keep-alive"));
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

pub fn http_read_request_from_shared_stream(
    stream: &Arc<Mutex<super::SharedTcpStream>>,
    max_body_bytes: usize,
) -> Result<HttpReadOutcome, String> {
    let mut stream = stream
        .lock()
        .map_err(|_| "http.read_request: stream lock poisoned".to_string())?;
    let stream = &mut *stream;
    let (reader, read_buffer) = (&mut stream.stream, &mut stream.read_buffer);
    http_read_request_from_reader(reader, read_buffer, max_body_bytes)
}

fn http_read_request_from_reader<R: Read>(
    reader: &mut R,
    read_buffer: &mut Vec<u8>,
    max_body_bytes: usize,
) -> Result<HttpReadOutcome, String> {
    let mut buf = std::mem::take(read_buffer);
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

                let read = reader
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

    let remaining = if parsed_head.bytes_consumed < buf.len() {
        &buf[parsed_head.bytes_consumed..]
    } else {
        &[]
    };
    let split_at = remaining.len().min(parsed_head.content_length);
    let mut body_bytes = remaining[..split_at].to_vec();
    if remaining.len() > parsed_head.content_length {
        read_buffer.extend_from_slice(&remaining[parsed_head.content_length..]);
    }

    while body_bytes.len() < parsed_head.content_length {
        let remaining = parsed_head.content_length - body_bytes.len();
        let to_read = remaining.min(HTTP_READ_CHUNK_BYTES);

        let read = reader
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
