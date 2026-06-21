use std::marker::PhantomData;
use std::sync::Arc;

use crate::error::ErrorContext;
use crate::span::Span;

#[derive(Debug, Clone)]
pub struct SourceRef<'a> {
    source: Arc<str>,
    filename: Arc<str>,
    _lifetime: PhantomData<&'a str>,
}

impl<'a> SourceRef<'a> {
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self {
            source: Arc::from(source),
            filename: Arc::from(filename),
            _lifetime: PhantomData,
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn source_handle(&self) -> Arc<str> {
        Arc::clone(&self.source)
    }

    pub fn filename_handle(&self) -> Arc<str> {
        Arc::clone(&self.filename)
    }

    pub fn error_context(&self, span: Span) -> ErrorContext {
        ErrorContext::from_shared(span, self.source_handle(), self.filename_handle())
    }
}

pub struct OwnedSourceRef {
    pub src: Arc<str>,
    pub filename: Arc<str>,
}

impl OwnedSourceRef {
    pub fn new(src: String, filename: String) -> Self {
        Self {
            src: Arc::from(src),
            filename: Arc::from(filename),
        }
    }
}

impl<'a> From<&'a OwnedSourceRef> for SourceRef<'a> {
    fn from(src: &'a OwnedSourceRef) -> Self {
        Self {
            source: Arc::clone(&src.src),
            filename: Arc::clone(&src.filename),
            _lifetime: PhantomData,
        }
    }
}
