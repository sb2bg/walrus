use std::marker::PhantomData;
use std::rc::Rc;

use crate::error::ErrorContext;
use crate::span::Span;

#[derive(Debug, Clone)]
pub struct SourceRef<'a> {
    source: Rc<str>,
    filename: Rc<str>,
    _lifetime: PhantomData<&'a str>,
}

impl<'a> SourceRef<'a> {
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self {
            source: Rc::from(source),
            filename: Rc::from(filename),
            _lifetime: PhantomData,
        }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn source_handle(&self) -> Rc<str> {
        Rc::clone(&self.source)
    }

    pub fn filename_handle(&self) -> Rc<str> {
        Rc::clone(&self.filename)
    }

    pub fn error_context(&self, span: Span) -> ErrorContext {
        ErrorContext::from_shared(span, self.source_handle(), self.filename_handle())
    }
}

pub struct OwnedSourceRef {
    pub src: Rc<str>,
    pub filename: Rc<str>,
}

impl OwnedSourceRef {
    pub fn new(src: String, filename: String) -> Self {
        Self {
            src: Rc::from(src),
            filename: Rc::from(filename),
        }
    }
}

impl<'a> From<&'a OwnedSourceRef> for SourceRef<'a> {
    fn from(src: &'a OwnedSourceRef) -> Self {
        Self {
            source: Rc::clone(&src.src),
            filename: Rc::clone(&src.filename),
            _lifetime: PhantomData,
        }
    }
}
