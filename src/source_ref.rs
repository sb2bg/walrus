use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::error::ErrorContext;
use crate::span::{FileId, Span};

#[derive(Debug, Clone)]
pub struct SourceMap {
    files: Rc<RefCell<Vec<SourceFile>>>,
}

#[derive(Debug, Clone)]
struct SourceFile {
    source: Rc<str>,
    filename: Rc<str>,
}

#[derive(Debug, Clone)]
pub struct SourceFileRef {
    pub source: Rc<str>,
    pub filename: Rc<str>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            files: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn add_source(
        &self,
        source: impl Into<Rc<str>>,
        filename: impl Into<Rc<str>>,
    ) -> (FileId, SourceFileRef) {
        let source = source.into();
        let filename = filename.into();
        let mut files = self.files.borrow_mut();
        let file_id = FileId(files.len());
        files.push(SourceFile {
            source: Rc::clone(&source),
            filename: Rc::clone(&filename),
        });
        (file_id, SourceFileRef { source, filename })
    }

    pub fn get(&self, file_id: FileId) -> Option<SourceFileRef> {
        if file_id.is_unknown() {
            return None;
        }

        self.files
            .borrow()
            .get(file_id.0)
            .map(|file| SourceFileRef {
                source: Rc::clone(&file.source),
                filename: Rc::clone(&file.filename),
            })
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SourceRef<'a> {
    source_map: SourceMap,
    file_id: FileId,
    source: Rc<str>,
    filename: Rc<str>,
    _lifetime: PhantomData<&'a str>,
}

impl<'a> SourceRef<'a> {
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self::new_in(SourceMap::new(), source, filename)
    }

    pub fn new_in(
        source_map: SourceMap,
        source: impl Into<Rc<str>>,
        filename: impl Into<Rc<str>>,
    ) -> Self {
        let (file_id, file) = source_map.add_source(source, filename);
        Self {
            source_map,
            file_id,
            source: file.source,
            filename: file.filename,
            _lifetime: PhantomData,
        }
    }

    pub fn source_map(&self) -> SourceMap {
        self.source_map.clone()
    }

    pub fn file_id(&self) -> FileId {
        self.file_id
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
        let span = if span.file_id().is_unknown() {
            span.with_file_id(self.file_id)
        } else {
            span
        };
        ErrorContext::from_source_map(span, self.source_map())
    }
}

pub struct OwnedSourceRef {
    pub source_map: SourceMap,
    pub file_id: FileId,
    pub src: Rc<str>,
    pub filename: Rc<str>,
}

impl OwnedSourceRef {
    pub fn new(src: String, filename: String) -> Self {
        Self::new_in(SourceMap::new(), src, filename)
    }

    pub fn new_in(source_map: SourceMap, src: String, filename: String) -> Self {
        let (file_id, file) = source_map.add_source(src, filename);
        Self {
            source_map,
            file_id,
            src: file.source,
            filename: file.filename,
        }
    }
}

impl<'a> From<&'a OwnedSourceRef> for SourceRef<'a> {
    fn from(src: &'a OwnedSourceRef) -> Self {
        Self {
            source_map: src.source_map.clone(),
            file_id: src.file_id,
            source: Rc::clone(&src.src),
            filename: Rc::clone(&src.filename),
            _lifetime: PhantomData,
        }
    }
}
