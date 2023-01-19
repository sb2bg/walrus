#[derive(Debug, Copy, Clone)]
pub struct SourceRef<'a> {
    source: &'a str,
    filename: &'a str,
}

impl<'a> SourceRef<'a> {
    pub fn new(source: &'a str, filename: &'a str) -> Self {
        Self { source, filename }
    }

    pub fn source(&self) -> &'a str {
        self.source
    }

    pub fn filename(&self) -> &'a str {
        self.filename
    }
}

pub struct OwnedSourceRef {
    pub src: String,
    pub filename: String,
}

impl OwnedSourceRef {
    pub fn new(src: String, filename: String) -> Self {
        Self { src, filename }
    }
}

impl<'a> From<&'a OwnedSourceRef> for SourceRef<'a> {
    fn from(src: &'a OwnedSourceRef) -> Self {
        Self {
            source: &src.src,
            filename: &src.filename,
        }
    }
}
