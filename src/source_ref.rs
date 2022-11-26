#[derive(Debug)]
pub struct SourceRef<'a> {
    source: &'a str,
    filename: &'a str,
}

impl<'a> SourceRef<'a> {
    pub fn new(filename: &'a str, source: &'a str) -> Self {
        Self { source, filename }
    }

    pub fn source(&self) -> &'a str {
        self.source
    }

    pub fn filename(&self) -> &'a str {
        self.filename
    }
}
