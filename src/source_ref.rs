#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)] // todo: remove PartialEq, Eq, Hash, PartialOrd, Ord
pub struct SourceRef<'a> {
    pub source: &'a str,
    pub filename: &'a str,
}

impl<'a> SourceRef<'a> {
    pub fn new(filename: &'a str, source: &'a str) -> Self {
        Self { source, filename }
    }
}
