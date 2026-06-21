use std::fmt::Display;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
pub struct FileId(pub usize);

impl FileId {
    pub const UNKNOWN: Self = Self(usize::MAX);

    pub fn is_unknown(self) -> bool {
        self == Self::UNKNOWN
    }
}

impl Default for FileId {
    fn default() -> Self {
        Self::UNKNOWN
    }
}

#[derive(Default, Debug, PartialEq, Eq, Copy, Clone)]
pub struct Span(pub usize, pub usize, pub FileId);

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.0, self.1)
    }
}

impl Span {
    pub fn new(file_id: FileId, start: usize, end: usize) -> Span {
        Span(start, end, file_id)
    }

    pub fn unknown(start: usize, end: usize) -> Span {
        Span(start, end, FileId::UNKNOWN)
    }

    pub fn file_id(self) -> FileId {
        self.2
    }

    pub fn with_file_id(self, file_id: FileId) -> Span {
        Span(self.0, self.1, file_id)
    }

    pub fn in_same_file(self, start: usize, end: usize) -> Span {
        Span(start, end, self.2)
    }

    pub fn extend(self, other: Span) -> Span {
        let file_id = if self.2 == other.2 {
            self.2
        } else {
            FileId::UNKNOWN
        };
        Span(self.0.min(other.0), self.1.max(other.1), file_id)
    }
}

impl From<Span> for Range<usize> {
    fn from(Span(l, r, _): Span) -> Self {
        l..r
    }
}

impl From<Range<usize>> for Span {
    fn from(r: Range<usize>) -> Self {
        Span::unknown(r.start, r.end)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    value: T,
    span: Span,
}

impl<T> Spanned<T> {
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn into_value(self) -> T {
        self.value
    }

    pub fn span(&self) -> Span {
        self.span
    }
}
