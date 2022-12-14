use std::ops::Range;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Hash)] // todo: remove PartialEq, Eq, Hash, PartialOrd, Ord
pub struct Span(pub usize, pub usize);

impl Span {
    pub fn extend(self, other: Span) -> Span {
        Span(self.0.min(other.0), self.1.max(other.1))
    }
}

impl Default for Span {
    fn default() -> Self {
        Span(0, 0)
    }
}

impl From<Span> for Range<usize> {
    fn from(Span(l, r): Span) -> Self {
        l..r
    }
}

impl From<Range<usize>> for Span {
    fn from(r: Range<usize>) -> Self {
        Span(r.start, r.end)
    }
}

pub struct Spanned<T> {
    pub value: T,
    pub span: Span,
}
