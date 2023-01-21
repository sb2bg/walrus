use std::fmt::Display;
use std::ops::Range;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub struct Span(pub usize, pub usize);

impl Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}..{}", self.0, self.1)
    }
}

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
