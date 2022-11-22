use get_size::GetSize;
use std::ops::Range;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Span(pub usize, pub usize);

impl GetSize for Span {
    fn get_size(&self) -> usize {
        std::mem::size_of::<Self>()
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