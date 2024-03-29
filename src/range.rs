use std::hash::{Hash, Hasher};

use crate::span::Span;

#[derive(Copy, Clone, Debug)]
pub struct RangeValue {
    pub start: i64,
    start_span: Span,
    pub end: i64,
    end_span: Span,
}

impl Hash for RangeValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
    }
}

impl PartialEq for RangeValue {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}

impl Eq for RangeValue {}

impl RangeValue {
    pub fn new(start: i64, start_span: Span, end: i64, end_span: Span) -> Self {
        Self {
            start,
            start_span,
            end,
            end_span,
        }
    }

    pub fn start(&self) -> i64 {
        self.start
    }

    pub fn start_span(&self) -> Span {
        self.start_span
    }

    pub fn end(&self) -> i64 {
        self.end
    }

    pub fn end_span(&self) -> Span {
        self.end_span
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl std::fmt::Display for RangeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
