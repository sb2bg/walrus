use crate::span::Span;
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RangeValue {
    start: i64,
    start_span: Span,
    end: i64,
    end_span: Span,
}

impl Hash for RangeValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.start.hash(state);
        self.end.hash(state);
    }
}

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
}

impl std::fmt::Display for RangeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}
