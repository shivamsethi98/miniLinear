//! Compiler error type. The pretty-printing implementation is intentionally
//! stubbed for now (Phase 0, deferred step) — only the data shape is here so
//! that the parser and type checker can produce errors today.

use crate::ast::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    ParseError,
    TypeError,
    // Later phases add: LinearityViolation, BorrowViolation,
    // BranchDisagreement.
}

impl ErrorCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCategory::ParseError => "parse error",
            ErrorCategory::TypeError => "type error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub category: ErrorCategory,
    pub span: Span,
    pub note: Option<String>,
}

impl Error {
    pub fn new(category: ErrorCategory, span: Span, note: Option<String>) -> Self {
        Self {
            category,
            span,
            note,
        }
    }
}
