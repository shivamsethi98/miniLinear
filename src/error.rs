//! Compiler error type and pretty-printer.
//!
//! Output format (matches the handoff spec):
//!
//! ```text
//! error: <category>
//!   --> <file>:<line>:<col>
//!    |
//!  N | <source line>
//!    |     ^^^
//!    |
//!    = note: <human-readable explanation>
//! ```
//!
//! The inline pointer text after the carets is left blank; the explanation
//! goes in the trailing `= note:` line. Line/column are computed from byte
//! offsets at display time (option (1) of the handoff's clarifying
//! questions). Columns are byte-based — fine for ASCII source, which is all
//! Phase 0 produces; revisit if non-ASCII identifiers are ever allowed.

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

// -----------------------------------------------------------------------------
// Pretty-printing
// -----------------------------------------------------------------------------

/// Format `err` for display against the original `source` text. `file` is
/// the path string shown in the `-->` location line; the caller is
/// responsible for using the same string the user passed on the CLI.
pub fn format_error(err: &Error, source: &str, file: &str) -> String {
    let (line_num, col, line_text, line_start_offset) = locate(source, err.span.start);

    // Cap caret length at the end of the start line. Multi-line spans get
    // pointed-at on their first line only, which is fine for Phase 0 errors
    // (none of them span line boundaries today).
    let line_end_offset = line_start_offset + line_text.len();
    let span_end_in_line = err.span.end.min(line_end_offset);
    let caret_len = span_end_in_line.saturating_sub(err.span.start).max(1);

    let line_num_str = line_num.to_string();
    // Gutter convention (matching the spec and rust's compiler output):
    // the line number is right-padded with one space on each side. The
    // empty gutter on context-only lines is the same width.
    let gutter_width = line_num_str.len() + 2;
    let empty_gutter = " ".repeat(gutter_width);
    let line_gutter = format!(" {line_num_str} ");
    // `-->` is positioned so that its `>` lands one column past where `|`
    // would appear, matching the spec layout.
    let arrow_indent = " ".repeat(gutter_width - 1);
    let pre_caret = " ".repeat(col.saturating_sub(1));
    let carets = "^".repeat(caret_len);

    let mut out = String::new();
    out.push_str(&format!("error: {}\n", err.category.as_str()));
    out.push_str(&format!("{arrow_indent}--> {file}:{line_num}:{col}\n"));
    out.push_str(&format!("{empty_gutter}|\n"));
    out.push_str(&format!("{line_gutter}| {line_text}\n"));
    out.push_str(&format!("{empty_gutter}| {pre_caret}{carets}\n"));
    if let Some(note) = &err.note {
        out.push_str(&format!("{empty_gutter}|\n"));
        out.push_str(&format!("{empty_gutter}= note: {note}\n"));
    }
    out
}

/// Map a byte offset into `source` to a (line_number, column, line_text,
/// line_start_offset) tuple. Lines and columns are 1-indexed; column is a
/// byte offset within the line. If the offset is past EOF it is clamped.
fn locate(source: &str, byte_offset: usize) -> (usize, usize, &str, usize) {
    let bytes = source.as_bytes();
    let offset = byte_offset.min(bytes.len());

    let mut line_start = 0;
    let mut line_num = 1;
    for (i, &b) in bytes[..offset].iter().enumerate() {
        if b == b'\n' {
            line_num += 1;
            line_start = i + 1;
        }
    }

    let line_end = bytes[offset..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| offset + p)
        .unwrap_or(bytes.len());

    let line_text = &source[line_start..line_end];
    let col = offset - line_start + 1;
    (line_num, col, line_text, line_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(span_start: usize, span_end: usize, note: &str) -> Error {
        Error::new(
            ErrorCategory::TypeError,
            Span::new(span_start, span_end),
            Some(note.to_string()),
        )
    }

    #[test]
    fn simple_first_line() {
        let src = "x";
        let formatted = format_error(&err(0, 1, "unbound variable `x`"), src, "input.lin");
        let expected = "\
error: type error
  --> input.lin:1:1
   |
 1 | x
   | ^
   |
   = note: unbound variable `x`
";
        assert_eq!(formatted, expected);
    }

    #[test]
    fn second_line_with_offset() {
        let src = "let ordinary x = 1 in\n  x + missing";
        // Span over `missing` (offset 30..37 in the second line)
        let mis_start = src.find("missing").unwrap();
        let mis_end = mis_start + "missing".len();
        let formatted = format_error(
            &err(mis_start, mis_end, "unbound variable `missing`"),
            src,
            "snippet.lin",
        );
        // Column is 7 (after "  x + ").
        assert!(formatted.contains("--> snippet.lin:2:7"));
        assert!(formatted.contains("2 |   x + missing"));
        assert!(formatted.contains("|       ^^^^^^^"));
        assert!(formatted.contains("= note: unbound variable `missing`"));
    }

    #[test]
    fn eof_span_does_not_panic() {
        let src = "1 +";
        let formatted = format_error(
            &err(src.len(), src.len(), "expected expression"),
            src,
            "x.lin",
        );
        // Should still produce something coherent.
        assert!(formatted.contains("error: type error"));
        assert!(formatted.contains("= note: expected expression"));
    }

    #[test]
    fn no_note_produces_no_note_line() {
        let src = "x";
        let mut e = err(0, 1, "ignored");
        e.note = None;
        let formatted = format_error(&e, src, "x.lin");
        assert!(!formatted.contains("= note:"));
    }
}
