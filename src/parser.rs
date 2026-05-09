//! Hand-rolled recursive-descent parser. Phase 0 only.
//!
//! Grammar (Phase 0):
//!
//! ```text
//! program  ::= expr
//! expr     ::= 'let' 'ordinary' ident '=' expr 'in' expr
//!            | add_expr
//! add_expr ::= atom ('+' atom)*
//! atom     ::= int_literal
//!            | ident
//!            | '(' expr ')'
//! ```
//!
//! Future operators (sequencing `;`, comparisons, subtraction, etc.) will
//! need an explicit precedence layering between `expr` and `atom`; the
//! current two-level grammar is intentionally minimal for Phase 0.
//!
//! Comments (`// ...`) and ASCII whitespace are skipped between tokens.
//! Reserved keywords (`linear`, `shared`, `method`, `returns`, `enum`,
//! `match`, `inout`, `int`) are recognised by the lexer and rejected by the
//! parser, so e.g. `let linear x = ...` is a parse error rather than parsing
//! `linear` as a variable.

use crate::ast::{Expr, Program, Span, Usage};
use crate::error::{Error, ErrorCategory};

/// Parse a complete program from `source`. The Phase 0 program is a single
/// expression; future phases will accept a list of declarations followed by
/// a top-level expression.
pub fn parse(source: &str) -> Result<Program, Error> {
    let mut p = Parser::new(source);
    p.skip_trivia();
    let start = p.pos;
    let main_expr = p.parse_expr()?;
    p.skip_trivia();
    if !p.at_eof() {
        return Err(p.error_here("unexpected trailing input"));
    }
    let span = Span::new(start, p.pos);
    Ok(Program {
        decls: Vec::new(),
        main_expr,
        span,
    })
}

// -----------------------------------------------------------------------------
// Parser state
// -----------------------------------------------------------------------------

struct Parser<'src> {
    src: &'src [u8],
    pos: usize,
}

/// Reserved keywords across all phases. Listed here so the lexer can reject
/// them anywhere an identifier is expected, even though Phase 0 itself only
/// consumes `let`, `in`, and `ordinary`.
const RESERVED: &[&str] = &[
    "let", "in", "linear", "shared", "ordinary", "method", "returns", "enum", "match", "inout",
    "int",
];

fn is_reserved(text: &str) -> bool {
    RESERVED.contains(&text)
}

impl<'src> Parser<'src> {
    fn new(source: &'src str) -> Self {
        Self {
            src: source.as_bytes(),
            pos: 0,
        }
    }

    // -- Position primitives --

    fn at_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.src.get(self.pos + offset).copied()
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(b) if b.is_ascii_whitespace() => {
                    self.pos += 1;
                }
                Some(b'/') if self.peek_at(1) == Some(b'/') => {
                    while let Some(b) = self.peek() {
                        self.pos += 1;
                        if b == b'\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn error_here(&self, note: impl Into<String>) -> Error {
        let end = (self.pos + 1).min(self.src.len()).max(self.pos);
        Error::new(
            ErrorCategory::ParseError,
            Span::new(self.pos, end),
            Some(note.into()),
        )
    }

    fn error_at(&self, span: Span, note: impl Into<String>) -> Error {
        Error::new(ErrorCategory::ParseError, span, Some(note.into()))
    }

    // -- Lexing primitives --

    /// Lex an identifier (or keyword) at the current position. Does not skip
    /// trivia first; callers handle that. On match, advances `pos`. On
    /// non-match, leaves `pos` untouched.
    fn lex_ident(&mut self) -> Option<(&'src str, Span)> {
        let start = self.pos;
        let first = self.peek()?;
        if !(first.is_ascii_alphabetic() || first == b'_') {
            return None;
        }
        self.pos += 1;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        // Identifier characters are ASCII so this is always valid UTF-8.
        let text = std::str::from_utf8(&self.src[start..self.pos]).expect("ident is ASCII");
        Some((text, Span::new(start, self.pos)))
    }

    /// Lex an integer literal (`-?[0-9]+`). Returns `Ok(None)` if there is
    /// no integer here (without advancing); returns `Err` if digits parse
    /// but overflow `i64`.
    fn lex_int(&mut self) -> Result<Option<(i64, Span)>, Error> {
        let start = self.pos;
        let mut p = self.pos;
        if self.src.get(p).copied() == Some(b'-') {
            match self.src.get(p + 1).copied() {
                Some(b) if b.is_ascii_digit() => {
                    p += 1;
                }
                _ => return Ok(None),
            }
        }
        match self.src.get(p).copied() {
            Some(b) if b.is_ascii_digit() => {}
            _ => return Ok(None),
        }
        while let Some(b) = self.src.get(p).copied() {
            if b.is_ascii_digit() {
                p += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..p]).expect("digits are ASCII");
        let value: i64 = text.parse().map_err(|_| {
            Error::new(
                ErrorCategory::ParseError,
                Span::new(start, p),
                Some(format!("integer literal `{text}` does not fit in i64")),
            )
        })?;
        self.pos = p;
        Ok(Some((value, Span::new(start, p))))
    }

    /// Try to consume the keyword `kw`. Requires that the keyword be
    /// followed by a non-identifier character (so `inert` doesn't match
    /// `in`). Returns the matched span on success.
    fn try_keyword(&mut self, kw: &str) -> Option<Span> {
        let saved = self.pos;
        match self.lex_ident() {
            Some((text, span)) if text == kw => Some(span),
            _ => {
                self.pos = saved;
                None
            }
        }
    }

    /// Try to consume a single ASCII byte token, returning its span.
    fn try_byte(&mut self, b: u8) -> Option<Span> {
        if self.peek() == Some(b) {
            let span = Span::new(self.pos, self.pos + 1);
            self.pos += 1;
            Some(span)
        } else {
            None
        }
    }

    // -- Grammar --

    fn parse_expr(&mut self) -> Result<Expr, Error> {
        self.skip_trivia();
        if let Some(let_span) = self.try_keyword("let") {
            return self.parse_let_tail(let_span);
        }
        self.parse_add_expr()
    }

    /// Parse the part of `let u x = e1 in e2` after the `let` keyword.
    fn parse_let_tail(&mut self, let_span: Span) -> Result<Expr, Error> {
        self.skip_trivia();

        // Phase 0 allows only `ordinary`; `linear` and `shared` are reserved
        // and explicitly rejected here so the test corpus's
        // reject/03_linear_unsupported.lin gets a parse error.
        let usage = if self.try_keyword("ordinary").is_some() {
            Usage::Ordinary
        } else {
            // Peek at whatever identifier is here to give a targeted error.
            let saved = self.pos;
            let (kw_text, kw_span) = match self.lex_ident() {
                Some(found) => found,
                None => {
                    self.pos = saved;
                    return Err(self.error_here("expected `ordinary` after `let`"));
                }
            };
            let msg = match kw_text {
                "linear" | "shared" => format!(
                    "`{kw_text}` let bindings are not supported in Phase 0; only `ordinary` is allowed",
                ),
                _ => format!("expected `ordinary` after `let`, found `{kw_text}`"),
            };
            return Err(self.error_at(kw_span, msg));
        };

        self.skip_trivia();
        let (name, name_span) = self.parse_binder()?;

        self.skip_trivia();
        self.try_byte(b'=')
            .ok_or_else(|| self.error_here("expected `=` after let-binding name"))?;

        self.skip_trivia();
        let value = self.parse_expr()?;

        self.skip_trivia();
        self.try_keyword("in")
            .ok_or_else(|| self.error_here("expected `in` after let-binding value"))?;

        self.skip_trivia();
        let body = self.parse_expr()?;

        let span = let_span.join(body.span());
        Ok(Expr::Let {
            usage,
            name,
            name_span,
            value: Box::new(value),
            body: Box::new(body),
            span,
        })
    }

    fn parse_add_expr(&mut self) -> Result<Expr, Error> {
        self.skip_trivia();
        let mut lhs = self.parse_atom()?;
        loop {
            self.skip_trivia();
            if self.try_byte(b'+').is_none() {
                break;
            }
            self.skip_trivia();
            let rhs = self.parse_atom()?;
            let span = lhs.span().join(rhs.span());
            lhs = Expr::Add {
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    fn parse_atom(&mut self) -> Result<Expr, Error> {
        self.skip_trivia();

        if let Some((value, span)) = self.lex_int()? {
            return Ok(Expr::IntLit { value, span });
        }

        if let Some(open_span) = self.try_byte(b'(') {
            self.skip_trivia();
            let inner = self.parse_expr()?;
            self.skip_trivia();
            // A comma here would start a tuple; tuples are out of scope for
            // Phase 0.
            if self.peek() == Some(b',') {
                let comma_span = Span::new(self.pos, self.pos + 1);
                return Err(self.error_at(
                    comma_span,
                    "tuples are not supported in Phase 0; use a single grouped expression",
                ));
            }
            if self.try_byte(b')').is_none() {
                return Err(self.error_here("expected `)` to close grouping"));
            }
            // Inner expression keeps its own span; grouping is invisible in
            // the AST.
            let _ = open_span;
            return Ok(inner);
        }

        // Identifier or reserved keyword.
        let saved = self.pos;
        if let Some((text, span)) = self.lex_ident() {
            if is_reserved(text) {
                self.pos = saved;
                return Err(self.error_at(
                    span,
                    format!("`{text}` is a reserved keyword and cannot appear here"),
                ));
            }
            return Ok(Expr::Var {
                name: text.to_string(),
                span,
            });
        }

        Err(self.error_here("expected expression"))
    }

    fn parse_binder(&mut self) -> Result<(String, Span), Error> {
        match self.lex_ident() {
            Some((text, span)) => {
                if is_reserved(text) {
                    Err(self.error_at(
                        span,
                        format!("`{text}` is a reserved keyword and cannot be used as a binder"),
                    ))
                } else {
                    Ok((text.to_string(), span))
                }
            }
            None => Err(self.error_here("expected identifier")),
        }
    }
}

// -----------------------------------------------------------------------------
// Inline sanity tests. The full corpus lives under `tests/`; these just
// guard the parser's structural invariants.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> Expr {
        parse(src).expect("expected parse success").main_expr
    }

    fn parse_err(src: &str) -> Error {
        parse(src).expect_err("expected parse failure")
    }

    #[test]
    fn int_literal() {
        match parse_ok("42") {
            Expr::IntLit { value: 42, .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn negative_int_literal() {
        match parse_ok("-7") {
            Expr::IntLit { value: -7, .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn simple_add() {
        match parse_ok("1 + 2") {
            Expr::Add { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn add_left_assoc() {
        // `1 + 2 + 3` should be `(1 + 2) + 3`.
        match parse_ok("1 + 2 + 3") {
            Expr::Add { lhs, rhs, .. } => {
                assert!(matches!(*lhs, Expr::Add { .. }));
                assert!(matches!(*rhs, Expr::IntLit { value: 3, .. }));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parenthesised_grouping() {
        match parse_ok("(1 + 2)") {
            Expr::Add { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn let_ordinary() {
        match parse_ok("let ordinary x = 1 in x + 1") {
            Expr::Let { usage, name, .. } => {
                assert_eq!(usage, Usage::Ordinary);
                assert_eq!(name, "x");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn rejects_linear_let() {
        let err = parse_err("let linear x = 1 in x");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn rejects_shared_let() {
        let err = parse_err("let shared x = 1 in x");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn rejects_repeated_let() {
        let err = parse_err("let let let");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn rejects_tuple_syntax() {
        let err = parse_err("(1, 2)");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn rejects_keyword_as_binder() {
        let err = parse_err("let ordinary in = 1 in 1");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn comments_and_whitespace() {
        // Comment and whitespace handling.
        let expr = parse_ok("// hello\n  1 + 1 // trailing\n");
        assert!(matches!(expr, Expr::Add { .. }));
    }

    #[test]
    fn unbound_variable_parses_ok() {
        // The parser doesn't know about scoping; `x` is just a Var node.
        match parse_ok("x") {
            Expr::Var { name, .. } => assert_eq!(name, "x"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn keyword_prefix_is_not_keyword() {
        // `inertia` should lex as a single identifier, not as `in` + `ertia`.
        match parse_ok("inertia") {
            Expr::Var { name, .. } => assert_eq!(name, "inertia"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn integer_overflow_is_parse_error() {
        let err = parse_err("99999999999999999999");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn nested_let() {
        // `let ordinary x = (let ordinary y = 1 in y) in x + 1`
        let expr = parse_ok("let ordinary x = let ordinary y = 1 in y in x + 1");
        let Expr::Let {
            name: outer_name,
            value,
            body,
            ..
        } = expr
        else {
            panic!("expected outer Let");
        };
        assert_eq!(outer_name, "x");
        // The value of the outer let must itself be a Let binding `y`.
        let Expr::Let {
            name: inner_name, ..
        } = *value
        else {
            panic!("expected inner Let in value position");
        };
        assert_eq!(inner_name, "y");
        // The body of the outer let is `x + 1`.
        assert!(matches!(*body, Expr::Add { .. }));
    }

    #[test]
    fn trailing_input_is_parse_error() {
        let err = parse_err("1 + 1 garbage");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }
}
