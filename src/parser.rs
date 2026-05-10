//! Hand-rolled recursive-descent parser. Phases 0 + 1 + 2 + 3.
//!
//! Grammar:
//!
//! ```text
//! program       ::= decl* expr
//! decl          ::= method_decl
//! method_decl   ::= 'method' ident '(' params? ')' 'returns' '(' params? ')' '{' expr '}'
//! params        ::= param (',' param)*
//! param         ::= usage ('inout')? ident ':' type
//!                   (parser rejects `inout` on non-linear params and on
//!                    return params; type checker rejects `shared` returns)
//!
//! type          ::= 'int' | tuple_type
//! tuple_type    ::= '(' (tuple_field (',' tuple_field)*)? ')'
//!                   (zero fields = empty tuple `()`; one field is a parse
//!                    error since the language has no 1-tuples)
//! tuple_field   ::= [usage] type        (usage defaults to `ordinary`)
//!
//! expr       ::= seq_expr
//! seq_expr   ::= seq_atom (';' seq_atom)*
//! seq_atom   ::= 'let' usage ident '=' expr 'in' expr
//!              | 'let' '(' (ident (',' ident)*)? ')' '=' expr 'in' expr
//!              | add_expr
//! add_expr   ::= postfix ('+' postfix)*
//! postfix    ::= atom ('.' nonneg_int)*
//! atom       ::= int_literal
//!              | ident                                 (variable reference)
//!              | ident '(' (expr (',' expr)*)? ')'     (method call)
//!              | '(' paren_body ')'
//! paren_body ::= /* empty */                  // empty tuple
//!              | expr                          // grouping (no comma)
//!              | expr (',' expr)+              // tuple with ≥ 2 elements
//! usage      ::= 'ordinary' | 'linear' | 'shared'
//! ```
//!
//! Precedence (high → low): tuple projection `e.i`, addition `+`,
//! sequencing `;`, then `let` (which extends as far right as possible
//! because the let body is parsed via `expr`). Both `+` and `;` are
//! left-associative.
//!
//! Method calls live at the `atom` level rather than at `postfix`: only
//! bare names can be called, never the result of an arbitrary expression.
//! `f(x).0` is fine (project from a call result); `f(x).0(y)` is not (the
//! second `(...)` doesn't bind to anything because postfix only handles
//! `.i`).
//!
//! Future operators (subtraction, comparisons, etc.) will need additional
//! precedence layers between `expr` and `atom`.
//!
//! Comments (`// ...`) and ASCII whitespace are skipped between tokens.
//! Reserved keywords are recognised by the lexer (see `RESERVED`) and
//! rejected as binders / variables anywhere they would otherwise look like
//! identifiers.

use crate::ast::{
    Decl, Expr, MethodDecl, Param, Program, Span, TupleField, Type, Usage,
};
use crate::error::{Error, ErrorCategory};

/// Parse a complete program from `source`: zero or more top-level method
/// declarations, followed by a single main expression.
pub fn parse(source: &str) -> Result<Program, Error> {
    let mut p = Parser::new(source);
    p.skip_trivia();
    let start = p.pos;

    let mut decls = Vec::new();
    loop {
        p.skip_trivia();
        if let Some(method_span) = p.try_keyword("method") {
            let decl = p.parse_method_decl_tail(method_span)?;
            decls.push(Decl::Method(decl));
        } else {
            break;
        }
    }

    p.skip_trivia();
    let main_expr = p.parse_expr()?;
    p.skip_trivia();
    if !p.at_eof() {
        return Err(p.error_here("unexpected trailing input"));
    }
    let span = Span::new(start, p.pos);
    Ok(Program {
        decls,
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
/// them anywhere an identifier is expected, even though Phases 0 + 1 only
/// directly consume `let`, `in`, `ordinary`, and `linear`.
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

    /// Lex an identifier (or keyword) at the current position. Does not
    /// skip trivia first; callers handle that. On match, advances `pos`.
    /// On non-match, leaves `pos` untouched.
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
        let text = std::str::from_utf8(&self.src[start..self.pos]).expect("ident is ASCII");
        Some((text, Span::new(start, self.pos)))
    }

    /// Lex an integer literal (`-?[0-9]+`). Returns `Ok(None)` if there is
    /// no integer here (without advancing); returns `Err` if digits parse
    /// but overflow `i64`.
    ///
    /// NOTE: this lexes a leading `-` as part of the integer because there
    /// is no `-` operator in the language at Phases 0–1, so there's no
    /// ambiguity. When subtraction (or any other use of `-`) is added in a
    /// later phase, this rule must be replaced with a unary-minus operator
    /// at the expression level — keeping the sign here would cause
    /// `1 - 5` to mis-lex.
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

    /// Lex a non-negative integer (used for tuple projection indices like
    /// `e.0`). Errors out if no digits or if the value doesn't fit `usize`.
    fn lex_nonneg_int(&mut self) -> Result<(usize, Span), Error> {
        let start = self.pos;
        match self.peek() {
            Some(b) if b.is_ascii_digit() => {}
            _ => return Err(self.error_here("expected non-negative integer index")),
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos]).expect("digits are ASCII");
        let value: usize = text.parse().map_err(|_| {
            Error::new(
                ErrorCategory::ParseError,
                Span::new(start, self.pos),
                Some(format!("tuple index `{text}` does not fit in usize")),
            )
        })?;
        Ok((value, Span::new(start, self.pos)))
    }

    /// Try to consume the keyword `kw`. Requires that the keyword be
    /// followed by a non-identifier character (so `inertia` doesn't match
    /// `in`).
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
        self.parse_seq_expr()
    }

    /// `seq_expr ::= seq_atom (';' seq_atom)*` — left-associative.
    fn parse_seq_expr(&mut self) -> Result<Expr, Error> {
        let mut lhs = self.parse_seq_atom()?;
        loop {
            self.skip_trivia();
            if self.try_byte(b';').is_none() {
                break;
            }
            self.skip_trivia();
            let rhs = self.parse_seq_atom()?;
            let span = lhs.span().join(rhs.span());
            lhs = Expr::Seq {
                first: Box::new(lhs),
                second: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    /// `seq_atom` is either a `let` form or an `add_expr`. Splitting the
    /// `let` cases out at this level (rather than at `expr`) is what gives
    /// `let extends as far right as possible` and `1 ; let X in 2` both
    /// the right behaviour.
    fn parse_seq_atom(&mut self) -> Result<Expr, Error> {
        self.skip_trivia();
        if let Some(let_span) = self.try_keyword("let") {
            return self.parse_let_tail(let_span);
        }
        self.parse_add_expr()
    }

    /// Dispatch on what follows `let`:
    /// - `(` → tuple deconstruction `let (x1, ..., xn) = e1 in e2`
    /// - usage keyword → variable let `let u x = e1 in e2`
    fn parse_let_tail(&mut self, let_span: Span) -> Result<Expr, Error> {
        self.skip_trivia();
        if self.peek() == Some(b'(') {
            return self.parse_let_destructure_tail(let_span);
        }
        self.parse_let_var_tail(let_span)
    }

    fn parse_let_var_tail(&mut self, let_span: Span) -> Result<Expr, Error> {
        let usage = if self.try_keyword("ordinary").is_some() {
            Usage::Ordinary
        } else if self.try_keyword("linear").is_some() {
            Usage::Linear
        } else if self.try_keyword("shared").is_some() {
            Usage::Shared
        } else {
            // Peek at the offender for a targeted error.
            let saved = self.pos;
            let (kw_text, kw_span) = match self.lex_ident() {
                Some(found) => found,
                None => {
                    self.pos = saved;
                    return Err(self.error_here(
                        "expected `ordinary`, `linear`, `shared`, or `(` after `let`",
                    ));
                }
            };
            let msg = format!(
                "expected `ordinary`, `linear`, `shared`, or `(` after `let`, found `{kw_text}`"
            );
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

    /// Parse `let (x1, ..., xn) = e1 in e2`. Caller has already verified
    /// that the next non-trivia byte is `(`.
    fn parse_let_destructure_tail(&mut self, let_span: Span) -> Result<Expr, Error> {
        self.try_byte(b'(')
            .expect("caller verified opening paren");

        self.skip_trivia();
        let mut names: Vec<(String, Span)> = Vec::new();
        if self.peek() != Some(b')') {
            let (name, span) = self.parse_binder()?;
            names.push((name, span));
            loop {
                self.skip_trivia();
                if self.try_byte(b',').is_none() {
                    break;
                }
                self.skip_trivia();
                if self.peek() == Some(b')') {
                    return Err(
                        self.error_here("trailing commas in deconstruction patterns are not supported")
                    );
                }
                let (name, span) = self.parse_binder()?;
                names.push((name, span));
            }
        }

        self.skip_trivia();
        self.try_byte(b')')
            .ok_or_else(|| self.error_here("expected `)` to close deconstruction pattern"))?;

        // Singleton deconstruction patterns (`let (x) = ...`) are rejected
        // because the language has no 1-tuple type for the value to inhabit.
        // Catching it here gives a clearer message than the type checker's
        // arity mismatch.
        if names.len() == 1 {
            let name_span = names[0].1;
            return Err(self.error_at(
                name_span,
                "1-element deconstruction patterns are not supported (no 1-tuples in the language)",
            ));
        }

        self.skip_trivia();
        self.try_byte(b'=')
            .ok_or_else(|| self.error_here("expected `=` after deconstruction pattern"))?;

        self.skip_trivia();
        let value = self.parse_expr()?;

        self.skip_trivia();
        self.try_keyword("in")
            .ok_or_else(|| self.error_here("expected `in` after deconstruction value"))?;

        self.skip_trivia();
        let body = self.parse_expr()?;

        let span = let_span.join(body.span());
        Ok(Expr::TupleDestructure {
            names,
            value: Box::new(value),
            body: Box::new(body),
            span,
        })
    }

    fn parse_add_expr(&mut self) -> Result<Expr, Error> {
        self.skip_trivia();
        let mut lhs = self.parse_postfix()?;
        loop {
            self.skip_trivia();
            if self.try_byte(b'+').is_none() {
                break;
            }
            self.skip_trivia();
            let rhs = self.parse_postfix()?;
            let span = lhs.span().join(rhs.span());
            lhs = Expr::Add {
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            };
        }
        Ok(lhs)
    }

    /// `postfix ::= atom ('.' nonneg_int)*` — left-associative tuple
    /// projection. Whitespace around `.` is permitted and behaves like
    /// trivia between any other tokens.
    fn parse_postfix(&mut self) -> Result<Expr, Error> {
        let mut expr = self.parse_atom()?;
        loop {
            self.skip_trivia();
            if self.try_byte(b'.').is_none() {
                break;
            }
            self.skip_trivia();
            let (index, idx_span) = self.lex_nonneg_int()?;
            let span = expr.span().join(idx_span);
            expr = Expr::TupleProj {
                tuple: Box::new(expr),
                index,
                span,
            };
        }
        Ok(expr)
    }

    fn parse_atom(&mut self) -> Result<Expr, Error> {
        self.skip_trivia();

        if let Some((value, span)) = self.lex_int()? {
            return Ok(Expr::IntLit { value, span });
        }

        if let Some(open_span) = self.try_byte(b'(') {
            return self.parse_paren_body(open_span);
        }

        // Identifier — either a variable reference or a method call.
        let saved = self.pos;
        if let Some((text, ident_span)) = self.lex_ident() {
            if is_reserved(text) {
                self.pos = saved;
                return Err(self.error_at(
                    ident_span,
                    format!("`{text}` is a reserved keyword and cannot appear here"),
                ));
            }
            let name = text.to_string();
            // Method call: identifier immediately (modulo trivia) followed
            // by `(`. Trivia between the name and `(` is allowed,
            // consistent with our trivia-anywhere policy.
            self.skip_trivia();
            if self.peek() == Some(b'(') {
                return self.parse_method_call_tail(name, ident_span);
            }
            return Ok(Expr::Var {
                name,
                span: ident_span,
            });
        }

        Err(self.error_here("expected expression"))
    }

    /// Parse the `(args)` tail of a method call. Caller has consumed the
    /// method name and any trivia; current position is at the opening `(`.
    fn parse_method_call_tail(
        &mut self,
        name: String,
        name_span: Span,
    ) -> Result<Expr, Error> {
        self.try_byte(b'(').expect("caller verified opening paren");

        self.skip_trivia();
        let mut args = Vec::new();
        if self.peek() != Some(b')') {
            args.push(self.parse_expr()?);
            loop {
                self.skip_trivia();
                if self.try_byte(b',').is_none() {
                    break;
                }
                self.skip_trivia();
                if self.peek() == Some(b')') {
                    return Err(
                        self.error_here("trailing commas in argument lists are not supported")
                    );
                }
                args.push(self.parse_expr()?);
            }
        }
        self.skip_trivia();
        let close_span = self
            .try_byte(b')')
            .ok_or_else(|| self.error_here("expected `)` to close argument list"))?;
        let span = name_span.join(close_span);
        Ok(Expr::MethodCall {
            name,
            name_span,
            args,
            span,
        })
    }

    /// After consuming `(`, decide between empty tuple, grouping, or
    /// tuple construction based on what follows.
    fn parse_paren_body(&mut self, open_span: Span) -> Result<Expr, Error> {
        self.skip_trivia();

        // `()` — empty tuple.
        if let Some(close_span) = self.try_byte(b')') {
            let span = open_span.join(close_span);
            return Ok(Expr::Tuple {
                elems: Vec::new(),
                span,
            });
        }

        let first = self.parse_expr()?;

        self.skip_trivia();
        // `(e)` — grouping. The inner expression keeps its own span;
        // grouping is invisible in the AST.
        if self.try_byte(b')').is_some() {
            return Ok(first);
        }

        // `(e1, e2, ...)` — tuple. Anything else is a parse error.
        if self.peek() != Some(b',') {
            return Err(self.error_here("expected `,` or `)` in parenthesised expression"));
        }

        let mut elems = vec![first];
        loop {
            self.try_byte(b',').expect("checked by peek above");
            self.skip_trivia();
            if self.peek() == Some(b')') {
                // We saw at least one `,` already, so this is either a
                // singleton tuple (`(e,)`) or a trailing comma after a
                // longer tuple (`(e1, e2,)`). Both are unsupported.
                return Err(self.error_here(
                    "trailing commas in tuples are not supported (singleton tuples are not a thing)",
                ));
            }
            let next = self.parse_expr()?;
            elems.push(next);
            self.skip_trivia();
            if self.peek() != Some(b',') {
                break;
            }
        }

        let close_span = self
            .try_byte(b')')
            .ok_or_else(|| self.error_here("expected `)` to close tuple"))?;
        let span = open_span.join(close_span);
        Ok(Expr::Tuple { elems, span })
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

    // -- Method declarations --

    /// Parse the rest of a method declaration after the `method` keyword.
    /// Caller has consumed `method`; we parse name through closing `}`.
    fn parse_method_decl_tail(&mut self, method_span: Span) -> Result<MethodDecl, Error> {
        self.skip_trivia();
        let (name, name_span) = self.parse_binder()?;

        self.skip_trivia();
        self.try_byte(b'(')
            .ok_or_else(|| self.error_here("expected `(` after method name"))?;
        let params = self.parse_params(false)?;
        self.try_byte(b')')
            .ok_or_else(|| self.error_here("expected `)` to close parameter list"))?;

        self.skip_trivia();
        self.try_keyword("returns")
            .ok_or_else(|| self.error_here("expected `returns` after parameter list"))?;

        self.skip_trivia();
        self.try_byte(b'(')
            .ok_or_else(|| self.error_here("expected `(` after `returns`"))?;
        let returns = self.parse_params(true)?;
        self.try_byte(b')')
            .ok_or_else(|| self.error_here("expected `)` to close return list"))?;

        self.skip_trivia();
        self.try_byte(b'{')
            .ok_or_else(|| self.error_here("expected `{` to begin method body"))?;

        self.skip_trivia();
        let body = self.parse_expr()?;

        self.skip_trivia();
        let close_span = self
            .try_byte(b'}')
            .ok_or_else(|| self.error_here("expected `}` to close method body"))?;
        let span = method_span.join(close_span);
        Ok(MethodDecl {
            name,
            name_span,
            params,
            returns,
            body,
            span,
        })
    }

    /// Parse a comma-separated list of parameters inside `(...)`. Caller
    /// has consumed the opening paren and is responsible for consuming the
    /// closing paren. `is_return` swaps `inout`'s legality (allowed only
    /// in non-return position).
    fn parse_params(&mut self, is_return: bool) -> Result<Vec<Param>, Error> {
        let mut params = Vec::new();
        self.skip_trivia();
        if self.peek() == Some(b')') {
            return Ok(params);
        }
        params.push(self.parse_one_param(is_return)?);
        loop {
            self.skip_trivia();
            if self.try_byte(b',').is_none() {
                break;
            }
            self.skip_trivia();
            if self.peek() == Some(b')') {
                return Err(
                    self.error_here("trailing commas in parameter lists are not supported")
                );
            }
            params.push(self.parse_one_param(is_return)?);
        }
        self.skip_trivia();
        Ok(params)
    }

    fn parse_one_param(&mut self, is_return: bool) -> Result<Param, Error> {
        self.skip_trivia();
        let param_start = self.pos;

        // Usage keyword (required). Detect "inout-without-usage" and emit
        // a targeted error.
        let usage = if self.try_keyword("ordinary").is_some() {
            Usage::Ordinary
        } else if self.try_keyword("linear").is_some() {
            Usage::Linear
        } else if self.try_keyword("shared").is_some() {
            Usage::Shared
        } else {
            // Specific case: the user wrote `inout x: T` without a usage.
            let saved = self.pos;
            if let Some((text, span)) = self.lex_ident() {
                self.pos = saved;
                if text == "inout" {
                    return Err(self.error_at(
                        span,
                        "expected usage keyword (`linear`, `shared`, or `ordinary`) before `inout`",
                    ));
                }
                return Err(self.error_at(
                    span,
                    format!("expected usage keyword for parameter, found `{text}`"),
                ));
            }
            return Err(self.error_here("expected usage keyword for parameter"));
        };

        // Optional `inout`.
        self.skip_trivia();
        let inout = if let Some(inout_span) = self.try_keyword("inout") {
            if is_return {
                return Err(self.error_at(
                    inout_span,
                    "`inout` is not allowed in return parameters",
                ));
            }
            if usage != Usage::Linear {
                return Err(self.error_at(
                    inout_span,
                    "`inout` is only valid on `linear` parameters",
                ));
            }
            true
        } else {
            false
        };

        self.skip_trivia();
        let (name, name_span) = self.parse_binder()?;

        self.skip_trivia();
        self.try_byte(b':')
            .ok_or_else(|| self.error_here("expected `:` after parameter name"))?;

        self.skip_trivia();
        let ty = self.parse_type()?;

        let span = Span::new(param_start, self.pos);
        Ok(Param {
            usage,
            inout,
            name,
            name_span,
            ty,
            span,
        })
    }

    // -- Type syntax --

    /// `type ::= 'int' | tuple_type`.
    fn parse_type(&mut self) -> Result<Type, Error> {
        self.skip_trivia();
        if self.try_keyword("int").is_some() {
            return Ok(Type::Int);
        }
        if let Some(open_span) = self.try_byte(b'(') {
            return self.parse_tuple_type_body(open_span);
        }
        Err(self.error_here("expected type (`int` or a tuple type)"))
    }

    /// After consuming `(`, parse the body of a tuple type. Empty `()` is
    /// the empty tuple type; one field is rejected (no 1-tuples in the
    /// language); two or more fields produce a tuple type with each
    /// field's usage defaulting to `ordinary` if omitted.
    fn parse_tuple_type_body(&mut self, open_span: Span) -> Result<Type, Error> {
        self.skip_trivia();
        if self.try_byte(b')').is_some() {
            return Ok(Type::Tuple(Vec::new()));
        }
        let mut fields = Vec::new();
        fields.push(self.parse_tuple_type_field()?);
        loop {
            self.skip_trivia();
            if self.try_byte(b',').is_none() {
                break;
            }
            self.skip_trivia();
            if self.peek() == Some(b')') {
                return Err(
                    self.error_here("trailing commas in tuple types are not supported")
                );
            }
            fields.push(self.parse_tuple_type_field()?);
        }
        self.skip_trivia();
        let close_span = self
            .try_byte(b')')
            .ok_or_else(|| self.error_here("expected `)` to close tuple type"))?;
        if fields.len() == 1 {
            let pattern_span = open_span.join(close_span);
            return Err(self.error_at(
                pattern_span,
                "1-element tuple types are not supported (no 1-tuples in the language)",
            ));
        }
        Ok(Type::Tuple(fields))
    }

    /// One field of a tuple type. Usage is optional — defaults to
    /// `ordinary` when omitted, matching the spirit of "annotations should
    /// be optional when the natural default is unambiguous." Forms like
    /// `(linear int, ordinary int)` (explicit) and `(int, int)` (defaults)
    /// both parse.
    fn parse_tuple_type_field(&mut self) -> Result<TupleField, Error> {
        self.skip_trivia();
        let usage = if self.try_keyword("ordinary").is_some() {
            Usage::Ordinary
        } else if self.try_keyword("linear").is_some() {
            Usage::Linear
        } else if self.try_keyword("shared").is_some() {
            Usage::Shared
        } else {
            Usage::Ordinary
        };
        self.skip_trivia();
        let ty = self.parse_type()?;
        Ok(TupleField { usage, ty })
    }
}

// -----------------------------------------------------------------------------
// Inline sanity tests. The full corpus lives under `tests/`; these guard the
// parser's structural invariants (precedence, associativity, span composition).
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

    // -- Phase 0 carry-over tests --

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
        // `(1 + 2)` parses as just Add — grouping is invisible in the AST.
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
    fn rejects_repeated_let() {
        let err = parse_err("let let let");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn rejects_keyword_as_binder() {
        let err = parse_err("let ordinary in = 1 in 1");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn comments_and_whitespace() {
        let expr = parse_ok("// hello\n  1 + 1 // trailing\n");
        assert!(matches!(expr, Expr::Add { .. }));
    }

    #[test]
    fn unbound_variable_parses_ok() {
        match parse_ok("x") {
            Expr::Var { name, .. } => assert_eq!(name, "x"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn keyword_prefix_is_not_keyword() {
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
        let Expr::Let {
            name: inner_name, ..
        } = *value
        else {
            panic!("expected inner Let in value position");
        };
        assert_eq!(inner_name, "y");
        assert!(matches!(*body, Expr::Add { .. }));
    }

    #[test]
    fn trailing_input_is_parse_error() {
        let err = parse_err("1 + 1 garbage");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    // -- Phase 1: linear let --

    #[test]
    fn accepts_linear_let() {
        match parse_ok("let linear x = (1, 2) in let (a, b) = x in a + b") {
            Expr::Let { usage, name, .. } => {
                assert_eq!(usage, Usage::Linear);
                assert_eq!(name, "x");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn accepts_shared_let() {
        // Phase 2: `shared` is now a valid let-binding usage. The type
        // checker is responsible for rejecting cases where the value's
        // actual usage doesn't match the annotation.
        match parse_ok("let shared x = y in x") {
            Expr::Let { usage, name, .. } => {
                assert_eq!(usage, Usage::Shared);
                assert_eq!(name, "x");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    // -- Phase 1: tuples --

    #[test]
    fn empty_tuple() {
        match parse_ok("()") {
            Expr::Tuple { elems, .. } => assert!(elems.is_empty()),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn two_tuple() {
        match parse_ok("(1, 2)") {
            Expr::Tuple { elems, .. } => assert_eq!(elems.len(), 2),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn three_tuple() {
        match parse_ok("(1, 2, 3)") {
            Expr::Tuple { elems, .. } => assert_eq!(elems.len(), 3),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn singleton_tuple_is_parse_error() {
        let err = parse_err("(1,)");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn trailing_comma_in_tuple_is_parse_error() {
        let err = parse_err("(1, 2,)");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    // -- Phase 1: projection --

    #[test]
    fn tuple_projection() {
        match parse_ok("t.0") {
            Expr::TupleProj { index, .. } => assert_eq!(index, 0),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn projection_chain_is_left_assoc() {
        // `t.0.1` should parse as `(t.0).1`.
        let Expr::TupleProj {
            tuple: outer_tuple,
            index: outer_index,
            ..
        } = parse_ok("t.0.1")
        else {
            panic!("expected TupleProj");
        };
        assert_eq!(outer_index, 1);
        let Expr::TupleProj {
            index: inner_index, ..
        } = *outer_tuple
        else {
            panic!("expected inner TupleProj");
        };
        assert_eq!(inner_index, 0);
    }

    #[test]
    fn projection_higher_than_addition() {
        // `1 + t.0` should parse as `1 + (t.0)`.
        let Expr::Add { rhs, .. } = parse_ok("1 + t.0") else {
            panic!("expected Add");
        };
        assert!(matches!(*rhs, Expr::TupleProj { .. }));
    }

    #[test]
    fn negative_projection_index_rejected() {
        // `t.-1` would lex `-1` as a literal index, but the projection
        // parser explicitly rejects negatives at the lex_nonneg_int step.
        let err = parse_err("t.-1");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    // -- Phase 1: deconstruction --

    #[test]
    fn deconstruction_two_names() {
        match parse_ok("let (a, b) = t in a + b") {
            Expr::TupleDestructure { names, .. } => {
                assert_eq!(names.len(), 2);
                assert_eq!(names[0].0, "a");
                assert_eq!(names[1].0, "b");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn deconstruction_empty_names() {
        match parse_ok("let () = u in 5") {
            Expr::TupleDestructure { names, .. } => assert!(names.is_empty()),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn trailing_comma_in_pattern_is_parse_error() {
        let err = parse_err("let (a, b,) = t in 1");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn singleton_pattern_is_parse_error() {
        let err = parse_err("let (x) = t in x");
        assert_eq!(err.category, ErrorCategory::ParseError);
        let note = err.note.expect("note");
        assert!(
            note.contains("1-element"),
            "expected message about 1-element patterns, got: {note}"
        );
    }

    // -- Phase 1: sequencing --

    #[test]
    fn simple_seq() {
        match parse_ok("1 ; 2") {
            Expr::Seq { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn seq_left_assoc() {
        // `1 ; 2 ; 3` → `(1; 2); 3`
        let Expr::Seq { first, second, .. } = parse_ok("1 ; 2 ; 3") else {
            panic!("expected Seq");
        };
        assert!(matches!(*first, Expr::Seq { .. }));
        assert!(matches!(*second, Expr::IntLit { value: 3, .. }));
    }

    #[test]
    fn add_higher_than_seq() {
        // `1 + 2 ; 3` → `(1 + 2) ; 3`
        let Expr::Seq { first, second, .. } = parse_ok("1 + 2 ; 3") else {
            panic!("expected Seq");
        };
        assert!(matches!(*first, Expr::Add { .. }));
        assert!(matches!(*second, Expr::IntLit { value: 3, .. }));
    }

    #[test]
    fn let_extends_through_seq_in_body() {
        // `let ordinary x = 1 in x ; 2` → `let x = 1 in (x ; 2)`
        let Expr::Let { body, .. } = parse_ok("let ordinary x = 1 in x ; 2") else {
            panic!("expected Let");
        };
        assert!(matches!(*body, Expr::Seq { .. }));
    }

    #[test]
    fn seq_can_have_let_on_rhs() {
        // `1 ; let ordinary x = 2 in x` → `Seq(1, let x = 2 in x)`
        let Expr::Seq { first, second, .. } = parse_ok("1 ; let ordinary x = 2 in x") else {
            panic!("expected Seq");
        };
        assert!(matches!(*first, Expr::IntLit { value: 1, .. }));
        assert!(matches!(*second, Expr::Let { .. }));
    }

    // -- Phase 3: methods --

    fn parse_program(src: &str) -> Program {
        parse(src).expect("parse failed")
    }

    fn parse_program_err(src: &str) -> Error {
        parse(src).expect_err("expected parse failure")
    }

    #[test]
    fn method_decl_simple() {
        let prog = parse_program(
            "method add(ordinary x: int, ordinary y: int) returns (ordinary r: int) { x + y } 1",
        );
        assert_eq!(prog.decls.len(), 1);
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        assert_eq!(m.name, "add");
        assert_eq!(m.params.len(), 2);
        assert_eq!(m.params[0].name, "x");
        assert_eq!(m.params[0].usage, Usage::Ordinary);
        assert!(!m.params[0].inout);
        assert_eq!(m.returns.len(), 1);
        assert_eq!(m.returns[0].name, "r");
    }

    #[test]
    fn method_decl_with_inout() {
        let prog = parse_program(
            "method swap(linear inout a: (int, int), linear inout b: (int, int)) returns () { (b, a) } 1",
        );
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        assert_eq!(m.params.len(), 2);
        assert!(m.params[0].inout);
        assert_eq!(m.params[0].usage, Usage::Linear);
        assert_eq!(m.returns.len(), 0);
    }

    #[test]
    fn method_decl_no_params_no_returns() {
        let prog = parse_program("method nop() returns () { () } nop()");
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        assert_eq!(m.params.len(), 0);
        assert_eq!(m.returns.len(), 0);
    }

    #[test]
    fn method_call_at_top_level() {
        let prog = parse_program("method id(ordinary x: int) returns (ordinary r: int) { x } id(5)");
        match prog.main_expr {
            Expr::MethodCall { name, args, .. } => {
                assert_eq!(name, "id");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected MethodCall, got {other:?}"),
        }
    }

    #[test]
    fn method_call_zero_args() {
        let prog = parse_program("method nop() returns () { () } nop()");
        match prog.main_expr {
            Expr::MethodCall { args, .. } => assert_eq!(args.len(), 0),
            other => panic!("expected MethodCall, got {other:?}"),
        }
    }

    #[test]
    fn method_call_with_whitespace_before_paren() {
        // `id (5)` with a space between name and `(` should still parse as
        // a method call, consistent with our trivia-anywhere policy.
        let prog =
            parse_program("method id(ordinary x: int) returns (ordinary r: int) { x } id (5)");
        assert!(matches!(prog.main_expr, Expr::MethodCall { .. }));
    }

    #[test]
    fn bare_ident_remains_var() {
        // `foo` without parens is a Var, not a call.
        let prog = parse_program("foo");
        assert!(matches!(prog.main_expr, Expr::Var { .. }));
    }

    #[test]
    fn rejects_inout_on_ordinary_param() {
        let err = parse_program_err(
            "method bad(ordinary inout x: int) returns () { () } 1",
        );
        assert_eq!(err.category, ErrorCategory::ParseError);
        let note = err.note.expect("note");
        assert!(note.contains("`inout` is only valid on `linear`"), "got: {note}");
    }

    #[test]
    fn rejects_inout_on_shared_param() {
        let err = parse_program_err(
            "method bad(shared inout x: int) returns () { () } 1",
        );
        assert_eq!(err.category, ErrorCategory::ParseError);
        assert!(err.note.unwrap().contains("`inout` is only valid on `linear`"));
    }

    #[test]
    fn rejects_inout_in_returns() {
        let err = parse_program_err(
            "method bad() returns (linear inout x: int) { x } 1",
        );
        assert_eq!(err.category, ErrorCategory::ParseError);
        assert!(err.note.unwrap().contains("`inout` is not allowed in return"));
    }

    #[test]
    fn rejects_inout_without_usage() {
        let err = parse_program_err(
            "method bad(inout x: int) returns () { () } 1",
        );
        assert_eq!(err.category, ErrorCategory::ParseError);
        let note = err.note.unwrap();
        assert!(
            note.contains("usage keyword") && note.contains("inout"),
            "got: {note}",
        );
    }

    #[test]
    fn type_int() {
        // Round-trip through a method decl to exercise parse_type.
        let prog = parse_program("method m(ordinary x: int) returns () { () } 1");
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        assert_eq!(m.params[0].ty, Type::Int);
    }

    #[test]
    fn type_empty_tuple() {
        let prog = parse_program("method m(ordinary x: ()) returns () { () } 1");
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        assert_eq!(m.params[0].ty, Type::Tuple(Vec::new()));
    }

    #[test]
    fn type_tuple_with_implicit_ordinary() {
        // `(int, int)` should parse as `Tuple([{Ordinary, Int}, {Ordinary, Int}])`.
        let prog = parse_program("method m(ordinary x: (int, int)) returns () { () } 1");
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        if let Type::Tuple(fields) = &m.params[0].ty {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].usage, Usage::Ordinary);
            assert_eq!(fields[0].ty, Type::Int);
            assert_eq!(fields[1].usage, Usage::Ordinary);
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn type_tuple_with_explicit_usage() {
        let prog = parse_program(
            "method m(linear x: (linear int, ordinary int)) returns () { () } 1",
        );
        let Decl::Method(m) = &prog.decls[0] else {
            panic!("expected Method decl");
        };
        if let Type::Tuple(fields) = &m.params[0].ty {
            assert_eq!(fields[0].usage, Usage::Linear);
            assert_eq!(fields[1].usage, Usage::Ordinary);
        } else {
            panic!("expected Tuple");
        }
    }

    #[test]
    fn type_one_tuple_is_parse_error() {
        let err = parse_program_err(
            "method m(ordinary x: (int)) returns () { () } 1",
        );
        assert_eq!(err.category, ErrorCategory::ParseError);
        assert!(err.note.unwrap().contains("1-element"));
    }

    #[test]
    fn multiple_decls() {
        let prog = parse_program(
            "method first() returns (ordinary r: int) { 1 } \
             method second() returns (ordinary r: int) { 2 } \
             first()",
        );
        assert_eq!(prog.decls.len(), 2);
    }

    #[test]
    fn no_decls_still_works() {
        // Phase 0-2 backward compat: programs with no decls parse the same
        // as before.
        let prog = parse_program("1 + 1");
        assert!(prog.decls.is_empty());
        assert!(matches!(prog.main_expr, Expr::Add { .. }));
    }

    #[test]
    fn missing_main_expr_is_parse_error() {
        let err = parse_program_err("method nop() returns () { () }");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }

    #[test]
    fn rejects_reserved_method_name() {
        let err = parse_program_err("method let() returns () { () } 1");
        assert_eq!(err.category, ErrorCategory::ParseError);
    }
}
