//! Abstract syntax tree for the mini-linear language.
//!
//! This module defines the AST for the *entire* language across all phases
//! (Phases 0–4). Later phases extend the parser and type checker; the AST
//! shape itself should not need to change.
//!
//! Every node carries a `Span` (byte offsets into the source) so error
//! reporting can map back to a line/column at display time.

use std::fmt;

// -----------------------------------------------------------------------------
// Spans
// -----------------------------------------------------------------------------

/// A half-open byte range `[start, end)` into the source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        debug_assert!(start <= end, "span start must be <= end");
        Self { start, end }
    }

    /// Span covering both `self` and `other`. Useful when combining child
    /// spans to span a parent node.
    pub fn join(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

// -----------------------------------------------------------------------------
// Usages
// -----------------------------------------------------------------------------

/// Variable usage qualifier. Ordinary values can be freely duplicated and
/// discarded; linear values must be consumed exactly once; shared values are
/// temporarily borrowed views of linear values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Usage {
    Linear,
    Shared,
    Ordinary,
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Usage::Linear => f.write_str("linear"),
            Usage::Shared => f.write_str("shared"),
            Usage::Ordinary => f.write_str("ordinary"),
        }
    }
}

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// A type annotation as written in source, or a type produced by the type
/// checker. Note: tuple field usages are syntactically permissive here — the
/// type checker is responsible for rejecting `shared` in tuple fields (the
/// paper's `u_lo` restriction).
///
/// Methods are not first-class and have their own signature type
/// (`MethodSig`) kept in a side environment by the type checker, so they
/// don't appear in this enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    /// `int`
    Int,
    /// `(u1 t1, u2 t2, ...)` — a tuple type with a usage on each field.
    Tuple(Vec<TupleField>),
    /// Reference to a user-defined datatype, e.g. `List`.
    Datatype { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TupleField {
    pub usage: Usage,
    pub ty: Type,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => f.write_str("int"),
            Type::Datatype { name } => f.write_str(name),
            Type::Tuple(fields) => {
                f.write_str("(")?;
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{} {}", field.usage, field.ty)?;
                }
                f.write_str(")")
            }
        }
    }
}

/// Shape of a method's parameters and returns. Methods are not first-class,
/// so this lives outside `Type`; the type checker keeps a separate method
/// environment keyed by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodSig {
    pub params: Vec<ParamSig>,
    pub returns: Vec<ParamSig>,
}

/// A single parameter or return slot in a method signature: usage, optional
/// `inout` borrow, and type. Names are not part of the signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSig {
    pub usage: Usage,
    pub inout: bool,
    pub ty: Type,
}

// -----------------------------------------------------------------------------
// Patterns (for `match`)
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    /// `_`
    Wildcard { span: Span },
    /// `x` — bind the scrutinee to a fresh variable.
    Var { name: String, span: Span },
    /// `Cons { data, tail }` — a datatype constructor with named field
    /// bindings (Rust struct-pattern style).
    Constructor {
        name: String,
        fields: Vec<FieldBinding>,
        span: Span,
    },
}

/// One field of a constructor pattern: the field name, the variable it
/// binds, and the span of the binding. In the shorthand form
/// `Cons { data, tail }`, `field` and `var` are the same string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldBinding {
    pub field: String,
    pub var: String,
    pub span: Span,
}

// -----------------------------------------------------------------------------
// Expressions
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// Integer literal.
    IntLit { value: i64, span: Span },

    /// Variable reference.
    Var { name: String, span: Span },

    /// `e1 + e2`
    Add {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },

    /// `e1 ; e2` — evaluate `e1` for effect/borrow, then `e2`.
    Seq {
        first: Box<Expr>,
        second: Box<Expr>,
        span: Span,
    },

    /// `let u x = e1 in e2`
    Let {
        usage: Usage,
        name: String,
        /// Span of just the binder identifier, for "unused linear `x`"-
        /// style errors that want to point at the name rather than the
        /// whole `let`.
        name_span: Span,
        value: Box<Expr>,
        body: Box<Expr>,
        span: Span,
    },

    /// `(e1, e2, ...)` — tuple construction. Single-element tuples are not
    /// expressible; `(e)` parses as grouping in Phase 0+.
    Tuple { elems: Vec<Expr>, span: Span },

    /// `e.i` — zero-indexed projection.
    TupleProj {
        tuple: Box<Expr>,
        index: usize,
        span: Span,
    },

    /// `let (x1, ..., xn) = e1 in e2` — tuple deconstruction.
    TupleDestructure {
        names: Vec<(String, Span)>,
        value: Box<Expr>,
        body: Box<Expr>,
        span: Span,
    },

    /// `f(e1, ..., en)` — call to a top-level method. (Methods are not
    /// first-class, so the callee is always a name.)
    MethodCall {
        name: String,
        name_span: Span,
        args: Vec<Expr>,
        span: Span,
    },

    /// `Cons { data: e1, tail: e2 }` — datatype constructor with named
    /// field initialisers. Zero-field variants may be written as
    /// `Nil` (parser accepts both `Nil` and `Nil {}` and produces this
    /// node either way).
    ConstructorApp {
        name: String,
        name_span: Span,
        fields: Vec<FieldInit>,
        span: Span,
    },

    /// `match e { p1 => e1, p2 => e2, ... }`
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
}

impl Expr {
    /// Span covering this expression in the source.
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit { span, .. }
            | Expr::Var { span, .. }
            | Expr::Add { span, .. }
            | Expr::Seq { span, .. }
            | Expr::Let { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::TupleProj { span, .. }
            | Expr::TupleDestructure { span, .. }
            | Expr::MethodCall { span, .. }
            | Expr::ConstructorApp { span, .. }
            | Expr::Match { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInit {
    pub field: String,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

// -----------------------------------------------------------------------------
// Top-level declarations
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decl {
    Method(MethodDecl),
    Datatype(DatatypeDecl),
}

/// A method declaration:
/// `method foo(linear inout x: int, ordinary y: int) returns (a: int, b: int) { body }`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MethodDecl {
    pub name: String,
    pub name_span: Span,
    pub params: Vec<Param>,
    pub returns: Vec<Param>,
    pub body: Expr,
    pub span: Span,
}

/// A named parameter or return slot, carrying source-level information that
/// `ParamSig` discards (name, identifier span).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub usage: Usage,
    pub inout: bool,
    pub name: String,
    pub name_span: Span,
    pub ty: Type,
    pub span: Span,
}

/// An algebraic datatype declaration:
/// ```text
/// enum List {
///     Nil,
///     Cons { data: int, linear tail: List },
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatatypeDecl {
    pub name: String,
    pub name_span: Span,
    pub variants: Vec<VariantDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantDecl {
    pub name: String,
    pub name_span: Span,
    pub fields: Vec<VariantField>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantField {
    pub usage: Usage,
    pub name: String,
    pub name_span: Span,
    pub ty: Type,
    pub span: Span,
}

// -----------------------------------------------------------------------------
// Program
// -----------------------------------------------------------------------------

/// A whole compilation unit: zero or more top-level declarations followed by
/// a single top-level expression that is the "main" expression to type-check.
///
/// In Phase 0, `decls` is always empty and only `main_expr` is parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub decls: Vec<Decl>,
    pub main_expr: Expr,
    pub span: Span,
}
