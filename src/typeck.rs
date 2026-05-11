//! Type checker for the mini-linear language. Phases 0 + 1 + 2.
//!
//! Phase 2 implements the full inference algorithm from §2.6 of the paper:
//! both `borrows(B)` and `consumes(C)` tracking, with the mode-sensitive
//! variable rule that turns linear variables into shared views in Borrow
//! mode.
//!
//! `consumes` and `borrows` share the same shape — `BTreeMap<String, Span>`
//! mapping a variable name to the use-site span. The two have *different*
//! merge policies though: `consumes` requires disjointness (a name in both
//! halves is a "consumed twice" error), while `borrows` simply unions and
//! keeps the first-seen span on collision (multiple borrows of the same
//! variable are valid).
//!
//! Top-level invariant: with no methods or parameters, the initial env is
//! empty, the let / decon rules' "linear must be consumed" check
//! transitively prevents leakage, and the top-level program's `consumes`
//! and `borrows` are both empty by construction.

use std::collections::BTreeMap;
use std::fmt;

use crate::ast::{Expr, Program, Span, TupleField, Type, Usage};
use crate::error::{Error, ErrorCategory};

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// A typed value: a usage qualifier plus a type (paper's `u τ`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Typed {
    pub usage: Usage,
    pub ty: Type,
}

impl fmt::Display for Typed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.usage, self.ty)
    }
}

/// Expected consumption mode passed down into each subexpression (paper's
/// `c`). In Borrow mode, linear variable lookups produce shared views and
/// record into `borrows` instead of `consumes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsumeMode {
    Consume,
    Borrow,
}

/// Type-check a complete program.
pub fn check_program(program: &Program) -> Result<Typed, Error> {
    debug_assert!(
        program.decls.is_empty(),
        "Phase 0/1/2 parser does not produce declarations",
    );
    let env = Env::new();
    let result = check_expr(&program.main_expr, &env, None, ConsumeMode::Consume)?;
    debug_assert!(
        result.consumes.is_empty(),
        "top-level program's consumes set must be empty (initial env is empty)",
    );
    debug_assert!(
        result.borrows.is_empty(),
        "top-level program's borrows set must be empty (initial env is empty)",
    );
    Ok(result.typed)
}

// -----------------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------------

type Env = BTreeMap<String, Typed>;

/// Per-expression checking result. Both `consumes` and `borrows` are
/// keyed by variable name with the use-site span as the value. See module
/// docs for their differing merge policies.
#[derive(Debug, Clone)]
struct Check {
    typed: Typed,
    consumes: BTreeMap<String, Span>,
    borrows: BTreeMap<String, Span>,
}

impl Check {
    fn ordinary_int() -> Self {
        Self {
            typed: Typed {
                usage: Usage::Ordinary,
                ty: Type::Int,
            },
            consumes: BTreeMap::new(),
            borrows: BTreeMap::new(),
        }
    }
}

// -----------------------------------------------------------------------------
// The recursive checker
// -----------------------------------------------------------------------------

fn check_expr(
    expr: &Expr,
    env: &Env,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    match expr {
        Expr::IntLit { .. } => Ok(Check::ordinary_int()),

        Expr::Var { name, span } => check_var(name, *span, env, mode),

        Expr::Add { lhs, rhs, .. } => check_add(lhs, rhs, env, mode),

        Expr::Seq {
            first, second, ..
        } => check_seq(first, second, env, expected_usage, mode),

        Expr::Tuple { elems, span } => {
            check_tuple_construction(elems, env, expected_usage, *span, mode)
        }

        Expr::TupleProj {
            tuple,
            index,
            span,
        } => check_projection(tuple, *index, *span, env, mode),

        Expr::TupleDestructure {
            names,
            value,
            body,
            span,
        } => check_destructure(names, value, body, *span, env, expected_usage, mode),

        Expr::Let {
            usage,
            name,
            name_span,
            value,
            body,
            span,
        } => check_let(
            *usage,
            name,
            *name_span,
            value,
            body,
            *span,
            env,
            expected_usage,
            mode,
        ),

        Expr::MethodCall { span, .. }
        | Expr::ConstructorApp { span, .. }
        | Expr::Match { span, .. } => Err(Error::new(
            ErrorCategory::TypeError,
            *span,
            Some("this expression form is not supported until a later phase".to_string()),
        )),
    }
}

// -----------------------------------------------------------------------------
// Per-form checkers
// -----------------------------------------------------------------------------

fn check_var(name: &str, span: Span, env: &Env, mode: ConsumeMode) -> Result<Check, Error> {
    let typed = env.get(name).cloned().ok_or_else(|| {
        Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!("unbound variable `{name}`")),
        )
    })?;

    match mode {
        ConsumeMode::Consume => {
            // Linear variables consume; shared/ordinary do not.
            let mut consumes = BTreeMap::new();
            if typed.usage == Usage::Linear {
                consumes.insert(name.to_string(), span);
            }
            Ok(Check {
                typed,
                consumes,
                borrows: BTreeMap::new(),
            })
        }
        ConsumeMode::Borrow => {
            // Linear variables become shared views and record a borrow;
            // shared/ordinary behave the same as in Consume mode.
            if typed.usage == Usage::Linear {
                let mut borrows = BTreeMap::new();
                borrows.insert(name.to_string(), span);
                Ok(Check {
                    typed: Typed {
                        usage: Usage::Shared,
                        ty: typed.ty,
                    },
                    consumes: BTreeMap::new(),
                    borrows,
                })
            } else {
                Ok(Check {
                    typed,
                    consumes: BTreeMap::new(),
                    borrows: BTreeMap::new(),
                })
            }
        }
    }
}

fn check_add(lhs: &Expr, rhs: &Expr, env: &Env, mode: ConsumeMode) -> Result<Check, Error> {
    // Both operands in outer mode, expected to be ordinary int.
    let l = check_expr(lhs, env, Some(Usage::Ordinary), mode)?;
    require_ordinary_int(&l.typed, lhs.span(), "left operand of `+`")?;
    let r = check_expr(rhs, env, Some(Usage::Ordinary), mode)?;
    require_ordinary_int(&r.typed, rhs.span(), "right operand of `+`")?;
    let consumes = merge_consumes_disjoint(l.consumes, r.consumes)?;
    let borrows = merge_borrows_union(l.borrows, r.borrows);
    Ok(Check {
        typed: Typed {
            usage: Usage::Ordinary,
            ty: Type::Int,
        },
        consumes,
        borrows,
    })
}

fn check_seq(
    first: &Expr,
    second: &Expr,
    env: &Env,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // e1 always in Borrow mode (the borrow scope is e1; e2 sees `e1` as
    // having "completed and discarded").
    let f = check_expr(first, env, Some(Usage::Ordinary), ConsumeMode::Borrow)?;
    if f.typed.usage != Usage::Ordinary {
        let why = match f.typed.usage {
            Usage::Linear => "would silently leak (Phase 0/1 invariant)",
            Usage::Shared => "would escape its borrow scope (Phase 2 invariant)",
            Usage::Ordinary => unreachable!(),
        };
        return Err(Error::new(
            ErrorCategory::TypeError,
            first.span(),
            Some(format!(
                "the first expression in a sequence `e1 ; e2` must produce an `ordinary` value; this one produces `{}` and {why}",
                f.typed.usage,
            )),
        ));
    }

    // e2 inherits the outer mode and outer expected_usage.
    let s = check_expr(second, env, expected_usage, mode)?;

    // Disjointness: e1's consumes vs e2's consumes; e1's consumes vs e2's
    // borrows (e2 can't borrow what e1 just consumed).
    let consumes = merge_consumes_disjoint(f.consumes.clone(), s.consumes.clone())?;
    if let Some((name, b_span, c_span)) = find_consume_borrow_collision(&f.consumes, &s.borrows) {
        return Err(Error::new(
            ErrorCategory::TypeError,
            b_span,
            Some(format!(
                "the right-hand side of `;` cannot borrow `{name}` because it was consumed by the left-hand side"
            )),
        )
        .with_related(c_span, format!("`{name}` was consumed here")));
    }

    // Result borrows: B1 minus anything e2 consumed (those nested borrows
    // got "resolved" by the consume) ∪ B2.
    let mut b1 = f.borrows;
    for name in s.consumes.keys() {
        b1.remove(name);
    }
    let borrows = merge_borrows_union(b1, s.borrows);

    Ok(Check {
        typed: s.typed,
        consumes,
        borrows,
    })
}

/// Tuple construction. The `mode` parameter is threaded through to
/// components but doesn't change the accept/reject set for ordinary
/// tuples in Phase 2: any linear-var component in Borrow mode becomes
/// shared, and the ordinary-tuple's "components must be ordinary" check
/// rejects shared just as it would have rejected linear in Consume mode.
/// Mode is here for forward-compat — Phase 3+'s method calls will pass
/// linear arguments to ordinary-typed callees in Borrow mode, and the
/// var rule's mode-sensitivity will start mattering at component
/// granularity then. Don't drop the parameter.
fn check_tuple_construction(
    elems: &[Expr],
    env: &Env,
    expected_usage: Option<Usage>,
    span: Span,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    let outer_usage = match expected_usage {
        Some(Usage::Linear) => Usage::Linear,
        Some(Usage::Ordinary) | None => Usage::Ordinary,
        Some(Usage::Shared) => {
            return Err(Error::new(
                ErrorCategory::TypeError,
                span,
                Some(
                    "shared tuples are produced by borrowing, not by direct construction"
                        .to_string(),
                ),
            ));
        }
    };

    let component_expected = match outer_usage {
        Usage::Linear => None,
        Usage::Ordinary => Some(Usage::Ordinary),
        Usage::Shared => unreachable!(),
    };

    let mut field_types = Vec::with_capacity(elems.len());
    let mut consumes: BTreeMap<String, Span> = BTreeMap::new();
    let mut borrows: BTreeMap<String, Span> = BTreeMap::new();
    for elem in elems {
        let c = check_expr(elem, env, component_expected, mode)?;
        if outer_usage == Usage::Ordinary && c.typed.usage != Usage::Ordinary {
            return Err(Error::new(
                ErrorCategory::TypeError,
                elem.span(),
                Some(format!(
                    "an ordinary tuple may only contain ordinary values; this component has usage `{}`",
                    c.typed.usage,
                )),
            ));
        }
        field_types.push(TupleField {
            usage: c.typed.usage,
            ty: c.typed.ty,
        });
        consumes = merge_consumes_disjoint(consumes, c.consumes)?;
        borrows = merge_borrows_union(borrows, c.borrows);
    }

    Ok(Check {
        typed: Typed {
            usage: outer_usage,
            ty: Type::Tuple(field_types),
        },
        consumes,
        borrows,
    })
}

fn check_projection(
    tuple: &Expr,
    index: usize,
    span: Span,
    env: &Env,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // Inner in outer mode, expected_usage = None (projection accepts
    // ordinary or shared input; mode + variable rule decide which).
    let t = check_expr(tuple, env, None, mode)?;

    if t.typed.usage == Usage::Linear {
        // In Consume mode, projecting a linear var lookup keeps the var
        // linear. The user wanted an implicit borrow; tell them why this
        // didn't happen. (In Borrow mode the lookup would have already
        // produced shared, so we never reach this branch.)
        return Err(Error::new(
            ErrorCategory::TypeError,
            tuple.span(),
            Some(
                "cannot project from a `linear` tuple; project from a borrowed (shared) view instead"
                    .to_string(),
            ),
        ));
    }

    let fields = match &t.typed.ty {
        Type::Tuple(fields) => fields,
        _ => {
            return Err(Error::new(
                ErrorCategory::TypeError,
                tuple.span(),
                Some(format!(
                    "cannot project field `.{index}` from non-tuple type `{}`",
                    t.typed.ty,
                )),
            ));
        }
    };

    if index >= fields.len() {
        return Err(Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!(
                "tuple index {index} out of bounds (tuple has {} field{})",
                fields.len(),
                if fields.len() == 1 { "" } else { "s" },
            )),
        ));
    }

    let field_usage = share_as(t.typed.usage, fields[index].usage);
    Ok(Check {
        typed: Typed {
            usage: field_usage,
            ty: fields[index].ty.clone(),
        },
        consumes: t.consumes,
        borrows: t.borrows,
    })
}

fn check_destructure(
    names: &[(String, Span)],
    value: &Expr,
    body: &Expr,
    _span: Span,
    env: &Env,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // Value always in Consume mode — deconstruction needs to own the
    // tuple. Outer mode is irrelevant here.
    let v = check_expr(value, env, Some(Usage::Linear), ConsumeMode::Consume)?;

    if v.typed.usage != Usage::Linear {
        return Err(Error::new(
            ErrorCategory::TypeError,
            value.span(),
            Some(format!(
                "deconstruction `let (...) = e in ...` requires a linear tuple, found `{}`",
                v.typed,
            )),
        ));
    }

    let fields = match &v.typed.ty {
        Type::Tuple(fields) => fields.clone(),
        _ => {
            return Err(Error::new(
                ErrorCategory::TypeError,
                value.span(),
                Some(format!(
                    "deconstruction requires a tuple type, found `{}`",
                    v.typed.ty,
                )),
            ));
        }
    };

    if fields.len() != names.len() {
        return Err(Error::new(
            ErrorCategory::TypeError,
            value.span(),
            Some(format!(
                "deconstruction pattern has {} name{} but value is a tuple with {} field{}",
                names.len(),
                if names.len() == 1 { "" } else { "s" },
                fields.len(),
                if fields.len() == 1 { "" } else { "s" },
            )),
        ));
    }

    let mut body_env = env.clone();
    for ((name, _), field) in names.iter().zip(fields.iter()) {
        body_env.insert(
            name.clone(),
            Typed {
                usage: field.usage,
                ty: field.ty.clone(),
            },
        );
    }

    let b = check_expr(body, &body_env, expected_usage, mode)?;

    // Linear field bindings must be consumed in body.
    for ((name, name_span), field) in names.iter().zip(fields.iter()) {
        if field.usage == Usage::Linear && !b.consumes.contains_key(name) {
            return Err(Error::new(
                ErrorCategory::TypeError,
                *name_span,
                Some(format!(
                    "linear field binding `{name}` is not consumed in the body"
                )),
            ));
        }
    }

    // Disjointness: C1 ∩ keys(B2) — body can't borrow what value consumed.
    if let Some((name, b_span, c_span)) = find_consume_borrow_collision(&v.consumes, &b.borrows) {
        return Err(Error::new(
            ErrorCategory::TypeError,
            b_span,
            Some(format!(
                "the body of deconstruction cannot borrow `{name}`: the deconstruction's value already consumed it"
            )),
        )
        .with_related(c_span, format!("`{name}` was consumed here")));
    }

    // Disjointness: B1 ∩ keys(C2) — body can't consume what value borrows.
    // (e1's borrows are not in scope for e2 to resolve via consumption,
    // unlike the let-linear/ordinary case.)
    if let Some((name, c_span, b_span)) = find_borrow_consume_collision(&v.borrows, &b.consumes) {
        return Err(Error::new(
            ErrorCategory::TypeError,
            c_span,
            Some(format!(
                "the body cannot consume `{name}`: it is borrowed by the deconstruction's value"
            )),
        )
        .with_related(b_span, format!("`{name}` is borrowed here")));
    }

    // Strip the field bindings from body's consumes (per the rule
    // `C2 \ {x1, ..., xn}`) before merging — these are local to the
    // deconstruction.
    let mut body_consumes = b.consumes;
    let mut body_borrows = b.borrows;
    for (name, _) in names {
        body_consumes.remove(name);
        body_borrows.remove(name);
    }

    let consumes = merge_consumes_disjoint(v.consumes, body_consumes)?;
    // B1 propagates entirely (no `\ keys(C2)`); body's borrows merged on top.
    let borrows = merge_borrows_union(v.borrows, body_borrows);

    Ok(Check {
        typed: b.typed,
        consumes,
        borrows,
    })
}

#[allow(clippy::too_many_arguments)]
fn check_let(
    usage: Usage,
    name: &str,
    name_span: Span,
    value: &Expr,
    body: &Expr,
    _span: Span,
    env: &Env,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // Mode and expected_usage for `e1` depend on the annotation.
    let (value_mode, value_expected) = match usage {
        Usage::Linear | Usage::Ordinary => (ConsumeMode::Consume, Some(usage)),
        Usage::Shared => {
            // For shared lets, e1 runs in Borrow mode; expected_usage =
            // None (don't drive tuple construction toward shared).
            (ConsumeMode::Borrow, None)
        }
    };

    let v = check_expr(value, env, value_expected, value_mode)?;

    if v.typed.usage != usage {
        return Err(Error::new(
            ErrorCategory::TypeError,
            value.span(),
            Some(format!(
                "let-binding annotated `{usage}` but value has usage `{}`",
                v.typed.usage,
            )),
        ));
    }

    let mut body_env = env.clone();
    body_env.insert(name.to_string(), v.typed.clone());

    let b = check_expr(body, &body_env, expected_usage, mode)?;

    // Linear binding must be consumed in the body.
    if usage == Usage::Linear && !b.consumes.contains_key(name) {
        return Err(Error::new(
            ErrorCategory::TypeError,
            name_span,
            Some(format!("linear binding `{name}` is not consumed in the body")),
        ));
    }

    // Disjointness: C1 ∩ C2 (handled by merge), C1 ∩ keys(B2).
    if let Some((conflict_name, b_span, c_span)) =
        find_consume_borrow_collision(&v.consumes, &b.borrows)
    {
        return Err(Error::new(
            ErrorCategory::TypeError,
            b_span,
            Some(format!(
                "the body of let cannot borrow `{conflict_name}`: the let's value already consumed it"
            )),
        )
        .with_related(c_span, format!("`{conflict_name}` was consumed here")));
    }

    // For shared lets, also check B1 ∩ keys(C2): the body cannot consume a
    // variable that the shared binding is borrowing.
    let outgoing_value_borrows = if usage == Usage::Shared {
        if let Some((conflict_name, c_span, b_span)) =
            find_borrow_consume_collision(&v.borrows, &b.consumes)
        {
            return Err(Error::new(
                ErrorCategory::TypeError,
                c_span,
                Some(format!(
                    "cannot consume `{conflict_name}` while the shared binding `{name}` borrows it"
                )),
            )
            .with_related(b_span, format!("`{conflict_name}` is borrowed by `{name}` here")));
        }
        // Shared binding extends e1's borrows through the let; B' = B1.
        v.borrows
    } else {
        // For linear/ordinary: e1's borrows can be resolved by anything
        // e2 consumed (their scope ended at the end of e1).
        let mut b1 = v.borrows;
        for n in b.consumes.keys() {
            b1.remove(n);
        }
        b1
    };

    // Strip the bound name from body's maps before merging (it's local to
    // the let).
    let mut body_consumes = b.consumes;
    let mut body_borrows = b.borrows;
    body_consumes.remove(name);
    body_borrows.remove(name);

    let consumes = merge_consumes_disjoint(v.consumes, body_consumes)?;
    let borrows = merge_borrows_union(outgoing_value_borrows, body_borrows);

    Ok(Check {
        typed: b.typed,
        consumes,
        borrows,
    })
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn require_ordinary_int(typed: &Typed, span: Span, what: &str) -> Result<(), Error> {
    if typed.usage == Usage::Ordinary && typed.ty == Type::Int {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!(
                "{what} must have type `ordinary int`, found `{typed}`"
            )),
        ))
    }
}

/// `share_as(outer_usage, field_usage)` per the Phase 2 projection rule:
/// projecting through an ordinary tuple keeps each field's usage; through
/// a shared tuple, every linear/shared field becomes shared.
fn share_as(outer: Usage, inner: Usage) -> Usage {
    match (outer, inner) {
        // Ordinary tuple: field's own usage stands.
        (Usage::Ordinary, u) => u,
        // Shared tuple: ordinary stays ordinary; linear/shared collapse to shared.
        (Usage::Shared, Usage::Ordinary) => Usage::Ordinary,
        (Usage::Shared, Usage::Linear) | (Usage::Shared, Usage::Shared) => Usage::Shared,
        // Linear outer is rejected by the projection rule before reaching here.
        (Usage::Linear, _) => unreachable!("projection rule rejects linear outer"),
    }
}

/// Merge two consumes maps. On collision, error: the variable was
/// consumed twice. Primary span at the second site, related at the first.
fn merge_consumes_disjoint(
    mut a: BTreeMap<String, Span>,
    b: BTreeMap<String, Span>,
) -> Result<BTreeMap<String, Span>, Error> {
    for (name, b_span) in b {
        if let Some(&a_span) = a.get(&name) {
            return Err(Error::new(
                ErrorCategory::TypeError,
                b_span,
                Some(format!("linear variable `{name}` is consumed twice")),
            )
            .with_related(a_span, format!("`{name}` previously consumed here")));
        }
        a.insert(name, b_span);
    }
    Ok(a)
}

/// Merge two borrows maps. **No disjointness check** — multiple borrows of
/// the same variable are valid (that's the whole point of `shared`). On a
/// name collision, the *first-seen* span is kept (the existing entry
/// wins; the new one is dropped). This is a deliberate choice — it gives
/// stable error spans regardless of which subexpression was processed
/// last. Don't accidentally flip this to last-seen.
fn merge_borrows_union(
    mut a: BTreeMap<String, Span>,
    b: BTreeMap<String, Span>,
) -> BTreeMap<String, Span> {
    for (name, span) in b {
        a.entry(name).or_insert(span);
    }
    a
}

/// Find a name that appears in both `consumes` and `borrows`. Returns
/// `(name, borrow_span, consume_span)` for the first conflict, or `None`.
/// Used by the seq / decon / let rules' `C ∩ keys(B) = ∅` checks.
fn find_consume_borrow_collision(
    consumes: &BTreeMap<String, Span>,
    borrows: &BTreeMap<String, Span>,
) -> Option<(String, Span, Span)> {
    for (name, &b_span) in borrows {
        if let Some(&c_span) = consumes.get(name) {
            return Some((name.clone(), b_span, c_span));
        }
    }
    None
}

/// Find a name that appears in both `borrows` and `consumes`. Returns
/// `(name, consume_span, borrow_span)` for the first conflict, or `None`.
/// Used by the decon / let-shared rules' `B ∩ keys(C) = ∅` checks where
/// the consume site is the action being rejected.
fn find_borrow_consume_collision(
    borrows: &BTreeMap<String, Span>,
    consumes: &BTreeMap<String, Span>,
) -> Option<(String, Span, Span)> {
    for (name, &c_span) in consumes {
        if let Some(&b_span) = borrows.get(name) {
            return Some((name.clone(), c_span, b_span));
        }
    }
    None
}

// -----------------------------------------------------------------------------
// Inline structural tests. End-to-end accept/reject coverage lives in the
// test corpus.
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn check_ok(src: &str) -> Typed {
        let prog = parse(src).expect("parse failed");
        check_program(&prog).expect("type check failed")
    }

    fn check_err(src: &str) -> Error {
        let prog = parse(src).expect("parse failed");
        check_program(&prog).expect_err("expected type error")
    }

    /// For tests that need to inspect the `Check` struct directly.
    fn run_check(src: &str) -> Check {
        let prog = parse(src).expect("parse failed");
        let env = Env::new();
        check_expr(&prog.main_expr, &env, None, ConsumeMode::Consume)
            .expect("type check failed")
    }

    // -- Phase 0 carry-over --

    #[test]
    fn int_literal_is_ordinary_int() {
        let typed = check_ok("42");
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn negative_literal_is_ordinary_int() {
        let typed = check_ok("-3");
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn add_of_two_ints_is_int() {
        let typed = check_ok("1 + 2");
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn let_binding_resolves_var() {
        let typed = check_ok("let ordinary x = 1 in x + 1");
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn nested_let_binding() {
        let typed = check_ok("let ordinary x = 1 in let ordinary y = x + 2 in y + x");
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn unbound_variable_is_type_error() {
        let err = check_err("x");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("unbound"));
    }

    #[test]
    fn unbound_variable_inside_add_is_type_error() {
        let err = check_err("1 + missing");
        assert_eq!(err.category, ErrorCategory::TypeError);
    }

    #[test]
    fn let_body_sees_binding() {
        let typed = check_ok("let ordinary x = 100 in x");
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn outer_binding_does_not_escape() {
        let err = check_err("let ordinary x = y in x");
        assert_eq!(err.category, ErrorCategory::TypeError);
    }

    #[test]
    fn typed_display_phase0() {
        let typed = check_ok("1 + 1");
        assert_eq!(format!("{typed}"), "ordinary int");
    }

    #[test]
    fn shadowing_uses_inner_binding() {
        let typed = check_ok("let ordinary x = 1 in let ordinary x = x + 1 in x");
        assert_eq!(typed.ty, Type::Int);
    }

    // -- Phase 1 carry-over (consumes empty regression) --

    #[test]
    fn ordinary_program_has_empty_consumes_and_borrows() {
        let check = run_check("let ordinary x = 1 in x + 1");
        assert!(check.consumes.is_empty());
        assert!(check.borrows.is_empty());
    }

    #[test]
    fn linear_tuple_decon() {
        let typed = check_ok("let linear t = (1, 2) in let (a, b) = t in a + b");
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn empty_tuple_defaults_to_ordinary() {
        let typed = check_ok("()");
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Tuple(Vec::new()));
    }

    #[test]
    fn empty_linear_tuple_via_annotation() {
        let typed = check_ok("let linear u = () in let () = u in 5");
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn ordinary_tuple_projection() {
        let typed = check_ok("let ordinary t = (1, 2) in t.0 + t.1");
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn nested_linear_tuple() {
        let typed = check_ok(
            "let linear inner = (1, 2) in \
             let linear outer = (inner, 3) in \
             let (i, x) = outer in \
             let (a, b) = i in a + b + x",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn linear_unused_is_type_error() {
        let err = check_err("let linear t = (1, 2) in 5");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("not consumed"));
    }

    #[test]
    fn linear_consumed_twice_includes_related_span() {
        let err = check_err(
            "let linear t = (1, 2) in let (a, b) = t in let (c, d) = t in a + b + c + d",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("consumed twice"));
        assert_eq!(err.related.len(), 1);
        assert!(err.related[0].label.contains("previously consumed"));
        assert_ne!(err.related[0].span, err.span);
    }

    #[test]
    fn project_linear_is_type_error() {
        // Phase 2: projecting from a linear var (in Consume mode) is
        // still a type error — the user should borrow first. The Phase 1
        // message changed slightly; assert intent rather than exact text.
        let err = check_err("let linear t = (1, 2) in t.0");
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(note.contains("linear") && note.contains("project"));
    }

    #[test]
    fn decon_ordinary_is_type_error() {
        let err = check_err("let ordinary t = (1, 2) in let (a, b) = t in a + b");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("linear tuple"));
    }

    /// Phase 1 spelling: "must not produce a linear value". Phase 2: the
    /// same program now rejects because `t` becomes shared in Borrow mode
    /// and the seq rule rejects shared. Loosen the assertion to match the
    /// rule's intent (sequencing rejecting based on e1's usage), not the
    /// specific blocking usage word.
    #[test]
    fn seq_linear_is_type_error() {
        let err = check_err("let linear t = (1, 2) in t ; 5");
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(
            note.contains("sequence") || note.contains("first expression"),
            "expected a sequencing-related message, got: {note}",
        );
    }

    // -- Phase 2 specific --

    #[test]
    fn linear_var_in_borrow_mode_returns_shared() {
        // Build env directly to bypass the parser.
        let mut env = Env::new();
        env.insert(
            "y".to_string(),
            Typed {
                usage: Usage::Linear,
                ty: Type::Int, // shape doesn't matter for the lookup
            },
        );
        let var_expr = Expr::Var {
            name: "y".to_string(),
            span: Span::new(0, 1),
        };
        let in_consume =
            check_expr(&var_expr, &env, None, ConsumeMode::Consume).unwrap();
        assert_eq!(in_consume.typed.usage, Usage::Linear);
        assert_eq!(in_consume.consumes.len(), 1);
        assert!(in_consume.borrows.is_empty());

        let in_borrow =
            check_expr(&var_expr, &env, None, ConsumeMode::Borrow).unwrap();
        assert_eq!(in_borrow.typed.usage, Usage::Shared);
        assert!(in_borrow.consumes.is_empty());
        assert_eq!(in_borrow.borrows.len(), 1);
    }

    #[test]
    fn seq_borrow_basic_resolves_borrow() {
        // The seq's e1 borrows y; e2 consumes y; the borrow gets resolved
        // by the consume so the outer let is happy.
        let typed = check_ok(
            "let linear y = (1, 2) in (y.0 + y.1) ; let (a, b) = y in a + b",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn seq_borrow_repeated_in_e1() {
        let typed = check_ok(
            "let linear y = (1, 2) in (y.0 + y.0 + y.1) ; let (a, b) = y in a + b",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn let_shared_basic() {
        let typed = check_ok(
            "let linear y = (1, 2) in (let shared s = y in s.0 + s.0) ; let (a, b) = y in a + b",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn nested_let_shared() {
        let typed = check_ok(
            "let linear y = (1, 2) in \
             (let shared s1 = y in let shared s2 = y in s1.0 + s2.1) ; \
             let (a, b) = y in a + b",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn share_escape_is_type_error() {
        // `let shared s = y` borrows y; the body then tries to consume y
        // via deconstruction. The let-shared rule's `B1 ∩ keys(C2)` check
        // catches this.
        let err = check_err(
            "let linear y = (1, 2) in let shared s = y in let (a, b) = y in a + b",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(note.contains("shared") || note.contains("borrow"));
        // Should have a related span pointing at where y was borrowed.
        assert!(!err.related.is_empty());
    }

    #[test]
    fn borrow_unresolved_falls_through_to_linear_unused() {
        // `(y.0 + y.0) ; 5` — the borrow of y is *not* resolved (e2 doesn't
        // consume y). The outer linear-must-be-consumed check fires
        // because y was never consumed at all.
        let err = check_err("let linear y = (1, 2) in (y.0 + y.0) ; 5");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("not consumed"));
    }

    #[test]
    fn seq_returning_shared_is_type_error() {
        // Phase 2-specific: `t ; 5` where t is linear. In Borrow mode, t
        // becomes shared. seq rejects because u1 is shared.
        let err = check_err("let linear t = (1, 2) in t ; 5");
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(
            note.contains("ordinary") && (note.contains("shared") || note.contains("escape")),
            "expected a shared-escape message, got: {note}",
        );
    }

    #[test]
    fn share_as_table() {
        // Direct unit test of the `share_as` function.
        assert_eq!(share_as(Usage::Ordinary, Usage::Ordinary), Usage::Ordinary);
        assert_eq!(share_as(Usage::Ordinary, Usage::Linear), Usage::Linear);
        assert_eq!(share_as(Usage::Ordinary, Usage::Shared), Usage::Shared);
        assert_eq!(share_as(Usage::Shared, Usage::Ordinary), Usage::Ordinary);
        assert_eq!(share_as(Usage::Shared, Usage::Linear), Usage::Shared);
        assert_eq!(share_as(Usage::Shared, Usage::Shared), Usage::Shared);
    }

    #[test]
    fn merge_borrows_first_seen_wins() {
        let mut a = BTreeMap::new();
        a.insert("y".to_string(), Span::new(0, 1));
        let mut b = BTreeMap::new();
        b.insert("y".to_string(), Span::new(10, 11));
        let merged = merge_borrows_union(a, b);
        // First-seen (the entry from `a`) should win.
        assert_eq!(merged.get("y").copied(), Some(Span::new(0, 1)));
    }
}
