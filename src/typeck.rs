//! Type checker for the mini-linear language. Phases 0 + 1.
//!
//! Phase 1 implements the consumption-tracking algorithm from §2.6 of the
//! paper, *without* borrowing. Each call to [`check_expr`] returns a
//! [`Check`] containing both the resulting [`Typed`] value and a `consumes`
//! map recording which linear variables were consumed (and *where* — so
//! "consumed twice" errors can point at both sites).
//!
//! Top-level invariant (per the Phase 1 handoff): with no methods or
//! parameters, the initial environment is empty, the let rule's
//! linear-must-be-consumed check transitively prevents leakage, and the
//! top-level program's `consumes` set is always empty.

use std::collections::BTreeMap;
use std::fmt;

use crate::ast::{Expr, Program, Span, TupleField, Type, Usage};
use crate::error::{Error, ErrorCategory};

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// A typed value: a usage qualifier plus a type. This is what the paper
/// writes as `u τ`.
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

/// The expected consumption mode passed down into each subexpression. The
/// paper calls this `c`. Phase 2 will add a `Borrow` variant alongside
/// shared usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsumeMode {
    Consume,
}

/// Type-check a complete program.
pub fn check_program(program: &Program) -> Result<Typed, Error> {
    debug_assert!(
        program.decls.is_empty(),
        "Phase 0/1 parser does not produce declarations",
    );
    let env = Env::new();
    let result = check_expr(&program.main_expr, &env, None, ConsumeMode::Consume)?;
    debug_assert!(
        result.consumes.is_empty(),
        "top-level program's consumes set must be empty (initial env is empty, so there's nothing to consume)",
    );
    Ok(result.typed)
}

// -----------------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------------

/// Variable typing environment `X` from the paper.
type Env = BTreeMap<String, Typed>;

/// Per-expression checking result. `consumes` maps each linear variable
/// name consumed by this expression to the span where it was consumed
/// (i.e. the use-site). The map shape — rather than a plain set — lets
/// disjointness-violation errors point at *both* consume sites.
#[derive(Debug, Clone)]
struct Check {
    typed: Typed,
    consumes: BTreeMap<String, Span>,
}

impl Check {
    fn ordinary_int() -> Self {
        Self {
            typed: Typed {
                usage: Usage::Ordinary,
                ty: Type::Int,
            },
            consumes: BTreeMap::new(),
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

        Expr::Var { name, span } => check_var(name, *span, env),

        Expr::Add { lhs, rhs, .. } => check_add(lhs, rhs, env),

        Expr::Seq {
            first, second, ..
        } => check_seq(first, second, env, expected_usage, mode),

        Expr::Tuple { elems, span } => {
            check_tuple_construction(elems, env, expected_usage, *span)
        }

        Expr::TupleProj {
            tuple,
            index,
            span,
        } => check_projection(tuple, *index, *span, env),

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

        // Phase 2+ adds methods, datatypes, match. Defensive arm for
        // programmatically-constructed ASTs.
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

fn check_var(name: &str, span: Span, env: &Env) -> Result<Check, Error> {
    let typed = env.get(name).cloned().ok_or_else(|| {
        Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!("unbound variable `{name}`")),
        )
    })?;
    let mut consumes = BTreeMap::new();
    if typed.usage == Usage::Linear {
        consumes.insert(name.to_string(), span);
    }
    Ok(Check { typed, consumes })
}

fn check_add(lhs: &Expr, rhs: &Expr, env: &Env) -> Result<Check, Error> {
    let l = check_expr(lhs, env, Some(Usage::Ordinary), ConsumeMode::Consume)?;
    require_ordinary_int(&l.typed, lhs.span(), "left operand of `+`")?;
    let r = check_expr(rhs, env, Some(Usage::Ordinary), ConsumeMode::Consume)?;
    require_ordinary_int(&r.typed, rhs.span(), "right operand of `+`")?;
    let consumes = merge_disjoint(l.consumes, r.consumes)?;
    Ok(Check {
        typed: Typed {
            usage: Usage::Ordinary,
            ty: Type::Int,
        },
        consumes,
    })
}

fn check_seq(
    first: &Expr,
    second: &Expr,
    env: &Env,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // e1's expected_usage is Some(Ordinary) because e1's value will be
    // discarded — a linear value here would be silently leaked.
    let f = check_expr(first, env, Some(Usage::Ordinary), ConsumeMode::Consume)?;
    if f.typed.usage == Usage::Linear {
        return Err(Error::new(
            ErrorCategory::TypeError,
            first.span(),
            Some(format!(
                "the first expression in a sequence `e1 ; e2` must not produce a linear value (it would be silently discarded), found `{}`",
                f.typed,
            )),
        ));
    }
    // e2 inherits the outer expected_usage and outer mode.
    let s = check_expr(second, env, expected_usage, mode)?;
    let consumes = merge_disjoint(f.consumes, s.consumes)?;
    Ok(Check {
        typed: s.typed,
        consumes,
    })
}

fn check_tuple_construction(
    elems: &[Expr],
    env: &Env,
    expected_usage: Option<Usage>,
    span: Span,
) -> Result<Check, Error> {
    // The handoff: Some(Linear) → linear tuple, components free; Some(Ord)
    // or None → ordinary tuple, components must be ordinary; Some(Shared)
    // → unreachable in Phase 1 but defensively rejected.
    let outer_usage = match expected_usage {
        Some(Usage::Linear) => Usage::Linear,
        Some(Usage::Ordinary) | None => Usage::Ordinary,
        Some(Usage::Shared) => {
            return Err(Error::new(
                ErrorCategory::TypeError,
                span,
                Some("shared tuples are not supported until Phase 2".to_string()),
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
    for elem in elems {
        let c = check_expr(elem, env, component_expected, ConsumeMode::Consume)?;
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
        consumes = merge_disjoint(consumes, c.consumes)?;
    }

    Ok(Check {
        typed: Typed {
            usage: outer_usage,
            ty: Type::Tuple(field_types),
        },
        consumes,
    })
}

fn check_projection(
    tuple: &Expr,
    index: usize,
    span: Span,
    env: &Env,
) -> Result<Check, Error> {
    let t = check_expr(tuple, env, Some(Usage::Ordinary), ConsumeMode::Consume)?;

    if t.typed.usage != Usage::Ordinary {
        return Err(Error::new(
            ErrorCategory::TypeError,
            tuple.span(),
            Some(format!(
                "cannot project from a `{}` tuple; Phase 1 only supports projection from ordinary tuples",
                t.typed.usage,
            )),
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

    // Phase 1 invariant: ordinary tuples have ordinary fields. This is
    // already enforced at construction, but verifying here keeps the
    // projection rule self-contained.
    debug_assert!(
        fields.iter().all(|f| f.usage == Usage::Ordinary),
        "ordinary tuple has a non-ordinary field — invariant violation",
    );

    Ok(Check {
        typed: Typed {
            usage: Usage::Ordinary,
            ty: fields[index].ty.clone(),
        },
        consumes: t.consumes,
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
    // Value must produce a linear tuple.
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

    // Per the rule: consumes = C1 ∪ (C2 \ {x1, ..., xn}).
    // Removing the field names from the body's consumes is what makes the
    // disjointness check meaningful (any name still left after removal is
    // an "outer" consume that conflicts with C1 if shared with v).
    let mut body_consumes = b.consumes;
    for (name, _) in names {
        body_consumes.remove(name);
    }
    let consumes = merge_disjoint(v.consumes, body_consumes)?;

    Ok(Check {
        typed: b.typed,
        consumes,
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
    // Phase 1 supports `ordinary` and `linear`. `shared` is reserved at the
    // parser, but defensively reject here too in case of programmatic AST
    // construction.
    if usage == Usage::Shared {
        return Err(Error::new(
            ErrorCategory::TypeError,
            name_span,
            Some("`shared` let bindings are not supported until Phase 2".to_string()),
        ));
    }

    let v = check_expr(value, env, Some(usage), ConsumeMode::Consume)?;

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

    // If linear, the binding must be consumed in the body.
    if usage == Usage::Linear && !b.consumes.contains_key(name) {
        return Err(Error::new(
            ErrorCategory::TypeError,
            name_span,
            Some(format!("linear binding `{name}` is not consumed in the body")),
        ));
    }

    // consumes = C1 ∪ (C2 \ {x})
    let mut body_consumes = b.consumes;
    body_consumes.remove(name);
    let consumes = merge_disjoint(v.consumes, body_consumes)?;

    Ok(Check {
        typed: b.typed,
        consumes,
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

/// Merge two consumes maps. If a name appears in both, that's a
/// linearity violation: the variable was consumed twice, once at each
/// recorded span. The resulting error points at the *second* (later)
/// site as primary, with a related-span pointer at the *first* site.
fn merge_disjoint(
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

    /// For tests that need to inspect the `Check` struct directly (rather
    /// than just the program-level result).
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

    // -- Phase 1: consumes regression --

    /// Phase 0 carry-forward: `consumes` must be empty for a program that
    /// uses no linear values. Smallest test that the new field doesn't
    /// silently break Phase 0–style code paths.
    #[test]
    fn ordinary_program_has_empty_consumes() {
        let check = run_check("let ordinary x = 1 in x + 1");
        assert!(check.consumes.is_empty());
    }

    #[test]
    fn linear_var_use_records_consume() {
        // A bare `t` reference to a linear variable: consumes should
        // contain `t` mapped to the use-site span.
        let prog = parse("let linear t = () in let () = t in 5").expect("parse");
        // Pull out the inner `let () = t in 5` to inspect its consumes.
        let Expr::Let { body, .. } = &prog.main_expr else {
            panic!("expected outer let");
        };
        let Expr::TupleDestructure { value, .. } = body.as_ref() else {
            panic!("expected destructure");
        };
        // value is `t` — a Var. Build a minimal env.
        let mut env = Env::new();
        env.insert(
            "t".to_string(),
            Typed {
                usage: Usage::Linear,
                ty: Type::Tuple(Vec::new()),
            },
        );
        let check = check_expr(value, &env, Some(Usage::Linear), ConsumeMode::Consume).unwrap();
        assert_eq!(check.consumes.len(), 1);
        assert!(check.consumes.contains_key("t"));
    }

    // -- Phase 1: linear / tuple happy paths --

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
        // `let linear u = ()` should bind u as `linear ()`.
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

    // -- Phase 1: rejection paths --

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
        // The related span should NOT be the same as the primary span.
        assert_ne!(err.related[0].span, err.span);
    }

    #[test]
    fn project_linear_is_type_error() {
        let err = check_err("let linear t = (1, 2) in t.0");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("linear"));
    }

    #[test]
    fn decon_ordinary_is_type_error() {
        let err = check_err("let ordinary t = (1, 2) in let (a, b) = t in a + b");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("linear tuple"));
    }

    #[test]
    fn seq_linear_is_type_error() {
        let err = check_err("let linear t = (1, 2) in t ; 5");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("linear"));
    }

    #[test]
    fn linear_in_ordinary_tuple_is_type_error() {
        let err = check_err("let linear t = (1, 2) in let ordinary outer = (t, 3) in 5");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("ordinary tuple"));
    }

    #[test]
    fn int_literal_as_linear_is_type_error() {
        let err = check_err("let linear x = 5 in 6");
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(note.contains("linear") && note.contains("ordinary"));
    }
}
