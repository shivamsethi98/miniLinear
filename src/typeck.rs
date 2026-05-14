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

use crate::ast::{Decl, Expr, ParamSig, Program, Span, TupleField, Type, Usage};
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

/// Per-method information available during type checking. The handoff
/// suggests `BTreeMap<String, MethodSig>`, but a small wrapper here lets
/// us carry the precomputed `effective_returns` (declared returns ++ the
/// inout-rebind contributions) and the declaration's name span — the
/// latter is needed for "method declared more than once" errors that point
/// at both decl sites.
/// Per-method information available during type checking.
#[derive(Debug, Clone)]
pub(crate) struct MethodEntry {
    /// Declared parameter signatures (names dropped, inout flag preserved).
    pub(crate) params: Vec<ParamSig>,
    /// Effective returns: `declared_returns ++ [(p.usage, p.ty) for p in params if p.inout]`.
    /// This is what the call-site rule reads; bodies expect a value
    /// matching the type derived from these.
    pub(crate) effective_returns: Vec<ParamSig>,
    /// Span of the method's declared name, for "method declared here"
    /// notes attached to duplicate-name errors.
    pub(crate) name_span: Span,
}

/// Method environment: name → method entry.
pub(crate) type MethodEnv = BTreeMap<String, MethodEntry>;

/// Pass 1: walk every method declaration in `decls`, validate signature
/// invariants (no `shared` returns; no duplicate method names), and
/// return the populated method env. Bodies are *not* checked; that's pass
/// 2's job.
pub(crate) fn collect_method_env(decls: &[Decl]) -> Result<MethodEnv, Error> {
    let mut env: MethodEnv = BTreeMap::new();

    for decl in decls {
        let m = match decl {
            Decl::Method(m) => m,
            // Phase 4 will add Datatype; nothing to collect for them in
            // this pass.
            Decl::Datatype(_) => continue,
        };

        // Duplicate-name check. Look up before insert so we still have the
        // previous entry's name_span for the related-span pointer.
        if let Some(prev) = env.get(&m.name) {
            return Err(Error::new(
                ErrorCategory::TypeError,
                m.name_span,
                Some(format!("method `{}` declared more than once", m.name)),
            )
            .with_related(prev.name_span, format!("`{}` first declared here", m.name)));
        }

        // Shared-in-returns check. Returns can be linear or ordinary; a
        // shared return slot is unreachable in any well-typed body
        // (borrows can't escape the method's scope) so we surface this at
        // signature-collection time. The error span covers the offending
        // return parameter — the AST doesn't carry a separate
        // `usage_span`, and the whole-param span is informative enough.
        for ret in &m.returns {
            if ret.usage == Usage::Shared {
                return Err(Error::new(
                    ErrorCategory::TypeError,
                    ret.span,
                    Some(
                        "method returns cannot have `shared` usage; use `linear` or `ordinary`"
                            .to_string(),
                    ),
                ));
            }
        }

        // Build the entry. Declared params drop their names but keep
        // usage/inout/type. Effective returns = declared ++ inout-rebinds.
        let params: Vec<ParamSig> = m
            .params
            .iter()
            .map(|p| ParamSig {
                usage: p.usage,
                inout: p.inout,
                ty: p.ty.clone(),
            })
            .collect();

        let mut effective_returns: Vec<ParamSig> = m
            .returns
            .iter()
            .map(|p| ParamSig {
                usage: p.usage,
                inout: false, // returns don't carry inout
                ty: p.ty.clone(),
            })
            .collect();
        for p in &m.params {
            if p.inout {
                effective_returns.push(ParamSig {
                    usage: p.usage,
                    inout: false, // inout-to-return strips the flag
                    ty: p.ty.clone(),
                });
            }
        }

        env.insert(
            m.name.clone(),
            MethodEntry {
                params,
                effective_returns,
                name_span: m.name_span,
            },
        );
    }

    Ok(env)
}

/// Type-check a complete program in three passes:
///
/// 1. Collect method signatures into `MethodEnv`. Validates duplicate-name
///    and shared-return invariants; bodies are *not* checked yet.
/// 2. For each method declaration, check the body against its expected
///    return type, verify all linear params are consumed, and verify the
///    body's outgoing borrows are empty (borrows can't escape a method).
/// 3. Check the main expression in the populated method env. The
///    top-level invariant — empty outgoing consumes/borrows — is now a
///    real error rather than a `debug_assert`.
pub fn check_program(program: &Program) -> Result<Typed, Error> {
    let method_env = collect_method_env(&program.decls)?;

    for decl in &program.decls {
        let m = match decl {
            Decl::Method(m) => m,
            Decl::Datatype(_) => continue, // Phase 4
        };
        check_method_body(m, &method_env)?;
    }

    let env = Env::new();
    let result = check_expr(
        &program.main_expr,
        &env,
        &method_env,
        None,
        ConsumeMode::Consume,
    )?;
    if let Some((name, &span)) = result.consumes.iter().next() {
        return Err(Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!(
                "top-level expression consumes `{name}` but there is no enclosing binding"
            )),
        ));
    }
    if let Some((name, &span)) = result.borrows.iter().next() {
        return Err(Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!(
                "top-level expression borrows `{name}` but there is no enclosing binding"
            )),
        ));
    }
    Ok(result.typed)
}

// -----------------------------------------------------------------------------
// Pass 2: method-body checking
// -----------------------------------------------------------------------------

/// Compute the expected (and call-result) type for a list of effective
/// returns:
/// - 0 returns → `ordinary ()`
/// - 1 return → that return's `usage τ`
/// - 2+ returns → `outer (u1 τ1, ..., un τn)` where outer is `linear` if
///   any return is linear, else `ordinary`. (Shared returns are already
///   rejected in pass 1, so this collapse is sound.)
fn body_expected_type(effective_returns: &[ParamSig]) -> Typed {
    match effective_returns.len() {
        0 => Typed {
            usage: Usage::Ordinary,
            ty: Type::Tuple(Vec::new()),
        },
        1 => Typed {
            usage: effective_returns[0].usage,
            ty: effective_returns[0].ty.clone(),
        },
        _ => {
            let outer = if effective_returns
                .iter()
                .any(|r| r.usage == Usage::Linear)
            {
                Usage::Linear
            } else {
                Usage::Ordinary
            };
            let fields = effective_returns
                .iter()
                .map(|r| TupleField {
                    usage: r.usage,
                    ty: r.ty.clone(),
                })
                .collect();
            Typed {
                usage: outer,
                ty: Type::Tuple(fields),
            }
        }
    }
}

fn check_method_body(
    decl: &crate::ast::MethodDecl,
    method_env: &MethodEnv,
) -> Result<(), Error> {
    // Build the body's variable env: each declared param becomes a
    // binding `name ↦ usage τ`. (inout flag doesn't affect the binding's
    // shape inside the body — the body sees an inout param as an ordinary
    // linear binding that must be consumed.)
    let mut env = Env::new();
    for p in &decl.params {
        env.insert(
            p.name.clone(),
            Typed {
                usage: p.usage,
                ty: p.ty.clone(),
            },
        );
    }

    let entry = method_env
        .get(&decl.name)
        .expect("pass 1 collected this method's signature");
    let expected = body_expected_type(&entry.effective_returns);

    let result = check_expr(
        &decl.body,
        &env,
        method_env,
        Some(expected.usage),
        ConsumeMode::Consume,
    )?;

    // Type match — most fundamental error first.
    if result.typed != expected {
        return Err(Error::new(
            ErrorCategory::TypeError,
            decl.body.span(),
            Some(format!(
                "method `{}` body produces `{}` but expected return type is `{}`",
                decl.name, result.typed, expected,
            )),
        ));
    }

    // Linear params (regular and inout — both have usage = Linear) must
    // appear in the body's outgoing consumes.
    for p in &decl.params {
        if p.usage == Usage::Linear && !result.consumes.contains_key(&p.name) {
            let kind = if p.inout {
                "linear inout parameter"
            } else {
                "linear parameter"
            };
            return Err(Error::new(
                ErrorCategory::TypeError,
                p.name_span,
                Some(format!(
                    "{kind} `{}` is not consumed in the body of method `{}`",
                    p.name, decl.name,
                )),
            ));
        }
    }

    // Borrows can't escape a method. By construction this is hard to
    // trigger (a method body that leaves a borrow non-empty has typically
    // already failed the linear-must-consume check), but we keep it as a
    // defensive check matching the handoff's spec.
    if let Some((name, &span)) = result.borrows.iter().next() {
        return Err(Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!(
                "borrow of `{name}` escapes the body of method `{}`",
                decl.name,
            )),
        ));
    }

    Ok(())
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
    method_env: &MethodEnv,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    match expr {
        Expr::IntLit { .. } => Ok(Check::ordinary_int()),

        Expr::Var { name, span } => check_var(name, *span, env, mode),

        Expr::Add { lhs, rhs, .. } => check_add(lhs, rhs, env, method_env, mode),

        Expr::Seq {
            first, second, ..
        } => check_seq(first, second, env, method_env, expected_usage, mode),

        Expr::Tuple { elems, span } => {
            check_tuple_construction(elems, env, method_env, expected_usage, *span, mode)
        }

        Expr::TupleProj {
            tuple,
            index,
            span,
        } => check_projection(tuple, *index, *span, env, method_env, mode),

        Expr::TupleDestructure {
            names,
            value,
            body,
            span,
        } => check_destructure(
            names,
            value,
            body,
            *span,
            env,
            method_env,
            expected_usage,
            mode,
        ),

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
            method_env,
            expected_usage,
            mode,
        ),

        Expr::MethodCall {
            name,
            name_span,
            args,
            span,
        } => check_method_call(name, *name_span, args, *span, env, method_env),

        // Phase 4 territory.
        Expr::ConstructorApp { span, .. } | Expr::Match { span, .. } => Err(Error::new(
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

fn check_add(
    lhs: &Expr,
    rhs: &Expr,
    env: &Env,
    method_env: &MethodEnv,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // Both operands in outer mode, expected to be ordinary int.
    let l = check_expr(lhs, env, method_env, Some(Usage::Ordinary), mode)?;
    require_ordinary_int(&l.typed, lhs.span(), "left operand of `+`")?;
    let r = check_expr(rhs, env, method_env, Some(Usage::Ordinary), mode)?;
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
    method_env: &MethodEnv,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // e1 always in Borrow mode (the borrow scope is e1; e2 sees `e1` as
    // having "completed and discarded").
    let f = check_expr(
        first,
        env,
        method_env,
        Some(Usage::Ordinary),
        ConsumeMode::Borrow,
    )?;
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
    let s = check_expr(second, env, method_env, expected_usage, mode)?;

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
    method_env: &MethodEnv,
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
        let c = check_expr(elem, env, method_env, component_expected, mode)?;
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
    method_env: &MethodEnv,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // Inner in outer mode, expected_usage = None (projection accepts
    // ordinary or shared input; mode + variable rule decide which).
    let t = check_expr(tuple, env, method_env, None, mode)?;

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

#[allow(clippy::too_many_arguments)]
fn check_destructure(
    names: &[(String, Span)],
    value: &Expr,
    body: &Expr,
    _span: Span,
    env: &Env,
    method_env: &MethodEnv,
    expected_usage: Option<Usage>,
    mode: ConsumeMode,
) -> Result<Check, Error> {
    // Value always in Consume mode — deconstruction needs to own the
    // tuple. Outer mode is irrelevant here.
    let v = check_expr(
        value,
        env,
        method_env,
        Some(Usage::Linear),
        ConsumeMode::Consume,
    )?;

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

    let b = check_expr(body, &body_env, method_env, expected_usage, mode)?;

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
    method_env: &MethodEnv,
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

    let v = check_expr(value, env, method_env, value_expected, value_mode)?;

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

    let b = check_expr(body, &body_env, method_env, expected_usage, mode)?;

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
// Method calls
// -----------------------------------------------------------------------------

/// `NAME(arg1, ..., argn)`. Per-arg mode dispatch:
///
/// - linear param (incl. inout) → Consume, expected_usage = Some(Linear)
/// - ordinary param             → Consume, expected_usage = Some(Ordinary)
/// - shared param               → Borrow,  expected_usage = None
///
/// Cross-arg disjointness: pairwise consumes (no double-consume) and
/// consume-vs-borrow (no consume-of-something-also-borrowed). Result
/// type comes from the method's effective_returns.
fn check_method_call(
    name: &str,
    name_span: Span,
    args: &[Expr],
    span: Span,
    env: &Env,
    method_env: &MethodEnv,
) -> Result<Check, Error> {
    let entry = method_env.get(name).ok_or_else(|| {
        Error::new(
            ErrorCategory::TypeError,
            name_span,
            Some(format!("unknown method `{name}`")),
        )
    })?;

    if args.len() != entry.params.len() {
        return Err(Error::new(
            ErrorCategory::TypeError,
            span,
            Some(format!(
                "method `{name}` takes {} argument{} but {} {} provided",
                entry.params.len(),
                if entry.params.len() == 1 { "" } else { "s" },
                args.len(),
                if args.len() == 1 { "was" } else { "were" },
            )),
        ));
    }

    // Check each arg against its parameter, accumulating consumes/borrows.
    let mut combined_consumes: BTreeMap<String, Span> = BTreeMap::new();
    let mut combined_borrows: BTreeMap<String, Span> = BTreeMap::new();
    for (i, (arg, param)) in args.iter().zip(entry.params.iter()).enumerate() {
        let (arg_mode, arg_expected) = match param.usage {
            Usage::Linear => (ConsumeMode::Consume, Some(Usage::Linear)),
            Usage::Ordinary => (ConsumeMode::Consume, Some(Usage::Ordinary)),
            Usage::Shared => (ConsumeMode::Borrow, None),
        };
        let c = check_expr(arg, env, method_env, arg_expected, arg_mode)?;

        if c.typed.usage != param.usage {
            return Err(Error::new(
                ErrorCategory::TypeError,
                arg.span(),
                Some(format!(
                    "argument {} to `{name}` has usage `{}`, but the parameter expects `{}`",
                    i + 1,
                    c.typed.usage,
                    param.usage,
                )),
            ));
        }
        if c.typed.ty != param.ty {
            return Err(Error::new(
                ErrorCategory::TypeError,
                arg.span(),
                Some(format!(
                    "argument {} to `{name}` has type `{}`, but the parameter expects `{}`",
                    i + 1,
                    c.typed.ty,
                    param.ty,
                )),
            ));
        }

        combined_consumes = merge_consumes_disjoint(combined_consumes, c.consumes)?;
        combined_borrows = merge_borrows_union(combined_borrows, c.borrows);
    }

    // Cross-arg consume-vs-borrow: a name appearing in both `combined_*`
    // sets means it was consumed by one arg and borrowed by another.
    // (Within a single arg, `consumes` and `borrows` are disjoint by the
    // recursive checker; so any overlap here is necessarily cross-arg.)
    if let Some((conflict, c_span, b_span)) =
        find_borrow_consume_collision(&combined_borrows, &combined_consumes)
    {
        return Err(Error::new(
            ErrorCategory::TypeError,
            c_span,
            Some(format!(
                "cannot consume `{conflict}` in a call to `{name}`: another argument borrows it"
            )),
        )
        .with_related(b_span, format!("`{conflict}` is borrowed here")));
    }

    let result_type = body_expected_type(&entry.effective_returns);
    Ok(Check {
        typed: result_type,
        consumes: combined_consumes,
        borrows: combined_borrows,
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
        let method_env = MethodEnv::new();
        check_expr(&prog.main_expr, &env, &method_env, None, ConsumeMode::Consume)
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
        let method_env = MethodEnv::new();
        let in_consume =
            check_expr(&var_expr, &env, &method_env, None, ConsumeMode::Consume).unwrap();
        assert_eq!(in_consume.typed.usage, Usage::Linear);
        assert_eq!(in_consume.consumes.len(), 1);
        assert!(in_consume.borrows.is_empty());

        let in_borrow =
            check_expr(&var_expr, &env, &method_env, None, ConsumeMode::Borrow).unwrap();
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

    // -- Phase 3 pass 1: method environment collection --

    fn collect_ok(src: &str) -> MethodEnv {
        let prog = parse(src).expect("parse failed");
        collect_method_env(&prog.decls).expect("collection failed")
    }

    fn collect_err(src: &str) -> Error {
        let prog = parse(src).expect("parse failed");
        collect_method_env(&prog.decls).expect_err("expected collection error")
    }

    #[test]
    fn empty_program_yields_empty_method_env() {
        let env = collect_ok("1");
        assert!(env.is_empty());
    }

    #[test]
    fn single_method_collected() {
        let env = collect_ok(
            "method add(ordinary x: int, ordinary y: int) returns (ordinary r: int) { x + y } 1",
        );
        let entry = env.get("add").expect("`add` should be in env");
        assert_eq!(entry.params.len(), 2);
        assert_eq!(entry.params[0].usage, Usage::Ordinary);
        assert_eq!(entry.params[0].ty, Type::Int);
        assert!(!entry.params[0].inout);
        assert_eq!(entry.effective_returns.len(), 1);
        assert_eq!(entry.effective_returns[0].usage, Usage::Ordinary);
    }

    #[test]
    fn multiple_methods_collected() {
        let env = collect_ok(
            "method first() returns (ordinary r: int) { 1 } \
             method second() returns (ordinary r: int) { 2 } \
             1",
        );
        assert!(env.contains_key("first"));
        assert!(env.contains_key("second"));
        assert_eq!(env.len(), 2);
    }

    #[test]
    fn inout_appended_to_effective_returns() {
        // `linear inout p: T` becomes a trailing linear return.
        let env = collect_ok(
            "method incr(linear inout p: (int, int)) returns (ordinary first: int) { \
               let (a, b) = p in let linear new_p = (a + 1, b + 1) in (a, new_p) \
             } 1",
        );
        let entry = env.get("incr").unwrap();
        // params: 1 (the inout)
        assert_eq!(entry.params.len(), 1);
        assert!(entry.params[0].inout);
        assert_eq!(entry.params[0].usage, Usage::Linear);
        // effective_returns: [declared_first (ordinary int), inout_p (linear (int,int))]
        assert_eq!(entry.effective_returns.len(), 2);
        assert_eq!(entry.effective_returns[0].usage, Usage::Ordinary);
        assert_eq!(entry.effective_returns[0].ty, Type::Int);
        assert!(!entry.effective_returns[0].inout); // inout flag is stripped
        assert_eq!(entry.effective_returns[1].usage, Usage::Linear);
        assert!(!entry.effective_returns[1].inout);
    }

    #[test]
    fn inout_only_method_has_one_effective_return() {
        // No declared returns, one inout → one effective return.
        let env = collect_ok(
            "method passthrough(linear inout p: (int, int)) returns () { p } 1",
        );
        let entry = env.get("passthrough").unwrap();
        assert_eq!(entry.effective_returns.len(), 1);
        assert_eq!(entry.effective_returns[0].usage, Usage::Linear);
    }

    #[test]
    fn no_returns_no_inout_yields_empty_effective_returns() {
        let env = collect_ok("method nop() returns () { () } 1");
        let entry = env.get("nop").unwrap();
        assert!(entry.params.is_empty());
        assert!(entry.effective_returns.is_empty());
    }

    #[test]
    fn duplicate_method_name_is_type_error_with_related_span() {
        let err = collect_err(
            "method foo() returns (ordinary r: int) { 1 } \
             method foo() returns (ordinary r: int) { 2 } \
             1",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.expect("note");
        assert!(note.contains("declared more than once"), "got: {note}");
        // Related span should point at the *first* declaration.
        assert_eq!(err.related.len(), 1);
        assert!(err.related[0].label.contains("first declared here"));
        // Primary and related spans should be different (different decls).
        assert_ne!(err.related[0].span, err.span);
    }

    #[test]
    fn shared_return_is_type_error() {
        let err = collect_err(
            "method bad() returns (shared r: int) { 5 } 1",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("`shared`"));
    }

    #[test]
    fn shared_param_is_fine() {
        // Shared usage on params is allowed (it's borrows that can't
        // escape returns). Just verify collection succeeds.
        let env = collect_ok(
            "method peek(shared p: (int, int)) returns (ordinary r: int) { p.0 + p.1 } 1",
        );
        let entry = env.get("peek").unwrap();
        assert_eq!(entry.params[0].usage, Usage::Shared);
    }

    #[test]
    fn inout_in_returns_doesnt_reach_collection() {
        // `inout` in returns is rejected at the parser, not here. Sanity
        // check that the parser catches it before pass 1 sees it.
        let prog_err = parse("method bad() returns (linear inout x: int) { x } 1");
        assert!(prog_err.is_err());
    }

    // -- Phase 3 pass 2: body checking + method calls --

    #[test]
    fn body_expected_type_table() {
        // 0 returns → ordinary ()
        let zero = body_expected_type(&[]);
        assert_eq!(zero.usage, Usage::Ordinary);
        assert_eq!(zero.ty, Type::Tuple(Vec::new()));

        // 1 return → that return's usage τ
        let one = body_expected_type(&[ParamSig {
            usage: Usage::Linear,
            inout: false,
            ty: Type::Int,
        }]);
        assert_eq!(one.usage, Usage::Linear);
        assert_eq!(one.ty, Type::Int);

        // 2 returns, all ordinary → ordinary tuple
        let two_ord = body_expected_type(&[
            ParamSig {
                usage: Usage::Ordinary,
                inout: false,
                ty: Type::Int,
            },
            ParamSig {
                usage: Usage::Ordinary,
                inout: false,
                ty: Type::Int,
            },
        ]);
        assert_eq!(two_ord.usage, Usage::Ordinary);
        if let Type::Tuple(fields) = &two_ord.ty {
            assert_eq!(fields.len(), 2);
        } else {
            panic!("expected Tuple");
        }

        // 2 returns, one linear → linear outer
        let two_mixed = body_expected_type(&[
            ParamSig {
                usage: Usage::Ordinary,
                inout: false,
                ty: Type::Int,
            },
            ParamSig {
                usage: Usage::Linear,
                inout: false,
                ty: Type::Int,
            },
        ]);
        assert_eq!(two_mixed.usage, Usage::Linear);
    }

    #[test]
    fn method_simple_accept() {
        let typed = check_ok(
            "method add(ordinary x: int, ordinary y: int) returns (ordinary r: int) { x + y } \
             add(3, 4)",
        );
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn method_consume_linear_arg_accept() {
        let typed = check_ok(
            "method consume(linear t: (int, int)) returns (ordinary r: int) { \
               let (a, b) = t in a + b \
             } \
             let linear t = (1, 2) in consume(t)",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn method_with_shared_param_accept() {
        // Canonical accept/25: shared param + caller borrows then consumes.
        let typed = check_ok(
            "method peek(shared p: (int, int)) returns (ordinary r: int) { p.0 + p.1 } \
             let linear y = (1, 2) in peek(y) ; let (a, b) = y in a + b",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn inout_basic_accept() {
        let typed = check_ok(
            "method passthrough(linear inout p: (int, int)) returns () { p } \
             let linear t = (5, 5) in let linear new_t = passthrough(t) in \
             let (x, y) = new_t in x + y",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn inout_with_returns_accept() {
        // Canonical accept/24: declared return (ordinary) + inout (linear).
        // Effective returns: [ordinary int, linear (int, int)]. Body must
        // produce `linear (ordinary int, linear (int, int))`.
        let typed = check_ok(
            "method incr_and_first(linear inout p: (int, int)) returns (ordinary first: int) { \
               let (a, b) = p in let linear new_p = (a + 1, b + 1) in (a, new_p) \
             } \
             let linear t = (5, 7) in let (first, new_t) = incr_and_first(t) in \
             let (x, y) = new_t in first + x + y",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn mutual_recursion_signatures_accept() {
        // Pass 1 collects ALL signatures before pass 2 checks ANY body, so
        // a body calling a method declared *after* it works.
        let typed = check_ok(
            "method first(ordinary x: int) returns (ordinary r: int) { second(x + 1) } \
             method second(ordinary x: int) returns (ordinary r: int) { x + 10 } \
             first(5)",
        );
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn method_arg_arity_reject() {
        let err = check_err(
            "method add(ordinary x: int, ordinary y: int) returns (ordinary r: int) { x + y } \
             add(1, 2, 3)",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(note.contains("takes 2") && note.contains("3"), "got: {note}");
    }

    #[test]
    fn method_arg_type_mismatch_reject() {
        let err = check_err(
            "method square(ordinary x: int) returns (ordinary r: int) { x + x } \
             let ordinary t = (1, 2) in square(t)",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        // The argument's type is a tuple, not int.
        assert!(err.note.unwrap().contains("type"));
    }

    #[test]
    fn method_linear_param_unused_reject() {
        let err = check_err(
            "method bad(linear t: (int, int)) returns (ordinary r: int) { 5 } \
             bad((1, 2))",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(note.contains("not consumed"), "got: {note}");
    }

    #[test]
    fn inout_not_rebound_reject() {
        // Body returns ord int but expected linear (int, int) (from the
        // inout-rebind). The method-body-mismatch fires in pass 2 before
        // the main expression runs, so the (placeholder) main here just
        // needs to parse.
        let err = check_err(
            "method bad(linear inout p: (int, int)) returns () { let (a, b) = p in a + b } \
             1",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(
            note.contains("body produces") || note.contains("expected return"),
            "got: {note}"
        );
    }

    #[test]
    fn undefined_method_reject() {
        let err = check_err("unknown(1, 2)");
        assert_eq!(err.category, ErrorCategory::TypeError);
        assert!(err.note.unwrap().contains("unknown method"));
    }

    #[test]
    fn consume_borrow_arg_conflict_reject() {
        // Canonical reject/19: same y passed as linear and shared args.
        let err = check_err(
            "method M(linear t: (int, int), shared s: (int, int)) returns (ordinary r: int) { \
               let (a, b) = t in s.0 + a + b \
             } \
             let linear y = (1, 2) in M(y, y)",
        );
        assert_eq!(err.category, ErrorCategory::TypeError);
        let note = err.note.unwrap();
        assert!(note.contains("consume") && note.contains("borrow"), "got: {note}");
        // Both spans should be present.
        assert_eq!(err.related.len(), 1);
        assert_ne!(err.related[0].span, err.span);
    }

    #[test]
    fn empty_method_returns_unit() {
        let typed = check_ok("method nop() returns () { () } nop()");
        assert_eq!(typed.usage, Usage::Ordinary);
        assert_eq!(typed.ty, Type::Tuple(Vec::new()));
    }

    #[test]
    fn method_no_consume_in_body_for_ordinary_param() {
        // Ordinary params don't need to be "consumed" — the linear-must-
        // consume check only applies to linear params. Verify by passing
        // an ordinary arg whose body uses it once or not at all.
        let typed = check_ok(
            "method ignore(ordinary x: int) returns (ordinary r: int) { 42 } ignore(7)",
        );
        assert_eq!(typed.ty, Type::Int);
    }
}
