//! Type checker for the mini-linear language. Phase 0 only.
//!
//! Phase 0 typing rules (subset of Section 2.2 of the paper, with all
//! borrowing/linearity ignored because nothing here is linear or shared):
//!
//! ```text
//! !C ⊢ i : ordinary int                          (integer literal)
//! !C, x ↦→ u τ ⊢ x : u τ                         (variable lookup)
//!
//!   C1 ⊢ e1 : ordinary int    C2 ⊢ e2 : ordinary int
//!   ─────────────────────────────────────────────
//!         C1 # C2 ⊢ e1 + e2 : ordinary int        (addition)
//!
//!   C1 ⊢ e1 : ordinary τ1
//!   C2, x ↦→ ordinary τ1 ⊢ e2 : u2 τ2
//!   ─────────────────────────────────────────────
//!     C1 # C2 ⊢ let ordinary x = e1 in e2 : u2 τ2 (let, ordinary)
//! ```
//!
//! In Phase 0, `C1 # C2` is just `C` because all variables are nonlinear
//! (ordinary), so both halves of the split share the entire environment.
//! The function signature already threads the paper's `c` (consumption
//! mode) so Phase 1+ can introduce `Borrow`/`Consume` without changing
//! call sites.

use std::collections::BTreeMap;
use std::fmt;

use crate::ast::{Expr, Program, Span, Type, Usage};
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
/// paper calls this `c`. In Phase 0 only `Consume` exists; `Borrow` arrives
/// in Phase 2 along with shared usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsumeMode {
    Consume,
}

/// Type-check a complete program. Returns the type of the top-level
/// expression, or the first error encountered.
pub fn check_program(program: &Program) -> Result<Typed, Error> {
    debug_assert!(
        program.decls.is_empty(),
        "Phase 0 parser does not produce declarations",
    );
    let env = Env::new();
    let result = check_expr(&program.main_expr, &env, ConsumeMode::Consume)?;
    Ok(result.typed)
}

// -----------------------------------------------------------------------------
// Internal state
// -----------------------------------------------------------------------------

/// Variable typing environment `X` from the paper. `BTreeMap` rather than
/// `HashMap` for deterministic iteration order in error messages.
type Env = BTreeMap<String, Typed>;

/// Result of checking a single expression. Phase 0 only carries the typed
/// value; Phase 1+ will add `borrows`/`consumes` sets here.
struct Check {
    typed: Typed,
}

// -----------------------------------------------------------------------------
// The recursive checker
// -----------------------------------------------------------------------------

fn check_expr(expr: &Expr, env: &Env, _mode: ConsumeMode) -> Result<Check, Error> {
    match expr {
        Expr::IntLit { .. } => Ok(Check {
            typed: Typed {
                usage: Usage::Ordinary,
                ty: Type::Int,
            },
        }),

        Expr::Var { name, span } => match env.get(name) {
            Some(typed) => Ok(Check {
                typed: typed.clone(),
            }),
            None => Err(Error::new(
                ErrorCategory::TypeError,
                *span,
                Some(format!("unbound variable `{name}`")),
            )),
        },

        Expr::Add { lhs, rhs, .. } => {
            let l = check_expr(lhs, env, ConsumeMode::Consume)?;
            require_ordinary_int(&l.typed, lhs.span(), "left operand of `+`")?;
            let r = check_expr(rhs, env, ConsumeMode::Consume)?;
            require_ordinary_int(&r.typed, rhs.span(), "right operand of `+`")?;
            Ok(Check {
                typed: Typed {
                    usage: Usage::Ordinary,
                    ty: Type::Int,
                },
            })
        }

        Expr::Let {
            usage,
            name,
            value,
            body,
            span,
            ..
        } => check_let(*usage, name, value, body, *span, env, _mode),

        // All other Expr variants are reserved for later phases. The Phase
        // 0 parser does not produce them, so this arm is defensive: if an
        // out-of-scope tree is constructed programmatically and fed to the
        // checker, it must reject rather than silently misbehave.
        Expr::Seq { span, .. }
        | Expr::Tuple { span, .. }
        | Expr::TupleProj { span, .. }
        | Expr::TupleDestructure { span, .. }
        | Expr::MethodCall { span, .. }
        | Expr::ConstructorApp { span, .. }
        | Expr::Match { span, .. } => Err(Error::new(
            ErrorCategory::TypeError,
            *span,
            Some("this expression form is not supported in Phase 0".to_string()),
        )),
    }
}

fn check_let(
    usage: Usage,
    name: &str,
    value: &Expr,
    body: &Expr,
    span: Span,
    env: &Env,
    outer_mode: ConsumeMode,
) -> Result<Check, Error> {
    // The paper picks `c` for `e1` based on the user's `u` annotation:
    //   u = linear  → c = consume
    //   u = shared  → c = borrow
    //   u = ordinary → c is irrelevant
    // Phase 0 only handles ordinary; reject the other usages defensively.
    let value_mode = match usage {
        Usage::Ordinary => ConsumeMode::Consume,
        Usage::Linear | Usage::Shared => {
            return Err(Error::new(
                ErrorCategory::TypeError,
                span,
                Some(format!(
                    "let-binding usage `{usage}` is not supported in Phase 0; only `ordinary` is allowed"
                )),
            ));
        }
    };

    let value_check = check_expr(value, env, value_mode)?;

    // The annotation must agree with the value's actual usage. In Phase 0
    // this is automatic (all values are ordinary), but doing the check now
    // means Phase 1+ inherits the right error site without restructuring.
    if value_check.typed.usage != usage {
        return Err(Error::new(
            ErrorCategory::TypeError,
            value.span(),
            Some(format!(
                "let-binding annotated `{usage}` but value has usage `{}`",
                value_check.typed.usage,
            )),
        ));
    }

    let mut body_env = env.clone();
    body_env.insert(name.to_string(), value_check.typed);
    check_expr(body, &body_env, outer_mode)
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
        // Body uses `x`; if the env weren't extended this would error.
        let typed = check_ok("let ordinary x = 100 in x");
        assert_eq!(typed.ty, Type::Int);
    }

    #[test]
    fn outer_binding_does_not_escape() {
        // After the `let`, `x` should not be in scope at the toplevel.
        // We exercise this via a let whose value uses an outer-scope var
        // that doesn't exist.
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
        // Inner `x` shadows outer `x`. Phase 0 has only ordinary so this
        // is just standard lexical scoping.
        let typed = check_ok("let ordinary x = 1 in let ordinary x = x + 1 in x");
        assert_eq!(typed.ty, Type::Int);
    }
}
