# mini_linear

A type checker for a small linear-typed language inspired by the formal core of Linear Dafny (Li et al., OOPSLA 2022). Reads a `.lin` source file and produces either a typing verdict (`well-typed: <type>`) or an error pointing at a line and column. No execution, no runtime — pure static analysis. Hand-rolled recursive-descent parser, single Rust crate, standard library only.

## Build

```sh
cargo build
```

## Run

```sh
cargo run -- path/to/file.lin
```

Exit codes:
- `0` — well-typed
- `1` — parse or type error
- `2` — usage error (wrong args, file unreadable)

## Test

```sh
cargo test
```

Runs unit tests in `src/` plus the accept/reject corpus under `tests/`.

## Status

This is Phase 0: integer literals, addition, `let ordinary` bindings, and parenthesised grouping. Linearity, borrowing, methods, and algebraic datatypes are intentionally absent from the parser and type checker (though their AST nodes already exist) and arrive in Phases 1–4.
