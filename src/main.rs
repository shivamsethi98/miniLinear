//! CLI entry point: read a `.lin` file, parse it, type-check it, and either
//! print the typing verdict or a formatted error.
//!
//! Exit codes:
//!   0 — well-typed
//!   1 — parse or type error
//!   2 — usage error (wrong args, file unreadable)

use std::io::Write;
use std::process::ExitCode;

use mini_linear::error::format_error;
use mini_linear::parser::parse;
use mini_linear::typeck::check_program;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        let prog = args.first().map(String::as_str).unwrap_or("mini_linear");
        let _ = writeln!(std::io::stderr(), "usage: {prog} <file.lin>");
        return ExitCode::from(2);
    }
    let path = &args[1];
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(io_err) => {
            let _ = writeln!(std::io::stderr(), "cannot read `{path}`: {io_err}");
            return ExitCode::from(2);
        }
    };

    let program = match parse(&source) {
        Ok(p) => p,
        Err(err) => {
            let _ = write!(std::io::stderr(), "{}", format_error(&err, &source, path));
            return ExitCode::from(1);
        }
    };

    match check_program(&program) {
        Ok(typed) => {
            println!("well-typed: {typed}");
            ExitCode::from(0)
        }
        Err(err) => {
            let _ = write!(std::io::stderr(), "{}", format_error(&err, &source, path));
            ExitCode::from(1)
        }
    }
}
