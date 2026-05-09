//! Corpus test runner. Walks `tests/accept/` and `tests/reject/` and
//! invokes the `mini_linear` binary on each `.lin` file:
//!
//! - `accept/*.lin` must exit with code 0.
//! - `reject/*.lin` must exit non-zero, and the printed error category must
//!   match the file's `// EXPECT: <category>` header (first matching line).
//!
//! The binary is located via `CARGO_BIN_EXE_mini_linear`, which Cargo sets
//! when building integration tests for a crate that has a `[[bin]]`
//! target.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mini_linear"))
}

struct Outcome {
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run(file: &Path) -> Outcome {
    let output = Command::new(binary_path())
        .arg(file)
        .output()
        .unwrap_or_else(|e| panic!("failed to run mini_linear on {}: {e}", file.display()));
    Outcome {
        exit_code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Read the first `// EXPECT: <category>` header out of `source`. Returns
/// `None` if no such header exists.
fn parse_expect_header(source: &str) -> Option<String> {
    for line in source.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("// EXPECT:") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn lin_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir({}): {e}", dir.display()))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("lin"))
        .collect();
    paths.sort();
    paths
}

#[test]
fn corpus_accept() {
    let dir = Path::new("tests/accept");
    let files = lin_files_in(dir);
    assert!(!files.is_empty(), "no .lin files found in {}", dir.display());
    for path in files {
        let outcome = run(&path);
        assert_eq!(
            outcome.exit_code,
            Some(0),
            "{}: expected exit code 0, got {:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            path.display(),
            outcome.exit_code,
            outcome.stdout,
            outcome.stderr,
        );
        assert!(
            outcome.stdout.starts_with("well-typed: "),
            "{}: expected stdout to start with `well-typed: `, got: {:?}",
            path.display(),
            outcome.stdout,
        );
    }
}

#[test]
fn corpus_reject() {
    let dir = Path::new("tests/reject");
    let files = lin_files_in(dir);
    assert!(!files.is_empty(), "no .lin files found in {}", dir.display());
    for path in files {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read({}): {e}", path.display()));
        let expected_category = parse_expect_header(&source).unwrap_or_else(|| {
            panic!(
                "{}: missing `// EXPECT: <category>` header",
                path.display()
            )
        });

        let outcome = run(&path);
        assert_ne!(
            outcome.exit_code,
            Some(0),
            "{}: expected non-zero exit, got 0\n--- stdout ---\n{}\n--- stderr ---\n{}",
            path.display(),
            outcome.stdout,
            outcome.stderr,
        );

        let expected_line = format!("error: {expected_category}");
        let combined = format!("{}{}", outcome.stdout, outcome.stderr);
        assert!(
            combined.contains(&expected_line),
            "{}: expected `{expected_line}` in output\n--- stdout ---\n{}\n--- stderr ---\n{}",
            path.display(),
            outcome.stdout,
            outcome.stderr,
        );
    }
}
