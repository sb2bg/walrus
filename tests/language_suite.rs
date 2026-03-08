use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

#[derive(Debug)]
enum Expectation {
    Stdout(String),
    StderrContains(String),
}

#[derive(Debug)]
struct Case {
    name: String,
    program: PathBuf,
    expectation: Expectation,
    expected_exit_code: Option<i32>,
    env_vars: Vec<(String, String)>,
    stdin_bytes: Option<Vec<u8>>,
}

#[test]
fn language_suite() {
    let mut failures = Vec::new();

    for case in discover_cases() {
        if let Err(message) = run_case(&case) {
            failures.push(message);
        }
    }

    if !failures.is_empty() {
        let mut message = format!("{} fixture(s) failed:\n", failures.len());

        for (index, failure) in failures.iter().enumerate() {
            let _ = writeln!(&mut message, "\n{}. {}", index + 1, failure);
        }

        panic!("{message}");
    }
}

fn discover_cases() -> Vec<Case> {
    let fixtures_root = fixtures_root();
    let mut cases = Vec::new();

    cases.extend(load_cases(
        &fixtures_root,
        &fixtures_root.join("pass"),
        true,
    ));
    cases.extend(load_cases(
        &fixtures_root,
        &fixtures_root.join("fail"),
        false,
    ));

    cases.sort_by(|a, b| a.name.cmp(&b.name));
    cases
}

fn load_cases(fixtures_root: &Path, directory: &Path, is_pass: bool) -> Vec<Case> {
    let mut files = Vec::new();
    let expect_ext = if is_pass { "stdout" } else { "stderr" };
    collect_walrus_files(directory, &mut files, expect_ext);
    files.sort();

    files
        .into_iter()
        .map(|program| load_case(fixtures_root, &program, is_pass))
        .collect()
}

/// Recursively collect `.walrus` files that have a companion expectation file.
fn collect_walrus_files(directory: &Path, files: &mut Vec<PathBuf>, expect_ext: &str) {
    let entries = fs::read_dir(directory).unwrap_or_else(|err| {
        panic!(
            "failed to read fixture directory '{}': {err}",
            directory.display()
        )
    });

    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| {
                panic!(
                    "failed to read fixture entry in '{}': {err}",
                    directory.display()
                )
            })
            .path();

        if path.is_dir() {
            collect_walrus_files(&path, files, expect_ext);
        } else if path.extension().is_some_and(|ext| ext == "walrus")
            && path.with_extension(expect_ext).exists()
        {
            files.push(path);
        }
    }
}

fn load_case(fixtures_root: &Path, program: &Path, is_pass: bool) -> Case {
    let expectation = if is_pass {
        let stdout_path = program.with_extension("stdout");
        let expected_stdout = fs::read_to_string(&stdout_path).unwrap_or_else(|err| {
            panic!(
                "missing stdout expectation for '{}': {} ({err})",
                program.display(),
                stdout_path.display()
            )
        });

        Expectation::Stdout(expected_stdout)
    } else {
        let stderr_path = program.with_extension("stderr");
        let expected_stderr = fs::read_to_string(&stderr_path).unwrap_or_else(|err| {
            panic!(
                "missing stderr expectation for '{}': {} ({err})",
                program.display(),
                stderr_path.display()
            )
        });

        Expectation::StderrContains(expected_stderr)
    };

    let expected_exit_code = read_exit_code(program);
    let env_vars = read_env_vars(program);
    let stdin_bytes = read_stdin(program);

    let relative = program.strip_prefix(fixtures_root).unwrap_or_else(|_| {
        panic!(
            "fixture path '{}' is not under '{}'",
            program.display(),
            fixtures_root.display()
        )
    });

    Case {
        name: relative.to_string_lossy().replace('\\', "/"),
        program: program.to_path_buf(),
        expectation,
        expected_exit_code,
        env_vars,
        stdin_bytes,
    }
}

fn read_exit_code(program: &Path) -> Option<i32> {
    let exit_path = program.with_extension("exitcode");
    if !exit_path.exists() {
        return None;
    }

    let text = fs::read_to_string(&exit_path).unwrap_or_else(|err| {
        panic!(
            "failed to read exit code file '{}': {err}",
            exit_path.display()
        )
    });

    let value = text.trim();
    if value.is_empty() {
        panic!(
            "exit code file '{}' is empty; expected an integer exit code",
            exit_path.display()
        );
    }

    let code = value.parse::<i32>().unwrap_or_else(|err| {
        panic!(
            "invalid exit code '{}' in '{}': {err}",
            value,
            exit_path.display()
        )
    });

    Some(code)
}

fn read_env_vars(program: &Path) -> Vec<(String, String)> {
    let env_path = program.with_extension("env");
    if !env_path.exists() {
        return Vec::new();
    }

    dotenvy::from_path_iter(&env_path)
        .unwrap_or_else(|err| panic!("failed to read env file '{}': {err}", env_path.display()))
        .map(|entry| {
            entry.unwrap_or_else(|err| {
                panic!("invalid env entry in '{}': {err}", env_path.display())
            })
        })
        .collect()
}

fn read_stdin(program: &Path) -> Option<Vec<u8>> {
    let stdin_path = program.with_extension("stdin");
    if !stdin_path.exists() {
        return None;
    }

    let bytes = fs::read(&stdin_path).unwrap_or_else(|err| {
        panic!(
            "failed to read stdin fixture '{}': {err}",
            stdin_path.display()
        )
    });

    Some(bytes)
}

fn run_case(case: &Case) -> Result<(), String> {
    let output = run_program(case).map_err(|err| format!("{}: {err}", case.name))?;

    let stdout = normalize(&String::from_utf8_lossy(&output.stdout));
    let stderr = normalize(&String::from_utf8_lossy(&output.stderr));
    let actual_exit = output.status.code();

    if let Some(expected_exit) = case.expected_exit_code {
        if actual_exit != Some(expected_exit) {
            return Err(format!(
                "{}: exit code mismatch\nexpected: {}\nactual: {}\n\nstderr:\n{}",
                case.name,
                expected_exit,
                format_exit_code(actual_exit),
                stderr
            ));
        }
    } else if matches!(case.expectation, Expectation::Stdout(_)) && !output.status.success() {
        return Err(format!(
            "{}: expected successful exit, got {}\n\nstderr:\n{}",
            case.name,
            format_exit_code(actual_exit),
            stderr
        ));
    }

    match &case.expectation {
        Expectation::Stdout(expected_stdout) => {
            let expected = normalize(expected_stdout);

            if !stderr.is_empty() {
                return Err(format!(
                    "{}: expected no stderr, but got:\n{}",
                    case.name, stderr
                ));
            }

            if stdout != expected {
                return Err(format!(
                    "{}: stdout mismatch\nexpected:\n{}\n\nactual:\n{}",
                    case.name, expected, stdout
                ));
            }
        }
        Expectation::StderrContains(expected_snippet) => {
            let expected = expected_snippet.trim();
            if !stderr.contains(expected) {
                return Err(format!(
                    "{}: stderr did not contain expected snippet\nexpected snippet:\n{}\n\nactual stderr:\n{}\n\nstdout:\n{}",
                    case.name, expected, stderr, stdout
                ));
            }
        }
    }

    Ok(())
}

fn run_program(case: &Case) -> Result<Output, String> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_walrus"));
    command.current_dir(repo_root());
    command.arg(&case.program);

    for (key, value) in &case.env_vars {
        command.env(key, value);
    }

    if let Some(stdin_bytes) = &case.stdin_bytes {
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|err| {
            format!(
                "failed to spawn '{}' for '{}': {err}",
                env!("CARGO_BIN_EXE_walrus"),
                case.program.display()
            )
        })?;

        if let Some(mut child_stdin) = child.stdin.take() {
            child_stdin.write_all(stdin_bytes).map_err(|err| {
                format!(
                    "failed to write stdin for '{}': {err}",
                    case.program.display()
                )
            })?;
        } else {
            return Err(format!(
                "stdin pipe was not available for '{}'",
                case.program.display()
            ));
        }

        child.wait_with_output().map_err(|err| {
            format!(
                "failed to execute '{}' for '{}': {err}",
                env!("CARGO_BIN_EXE_walrus"),
                case.program.display()
            )
        })
    } else {
        command.output().map_err(|err| {
            format!(
                "failed to execute '{}' for '{}': {err}",
                env!("CARGO_BIN_EXE_walrus"),
                case.program.display()
            )
        })
    }
}

fn format_exit_code(code: Option<i32>) -> String {
    match code {
        Some(value) => value.to_string(),
        None => "terminated by signal".to_string(),
    }
}

fn normalize(text: &str) -> String {
    text.replace("\r\n", "\n")
        .trim_end_matches('\n')
        .to_string()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures_root() -> PathBuf {
    repo_root().join("tests").join("fixtures")
}
