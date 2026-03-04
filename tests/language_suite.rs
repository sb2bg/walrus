use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecMode {
    Vm,
    Interpreted,
}

impl ExecMode {
    fn all() -> [Self; 2] {
        [Self::Vm, Self::Interpreted]
    }

    fn label(self) -> &'static str {
        match self {
            Self::Vm => "vm",
            Self::Interpreted => "interpreted",
        }
    }

    fn cli_args(self) -> &'static [&'static str] {
        match self {
            Self::Vm => &[],
            Self::Interpreted => &["--interpreted"],
        }
    }
}

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
    modes: Vec<ExecMode>,
    expected_exit_code: Option<i32>,
    env_vars: Vec<(String, String)>,
    stdin_bytes: Option<Vec<u8>>,
}

#[test]
fn vm_language_suite() {
    run_suite(ExecMode::Vm);
}

#[test]
fn interpreted_language_suite() {
    run_suite(ExecMode::Interpreted);
}

fn run_suite(mode: ExecMode) {
    let mut failures = Vec::new();

    for case in discover_cases() {
        if !case.modes.contains(&mode) {
            continue;
        }

        if let Err(message) = run_case(&case, mode) {
            failures.push(message);
        }
    }

    if !failures.is_empty() {
        let mut message = format!(
            "{} fixture(s) failed while running in {} mode:\n",
            failures.len(),
            mode.label()
        );

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
    let mut files = fs::read_dir(directory)
        .unwrap_or_else(|err| {
            panic!(
                "failed to read fixture directory '{}': {err}",
                directory.display()
            )
        })
        .map(|entry| {
            entry
                .unwrap_or_else(|err| {
                    panic!(
                        "failed to read fixture entry in '{}': {err}",
                        directory.display()
                    )
                })
                .path()
        })
        .filter(|path| path.extension().is_some_and(|ext| ext == "walrus"))
        .collect::<Vec<_>>();

    files.sort();

    files
        .into_iter()
        .map(|program| load_case(fixtures_root, &program, is_pass))
        .collect()
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

    let modes = read_modes(program);
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
        modes,
        expected_exit_code,
        env_vars,
        stdin_bytes,
    }
}

fn read_modes(program: &Path) -> Vec<ExecMode> {
    let modes_path = program.with_extension("modes");
    if !modes_path.exists() {
        return ExecMode::all().to_vec();
    }

    let text = fs::read_to_string(&modes_path)
        .unwrap_or_else(|err| panic!("failed to read mode file '{}': {err}", modes_path.display()));

    let mut modes = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        match line {
            "both" => return ExecMode::all().to_vec(),
            "vm" => push_unique_mode(&mut modes, ExecMode::Vm),
            "interpreted" => push_unique_mode(&mut modes, ExecMode::Interpreted),
            _ => panic!(
                "invalid mode '{line}' in '{}'; expected one of: vm, interpreted, both",
                modes_path.display()
            ),
        }
    }

    if modes.is_empty() {
        panic!(
            "mode file '{}' is empty; expected one of: vm, interpreted, both",
            modes_path.display()
        );
    }

    modes
}

fn push_unique_mode(modes: &mut Vec<ExecMode>, mode: ExecMode) {
    if !modes.contains(&mode) {
        modes.push(mode);
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

fn run_case(case: &Case, mode: ExecMode) -> Result<(), String> {
    let output = run_program(case, mode)
        .map_err(|err| format!("{} [{}]: {err}", case.name, mode.label()))?;

    let stdout = normalize(&String::from_utf8_lossy(&output.stdout));
    let stderr = normalize(&String::from_utf8_lossy(&output.stderr));
    let actual_exit = output.status.code();

    if let Some(expected_exit) = case.expected_exit_code {
        if actual_exit != Some(expected_exit) {
            return Err(format!(
                "{} [{}]: exit code mismatch\nexpected: {}\nactual: {}\n\nstderr:\n{}",
                case.name,
                mode.label(),
                expected_exit,
                format_exit_code(actual_exit),
                stderr
            ));
        }
    } else if matches!(case.expectation, Expectation::Stdout(_)) && !output.status.success() {
        return Err(format!(
            "{} [{}]: expected successful exit, got {}\n\nstderr:\n{}",
            case.name,
            mode.label(),
            format_exit_code(actual_exit),
            stderr
        ));
    }

    match &case.expectation {
        Expectation::Stdout(expected_stdout) => {
            let expected = normalize(expected_stdout);

            if !stderr.is_empty() {
                return Err(format!(
                    "{} [{}]: expected no stderr, but got:\n{}",
                    case.name,
                    mode.label(),
                    stderr
                ));
            }

            if stdout != expected {
                return Err(format!(
                    "{} [{}]: stdout mismatch\nexpected:\n{}\n\nactual:\n{}",
                    case.name,
                    mode.label(),
                    expected,
                    stdout
                ));
            }
        }
        Expectation::StderrContains(expected_snippet) => {
            let expected = expected_snippet.trim();
            if !stderr.contains(expected) {
                return Err(format!(
                    "{} [{}]: stderr did not contain expected snippet\nexpected snippet:\n{}\n\nactual stderr:\n{}\n\nstdout:\n{}",
                    case.name,
                    mode.label(),
                    expected,
                    stderr,
                    stdout
                ));
            }
        }
    }

    Ok(())
}

fn run_program(case: &Case, mode: ExecMode) -> Result<Output, String> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_walrus"));
    command.current_dir(repo_root());
    command.arg(&case.program);

    for arg in mode.cli_args() {
        command.arg(arg);
    }

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
