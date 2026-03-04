use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempProject {
    root: PathBuf,
}

impl TempProject {
    fn new(name: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("walrus-{name}-{}-{now}", std::process::id()));
        fs::create_dir_all(&root).expect("should create temporary project directory");
        Self { root }
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for TempProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn sync_lock_generates_pinned_entries_for_path_dependencies() {
    let project = TempProject::new("sync-lock");
    write_file(
        &project.path().join("Walrus.toml"),
        r#"[package]
name = "app"
version = "0.1.0"

[dependencies]
greeter = { version = "1.2.3", path = "./deps/greeter" }
"#,
    );
    write_file(
        &project.path().join("deps/greeter/Walrus.toml"),
        r#"[package]
name = "greeter"
version = "1.2.3"
"#,
    );
    write_file(
        &project.path().join("deps/greeter/main.walrus"),
        r#"let message = "hello";"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_walrus"))
        .arg("--sync-lock")
        .current_dir(project.path())
        .output()
        .expect("sync-lock command should execute");

    assert!(
        output.status.success(),
        "sync-lock should succeed.\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let lock_path = project.path().join("Walrus.lock");
    assert!(lock_path.is_file(), "Walrus.lock should be created");

    let lock_text = fs::read_to_string(&lock_path).expect("Walrus.lock should be readable");
    assert!(
        lock_text.contains("[packages.greeter]"),
        "lock file should contain greeter entry:\n{lock_text}"
    );
    assert!(
        lock_text.contains("version = \"1.2.3\""),
        "lock file should pin the dependency version:\n{lock_text}"
    );
    assert!(
        lock_text.contains("path = \"deps/greeter\""),
        "lock file should store a relative dependency path:\n{lock_text}"
    );
}

#[test]
fn import_uses_walrus_lock_for_package_resolution() {
    let project = TempProject::new("locked-import");
    write_file(
        &project.path().join("Walrus.toml"),
        r#"[package]
name = "app"
version = "0.1.0"

[dependencies]
greeter = { version = "1.2.3", path = "./deps/greeter" }
"#,
    );
    write_file(
        &project.path().join("deps/greeter/Walrus.toml"),
        r#"[package]
name = "greeter"
version = "1.2.3"
"#,
    );
    write_file(
        &project.path().join("deps/greeter/main.walrus"),
        r#"let message = "from-lock";"#,
    );
    write_file(
        &project.path().join("main.walrus"),
        r#"import @greeter as g;

println(g["message"]);
"#,
    );

    let sync_output = Command::new(env!("CARGO_BIN_EXE_walrus"))
        .arg("--sync-lock")
        .current_dir(project.path())
        .output()
        .expect("sync-lock command should execute");
    assert!(
        sync_output.status.success(),
        "sync-lock should succeed before import test:\nstderr:\n{}",
        String::from_utf8_lossy(&sync_output.stderr)
    );

    let vm_output = Command::new(env!("CARGO_BIN_EXE_walrus"))
        .arg("main.walrus")
        .current_dir(project.path())
        .output()
        .expect("vm run should execute");
    assert_eq!(
        String::from_utf8_lossy(&vm_output.stdout).replace("\r\n", "\n"),
        "from-lock\n"
    );
    assert!(
        vm_output.stderr.is_empty(),
        "vm run should not write stderr:\n{}",
        String::from_utf8_lossy(&vm_output.stderr)
    );

    let interpreted_output = Command::new(env!("CARGO_BIN_EXE_walrus"))
        .arg("main.walrus")
        .arg("--interpreted")
        .current_dir(project.path())
        .output()
        .expect("interpreted run should execute");
    assert_eq!(
        String::from_utf8_lossy(&interpreted_output.stdout).replace("\r\n", "\n"),
        "from-lock\n"
    );
    assert!(
        interpreted_output.stderr.is_empty(),
        "interpreted run should not write stderr:\n{}",
        String::from_utf8_lossy(&interpreted_output.stderr)
    );
}

#[test]
fn import_reports_version_pin_mismatch() {
    let project = TempProject::new("version-mismatch");
    write_file(
        &project.path().join("Walrus.toml"),
        r#"[package]
name = "app"
version = "0.1.0"

[dependencies]
greeter = { version = "1.2.3", path = "./deps/greeter" }
"#,
    );
    write_file(
        &project.path().join("deps/greeter/main.walrus"),
        r#"let message = "from-lock";"#,
    );
    write_file(
        &project.path().join("Walrus.lock"),
        r#"version = 1

[packages.greeter]
version = "9.9.9"
path = "deps/greeter"
"#,
    );
    write_file(
        &project.path().join("main.walrus"),
        r#"import @greeter as g;
println(g["message"]);
"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_walrus"))
        .arg("main.walrus")
        .current_dir(project.path())
        .output()
        .expect("run should execute");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("is pinned to version"),
        "stderr should report version mismatch.\nstderr:\n{stderr}"
    );
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent directories should be created");
    }
    fs::write(path, content).expect("file should be written");
}
