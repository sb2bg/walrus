pub fn env_get(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Get all command line arguments
pub fn args() -> Vec<String> {
    std::env::args().collect()
}

/// Get current working directory
pub fn cwd() -> Option<String> {
    std::env::current_dir()
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}
