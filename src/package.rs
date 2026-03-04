use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::WalrusError;

const MANIFEST_FILENAME: &str = "Walrus.toml";
const LOCK_FILENAME: &str = "Walrus.lock";
const LOCK_FILE_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
struct ManifestFile {
    #[serde(default)]
    package: Option<ManifestPackage>,
    #[serde(default)]
    dependencies: BTreeMap<String, ManifestDependency>,
}

#[derive(Debug, Deserialize)]
struct ManifestPackage {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ManifestDependency {
    Version(String),
    Detailed(ManifestDependencyDetail),
}

impl ManifestDependency {
    fn version(&self) -> Option<&str> {
        match self {
            Self::Version(version) => {
                let trimmed = version.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            }
            Self::Detailed(detail) => detail.version.as_deref().and_then(non_empty),
        }
    }

    fn path(&self) -> Option<&str> {
        match self {
            Self::Version(_) => None,
            Self::Detailed(detail) => detail.path.as_deref().and_then(non_empty),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ManifestDependencyDetail {
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LockFile {
    version: u32,
    #[serde(default)]
    packages: BTreeMap<String, LockPackage>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LockPackage {
    version: String,
    path: String,
}

pub fn resolve_package_main(
    base_dir: &Path,
    package_name: &str,
) -> Result<Option<PathBuf>, WalrusError> {
    let Some(project_root) = find_project_root(base_dir) else {
        return Ok(None);
    };

    let manifest = read_manifest(&project_root)?;
    let Some(dependency) = manifest.dependencies.get(package_name) else {
        return Err(WalrusError::GenericError {
            message: format!(
                "Package '@{package_name}' is not declared in '{}'",
                project_root.join(MANIFEST_FILENAME).display()
            ),
        });
    };

    let lock = read_lock(&project_root)?;
    let Some(locked) = lock.packages.get(package_name) else {
        return Err(WalrusError::GenericError {
            message: format!(
                "Package '@{package_name}' is missing from '{}'. Run `walrus --sync-lock`.",
                project_root.join(LOCK_FILENAME).display()
            ),
        });
    };

    if let Some(requested_version) = dependency.version() {
        if requested_version != locked.version {
            return Err(WalrusError::GenericError {
                message: format!(
                    "Package '@{package_name}' is pinned to version '{}' in {}, but '{}' is requested in {}",
                    locked.version, LOCK_FILENAME, requested_version, MANIFEST_FILENAME
                ),
            });
        }
    }

    let package_root = resolve_path(&project_root, &locked.path)?;
    let main_file = package_root.join("main.walrus");
    if !main_file.is_file() {
        return Err(WalrusError::FileNotFound {
            filename: main_file.to_string_lossy().to_string(),
        });
    }

    let canonical = main_file
        .canonicalize()
        .map_err(|_| WalrusError::FileNotFound {
            filename: main_file.to_string_lossy().to_string(),
        })?;

    Ok(Some(canonical))
}

pub fn sync_lock_from(start_dir: &Path) -> Result<PathBuf, WalrusError> {
    let project_root = find_project_root(start_dir).ok_or_else(|| WalrusError::GenericError {
        message: format!(
            "Could not find {} in '{}' or any parent directory",
            MANIFEST_FILENAME,
            start_dir.display()
        ),
    })?;

    let manifest = read_manifest(&project_root)?;
    let mut packages = BTreeMap::new();

    for (name, dependency) in manifest.dependencies {
        let Some(path_spec) = dependency.path() else {
            return Err(WalrusError::GenericError {
                message: format!(
                    "Dependency '{name}' in {} must use a local `path` for now",
                    MANIFEST_FILENAME
                ),
            });
        };

        let package_root = resolve_path(&project_root, path_spec)?;
        if !package_root.is_dir() {
            return Err(WalrusError::GenericError {
                message: format!(
                    "Dependency '{name}' path '{}' is not a directory",
                    package_root.display()
                ),
            });
        }

        let main_file = package_root.join("main.walrus");
        if !main_file.is_file() {
            return Err(WalrusError::FileNotFound {
                filename: main_file.to_string_lossy().to_string(),
            });
        }

        let dependency_manifest = read_manifest_optional(&package_root)?;
        let manifest_package = dependency_manifest
            .as_ref()
            .and_then(|entry| entry.package.as_ref());

        if let Some(actual_name) = manifest_package
            .and_then(|entry| entry.name.as_deref())
            .and_then(non_empty)
        {
            if actual_name != name {
                return Err(WalrusError::GenericError {
                    message: format!(
                        "Dependency key '{name}' does not match package name '{actual_name}' in '{}'",
                        package_root.join(MANIFEST_FILENAME).display()
                    ),
                });
            }
        }

        let requested_version = dependency.version();
        let detected_version = manifest_package
            .and_then(|entry| entry.version.as_deref())
            .and_then(non_empty);
        let locked_version = select_locked_version(&name, requested_version, detected_version)?;

        let path_for_lock = encode_lock_path(&project_root, &package_root);
        packages.insert(
            name,
            LockPackage {
                version: locked_version,
                path: path_for_lock,
            },
        );
    }

    let lock = LockFile {
        version: LOCK_FILE_VERSION,
        packages,
    };

    let mut lock_text = toml::to_string_pretty(&lock).map_err(|err| WalrusError::GenericError {
        message: format!("Failed to serialize {}: {err}", LOCK_FILENAME),
    })?;
    lock_text.push('\n');

    let lock_path = project_root.join(LOCK_FILENAME);
    fs::write(&lock_path, lock_text).map_err(|source| WalrusError::IOError { source })?;
    Ok(lock_path)
}

fn select_locked_version(
    dependency_name: &str,
    requested_version: Option<&str>,
    detected_version: Option<&str>,
) -> Result<String, WalrusError> {
    match (requested_version, detected_version) {
        (Some(requested), Some(detected)) if requested != detected => {
            Err(WalrusError::GenericError {
                message: format!(
                    "Dependency '{dependency_name}' requests version '{requested}' but path package declares '{detected}'"
                ),
            })
        }
        (Some(requested), _) => Ok(requested.to_string()),
        (None, Some(detected)) => Ok(detected.to_string()),
        (None, None) => Err(WalrusError::GenericError {
            message: format!(
                "Dependency '{dependency_name}' must define a version in {} or in its own {}",
                MANIFEST_FILENAME, MANIFEST_FILENAME
            ),
        }),
    }
}

fn resolve_path(project_root: &Path, raw_path: &str) -> Result<PathBuf, WalrusError> {
    let path = PathBuf::from(raw_path);
    let candidate = if path.is_absolute() {
        path
    } else {
        project_root.join(path)
    };

    candidate
        .canonicalize()
        .map_err(|_| WalrusError::FileNotFound {
            filename: candidate.to_string_lossy().to_string(),
        })
}

fn encode_lock_path(project_root: &Path, package_root: &Path) -> String {
    match package_root.strip_prefix(project_root) {
        Ok(relative) => relative.to_string_lossy().to_string(),
        Err(_) => package_root.to_string_lossy().to_string(),
    }
}

fn find_project_root(start_dir: &Path) -> Option<PathBuf> {
    let mut cursor = if start_dir.is_file() {
        start_dir
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    } else {
        start_dir.to_path_buf()
    };

    loop {
        let manifest_path = cursor.join(MANIFEST_FILENAME);
        if manifest_path.is_file() {
            return Some(cursor);
        }

        if !cursor.pop() {
            return None;
        }
    }
}

fn read_manifest(project_root: &Path) -> Result<ManifestFile, WalrusError> {
    let manifest_path = project_root.join(MANIFEST_FILENAME);
    let manifest_text =
        fs::read_to_string(&manifest_path).map_err(|_| WalrusError::FileNotFound {
            filename: manifest_path.to_string_lossy().to_string(),
        })?;

    toml::from_str(&manifest_text).map_err(|err| WalrusError::GenericError {
        message: format!("Failed to parse '{}': {err}", manifest_path.display()),
    })
}

fn read_manifest_optional(directory: &Path) -> Result<Option<ManifestFile>, WalrusError> {
    let manifest_path = directory.join(MANIFEST_FILENAME);
    if !manifest_path.is_file() {
        return Ok(None);
    }

    let manifest_text =
        fs::read_to_string(&manifest_path).map_err(|_| WalrusError::FileNotFound {
            filename: manifest_path.to_string_lossy().to_string(),
        })?;

    let manifest = toml::from_str(&manifest_text).map_err(|err| WalrusError::GenericError {
        message: format!("Failed to parse '{}': {err}", manifest_path.display()),
    })?;
    Ok(Some(manifest))
}

fn read_lock(project_root: &Path) -> Result<LockFile, WalrusError> {
    let lock_path = project_root.join(LOCK_FILENAME);
    let lock_text = fs::read_to_string(&lock_path).map_err(|_| WalrusError::GenericError {
        message: format!(
            "Missing '{}'. Run `walrus --sync-lock` in '{}'.",
            lock_path.display(),
            project_root.display()
        ),
    })?;

    let lock = toml::from_str::<LockFile>(&lock_text).map_err(|err| WalrusError::GenericError {
        message: format!("Failed to parse '{}': {err}", lock_path.display()),
    })?;

    if lock.version != LOCK_FILE_VERSION {
        return Err(WalrusError::GenericError {
            message: format!(
                "Unsupported {} format version '{}'. Expected '{}'.",
                LOCK_FILENAME, lock.version, LOCK_FILE_VERSION
            ),
        });
    }

    Ok(lock)
}

fn non_empty(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
