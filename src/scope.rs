use std::io::Write;
use std::ptr::NonNull;

use rustc_hash::FxHashMap;

use crate::arenas::{HeapValue, Resolve};
use crate::error::WalrusError;
use crate::rust_function::RustFunction;
use crate::value::ValueKind;

#[derive(Debug)]
pub struct Scope {
    name: String,
    // todo: add line and file name for stack trace
    vars: FxHashMap<String, ValueKind>,
    parent: Option<NonNull<Scope>>,
}

impl Scope {
    pub fn new() -> Self {
        Self {
            name: "global".to_string(),
            vars: Self::create_builtins(),
            parent: None,
        }
    }

    // fixme: this feels janky
    fn create_builtins() -> FxHashMap<String, ValueKind> {
        let mut builtins = FxHashMap::default();

        builtins.insert(
            "input".to_string(),
            HeapValue::RustFunction(RustFunction::new(
                "input".to_string(),
                Some(1),
                |args, _, _| {
                    print!("{}", args[0].stringify()?);
                    std::io::stdout()
                        .flush()
                        .map_err(|source| WalrusError::IOError { source })?;

                    let mut input = String::new();
                    std::io::stdin()
                        .read_line(&mut input)
                        .map_err(|source| WalrusError::IOError { source })?;

                    Ok(HeapValue::String(input).alloc())
                },
            ))
            .alloc(),
        );

        builtins.insert(
            "len".to_string(),
            HeapValue::RustFunction(RustFunction::new(
                "len".to_string(),
                Some(1),
                |args, source_ref, span| match args[0] {
                    ValueKind::String(key) => Ok(ValueKind::Int(key.resolve()?.len() as i64)),
                    ValueKind::List(key) => Ok(ValueKind::Int(key.resolve()?.len() as i64)),
                    ValueKind::Dict(key) => Ok(ValueKind::Int(key.resolve()?.len() as i64)),
                    _ => Err(Box::new(WalrusError::NoLength {
                        type_name: args[0].get_type().to_string(),
                        span,
                        src: source_ref.source().to_string(),
                        filename: source_ref.filename().to_string(),
                    })),
                },
            ))
            .alloc(),
        );

        builtins
    }

    pub fn new_child(&self, name: String) -> Self {
        Self {
            name,
            vars: FxHashMap::default(),
            parent: Some(NonNull::from(self)),
        }
    }

    pub fn get(&self, name: &str) -> Option<ValueKind> {
        if let Some(value) = self.vars.get(name) {
            Some(*value)
        } else {
            match self.parent {
                Some(parent) => unsafe { parent.as_ref().get(name) },
                _ => None,
            }
        }
    }

    pub fn assign(&mut self, name: String, value: ValueKind) {
        if name == "_" {
            return;
        }

        self.vars.insert(name, value);
    }

    pub fn is_defined(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    pub fn reassign<'a>(&'a mut self, name: &'a str, value: ValueKind) -> Result<(), &'a str> {
        if self.vars.contains_key(name) {
            if let Some((entry, _)) = self.vars.remove_entry(name) {
                self.vars.insert(entry, value);
            }

            self.vars.insert(name.to_string(), value); // should never happen
            Ok(())
        } else {
            match self.parent {
                Some(mut parent) => unsafe { parent.as_mut().reassign(name, value) },
                _ => Err(name),
            }
        }
    }

    // todo: make this print the right way around and do better than just a string
    // something like StackTraceEntry
    pub fn stack_trace(&self) -> String {
        if let Some(parent) = self.parent {
            format!("{} -> {}", self.name, unsafe {
                parent.as_ref().stack_trace()
            })
        } else {
            self.name.clone()
        }
    }
}
