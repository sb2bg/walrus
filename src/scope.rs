use crate::arenas::{DictKey, FuncKey, ListKey, RustFuncKey, RustFunction, ValueHolder};
use crate::ast::Node;
use crate::error::WalrusError;
use crate::value::{HeapValue, ValueKind};
use crate::WalrusResult;
use once_cell::sync::Lazy;
use rustc_hash::FxHashMap;
use std::io::Write;
use std::ptr::NonNull;
use string_interner::DefaultSymbol;

static mut ARENA: Lazy<ValueHolder> = Lazy::new(ValueHolder::new);

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
            Self::heap_alloc(HeapValue::RustFunction((
                |args, interpreter, _| {
                    print!("{}", interpreter.stringify(args[0])?);
                    std::io::stdout()
                        .flush()
                        .map_err(|source| WalrusError::IOError { source })?;

                    let mut input = String::new();
                    std::io::stdin()
                        .read_line(&mut input)
                        .map_err(|source| WalrusError::IOError { source })?;

                    Ok(Self::heap_alloc(HeapValue::String(input)))
                },
                Some(1),
                "input".to_string(),
            ))),
        );

        builtins.insert(
            "len".to_string(),
            Self::heap_alloc(HeapValue::RustFunction((
                |args, interpreter, span| match args[0] {
                    ValueKind::String(key) => {
                        Ok(ValueKind::Int(Self::get_string(key)?.len() as i64))
                    }
                    ValueKind::List(key) => Ok(ValueKind::Int(Self::get_list(key)?.len() as i64)),
                    ValueKind::Dict(key) => Ok(ValueKind::Int(Self::get_dict(key)?.len() as i64)),
                    _ => Err(WalrusError::NoLength {
                        type_name: args[0].get_type().to_string(),
                        span,
                        src: interpreter.source_ref().source().to_string(),
                        filename: interpreter.source_ref().filename().to_string(),
                    }),
                },
                Some(1),
                "len".to_string(),
            ))),
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

    pub fn heap_alloc(value: HeapValue) -> ValueKind {
        unsafe { ARENA.alloc(value) }
    }

    pub fn define(&mut self, name: String, value: ValueKind) {
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

    pub fn get_rust_function<'a>(key: RustFuncKey) -> WalrusResult<&'a RustFunction> {
        unsafe { ARENA.get_rust_function(key) }
    }

    pub fn get_mut_dict<'a>(key: DictKey) -> WalrusResult<&'a mut FxHashMap<ValueKind, ValueKind>> {
        unsafe { ARENA.get_mut_dict(key) }
    }

    pub fn get_dict<'a>(key: DictKey) -> WalrusResult<&'a FxHashMap<ValueKind, ValueKind>> {
        unsafe { ARENA.get_dict(key) }
    }

    pub fn get_mut_list<'a>(key: ListKey) -> WalrusResult<&'a mut Vec<ValueKind>> {
        unsafe { ARENA.get_mut_list(key) }
    }

    pub fn get_list<'a>(key: ListKey) -> WalrusResult<&'a Vec<ValueKind>> {
        unsafe { ARENA.get_list(key) }
    }

    pub fn get_string<'a>(key: DefaultSymbol) -> WalrusResult<&'a str> {
        unsafe { ARENA.get_string(key) }
    }

    pub fn get_function<'a>(key: FuncKey) -> WalrusResult<&'a (String, Vec<String>, Node)> {
        unsafe { ARENA.get_function(key) }
    }

    pub fn free(value: ValueKind) -> bool {
        unsafe { ARENA.free(value) }
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
