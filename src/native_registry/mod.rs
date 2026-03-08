use once_cell::sync::Lazy;

use crate::WalrusResult;
use crate::error::WalrusError;
use crate::function::NativeFunction;
use crate::span::Span;
use crate::value::Value;
use crate::vm::VM;

mod asyncx;
mod core;
mod helpers;
mod http;
mod io;
mod json;
mod math;
mod net;
mod sys;

pub type NativeHandler = for<'a> fn(&mut VM<'a>, &[Value], Span) -> WalrusResult<Value>;

#[derive(Clone, Copy)]
pub struct NativeSpec {
    pub id: NativeFunction,
    pub module: &'static str,
    pub name: &'static str,
    pub arity: usize,
    pub params: &'static [&'static str],
    pub docs: &'static str,
    pub handler: NativeHandler,
}

pub const fn define_native_spec(
    id: NativeFunction,
    module: &'static str,
    name: &'static str,
    params: &'static [&'static str],
    docs: &'static str,
    handler: NativeHandler,
) -> NativeSpec {
    NativeSpec {
        id,
        module,
        name,
        arity: params.len(),
        params,
        docs,
        handler,
    }
}

pub static NATIVE_SPECS: Lazy<Vec<NativeSpec>> = Lazy::new(|| {
    let mut specs = Vec::new();
    specs.extend_from_slice(core::SPECS);
    specs.extend_from_slice(asyncx::SPECS);
    specs.extend_from_slice(io::SPECS);
    specs.extend_from_slice(sys::SPECS);
    specs.extend_from_slice(net::SPECS);
    specs.extend_from_slice(http::SPECS);
    specs.extend_from_slice(math::SPECS);
    specs.extend_from_slice(json::SPECS);
    specs
});

pub fn native_spec(function: NativeFunction) -> &'static NativeSpec {
    NATIVE_SPECS
        .iter()
        .find(|spec| spec.id == function)
        .expect("native function not registered")
}

pub fn module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    let functions: Vec<NativeFunction> = NATIVE_SPECS
        .iter()
        .filter(|spec| spec.module == module)
        .map(|spec| spec.id)
        .collect();

    if functions.is_empty() {
        None
    } else {
        Some(functions)
    }
}

pub fn dispatch_native(
    vm: &mut VM<'_>,
    native_fn: NativeFunction,
    args: Vec<Value>,
    span: Span,
) -> WalrusResult<Value> {
    let spec = native_spec(native_fn);

    if args.len() != spec.arity {
        return Err(WalrusError::InvalidArgCount {
            name: spec.name.to_string(),
            expected: spec.arity,
            got: args.len(),
            span,
            src: vm.source_ref().source().into(),
            filename: vm.source_ref().filename().into(),
        });
    }

    (spec.handler)(vm, &args, span)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn native_specs_are_unique_and_well_formed() {
        let mut ids = HashSet::new();
        let mut module_names = HashSet::new();

        for spec in NATIVE_SPECS.iter() {
            assert!(
                ids.insert(spec.id),
                "duplicate native function id detected: {:?}",
                spec.id
            );
            assert!(
                module_names.insert((spec.module, spec.name)),
                "duplicate native function name detected: {}/{}",
                spec.module,
                spec.name
            );
            assert_eq!(
                spec.arity,
                spec.params.len(),
                "arity mismatch for {}/{}",
                spec.module,
                spec.name
            );
            assert!(
                !spec.docs.trim().is_empty(),
                "missing docs for {}/{}",
                spec.module,
                spec.name
            );
            assert!(
                spec.module.starts_with("std/"),
                "unexpected module prefix for {}/{}",
                spec.module,
                spec.name
            );
        }
    }

    #[test]
    fn module_lookup_matches_registered_specs() {
        let mut expected_by_module: HashMap<&str, Vec<NativeFunction>> = HashMap::new();
        for spec in NATIVE_SPECS.iter() {
            expected_by_module
                .entry(spec.module)
                .or_default()
                .push(spec.id);
        }

        for (module, expected_fns) in expected_by_module {
            let actual = module_functions(module)
                .unwrap_or_else(|| panic!("module '{module}' should have registered functions"));
            assert_eq!(
                actual.len(),
                expected_fns.len(),
                "function count mismatch for module '{module}'"
            );

            for expected in expected_fns {
                assert!(
                    actual.contains(&expected),
                    "module '{module}' missing function {:?}",
                    expected
                );
            }
        }

        assert!(
            module_functions("std/unknown").is_none(),
            "unknown module unexpectedly returned functions"
        );
    }
}
