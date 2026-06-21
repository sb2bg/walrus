use lalrpop_util::lalrpop_mod;

mod ast;
mod error;
pub mod lsp_support;
#[allow(dead_code)]
mod source_ref;
pub mod span;
pub mod vm {
    pub mod opcode;
}

pub type WalrusResult<T> = Result<T, error::WalrusError>;

lalrpop_mod!(#[allow(clippy::all, unused_imports)] pub grammar);
