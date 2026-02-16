use lalrpop_util::lalrpop_mod;

mod ast;
mod error;
pub mod lsp_support;
pub mod span;
pub mod vm {
    pub mod opcode;
}

pub type WalrusResult<T> = Result<T, error::WalrusError>;

lalrpop_mod!(#[allow(clippy::all)] pub grammar);
