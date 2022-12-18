use crate::ast::{Node, NodeKind, Op};
use crate::error::WalrusError;
use crate::error::WalrusError::UndefinedVariable;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::{Span, Spanned};
use crate::value::ValueKind;
use float_ord::FloatOrd;
use log::debug;
use std::collections::HashMap;
use uuid::Uuid;

pub struct Interpreter<'a> {
    scope: Scope<'a>,
    source_ref: SourceRef<'a>,
    returnable: bool,
}

pub type InterpreterResult = Result<ValueKind, WalrusError>;

impl<'a> Interpreter<'a> {
    pub fn new(src: &'a str, filename: &'a str) -> Self {
        Self {
            scope: Scope::new(),
            source_ref: SourceRef::new(src, filename),
            returnable: false,
        }
    }

    pub fn dump(&self) {
        debug!("Interpreter dump");
        debug!("Returnable: {}", self.returnable);
        self.scope.dump();
    }

    pub fn source_ref(&self) -> SourceRef<'a> {
        self.source_ref
    }

    // for REPL to use
    pub fn set_source_ref(&mut self, src: &'a str) {
        self.source_ref = SourceRef::new(src, self.source_ref.filename());
    }

    pub fn create_child(&'a self, name: String) -> Interpreter {
        Self {
            scope: self.scope.new_child(name),
            source_ref: self.source_ref,
            returnable: true,
        }
    }

    pub fn interpret(&mut self, node: Node) -> InterpreterResult {
        // fixme: using copy to avoid borrow checker error, but this is probably not the best way to do this
        let span = *node.span();

        let res = match node.into_kind() {
            NodeKind::Statements(nodes) => self.visit_statements(nodes),
            NodeKind::BinOp(left, op, right) => self.visit_bin_op(*left, op, *right, span),
            NodeKind::UnaryOp(op, value) => self.visit_unary_op(op, *value, span),
            NodeKind::Int(num) => Ok(ValueKind::Int(num)),
            NodeKind::Float(num) => Ok(ValueKind::Float(num)),
            NodeKind::Bool(boolean) => Ok(ValueKind::Bool(boolean)),
            NodeKind::String(string) => Ok(self.scope.mut_arena().insert_string(string)),
            NodeKind::Dict(dict) => self.visit_dict(dict),
            NodeKind::List(list) => self.visit_list(list),
            NodeKind::FunctionDefinition(name, args, body) => self.visit_fn_def(name, args, *body),
            NodeKind::AnonFunctionDefinition(args, body) => self.visit_anon_fn_def(args, *body),
            NodeKind::Ident(ident) => self.visit_var(ident, span),
            NodeKind::Void => Ok(ValueKind::Void),
            NodeKind::If(condition, then, otherwise) => self.visit_if(*condition, *then, otherwise),
            NodeKind::While(condition, body) => self.visit_while(*condition, *body),
            NodeKind::Assign(name, value) => self.visit_assign(name, *value),
            NodeKind::Reassign(ident, value, op) => self.visit_reassign(ident, *value, op),
            NodeKind::Print(value) => self.visit_print(*value),
            NodeKind::Throw(value) => self.visit_throw(*value, span),
            NodeKind::Free(value) => self.visit_free(*value),
            NodeKind::FunctionCall(value, args) => self.visit_fn_call(*value, args, span),
            NodeKind::Index(value, index) => self.visit_index(*value, *index),
            node => Err(WalrusError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        };

        debug!("{:?}", res);
        res
    }

    fn get_variable(&self, name: &str, span: Span) -> InterpreterResult {
        self.scope.get(name).ok_or_else(|| UndefinedVariable {
            name: name.to_string(),
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    fn visit_statements(&mut self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        for node in nodes {
            let span = *node.span();

            match *node {
                Node {
                    kind: NodeKind::Return(ret),
                    ..
                } => {
                    if !self.returnable {
                        return Err(WalrusError::ReturnOutsideFunction {
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        });
                    }

                    return self.interpret(*ret);
                }
                _ => self.interpret(*node)?,
            };
        }

        Ok(ValueKind::Void)
    }

    fn visit_bin_op(&mut self, left: Node, op: Op, right: Node, span: Span) -> InterpreterResult {
        let left_val = self.interpret(left)?;
        let right_val = self.interpret(right)?;

        match op {
            Op::Add => self.add(left_val, right_val, span),
            Op::Sub => self.sub(left_val, right_val, span),
            Op::Mul => self.mul(left_val, right_val, span),
            Op::Div => self.div(left_val, right_val, span),
            Op::Mod => self.rem(left_val, right_val, span),
            Op::Pow => self.pow(left_val, right_val, span),
            Op::Equal => self.equal(left_val, right_val),
            Op::NotEqual => self.not_equal(left_val, right_val),
            Op::Less => self.less(left_val, right_val, span),
            Op::LessEqual => self.less_equal(left_val, right_val, span),
            Op::Greater => self.greater(left_val, right_val, span),
            Op::GreaterEqual => self.greater_equal(left_val, right_val, span),
            Op::And => self.and(left_val, right_val, span),
            Op::Or => self.or(left_val, right_val, span),
            Op::Not => Err(WalrusError::UnknownError {
                message: format!("Operator '{}' requires one operand", op),
            })?,
        }
        // fixme: the spans can cause the error span to sometimes be greater than the actual span of the
        // operation taking place because the spans get extended to the left and right of the operation
        // as we traverse the tree. its not a huge deal but it would be nice to fix
    }

    fn visit_unary_op(&mut self, op: Op, value: Node, span: Span) -> InterpreterResult {
        let right_res = self.interpret(value)?;

        match op {
            Op::Sub => self.neg(right_res, span),
            Op::Not => self.not(right_res, span),
            _ => Err(WalrusError::UnknownError {
                message: "Invalid unary operator".into(),
            })?,
        }
    }

    fn visit_dict(&mut self, dict: Vec<(Box<Node>, Box<Node>)>) -> InterpreterResult {
        let mut map = HashMap::new();

        for (key, value) in dict {
            map.insert(self.interpret(*key)?, self.interpret(*value)?);
        }

        Ok(self.scope.mut_arena().insert_dict(map))
    }

    fn visit_list(&mut self, list: Vec<Box<Node>>) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.interpret(*node)?);
        }

        Ok(self.scope.mut_arena().insert_list(vec))
    }

    fn visit_anon_fn_def(&mut self, args: Vec<String>, body: Node) -> InterpreterResult {
        let fn_name = format!("anon_{}", Uuid::new_v4());
        Ok(self.scope.mut_arena().insert_function(fn_name, args, body))
    }

    fn visit_fn_def(&mut self, name: String, args: Vec<String>, body: Node) -> InterpreterResult {
        let value = self
            .scope
            .mut_arena()
            .insert_function(name.clone(), args, body);

        self.scope.define(name, value);

        Ok(ValueKind::Void)
    }

    fn visit_var(&self, name: String, span: Span) -> InterpreterResult {
        self.get_variable(&name, span)
    }

    fn visit_if(
        &mut self,
        condition: Node,
        body: Node,
        otherwise: Option<Box<Node>>,
    ) -> InterpreterResult {
        let cond_span = *condition.span();
        let condition = self.interpret(condition)?;

        if self.is_truthy(condition, cond_span)? {
            self.interpret(body)?;
        } else if let Some(otherwise) = otherwise {
            self.interpret(*otherwise)?;
        }

        Ok(ValueKind::Void)
    }

    fn visit_while(&mut self, condition: Node, body: Node) -> InterpreterResult {
        // this is more complicated because we need to be able to repeatedly
        // evaluate the condition and body, which involves a lot of cloning
        // that we should avoid if possible. we could store our nodes in an
        // arena and copy the arena index instead of the node itself, but
        // that would require a lot of refactoring and would be a lot of work
        // for a small optimization, so for now, we'll just clone and revisit
        // this later
        let cond_span = *condition.span();

        loop {
            let condition = self.interpret(condition.clone())?;

            if self.is_truthy(condition, cond_span)? {
                self.interpret(body.clone())?;
            } else {
                break;
            }
        }

        Ok(ValueKind::Void)
    }

    fn visit_assign(&mut self, name: String, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        self.scope.define(name, value);

        Ok(ValueKind::Void)
    }

    fn visit_reassign(&mut self, ident: Spanned<String>, value: Node, op: Op) -> InterpreterResult {
        let new_value = self.interpret(value)?;
        let old_value = self.get_variable(ident.value(), ident.span())?;

        // fixme: clone
        // fixme: operator such as +=, -=, etc should be handled here
        if !self.scope.reassign(ident.value().clone(), new_value) {
            return Err(UndefinedVariable {
                name: ident.value().clone(), // fixme: clone
                span: ident.span(),
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            });
        }

        Ok(ValueKind::Void)
    }

    fn visit_throw(&mut self, value: Node, span: Span) -> InterpreterResult {
        let value = self.interpret(value)?;

        Err(WalrusError::Exception {
            message: self.stringify(value)?,
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    fn visit_print(&mut self, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        println!("{}", self.stringify(value)?);

        Ok(ValueKind::Void)
    }

    fn stringify(&self, value: ValueKind) -> Result<String, WalrusError> {
        match value {
            ValueKind::Void => Ok("void".into()),
            ValueKind::String(s) => Ok(self.scope.arena().get_string(s)?.clone()),
            ValueKind::List(l) => {
                let list = self.scope.arena().get_list(l)?;
                let mut string = String::new();

                string.push('[');

                for (i, value) in list.iter().enumerate() {
                    string.push_str(&self.stringify(*value)?);

                    if i != list.len() - 1 {
                        string.push_str(", ");
                    }
                }

                string.push(']');

                Ok(string)
            }
            ValueKind::Dict(d) => {
                let dict = self.scope.arena().get_dict(d)?;
                let mut string = String::new();

                string.push('{');

                for (i, (key, value)) in dict.iter().enumerate() {
                    string.push_str(&self.stringify(*key)?);
                    string.push_str(": ");
                    string.push_str(&self.stringify(*value)?);

                    if i != dict.len() - 1 {
                        string.push_str(", ");
                    }
                }

                string.push('}');

                Ok(string)
            }
            ValueKind::Function(f) => {
                let function = self.scope.arena().get_function(f)?;
                Ok(format!("fn {}", function.0))
            }
            ValueKind::RustFunction(f) => {
                let function = self.scope.arena().get_rust_function(f)?;
                Ok(format!("rust fn {:?}", function)) // fixme
            }
            ValueKind::Int(n) => Ok(n.to_string()),
            ValueKind::Float(n) => Ok(n.0.to_string()),
            ValueKind::Bool(b) => Ok(b.to_string()),
        }
    }

    fn visit_free(&mut self, value: Node) -> InterpreterResult {
        let span = *value.span();
        let result = self.interpret(value)?;

        if !self.scope.mut_arena().free(result) {
            Err(WalrusError::FailedFree {
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })?
        }

        Ok(ValueKind::Void)
    }

    fn visit_fn_call(
        &mut self,
        value: Node,
        args: Vec<Box<Node>>,
        span: Span,
    ) -> InterpreterResult {
        let value = self.interpret(value)?;

        let args = args
            .into_iter()
            .map(|arg| self.interpret(*arg))
            .collect::<Result<Vec<_>, _>>()?;

        match value {
            ValueKind::Function(f) => {
                let function = self.scope.arena().get_function(f)?;

                if function.1.len() != args.len() {
                    Err(WalrusError::InvalidArgCount {
                        name: function.0.clone(),
                        expected: function.1.len(),
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    })?
                }

                // fixme: when sub_interpreter gets dropped, it frees values in the arena causing
                // valid memory addresses to become invalid. maybe we only want one arena that is
                // shared between all interpreters and owned by the top level interpreter using
                // a cow, or maybe we want to use a reference counted arena.
                let mut sub_interpreter = self.create_child(function.0.clone()); // todo: clone

                for (name, value) in function.1.iter().zip(args) {
                    sub_interpreter.scope.define(name.clone(), value); // todo: clone
                }

                // todo: I'm pretty sure this clone is required unless I want
                // to do some crazy optimization stuff such as putting it in an
                // arena but that's a lot of work for a small optimization
                // this comment is very similar to the one in visit_while
                sub_interpreter.interpret(function.2.clone())
            }
            ValueKind::RustFunction(f) => {
                let function = self.scope.arena().get_rust_function(f)?;
                function(args)
            }
            _ => Err(WalrusError::NotCallable {
                value: value.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    fn visit_index(&mut self, value: Node, index: Node) -> InterpreterResult {
        let index_span = *index.span();
        let value_span = *value.span();
        let value = self.interpret(value)?;
        let index = self.interpret(index)?;

        match value {
            ValueKind::List(l) => {
                let list = self.scope.arena().get_list(l)?;

                match index {
                    ValueKind::Int(n) => {
                        if n < 0 || n as usize >= list.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: n as usize, // todo: lossy conversion
                                len: list.len(),
                                span: index_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        Ok(list[n as usize])
                    }
                    _ => Err(WalrusError::InvalidIndexType {
                        non_indexable: value.get_type().to_string(),
                        index_type: index.get_type().to_string(),
                        span: index_span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    }),
                }
            }
            ValueKind::Dict(d) => {
                let dict = self.scope.arena().get_dict(d)?;

                // fixme: this doesn't work because it's comparing pointers instead of values
                // so when the variable is on the heap, it will be different. the only way to
                // fix this is to compare the actual values, which means we need to implement
                // PartialEq but we also can't do that because we won't be able to access
                // the arena from the ValueKind struct. we could make the arena a global
                // variable but that's not a good idea. we could also make the arena a
                // reference counted pointer and that's probably the best idea.
                match dict.get(&index) {
                    Some(value) => Ok(*value),
                    None => Ok(ValueKind::Void), // todo: maybe return an error? and let the function return void if it wants to
                }
            }
            _ => Err(WalrusError::NotIndexable {
                value: value.get_type().to_string(),
                span: value_span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    fn add(&mut self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Int(a + b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Float(FloatOrd(a + b)))
            }
            (ValueKind::String(a), ValueKind::String(b)) => {
                let a_str = self.scope.arena().get_string(a)?.clone();
                let b_str = self.scope.arena().get_string(b)?.clone();
                // above clones required because we need to move the strings out of the arena

                Ok(self.scope.mut_arena().insert_string(a_str + b_str.as_str()))
            }
            (a, b) => Err(self.construct_err(Op::Add, a, b, span)),
        }
    }

    fn sub(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Int(a - b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Float(FloatOrd(a - b)))
            }
            (a, b) => Err(self.construct_err(Op::Sub, a, b, span)),
        }
    }

    fn mul(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Int(a * b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Float(FloatOrd(a * b)))
            }
            (a, b) => Err(self.construct_err(Op::Mul, a, b, span)),
        }
    }

    fn div(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Int(a / b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Float(FloatOrd(a / b)))
            }
            (a, b) => Err(self.construct_err(Op::Div, a, b, span)),
        }
    }

    fn rem(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Int(a % b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Float(FloatOrd(a % b)))
            }
            (a, b) => Err(self.construct_err(Op::Mod, a, b, span)),
        }
    }

    fn pow(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Int(a.pow(b as u32))),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Float(FloatOrd(a.powf(b))))
            }
            (a, b) => Err(self.construct_err(Op::Pow, a, b, span)),
        }
    }

    fn neg(&self, value: ValueKind, span: Span) -> InterpreterResult {
        match value {
            ValueKind::Int(a) => Ok(ValueKind::Int(-a)),
            ValueKind::Float(FloatOrd(a)) => Ok(ValueKind::Float(FloatOrd(-a))),
            value => Err(WalrusError::InvalidUnaryOperation {
                op: Op::Sub,
                operand: value.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
        }
    }

    fn equal(&self, left: ValueKind, right: ValueKind) -> InterpreterResult {
        match (left, right) {
            (ValueKind::String(a), ValueKind::String(b)) => {
                let a_str = self.scope.arena().get_string(a)?;
                let b_str = self.scope.arena().get_string(b)?;

                Ok(ValueKind::Bool(a_str == b_str))
            }
            (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                let a_dict = self.scope.arena().get_dict(a)?;
                let b_dict = self.scope.arena().get_dict(b)?;

                Ok(ValueKind::Bool(a_dict == b_dict))
            }
            (ValueKind::List(a), ValueKind::List(b)) => {
                let a_list = self.scope.arena().get_list(a)?;
                let b_list = self.scope.arena().get_list(b)?;

                Ok(ValueKind::Bool(a_list == b_list))
            }
            (ValueKind::Function(a), ValueKind::Function(b)) => {
                let a_func = self.scope.arena().get_function(a)?;
                let b_func = self.scope.arena().get_function(b)?;

                Ok(ValueKind::Bool(a_func == b_func))
            }
            (ValueKind::RustFunction(a), ValueKind::RustFunction(b)) => {
                let a_func = self.scope.arena().get_rust_function(a)?;
                let b_func = self.scope.arena().get_rust_function(b)?;

                Ok(ValueKind::Bool(a_func == b_func))
            }
            _ => Ok(ValueKind::Bool(left == right)),
        }
    }

    fn not_equal(&self, left: ValueKind, right: ValueKind) -> InterpreterResult {
        match (left, right) {
            (ValueKind::String(a), ValueKind::String(b)) => {
                let a_str = self.scope.arena().get_string(a)?;
                let b_str = self.scope.arena().get_string(b)?;

                Ok(ValueKind::Bool(a_str != b_str))
            }
            (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                let a_dict = self.scope.arena().get_dict(a)?;
                let b_dict = self.scope.arena().get_dict(b)?;

                Ok(ValueKind::Bool(a_dict != b_dict))
            }
            (ValueKind::List(a), ValueKind::List(b)) => {
                let a_list = self.scope.arena().get_list(a)?;
                let b_list = self.scope.arena().get_list(b)?;

                Ok(ValueKind::Bool(a_list != b_list))
            }
            (ValueKind::Function(a), ValueKind::Function(b)) => {
                let a_func = self.scope.arena().get_function(a)?;
                let b_func = self.scope.arena().get_function(b)?;

                Ok(ValueKind::Bool(a_func != b_func))
            }
            (ValueKind::RustFunction(a), ValueKind::RustFunction(b)) => {
                let a_func = self.scope.arena().get_rust_function(a)?;
                let b_func = self.scope.arena().get_rust_function(b)?;

                Ok(ValueKind::Bool(a_func != b_func))
            }
            _ => Ok(ValueKind::Bool(left != right)),
        }
    }

    fn less(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Bool(a < b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Bool(a < b))
            }
            (a, b) => Err(self.construct_err(Op::Less, a, b, span)),
        }
    }

    fn less_equal(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Bool(a <= b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Bool(a <= b))
            }
            (a, b) => Err(self.construct_err(Op::LessEqual, a, b, span)),
        }
    }

    fn greater(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Bool(a > b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Bool(a > b))
            }
            (a, b) => Err(self.construct_err(Op::Greater, a, b, span)),
        }
    }

    fn greater_equal(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => Ok(ValueKind::Bool(a >= b)),
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Bool(a >= b))
            }
            (a, b) => Err(self.construct_err(Op::GreaterEqual, a, b, span)),
        }
    }

    fn and(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Bool(a), ValueKind::Bool(b)) => Ok(ValueKind::Bool(a && b)),
            (a, b) => Err(self.construct_err(Op::And, a, b, span)),
        }
    }

    fn or(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Bool(a), ValueKind::Bool(b)) => Ok(ValueKind::Bool(a || b)),
            (a, b) => Err(self.construct_err(Op::Or, a, b, span)),
        }
    }

    fn not(&self, value: ValueKind, span: Span) -> InterpreterResult {
        match value {
            ValueKind::Bool(a) => Ok(ValueKind::Bool(!a)),
            value => Err(WalrusError::InvalidUnaryOperation {
                op: Op::Not,
                operand: value.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
        }
    }

    fn is_truthy(&self, value: ValueKind, span: Span) -> Result<bool, WalrusError> {
        match value {
            ValueKind::Bool(b) => Ok(b),
            value => Err(WalrusError::TypeMismatch {
                expected: "bool".to_string(),
                found: value.get_type().to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            }),
        }
    }

    fn construct_err(&self, op: Op, left: ValueKind, right: ValueKind, span: Span) -> WalrusError {
        WalrusError::InvalidOperation {
            op,
            left: left.get_type().to_string(),
            right: right.get_type().to_string(),
            span,
            src: self.source_ref.source().to_string(),
            filename: self.source_ref.filename().to_string(),
        }
    }
}
