use crate::ast::{Node, NodeKind, Op};
use crate::error::WalrusError;
use crate::program::Program;
use crate::range::RangeValue;
use crate::scope::Scope;
use crate::source_ref::SourceRef;
use crate::span::{Span, Spanned};
use crate::value::{HeapValue, ValueKind};
use float_ord::FloatOrd;
use log::debug;
use rustc_hash::FxHashMap;
use uuid::Uuid;

pub struct Interpreter<'a> {
    scope: Scope,
    source_ref: SourceRef<'a>,
    program: &'a Program,
    is_returning: bool,
}

pub type InterpreterResult = Result<ValueKind, WalrusError>;

impl<'a> Interpreter<'a> {
    pub fn new(source_ref: SourceRef<'a>, program: &'a Program) -> Self {
        Self {
            scope: Scope::new(),
            source_ref,
            program,
            is_returning: false,
        }
    }

    pub fn source_ref(&self) -> SourceRef<'a> {
        self.source_ref
    }

    // for REPL to use
    pub fn set_source_ref(&mut self, src: &'a str) {
        self.source_ref = SourceRef::new(src, self.source_ref.filename());
    }

    pub fn create_child(&'a self, name: String) -> Interpreter<'a> {
        Self {
            scope: self.scope.new_child(name),
            source_ref: self.source_ref,
            is_returning: false,
            program: self.program,
        }
    }

    pub fn interpret(&mut self, node: Node) -> InterpreterResult {
        let span = *node.span();

        let res = match node.into_kind() {
            NodeKind::Statements(nodes) => Ok(self.visit_statements(nodes)?),
            NodeKind::BinOp(left, op, right) => Ok(self.visit_bin_op(*left, op, *right, span)?),
            NodeKind::UnaryOp(op, value) => Ok(self.visit_unary_op(op, *value, span)?),
            NodeKind::Int(num) => Ok(ValueKind::Int(num)),
            NodeKind::Float(num) => Ok(ValueKind::Float(num)),
            NodeKind::Bool(boolean) => Ok(ValueKind::Bool(boolean)),
            NodeKind::String(string) => Ok(Scope::heap_alloc(HeapValue::String(string))),
            NodeKind::Dict(dict) => Ok(self.visit_dict(dict)?),
            NodeKind::List(list) => Ok(self.visit_list(list)?),
            NodeKind::FunctionDefinition(name, args, body) => {
                Ok(self.visit_fn_def(name, args, *body)?)
            }
            NodeKind::AnonFunctionDefinition(args, body) => {
                Ok(self.visit_anon_fn_def(args, *body)?)
            }
            NodeKind::Ident(ident) => Ok(self.visit_variable(&ident, span)?),
            NodeKind::Void => Ok(ValueKind::Void),
            NodeKind::If(condition, then, otherwise) => {
                Ok(self.visit_if(*condition, *then, otherwise)?)
            }
            NodeKind::While(condition, body) => Ok(self.visit_while(*condition, *body)?),
            NodeKind::Assign(name, value) => Ok(self.visit_assign(name, *value)?),
            NodeKind::Reassign(ident, value, op) => {
                Ok(self.visit_reassign(ident, *value, op, span)?)
            }
            NodeKind::Print(value) => Ok(self.visit_print(*value)?),
            NodeKind::Println(value) => Ok(self.visit_println(*value)?),
            NodeKind::Throw(value) => Ok(self.visit_throw(*value, span)?),
            NodeKind::Free(value) => Ok(self.visit_free(*value)?),
            NodeKind::FunctionCall(value, args) => Ok(self.visit_fn_call(*value, args, span)?),
            NodeKind::Index(value, index) => Ok(self.visit_index(*value, *index)?),
            NodeKind::ModuleImport(name, as_name) => Ok(self.visit_module_import(name, as_name)?),
            NodeKind::PackageImport(name, as_name) => Ok(self.visit_package_import(name, as_name)?),
            NodeKind::For(var, iter, body) => Ok(self.visit_for(var, *iter, *body)?),
            NodeKind::Return(value) => Ok(self.visit_return(*value)?),
            NodeKind::IndexAssign(value, index, new_value) => {
                Ok(self.visit_index_assign(*value, *index, *new_value)?)
            }
            NodeKind::Range(start, end) => Ok(self.visit_range(start, end)?),
            node => Err(WalrusError::UnknownError {
                message: format!("Unknown node: {:?}", node),
            }),
        };

        debug!("Interpreted: {:?}", res);
        res
    }

    fn visit_variable(&self, name: &str, span: Span) -> InterpreterResult {
        self.scope
            .get(name)
            .ok_or_else(|| WalrusError::UndefinedVariable {
                name: name.to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            })
    }

    // fixme: returns in blocks that aren't immediately in a function don't return from the function, but from the block
    fn visit_statements(&mut self, nodes: Vec<Box<Node>>) -> InterpreterResult {
        let mut sub_interpreter = self.create_child("name".to_string()); // fixme: should be name of func, for loop, etc

        for node in nodes {
            let res = sub_interpreter.interpret(*node)?;

            if sub_interpreter.is_returning {
                self.is_returning = true;
                return Ok(res);
            }
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
        let mut map = FxHashMap::default();

        for (key, value) in dict {
            map.insert(self.interpret(*key)?, self.interpret(*value)?);
        }

        Ok(Scope::heap_alloc(HeapValue::Dict(map)))
    }

    fn visit_list(&mut self, list: Vec<Box<Node>>) -> InterpreterResult {
        let mut vec = vec![];

        for node in list {
            vec.push(self.interpret(*node)?);
        }

        Ok(Scope::heap_alloc(HeapValue::List(vec)))
    }

    fn visit_anon_fn_def(&mut self, args: Vec<String>, body: Node) -> InterpreterResult {
        let fn_name = format!("anon_{}", Uuid::new_v4());

        Ok(Scope::heap_alloc(HeapValue::Function((
            fn_name, args, body,
        ))))
    }

    fn visit_fn_def(&mut self, name: String, args: Vec<String>, body: Node) -> InterpreterResult {
        let value = Scope::heap_alloc(HeapValue::Function((name.clone(), args, body)));

        self.scope.define(name, value);
        Ok(ValueKind::Void)
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
            Ok(self.interpret(body)?)
        } else if let Some(otherwise) = otherwise {
            Ok(self.interpret(*otherwise)?)
        } else {
            Ok(ValueKind::Void)
        }
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

            // fixme: i believe this will fail to return a value if a return statement is
            // encountered in the body of the while loop.
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

    fn visit_reassign(
        &mut self,
        ident: Spanned<String>,
        value: Node,
        op: Op,
        span: Span,
    ) -> InterpreterResult {
        let new_value = self.interpret(value)?;
        let old_value = self.visit_variable(ident.value(), ident.span())?;

        let new_value = match op {
            Op::Add => self.add(old_value, new_value, span)?,
            Op::Sub => self.sub(old_value, new_value, span)?,
            Op::Mul => self.mul(old_value, new_value, span)?,
            Op::Div => self.div(old_value, new_value, span)?,
            Op::Mod => self.rem(old_value, new_value, span)?,
            Op::Pow => self.pow(old_value, new_value, span)?,
            Op::Or => self.or(old_value, new_value, span)?,
            Op::And => self.and(old_value, new_value, span)?,
            _ => new_value,
        };

        let span = ident.span();

        if let Err(name) = self.scope.reassign(ident.value(), new_value) {
            Err(WalrusError::UndefinedVariable {
                name: name.to_string(),
                span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
            })?
        };

        Ok(ValueKind::Void)
    }

    fn visit_throw(&mut self, value: Node, span: Span) -> InterpreterResult {
        let value = self.interpret(value)?;

        Err(WalrusError::Exception {
            message: value.stringify()?,
            span,
            src: self.source_ref.source().into(),
            filename: self.source_ref.filename().into(),
        })
    }

    fn visit_print(&mut self, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        print!("{}", value.stringify()?);

        Ok(ValueKind::Void)
    }

    fn visit_println(&mut self, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        println!("{}", value.stringify()?);

        Ok(ValueKind::Void)
    }

    fn visit_free(&mut self, value: Node) -> InterpreterResult {
        let span = *value.span();
        let result = self.interpret(value)?;

        if !Scope::free(result) {
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
        let fn_span = *value.span();
        let value = self.interpret(value)?;

        let args = args
            .into_iter()
            .map(|arg| self.interpret(*arg))
            .collect::<Result<Vec<_>, _>>()?;

        match value {
            ValueKind::Function(f) => {
                let function = Scope::get_function(f)?;

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

                // fixme: this creates a double nested interpreter child because statements also creates a new child
                // for now this is okay but for performance reasons we should probably avoid this
                let mut sub_interpreter = self.create_child(function.0.clone()); // todo: clone

                for (name, value) in function.1.iter().zip(args) {
                    sub_interpreter.scope.define(name.clone(), value); // todo: clone
                }

                // todo: I'm pretty sure this clone is required but see if we can avoid it
                Ok(sub_interpreter.interpret(function.2.clone())?)
            }
            ValueKind::RustFunction(f) => {
                let function = Scope::get_rust_function(f)?;

                if let Some(rust_args) = function.args() {
                    if rust_args != args.len() {
                        Err(WalrusError::InvalidArgCount {
                            name: "rust fn".to_string(), // fixme: get name
                            expected: args.len(),
                            got: rust_args,
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        })?
                    }
                }

                function.call(args, self.source_ref, span)
            }
            _ => Err(WalrusError::NotCallable {
                value: value.get_type().to_string(),
                span: fn_span,
                src: self.source_ref.source().to_string(),
                filename: self.source_ref.filename().to_string(),
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
                let list = Scope::get_list(l)?;

                match index {
                    // todo: make sure all these usize to i64 and vice versa conversions are safe
                    ValueKind::Int(n) => {
                        let index = if n < 0 { n + list.len() as i64 } else { n };

                        if index < 0 || index as usize >= list.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: n,
                                len: list.len(),
                                span: index_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        Ok(list[index as usize])
                    }
                    ValueKind::Range(range) => {
                        let start = range.start();
                        let end = range.end();
                        let start_span = range.start_span();
                        let end_span = range.end_span();

                        let end = if end < 0 {
                            end + list.len() as i64
                        } else {
                            end
                        };

                        if start < 0 || start as usize >= list.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: start,
                                len: list.len(),
                                span: start_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        if end < 0 || end as usize >= list.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: end,
                                len: list.len(),
                                span: end_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        let sublist = list[start as usize..(end + 1) as usize].to_vec();
                        Ok(Scope::heap_alloc(HeapValue::List(sublist)))
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
                let dict = Scope::get_dict(d)?;

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
            ValueKind::String(s) => {
                let string = Scope::get_string(s)?;

                match index {
                    ValueKind::Int(n) => {
                        let index = if n < 0 { n + string.len() as i64 } else { n };

                        if index < 0 || index as usize >= string.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: n, // fixme: lossy conversion
                                len: string.len(),
                                span: index_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        let index = index as usize;

                        Ok(Scope::heap_alloc(HeapValue::String(
                            (&string[index..index + 1]).to_string(),
                        )))
                    }
                    // fixme: this is a exact copy of the above code, make a function for this
                    ValueKind::Range(range) => {
                        let start = range.start();
                        let end = range.end();
                        let start_span = range.start_span();
                        let end_span = range.end_span();

                        let end = if end < 0 {
                            end + string.len() as i64
                        } else {
                            end
                        };

                        if start < 0 || start as usize >= string.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: start,
                                len: string.len(),
                                span: start_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        if end < 0 || end as usize >= string.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: end,
                                len: string.len(),
                                span: end_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        let substring = string[start as usize..(end + 1) as usize].to_string();
                        Ok(Scope::heap_alloc(HeapValue::String(substring)))
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
            _ => Err(WalrusError::NotIndexable {
                value: value.get_type().to_string(),
                span: value_span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    // fixme: when 2 files import each other, it loops
    // fixme: when importing a function, for example, it clones the function rather than just referencing it
    fn visit_module_import(&mut self, name: String, as_name: Option<String>) -> InterpreterResult {
        if let Some(as_name) = as_name {
            // fixme: we need to get program as mutable
            // self.scope.define(as_name, self.program.load_module(&name)?);
        }

        Ok(ValueKind::Void)
    }

    fn visit_package_import(&mut self, name: String, as_name: Option<String>) -> InterpreterResult {
        todo!("importing packages")
    }

    fn visit_for(&mut self, name: String, value: Node, body: Node) -> InterpreterResult {
        let value_span = *value.span();
        let value = self.interpret(value)?;

        // todo: see visit_for comment for info on possible optimization and why clone is required
        // fixme: i believe this will fail to return a value if a return statement is
        // encountered in the body of the for loop.
        match value {
            ValueKind::List(l) => {
                let list = Scope::get_list(l)?;

                for value in list {
                    // fixme: this is here because I need to be able to put values in the scope
                    // even though statements already get a new scope. same for the dict and string
                    // and range cases below
                    let mut sub_interpreter = self.create_child("for_list".to_string());

                    if sub_interpreter.scope.is_defined(&name) {
                        sub_interpreter.scope.reassign(&name, *value).unwrap_or(());
                    } else {
                        sub_interpreter.scope.define(name.clone(), *value);
                    };

                    sub_interpreter.interpret(body.clone())?;
                }

                Ok(ValueKind::Void)
            }
            ValueKind::Dict(d) => {
                let dict = Scope::get_dict(d)?;

                for (key, value) in dict {
                    let mut sub_interpreter = self.create_child("for_dict".to_string());

                    if sub_interpreter.scope.is_defined(&name) {
                        sub_interpreter.scope.reassign(&name, *key).unwrap_or(());
                    } else {
                        sub_interpreter.scope.define(name.clone(), *key);
                    };

                    // fixme: this gets overwritten, we need to support destructuring
                    sub_interpreter.scope.define(name.clone(), *value);
                    sub_interpreter.interpret(body.clone())?;
                }

                Ok(ValueKind::Void)
            }
            ValueKind::String(s) => {
                let string = Scope::get_string(s)?;

                for character in string.chars() {
                    let mut sub_interpreter = self.create_child("for_string".to_string());
                    if sub_interpreter.scope.is_defined(&name) {
                        sub_interpreter
                            .scope
                            .reassign(
                                &name,
                                Scope::heap_alloc(HeapValue::String(character.to_string())),
                            )
                            .unwrap_or(());
                    } else {
                        sub_interpreter.scope.define(
                            name.clone(),
                            Scope::heap_alloc(HeapValue::String(character.to_string())),
                        );
                    };
                    sub_interpreter.interpret(body.clone())?;
                }

                Ok(ValueKind::Void)
            }
            ValueKind::Range(range) => {
                let start = range.start();
                let end = range.end();

                for i in start..end {
                    let mut sub_interpreter = self.create_child("for_range".to_string());
                    sub_interpreter
                        .scope
                        .define(name.clone(), ValueKind::Int(i));

                    sub_interpreter.interpret(body.clone())?;
                }

                Ok(ValueKind::Void)
            }
            _ => Err(WalrusError::NotIterable {
                type_name: value.get_type().to_string(),
                span: value_span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    fn visit_return(&mut self, value: Node) -> InterpreterResult {
        let value = self.interpret(value)?;
        self.is_returning = true;
        Ok(value)
    }

    fn visit_index_assign(
        &mut self,
        value: Node,
        index: Node,
        new_value: Node,
    ) -> InterpreterResult {
        let value_span = *value.span();
        let value = self.interpret(value)?;
        let index_span = *index.span();
        let index = self.interpret(index)?;
        let new_value = self.interpret(new_value)?;

        match value {
            ValueKind::List(l) => {
                let list = Scope::get_mut_list(l)?;

                match index {
                    ValueKind::Int(n) => {
                        let index = if n < 0 { n + list.len() as i64 } else { n };

                        if index < 0 || index as usize >= list.len() {
                            Err(WalrusError::IndexOutOfBounds {
                                index: n, // fixme: lossy conversion
                                len: list.len(),
                                span: index_span,
                                src: self.source_ref.source().into(),
                                filename: self.source_ref.filename().into(),
                            })?
                        }

                        list[index as usize] = new_value;
                        Ok(ValueKind::Void)
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
                let dict = Scope::get_mut_dict(d)?;
                dict.insert(index, new_value);

                Ok(ValueKind::Void)
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

    fn visit_range(
        &mut self,
        start: Option<Box<Node>>,
        end: Option<Box<Node>>,
    ) -> InterpreterResult {
        let (start, start_span) = if let Some(start) = start {
            let start_span = *start.span();
            (self.interpret(*start)?, start_span)
        } else {
            (ValueKind::Int(0), Span::default())
        };

        let (end, end_span) = if let Some(end) = end {
            let end_span = *end.span();
            (self.interpret(*end)?, end_span)
        } else {
            (ValueKind::Int(-1), Span::default())
        };

        match (start, end) {
            (ValueKind::Int(start), ValueKind::Int(end)) => {
                Ok(ValueKind::Range(RangeValue::new(
                    start, start_span, end, end_span,
                ))) // todo: check if this is lossy
            }
            (ValueKind::Int(_), end) => Err(WalrusError::TypeMismatch {
                expected: "int".to_string(),
                found: end.get_type().to_string(),
                span: end_span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
            (start, ValueKind::Int(_)) => Err(WalrusError::TypeMismatch {
                expected: "int".to_string(),
                found: start.get_type().to_string(),
                span: start_span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
            (start, end) => Err(WalrusError::TypeMismatch {
                expected: "int".to_string(),
                found: format!("{} and {}", start.get_type(), end.get_type()),
                span: start_span.extend(end_span),
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
                let a_str = Scope::get_string(a)?.to_string();
                let b_str = Scope::get_string(b)?;

                Ok(Scope::heap_alloc(HeapValue::String(a_str + b_str)))
            }
            (ValueKind::List(a), ValueKind::List(b)) => {
                let mut a_list = Scope::get_list(a)?.clone();
                let b_list = Scope::get_list(b)?;

                a_list.extend(b_list);

                Ok(Scope::heap_alloc(HeapValue::List(a_list)))
            }
            (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                let mut a_dict = Scope::get_dict(a)?.clone();
                let b_dict = Scope::get_dict(b)?;

                a_dict.extend(b_dict);

                Ok(Scope::heap_alloc(HeapValue::Dict(a_dict)))
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
            (ValueKind::Int(a), ValueKind::Int(b)) => {
                if b == 0 {
                    Err(WalrusError::DivisionByZero {
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    })
                } else {
                    Ok(ValueKind::Int(a / b))
                }
            }

            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                if b == 0.0 {
                    Err(WalrusError::DivisionByZero {
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    })
                } else {
                    Ok(ValueKind::Float(FloatOrd(a / b)))
                }
            }
            (a, b) => Err(self.construct_err(Op::Div, a, b, span)),
        }
    }

    fn rem(&self, left: ValueKind, right: ValueKind, span: Span) -> InterpreterResult {
        match (left, right) {
            (ValueKind::Int(a), ValueKind::Int(b)) => {
                if b == 0 {
                    Err(WalrusError::DivisionByZero {
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    })
                } else {
                    Ok(ValueKind::Int(a % b))
                }
            }
            (ValueKind::Float(FloatOrd(a)), ValueKind::Float(FloatOrd(b))) => {
                if b == 0.0 {
                    Err(WalrusError::DivisionByZero {
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    })
                } else {
                    Ok(ValueKind::Float(FloatOrd(a % b)))
                }
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
                let a_str = Scope::get_string(a)?;
                let b_str = Scope::get_string(b)?;

                Ok(ValueKind::Bool(a_str == b_str))
            }
            (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                let a_dict = Scope::get_dict(a)?;
                let b_dict = Scope::get_dict(b)?;

                Ok(ValueKind::Bool(a_dict == b_dict))
            }
            (ValueKind::List(a), ValueKind::List(b)) => {
                let a_list = Scope::get_list(a)?;
                let b_list = Scope::get_list(b)?;

                Ok(ValueKind::Bool(a_list == b_list))
            }
            (ValueKind::Function(a), ValueKind::Function(b)) => {
                let a_func = Scope::get_function(a)?;
                let b_func = Scope::get_function(b)?;

                Ok(ValueKind::Bool(a_func == b_func))
            }
            (ValueKind::Int(a), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Bool(a == b as i64))
            }
            (ValueKind::Float(FloatOrd(a)), ValueKind::Int(b)) => {
                Ok(ValueKind::Bool(a as i64 == b))
            }
            _ => Ok(ValueKind::Bool(left == right)),
        }
    }

    fn not_equal(&self, left: ValueKind, right: ValueKind) -> InterpreterResult {
        match (left, right) {
            (ValueKind::String(a), ValueKind::String(b)) => {
                let a_str = Scope::get_string(a)?;
                let b_str = Scope::get_string(b)?;

                Ok(ValueKind::Bool(a_str != b_str))
            }
            (ValueKind::Dict(a), ValueKind::Dict(b)) => {
                let a_dict = Scope::get_dict(a)?;
                let b_dict = Scope::get_dict(b)?;

                Ok(ValueKind::Bool(a_dict != b_dict))
            }
            (ValueKind::List(a), ValueKind::List(b)) => {
                let a_list = Scope::get_list(a)?;
                let b_list = Scope::get_list(b)?;

                Ok(ValueKind::Bool(a_list != b_list))
            }
            (ValueKind::Function(a), ValueKind::Function(b)) => {
                let a_func = Scope::get_function(a)?;
                let b_func = Scope::get_function(b)?;

                Ok(ValueKind::Bool(a_func != b_func))
            }
            (ValueKind::Int(a), ValueKind::Float(FloatOrd(b))) => {
                Ok(ValueKind::Bool(a != b as i64))
            }
            (ValueKind::Float(FloatOrd(a)), ValueKind::Int(b)) => {
                Ok(ValueKind::Bool(a as i64 != b))
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
