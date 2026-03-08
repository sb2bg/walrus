use super::*;

impl<'a> VM<'a> {
    pub(super) fn create_task_for_function_key(
        &mut self,
        function_key: FuncKey,
        args: Vec<Value>,
    ) -> Value {
        let task = self
            .get_heap_mut()
            .push(HeapValue::Task(AsyncTask::Pending {
                function: function_key,
                args,
            }));
        if let Value::Task(task_key) = task {
            self.enqueue_task(task_key);
        }
        task
    }

    pub(super) fn create_task_for_cloned_function(
        &mut self,
        function: WalrusFunction,
        args: Vec<Value>,
    ) -> Value {
        let func_value = self.get_heap_mut().push(HeapValue::Function(function));
        let Value::Function(function_key) = func_value else {
            unreachable!("HeapValue::Function must allocate a function key")
        };
        self.create_task_for_function_key(function_key, args)
    }

    pub(super) fn create_non_runnable_task(&mut self, task: AsyncTask) -> Value {
        self.get_heap_mut().push(HeapValue::Task(task))
    }

    pub(crate) fn spawn_task_from_callable(
        &mut self,
        callable: Value,
        args: Vec<Value>,
        span: Span,
    ) -> WalrusResult<Value> {
        match callable {
            Value::Task(task_key) => {
                if !args.is_empty() {
                    return Err(WalrusError::InvalidArgCount {
                        name: "spawn".to_string(),
                        expected: 0,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }
                Ok(Value::Task(task_key))
            }
            Value::Function(function_key) => {
                let function = self.get_heap().get_function(function_key)?;
                let expected = match function {
                    WalrusFunction::Vm(func) => func.arity,
                    WalrusFunction::Native(func) => func.arity(),
                };
                if args.len() != expected {
                    return Err(WalrusError::InvalidArgCount {
                        name: function.to_string(),
                        expected,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }
                Ok(self.create_task_for_function_key(function_key, args))
            }
            other => Err(WalrusError::TypeMismatch {
                expected: "function or task".to_string(),
                found: other.get_type().to_string(),
                span,
                src: self.source_ref.source().into(),
                filename: self.source_ref.filename().into(),
            }),
        }
    }

    pub(crate) fn create_sleep_task(&mut self, delay_ms: u64) -> Value {
        let wake_at = Instant::now() + Duration::from_millis(delay_ms);
        self.create_non_runnable_task(AsyncTask::Sleep { wake_at })
    }

    pub(crate) fn create_timeout_task(
        &mut self,
        task_key: crate::arenas::TaskKey,
        timeout_ms: u64,
    ) -> Value {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        self.create_non_runnable_task(AsyncTask::Timeout {
            task: task_key,
            deadline,
        })
    }

    /// Spawn a background I/O operation on the bounded blocking I/O pool.
    /// Returns a Task value backed by a Channel that resolves when the worker completes.
    pub(crate) fn spawn_io<F>(&mut self, work: F) -> Value
    where
        F: FnOnce() -> Result<IoResult, String> + Send + 'static,
    {
        let rx = io_pool::submit_io(work, self.io_wakeup_tx.clone());
        self.create_non_runnable_task(AsyncTask::Channel(IoChannel::new(rx)))
    }

    /// Convert an IoResult from a worker thread into a Value on the VM thread.
    pub(super) fn io_result_to_value(&mut self, result: IoResult) -> WalrusResult<Value> {
        match result {
            IoResult::Stream(stream) => {
                let handle = crate::stdlib::store_tcp_stream(stream);
                Ok(Value::Int(handle))
            }
            IoResult::Listener(listener) => {
                let handle = crate::stdlib::store_tcp_listener(listener);
                Ok(Value::Int(handle))
            }
            IoResult::Bytes(bytes) => {
                let text = String::from_utf8(bytes).map_err(|e| WalrusError::GenericError {
                    message: format!("I/O result contains non-UTF8 data: {e}"),
                })?;
                Ok(self.get_heap_mut().push(HeapValue::String(&text)))
            }
            IoResult::ByteCount(n) => Ok(Value::Int(n as i64)),
            IoResult::HttpOutcome(outcome) => self.http_outcome_to_value(outcome),
            IoResult::Void => Ok(Value::Void),
        }
    }

    /// Convert an IoHttpOutcome into a Value (dict or Void).
    pub(super) fn http_outcome_to_value(
        &mut self,
        outcome: crate::value::IoHttpOutcome,
    ) -> WalrusResult<Value> {
        use crate::value::IoHttpOutcome;

        match outcome {
            IoHttpOutcome::Eof => Ok(Value::Void),
            IoHttpOutcome::BadRequest(message) => {
                let mut dict = FxHashMap::default();
                let ok_key = self.get_heap_mut().push(HeapValue::String("ok"));
                dict.insert(ok_key, Value::Bool(false));
                let err_key = self.get_heap_mut().push(HeapValue::String("error"));
                let err_val = self.get_heap_mut().push(HeapValue::String(&message));
                dict.insert(err_key, err_val);
                Ok(self.get_heap_mut().push(HeapValue::Dict(dict)))
            }
            IoHttpOutcome::Request(req) => {
                let mut headers = FxHashMap::default();
                let mut header_pairs = Vec::with_capacity(req.headers.len());
                for (name, value) in &req.headers {
                    let key = self.get_heap_mut().push(HeapValue::String(name));
                    let val = self.get_heap_mut().push(HeapValue::String(value));
                    headers.insert(key, val);
                    let pair = self.get_heap_mut().push(HeapValue::List(vec![key, val]));
                    header_pairs.push(pair);
                }
                let headers_value = self.get_heap_mut().push(HeapValue::Dict(headers));
                let header_pairs_value = self.get_heap_mut().push(HeapValue::List(header_pairs));

                let query_pairs = crate::stdlib::http_parse_query(&req.query);
                let mut query = FxHashMap::default();
                let mut query_pairs_values = Vec::with_capacity(query_pairs.len());
                for (name, value) in query_pairs {
                    let key = self.get_heap_mut().push(HeapValue::String(&name));
                    let val = self.get_heap_mut().push(HeapValue::String(&value));
                    query.insert(key, val);
                    let pair = self.get_heap_mut().push(HeapValue::List(vec![key, val]));
                    query_pairs_values.push(pair);
                }
                let query_value = self.get_heap_mut().push(HeapValue::Dict(query));
                let query_pairs_value = self
                    .get_heap_mut()
                    .push(HeapValue::List(query_pairs_values));

                let mut dict = FxHashMap::default();
                let ok_key = self.get_heap_mut().push(HeapValue::String("ok"));
                dict.insert(ok_key, Value::Bool(true));
                let method_key = self.get_heap_mut().push(HeapValue::String("method"));
                let method_val = self.get_heap_mut().push(HeapValue::String(&req.method));
                dict.insert(method_key, method_val);
                let target_key = self.get_heap_mut().push(HeapValue::String("target"));
                let target_val = self.get_heap_mut().push(HeapValue::String(&req.target));
                dict.insert(target_key, target_val);
                let path_key = self.get_heap_mut().push(HeapValue::String("path"));
                let path_val = self.get_heap_mut().push(HeapValue::String(&req.path));
                dict.insert(path_key, path_val);
                let query_key = self.get_heap_mut().push(HeapValue::String("query"));
                let query_text_val = self.get_heap_mut().push(HeapValue::String(&req.query));
                dict.insert(query_key, query_text_val);
                let qp_key = self.get_heap_mut().push(HeapValue::String("query_params"));
                dict.insert(qp_key, query_value);
                let qpp_key = self.get_heap_mut().push(HeapValue::String("query_pairs"));
                dict.insert(qpp_key, query_pairs_value);
                let ver_key = self.get_heap_mut().push(HeapValue::String("version"));
                let ver_val = self.get_heap_mut().push(HeapValue::String(&req.version));
                dict.insert(ver_key, ver_val);
                let hdr_key = self.get_heap_mut().push(HeapValue::String("headers"));
                dict.insert(hdr_key, headers_value);
                let hdr_pairs_key = self.get_heap_mut().push(HeapValue::String("header_pairs"));
                dict.insert(hdr_pairs_key, header_pairs_value);
                let body_key = self.get_heap_mut().push(HeapValue::String("body"));
                let body_val = self.get_heap_mut().push(HeapValue::String(&req.body));
                dict.insert(body_key, body_val);
                let cl_key = self
                    .get_heap_mut()
                    .push(HeapValue::String("content_length"));
                dict.insert(cl_key, Value::Int(req.content_length));

                Ok(self.get_heap_mut().push(HeapValue::Dict(dict)))
            }
        }
    }

    /// Check if a task tree has any pending I/O (Channel) tasks.
    pub(super) fn task_has_pending_io(
        &self,
        task_key: crate::arenas::TaskKey,
        visited: &mut FxHashSet<crate::arenas::TaskKey>,
    ) -> WalrusResult<bool> {
        if !visited.insert(task_key) {
            return Ok(false);
        }
        let task = self.get_heap().get_task(task_key)?;
        match task {
            AsyncTask::Channel(_) => Ok(true),
            AsyncTask::UserRecv { .. } => Ok(false),
            AsyncTask::Timeout { task, .. } => self.task_has_pending_io(*task, visited),
            AsyncTask::Gather { tasks }
            | AsyncTask::Race { tasks }
            | AsyncTask::AllSettled { tasks } => {
                for &child in tasks {
                    if self.task_has_pending_io(child, visited)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    pub(crate) fn create_gather_task(&mut self, tasks: Vec<crate::arenas::TaskKey>) -> Value {
        self.create_non_runnable_task(AsyncTask::Gather { tasks })
    }

    pub(crate) fn create_race_task(&mut self, tasks: Vec<crate::arenas::TaskKey>) -> Value {
        self.create_non_runnable_task(AsyncTask::Race { tasks })
    }

    pub(crate) fn create_all_settled_task(&mut self, tasks: Vec<crate::arenas::TaskKey>) -> Value {
        self.create_non_runnable_task(AsyncTask::AllSettled { tasks })
    }

    pub(crate) fn create_user_channel(&mut self) -> (Value, Value) {
        let id = self.user_channels.len();
        self.user_channels.push(UserChannel {
            buffer: Rc::new(RefCell::new(VecDeque::new())),
            closed: Rc::new(RefCell::new(false)),
        });
        let type_key = self.get_heap_mut().push(HeapValue::String("__type"));
        let id_key = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        let id_val = Value::Int(id as i64);

        let sender_type = self.get_heap_mut().push(HeapValue::String("sender"));
        let mut sender_dict = FxHashMap::default();
        sender_dict.insert(type_key, sender_type);
        sender_dict.insert(id_key, id_val);
        let sender = self.get_heap_mut().push(HeapValue::Dict(sender_dict));

        let receiver_type = self.get_heap_mut().push(HeapValue::String("receiver"));
        let mut receiver_dict = FxHashMap::default();
        receiver_dict.insert(type_key, receiver_type);
        receiver_dict.insert(id_key, id_val);
        let receiver = self.get_heap_mut().push(HeapValue::Dict(receiver_dict));

        (sender, receiver)
    }

    pub(crate) fn channel_send(
        &mut self,
        sender_key: crate::arenas::DictKey,
        value: Value,
    ) -> WalrusResult<bool> {
        let dict = self.get_heap().get_dict(sender_key)?.clone();
        let id_key_str = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        if let Some(Value::Int(id)) = dict.get(&id_key_str) {
            let id = *id as usize;
            if id < self.user_channels.len() {
                if *self.user_channels[id].closed.borrow() {
                    return Ok(false);
                }
                self.user_channels[id].buffer.borrow_mut().push_back(value);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn channel_recv(
        &mut self,
        receiver_key: crate::arenas::DictKey,
    ) -> WalrusResult<Value> {
        let dict = self.get_heap().get_dict(receiver_key)?.clone();
        let id_key_str = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        if let Some(Value::Int(id)) = dict.get(&id_key_str) {
            let id = *id as usize;
            if id < self.user_channels.len() {
                let buffered = self.user_channels[id].buffer.borrow_mut().pop_front();
                if let Some(value) = buffered {
                    return Ok(self.create_non_runnable_task(AsyncTask::Ready(value)));
                }
                return Ok(self.create_non_runnable_task(AsyncTask::UserRecv { channel_id: id }));
            }
        }
        Err(WalrusError::GenericError {
            message: "asyncx.recv: invalid receiver".to_string(),
        })
    }

    pub(crate) fn channel_close(
        &mut self,
        endpoint_key: crate::arenas::DictKey,
    ) -> WalrusResult<bool> {
        let dict = self.get_heap().get_dict(endpoint_key)?.clone();
        let id_key_str = self.get_heap_mut().push(HeapValue::String("__channel_id"));
        if let Some(Value::Int(id)) = dict.get(&id_key_str) {
            let id = *id as usize;
            if id < self.user_channels.len() {
                *self.user_channels[id].closed.borrow_mut() = true;
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(super) fn cancelled_task_error_value(&mut self) -> Value {
        self.get_heap_mut()
            .push(HeapValue::String("task cancelled"))
    }

    pub(super) fn cancel_task_recursive_internal(
        &mut self,
        task_key: crate::arenas::TaskKey,
        visited: &mut FxHashSet<crate::arenas::TaskKey>,
    ) -> WalrusResult<bool> {
        if !visited.insert(task_key) {
            return Ok(false);
        }

        let task = self.get_heap().get_task(task_key)?.clone();
        match task {
            AsyncTask::Pending { .. }
            | AsyncTask::Sleep { .. }
            | AsyncTask::Channel(_)
            | AsyncTask::UserRecv { .. } => {
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                self.suspended_tasks.remove(&task_key);
                Ok(true)
            }
            AsyncTask::Timeout { task, .. } => {
                let _ = self.cancel_task_recursive_internal(task, visited)?;
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                self.suspended_tasks.remove(&task_key);
                Ok(true)
            }
            AsyncTask::Gather { tasks }
            | AsyncTask::Race { tasks }
            | AsyncTask::AllSettled { tasks } => {
                for child in tasks {
                    let _ = self.cancel_task_recursive_internal(child, visited)?;
                }
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                self.suspended_tasks.remove(&task_key);
                Ok(true)
            }
            AsyncTask::Ready(_) | AsyncTask::Failed(_) | AsyncTask::Cancelled => Ok(false),
        }
    }

    pub(crate) fn cancel_task(&mut self, task_key: crate::arenas::TaskKey) -> WalrusResult<bool> {
        let mut visited = FxHashSet::default();
        let cancelled = self.cancel_task_recursive_internal(task_key, &mut visited)?;
        if cancelled {
            self.suspended_tasks.remove(&task_key);
            self.wake_task_waiters(task_key)?;
        }
        Ok(cancelled)
    }

    pub(crate) fn run_queued_tasks(&mut self, span: Span) -> WalrusResult<()> {
        self.refresh_waiting_tasks()?;
        while let Some(task_key) = self.next_runnable_task()? {
            self.run_pending_task_to_completion(task_key, span)?;
            self.refresh_waiting_tasks()?;
        }
        Ok(())
    }

    pub(crate) fn task_status_string(
        &self,
        task_key: crate::arenas::TaskKey,
    ) -> WalrusResult<&'static str> {
        let task = self.get_heap().get_task(task_key)?;
        Ok(match task {
            AsyncTask::Pending { .. }
            | AsyncTask::Sleep { .. }
            | AsyncTask::Timeout { .. }
            | AsyncTask::Gather { .. }
            | AsyncTask::Race { .. }
            | AsyncTask::AllSettled { .. }
            | AsyncTask::Channel(_)
            | AsyncTask::UserRecv { .. } => "pending",
            AsyncTask::Ready(_) => "ready",
            AsyncTask::Failed(_) => "failed",
            AsyncTask::Cancelled => "cancelled",
        })
    }

    pub(crate) fn task_is_cancelled(&self, task_key: crate::arenas::TaskKey) -> WalrusResult<bool> {
        let task = self.get_heap().get_task(task_key)?;
        Ok(matches!(task, AsyncTask::Cancelled))
    }

    pub(super) fn complete_task_on_frame_return(
        &mut self,
        task_key: Option<crate::arenas::TaskKey>,
        result: Value,
    ) -> WalrusResult<()> {
        if let Some(task_key) = task_key {
            let task = self.get_heap_mut().get_mut_task(task_key)?;
            if matches!(task, AsyncTask::Pending { .. }) {
                *task = AsyncTask::Ready(result);
            }
        }
        Ok(())
    }

    pub(super) fn set_task_failed(
        &mut self,
        task_key: crate::arenas::TaskKey,
        failure: Value,
    ) -> WalrusResult<()> {
        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(failure);
        self.wake_task_waiters(task_key)?;
        Ok(())
    }

    pub(super) fn task_failure_value_from_error(&mut self, err: WalrusError) -> Value {
        let message = match err {
            WalrusError::ThrownValue { message, .. } => message,
            WalrusError::RuntimeErrorWithStackTrace { error, stack_trace } => {
                format!("{error}\n{stack_trace}")
            }
            other => other.to_string(),
        };
        self.get_heap_mut().push(HeapValue::String(&message))
    }

    pub(super) fn fail_task_with_error(
        &mut self,
        task_key: crate::arenas::TaskKey,
        err: WalrusError,
    ) -> WalrusResult<()> {
        let failure = self.task_failure_value_from_error(err);
        self.set_task_failed(task_key, failure)
    }

    pub(super) fn enqueue_task(&mut self, task_key: crate::arenas::TaskKey) {
        if !self.async_task_queue.contains(&task_key) {
            self.async_task_queue.push_back(task_key);
        }
    }

    pub(super) fn wake_task_waiters(
        &mut self,
        task_key: crate::arenas::TaskKey,
    ) -> WalrusResult<()> {
        let Some(waiters) = self.task_waiters.remove(&task_key) else {
            return Ok(());
        };

        for waiter in waiters {
            if self.suspended_tasks.contains_key(&waiter)
                && !matches!(
                    self.get_heap().get_task(waiter)?,
                    AsyncTask::Ready(_) | AsyncTask::Failed(_) | AsyncTask::Cancelled
                )
            {
                self.enqueue_task(waiter);
            }
        }

        Ok(())
    }

    pub(super) fn refresh_waiting_tasks(&mut self) -> WalrusResult<()> {
        let mut watched = FxHashSet::default();
        if let Some(suspended) = &self.suspended_main {
            watched.insert(suspended.waiting_on);
        }
        watched.extend(self.task_waiters.keys().copied());

        for task_key in watched {
            if matches!(
                self.poll_task_resolution(task_key)?,
                TaskResolution::Ready(_) | TaskResolution::Failed(_) | TaskResolution::Cancelled
            ) {
                self.wake_task_waiters(task_key)?;
            }
        }

        Ok(())
    }

    pub(super) fn is_task_runnable(
        &mut self,
        task_key: crate::arenas::TaskKey,
    ) -> WalrusResult<bool> {
        let task = self.get_heap().get_task(task_key)?;
        if !matches!(task, AsyncTask::Pending { .. }) {
            return Ok(false);
        }

        if let Some(suspended) = self.suspended_tasks.get(&task_key) {
            return Ok(matches!(
                self.poll_task_resolution(suspended.waiting_on)?,
                TaskResolution::Ready(_) | TaskResolution::Failed(_) | TaskResolution::Cancelled
            ));
        }

        Ok(true)
    }

    pub(super) fn poll_task_resolution(
        &mut self,
        task_key: crate::arenas::TaskKey,
    ) -> WalrusResult<TaskResolution> {
        let task = self.get_heap().get_task(task_key)?.clone();
        match task {
            AsyncTask::Pending { .. } => Ok(TaskResolution::Pending),
            AsyncTask::Ready(value) => Ok(TaskResolution::Ready(value)),
            AsyncTask::Failed(value) => Ok(TaskResolution::Failed(value)),
            AsyncTask::Cancelled => Ok(TaskResolution::Cancelled),
            AsyncTask::Sleep { wake_at } => {
                if Instant::now() >= wake_at {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(Value::Void);
                    Ok(TaskResolution::Ready(Value::Void))
                } else {
                    Ok(TaskResolution::Pending)
                }
            }
            AsyncTask::Timeout { task, deadline } => match self.poll_task_resolution(task)? {
                TaskResolution::Ready(value) => {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                    Ok(TaskResolution::Ready(value))
                }
                TaskResolution::Failed(value) => {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(value);
                    Ok(TaskResolution::Failed(value))
                }
                TaskResolution::Cancelled => {
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                    Ok(TaskResolution::Cancelled)
                }
                TaskResolution::Pending => {
                    if Instant::now() >= deadline {
                        let message = self
                            .get_heap_mut()
                            .push(HeapValue::String("task timed out"));
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(message);
                        Ok(TaskResolution::Failed(message))
                    } else {
                        Ok(TaskResolution::Pending)
                    }
                }
            },
            AsyncTask::Channel(ref channel) => match channel.try_recv() {
                Ok(Ok(io_result)) => {
                    let value = self.io_result_to_value(io_result)?;
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                    Ok(TaskResolution::Ready(value))
                }
                Ok(Err(error_msg)) => {
                    let error = self.get_heap_mut().push(HeapValue::String(&error_msg));
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(error);
                    Ok(TaskResolution::Failed(error))
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => Ok(TaskResolution::Pending),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let error = self
                        .get_heap_mut()
                        .push(HeapValue::String("I/O worker thread disconnected"));
                    *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(error);
                    Ok(TaskResolution::Failed(error))
                }
            },
            AsyncTask::Gather { tasks } => {
                let mut values = Vec::with_capacity(tasks.len());
                for child in tasks {
                    match self.poll_task_resolution(child)? {
                        TaskResolution::Ready(value) => values.push(value),
                        TaskResolution::Failed(value) => {
                            *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(value);
                            return Ok(TaskResolution::Failed(value));
                        }
                        TaskResolution::Cancelled => {
                            *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                            return Ok(TaskResolution::Cancelled);
                        }
                        TaskResolution::Pending => {
                            return Ok(TaskResolution::Pending);
                        }
                    }
                }

                let list = self.get_heap_mut().push(HeapValue::List(values));
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(list);
                Ok(TaskResolution::Ready(list))
            }
            AsyncTask::Race { tasks } => {
                let mut all_failed = true;
                let mut last_failure = None;
                for child in tasks {
                    match self.poll_task_resolution(child)? {
                        TaskResolution::Ready(value) => {
                            *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                            return Ok(TaskResolution::Ready(value));
                        }
                        TaskResolution::Failed(value) => {
                            last_failure = Some(value);
                        }
                        TaskResolution::Cancelled => {}
                        TaskResolution::Pending => {
                            all_failed = false;
                        }
                    }
                }
                if all_failed {
                    if let Some(failure) = last_failure {
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Failed(failure);
                        Ok(TaskResolution::Failed(failure))
                    } else {
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Cancelled;
                        Ok(TaskResolution::Cancelled)
                    }
                } else {
                    Ok(TaskResolution::Pending)
                }
            }
            AsyncTask::AllSettled { tasks } => {
                let mut results = Vec::with_capacity(tasks.len());
                for child in tasks {
                    match self.poll_task_resolution(child)? {
                        TaskResolution::Ready(value) => {
                            let status_str = self.get_heap_mut().push(HeapValue::String("ok"));
                            let value_str = self.get_heap_mut().push(HeapValue::String("value"));
                            let status_key = self.get_heap_mut().push(HeapValue::String("status"));
                            let mut dict = FxHashMap::default();
                            dict.insert(status_key, status_str);
                            dict.insert(value_str, value);
                            results.push(self.get_heap_mut().push(HeapValue::Dict(dict)));
                        }
                        TaskResolution::Failed(value) => {
                            let status_str = self.get_heap_mut().push(HeapValue::String("error"));
                            let error_str = self.get_heap_mut().push(HeapValue::String("error"));
                            let status_key = self.get_heap_mut().push(HeapValue::String("status"));
                            let mut dict = FxHashMap::default();
                            dict.insert(status_key, status_str);
                            dict.insert(error_str, value);
                            results.push(self.get_heap_mut().push(HeapValue::Dict(dict)));
                        }
                        TaskResolution::Cancelled => {
                            let status_str =
                                self.get_heap_mut().push(HeapValue::String("cancelled"));
                            let status_key = self.get_heap_mut().push(HeapValue::String("status"));
                            let mut dict = FxHashMap::default();
                            dict.insert(status_key, status_str);
                            results.push(self.get_heap_mut().push(HeapValue::Dict(dict)));
                        }
                        TaskResolution::Pending => {
                            return Ok(TaskResolution::Pending);
                        }
                    }
                }
                let list = self.get_heap_mut().push(HeapValue::List(results));
                *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(list);
                Ok(TaskResolution::Ready(list))
            }
            AsyncTask::UserRecv { channel_id } => {
                if channel_id < self.user_channels.len() {
                    let mut buf = self.user_channels[channel_id].buffer.borrow_mut();
                    if let Some(value) = buf.pop_front() {
                        drop(buf);
                        *self.get_heap_mut().get_mut_task(task_key)? = AsyncTask::Ready(value);
                        return Ok(TaskResolution::Ready(value));
                    }
                    drop(buf);
                    if *self.user_channels[channel_id].closed.borrow() {
                        *self.get_heap_mut().get_mut_task(task_key)? =
                            AsyncTask::Ready(Value::Void);
                        return Ok(TaskResolution::Ready(Value::Void));
                    }
                }
                Ok(TaskResolution::Pending)
            }
        }
    }

    pub(super) fn next_deadline_for_task(
        &self,
        task_key: crate::arenas::TaskKey,
        visited: &mut FxHashSet<crate::arenas::TaskKey>,
    ) -> WalrusResult<Option<Instant>> {
        if !visited.insert(task_key) {
            return Ok(None);
        }

        let task = self.get_heap().get_task(task_key)?;
        match task {
            AsyncTask::Sleep { wake_at } => Ok(Some(*wake_at)),
            AsyncTask::Timeout { task, deadline } => {
                let nested = self.next_deadline_for_task(*task, visited)?;
                Ok(match nested {
                    Some(value) => Some((*deadline).min(value)),
                    None => Some(*deadline),
                })
            }
            AsyncTask::Gather { tasks }
            | AsyncTask::Race { tasks }
            | AsyncTask::AllSettled { tasks } => {
                let mut soonest: Option<Instant> = None;
                for &child in tasks {
                    if let Some(deadline) = self.next_deadline_for_task(child, visited)? {
                        soonest = Some(match soonest {
                            Some(current) => current.min(deadline),
                            None => deadline,
                        });
                    }
                }
                Ok(soonest)
            }
            AsyncTask::Pending { .. }
            | AsyncTask::Channel(_)
            | AsyncTask::UserRecv { .. }
            | AsyncTask::Ready(_)
            | AsyncTask::Failed(_)
            | AsyncTask::Cancelled => Ok(None),
        }
    }

    pub(super) fn next_scheduler_deadline(&self) -> WalrusResult<Option<Instant>> {
        let mut watched = FxHashSet::default();
        let mut deadline: Option<Instant> = None;

        if let Some(suspended) = &self.suspended_main {
            watched.insert(suspended.waiting_on);
        }
        watched.extend(self.task_waiters.keys().copied());

        for task_key in watched {
            let mut visited = FxHashSet::default();
            if let Some(next) = self.next_deadline_for_task(task_key, &mut visited)? {
                deadline = Some(match deadline {
                    Some(current) => current.min(next),
                    None => next,
                });
            }
        }

        Ok(deadline)
    }

    pub(super) fn scheduler_has_pending_io(&self) -> WalrusResult<bool> {
        let mut watched = FxHashSet::default();
        if let Some(suspended) = &self.suspended_main {
            watched.insert(suspended.waiting_on);
        }
        watched.extend(self.task_waiters.keys().copied());

        for task_key in watched {
            let mut visited = FxHashSet::default();
            if self.task_has_pending_io(task_key, &mut visited)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub(super) fn wait_for_scheduler_progress(&mut self) -> WalrusResult<()> {
        let deadline = self.next_scheduler_deadline()?;
        let has_io = self.scheduler_has_pending_io()?;

        if deadline.is_none() && !has_io {
            return Err(WalrusError::GenericError {
                message:
                    "Event loop deadlock: all suspended work is waiting with no runnable tasks"
                        .to_string(),
            });
        }

        if let Some(deadline) = deadline {
            let now = Instant::now();
            if deadline > now {
                let timeout = deadline.duration_since(now);
                let _ = self.io_wakeup_rx.recv_timeout(timeout);
            }
        } else {
            let _ = self.io_wakeup_rx.recv();
        }

        Ok(())
    }

    pub(super) fn main_waiting_task_resolved(&mut self) -> WalrusResult<bool> {
        let Some(suspended) = &self.suspended_main else {
            return Ok(false);
        };

        Ok(matches!(
            self.poll_task_resolution(suspended.waiting_on)?,
            TaskResolution::Ready(_) | TaskResolution::Failed(_) | TaskResolution::Cancelled
        ))
    }

    pub(super) fn resume_main_if_ready(&mut self) -> WalrusResult<bool> {
        if !self.main_waiting_task_resolved()? {
            return Ok(false);
        }

        let suspended = self
            .suspended_main
            .take()
            .expect("main suspension should exist when ready");
        self.restore_context(suspended.context);
        Ok(true)
    }

    pub(super) fn next_runnable_task(&mut self) -> WalrusResult<Option<crate::arenas::TaskKey>> {
        while let Some(task_key) = self.async_task_queue.pop_front() {
            if self.is_task_runnable(task_key)? {
                return Ok(Some(task_key));
            }
        }
        Ok(None)
    }

    pub(super) fn run_pending_task_to_completion(
        &mut self,
        task_key: crate::arenas::TaskKey,
        span: Span,
    ) -> WalrusResult<()> {
        let task_snapshot = self.get_heap().get_task(task_key)?.clone();
        let AsyncTask::Pending { function, args } = task_snapshot else {
            return Ok(());
        };

        let caller_context = self.take_context();
        let suspended_task = self.suspended_tasks.remove(&task_key);
        let function = self.get_heap().get_function(function)?.clone();
        match function {
            WalrusFunction::Vm(func) => {
                if args.len() != func.arity {
                    self.restore_context(caller_context);
                    return self.fail_task_with_error(
                        task_key,
                        WalrusError::InvalidArgCount {
                            name: func.name.clone(),
                            expected: func.arity,
                            got: args.len(),
                            span,
                            src: self.source_ref.source().into(),
                            filename: self.source_ref.filename().into(),
                        },
                    );
                }

                let context = if let Some(suspended) = suspended_task {
                    suspended.context
                } else {
                    ExecutionContext {
                        stack: Vec::new(),
                        locals: args,
                        local_string_builders: Vec::new(),
                        call_stack: vec![CallFrame {
                            return_ip: 0,
                            frame_pointer: 0,
                            stack_pointer: 0,
                            instructions: Rc::clone(&func.code),
                            function_name: func.name.clone(),
                            return_override: None,
                            module_binding: func.module_binding.clone(),
                            awaiting_task: Some(task_key),
                            memoize_result_key: None,
                        }],
                        exception_handlers: Vec::new(),
                        ip: 0,
                    }
                };

                self.restore_context(context);
                let outcome = self.run_inner();
                let task_context = self.take_context();
                self.restore_context(caller_context);

                match outcome {
                    Ok(RunSignal::Returned(_)) => {
                        if matches!(
                            self.poll_task_resolution(task_key)?,
                            TaskResolution::Pending
                        ) {
                            let failure = self.get_heap_mut().push(HeapValue::String(
                                "Async task exited without returning a result",
                            ));
                            self.set_task_failed(task_key, failure)?;
                        } else {
                            self.wake_task_waiters(task_key)?;
                        }
                    }
                    Ok(RunSignal::Suspended(waiting_on)) => {
                        self.suspended_tasks.insert(
                            task_key,
                            SuspendedExecution {
                                context: task_context,
                                waiting_on,
                            },
                        );
                        self.task_waiters
                            .entry(waiting_on)
                            .or_default()
                            .push(task_key);
                    }
                    Err(err) => {
                        let failure = self.task_failure_value_from_error(err);
                        self.set_task_failed(task_key, failure)?;
                    }
                }
            }
            WalrusFunction::Native(native_fn) => {
                let result = match self.call_native(native_fn, args, span) {
                    Ok(value) => value,
                    Err(err) => {
                        self.restore_context(caller_context);
                        return self.fail_task_with_error(task_key, err);
                    }
                };
                self.complete_task_on_frame_return(Some(task_key), result)?;
                self.restore_context(caller_context);
                self.wake_task_waiters(task_key)?;
            }
        }

        Ok(())
    }

    pub(super) fn call_exported_function(
        &mut self,
        function: WalrusFunction,
        args: Vec<Value>,
        span: Span,
    ) -> WalrusResult<Option<Value>> {
        match function {
            WalrusFunction::Native(native) => {
                let result = self.call_native(native, args, span)?;
                Ok(Some(result))
            }
            WalrusFunction::Vm(func) => {
                if args.len() != func.arity {
                    return Err(WalrusError::InvalidArgCount {
                        name: func.name.clone(),
                        expected: func.arity,
                        got: args.len(),
                        span,
                        src: self.source_ref.source().into(),
                        filename: self.source_ref.filename().into(),
                    });
                }

                if func.is_async {
                    let task = self
                        .create_task_for_cloned_function(WalrusFunction::Vm(func.clone()), args);
                    return Ok(Some(task));
                }

                let new_frame = CallFrame {
                    return_ip: self.ip,
                    frame_pointer: self.locals.len(),
                    stack_pointer: self.stack.len(),
                    instructions: Rc::clone(&func.code),
                    function_name: func.name.clone(),
                    return_override: None,
                    module_binding: func.module_binding.clone(),
                    awaiting_task: None,
                    memoize_result_key: None,
                };

                self.call_stack.push(new_frame);
                self.locals.extend(args);
                self.ip = 0;
                Ok(None)
            }
        }
    }
}
