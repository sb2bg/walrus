use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use once_cell::sync::Lazy;

use crate::value::IoResult;

const MIN_IO_WORKERS: usize = 2;
const MAX_IO_WORKERS: usize = 8;
const JOBS_PER_WORKER: usize = 64;

type IoJob = Box<dyn FnOnce() + Send + 'static>;

struct BlockingIoPool {
    sender: mpsc::SyncSender<IoJob>,
}

impl BlockingIoPool {
    fn new() -> Self {
        let worker_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4)
            .clamp(MIN_IO_WORKERS, MAX_IO_WORKERS);
        let queue_capacity = worker_count * JOBS_PER_WORKER;
        let (sender, receiver) = mpsc::sync_channel::<IoJob>(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));

        for index in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            thread::Builder::new()
                .name(format!("walrus-io-{index}"))
                .spawn(move || {
                    loop {
                        let job = {
                            let receiver = match receiver.lock() {
                                Ok(receiver) => receiver,
                                Err(_) => break,
                            };
                            match receiver.recv() {
                                Ok(job) => job,
                                Err(_) => break,
                            }
                        };
                        job();
                    }
                })
                .expect("failed to spawn walrus I/O worker thread");
        }

        Self { sender }
    }

    fn submit<F>(
        &self,
        work: F,
        wakeup: mpsc::Sender<()>,
    ) -> mpsc::Receiver<Result<IoResult, String>>
    where
        F: FnOnce() -> Result<IoResult, String> + Send + 'static,
    {
        let (result_tx, result_rx) = mpsc::channel();
        let job: IoJob = Box::new(move || {
            let result = work();
            let _ = result_tx.send(result);
            let _ = wakeup.send(());
        });

        self.sender
            .send(job)
            .expect("walrus I/O pool is unexpectedly unavailable");

        result_rx
    }
}

static BLOCKING_IO_POOL: Lazy<BlockingIoPool> = Lazy::new(BlockingIoPool::new);

pub(crate) fn submit_io<F>(
    work: F,
    wakeup: mpsc::Sender<()>,
) -> mpsc::Receiver<Result<IoResult, String>>
where
    F: FnOnce() -> Result<IoResult, String> + Send + 'static,
{
    BLOCKING_IO_POOL.submit(work, wakeup)
}
