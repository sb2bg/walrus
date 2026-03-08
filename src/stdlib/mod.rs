use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::function::NativeFunction;
mod http;
mod io;
mod math;
mod net;
mod sys;

pub use http::*;
pub use io::*;
pub use math::*;
pub use net::*;
pub use sys::*;

thread_local! {
    static FILE_TABLE: RefCell<FileTable> = RefCell::new(FileTable::new());
    static NET_TABLE: RefCell<NetState> = RefCell::new(NetState::new());
    static RNG_STATE: RefCell<StdRng> = RefCell::new(new_rng());
}

struct FileEntry {
    file: File,
}

struct FileTable {
    files: HashMap<i64, FileEntry>,
    next_handle: i64,
}

struct NetState {
    listeners: HashMap<i64, Arc<TcpListener>>,
    streams: HashMap<i64, Arc<Mutex<SharedTcpStream>>>,
    next_handle: i64,
}

pub(crate) struct SharedTcpStream {
    pub(crate) stream: TcpStream,
    pub(crate) read_buffer: Vec<u8>,
}

impl SharedTcpStream {
    fn new(stream: TcpStream) -> Self {
        Self {
            stream,
            read_buffer: Vec::new(),
        }
    }
}

impl Read for SharedTcpStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        if !self.read_buffer.is_empty() {
            let buffered = buf.len().min(self.read_buffer.len());
            buf[..buffered].copy_from_slice(&self.read_buffer[..buffered]);
            self.read_buffer.drain(..buffered);
            return Ok(buffered);
        }

        self.stream.read(buf)
    }
}

impl Write for SharedTcpStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

fn new_rng() -> StdRng {
    let mut seeder = rand::thread_rng();
    let mut seed = [0u8; 32];
    seeder.fill(&mut seed);
    StdRng::from_seed(seed)
}

impl FileTable {
    fn new() -> Self {
        Self {
            files: HashMap::new(),
            next_handle: 1,
        }
    }

    fn insert(&mut self, file: File) -> i64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.files.insert(handle, FileEntry { file });
        handle
    }

    fn get_mut(&mut self, handle: i64) -> Option<&mut FileEntry> {
        self.files.get_mut(&handle)
    }

    fn remove(&mut self, handle: i64) -> Option<FileEntry> {
        self.files.remove(&handle)
    }
}

impl NetState {
    fn new() -> Self {
        Self {
            listeners: HashMap::new(),
            streams: HashMap::new(),
            next_handle: 1,
        }
    }

    fn next(&mut self) -> i64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        handle
    }

    fn insert_listener(&mut self, listener: TcpListener) -> i64 {
        let handle = self.next();
        self.listeners.insert(handle, Arc::new(listener));
        handle
    }

    fn insert_stream(&mut self, stream: TcpStream) -> i64 {
        let handle = self.next();
        self.streams
            .insert(handle, Arc::new(Mutex::new(SharedTcpStream::new(stream))));
        handle
    }

    fn listener(&self, handle: i64) -> Option<Arc<TcpListener>> {
        self.listeners.get(&handle).cloned()
    }

    fn stream(&self, handle: i64) -> Option<Arc<Mutex<SharedTcpStream>>> {
        self.streams.get(&handle).cloned()
    }

    fn remove_listener(&mut self, handle: i64) -> Option<Arc<TcpListener>> {
        self.listeners.remove(&handle)
    }

    fn remove_stream(&mut self, handle: i64) -> Option<Arc<Mutex<SharedTcpStream>>> {
        self.streams.remove(&handle)
    }
}

pub fn get_module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    crate::native_registry::module_functions(module)
}
