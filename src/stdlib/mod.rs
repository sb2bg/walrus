use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::function::NativeFunction;

mod core;
mod http;
mod io;
mod math;
mod net;
mod sys;

pub use core::*;
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
    path: String,
    mode: String,
}

struct FileTable {
    files: HashMap<i64, FileEntry>,
    next_handle: i64,
}

struct NetState {
    listeners: HashMap<i64, Arc<TcpListener>>,
    streams: HashMap<i64, Arc<Mutex<TcpStream>>>,
    next_handle: i64,
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

    fn insert(&mut self, file: File, path: String, mode: String) -> i64 {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.files.insert(handle, FileEntry { file, path, mode });
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
        self.streams.insert(handle, Arc::new(Mutex::new(stream)));
        handle
    }

    fn listener(&self, handle: i64) -> Option<Arc<TcpListener>> {
        self.listeners.get(&handle).cloned()
    }

    fn stream(&self, handle: i64) -> Option<Arc<Mutex<TcpStream>>> {
        self.streams.get(&handle).cloned()
    }

    fn remove_listener(&mut self, handle: i64) -> Option<Arc<TcpListener>> {
        self.listeners.remove(&handle)
    }

    fn remove_stream(&mut self, handle: i64) -> Option<Arc<Mutex<TcpStream>>> {
        self.streams.remove(&handle)
    }
}

pub fn get_module_functions(module: &str) -> Option<Vec<NativeFunction>> {
    crate::native_registry::module_functions(module)
}
