//! Disk-backed capture bytes for content larger than the in-memory window.
//!
//! The relay path only hands reference-counted `Bytes` to a bounded channel;
//! one dedicated thread writes them. The destination worker later streams the
//! completed files into split records, so a capture's size is bounded by the
//! disk budget rather than by worker memory.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

use bytes::Bytes;
use serde::Deserialize;

/// File names this store owns inside its directory; nothing else is touched.
const FILE_PREFIX: &str = "exp-capture-";
const WRITE_BUFFER_BYTES: usize = 1024 * 1024;
/// Writer threads; each file is written by one of them, in order.
const WRITERS: usize = 4;
/// Message slots; the queued-byte budget is the real bound, so slots are many
/// (a message is a handle, never content) and small SSE frames cannot exhaust them first.
const QUEUE_MESSAGES: usize = 1 << 18;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Configuration {
    /// A directory this collector owns; stale capture files in it are removed at start.
    pub directory: String,
    /// Bytes of capture files held on disk at once.
    pub maximum_bytes: u64,
    /// Relay bytes queued for the writer threads at once.
    pub maximum_queued_bytes: usize,
    /// Parsed request memory background part writers may hold at once.
    pub maximum_inflight_bytes: usize,
    /// String values at least this long leave the record skeleton as their own part.
    pub part_threshold_bytes: usize,
}

impl Configuration {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if self.directory.trim().is_empty()
            || self.maximum_bytes < 1024 * 1024
            || !(64 * 1024..=1024 * 1024 * 1024).contains(&self.maximum_queued_bytes)
            || self.maximum_inflight_bytes > 1024 * 1024 * 1024
            || !(4 * 1024..=64 * 1024 * 1024).contains(&self.part_threshold_bytes)
        {
            return Err("invalid capture spill bounds");
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
enum State {
    Writing,
    Complete,
    /// The file could not be written, or its stream was abandoned unfinished.
    Lost,
}

struct FileInner {
    store: Arc<Shared>,
    path: PathBuf,
    state: Mutex<(State, u64)>,
    done: Condvar,
    /// A budget refused bytes: the file keeps the prefix written before it.
    cut: AtomicBool,
}

impl Drop for FileInner {
    fn drop(&mut self) {
        let bytes = self.state.lock().map(|state| state.1).unwrap_or(0);
        let _ = fs::remove_file(&self.path);
        self.store.used.fetch_sub(bytes, Ordering::AcqRel);
    }
}

/// One capture file. The last handle removes it and returns its disk budget.
#[derive(Clone)]
pub(crate) struct SpillFile(Arc<FileInner>);

impl SpillFile {
    pub(crate) fn path(&self) -> &Path {
        &self.0.path
    }

    /// Block until the writer has finished; `None` when bytes were lost.
    /// Only the destination worker waits, never a serving or relay task.
    pub(crate) fn wait(&self, timeout: Duration) -> Option<u64> {
        let guard = self.0.state.lock().ok()?;
        let (guard, _) = self
            .0
            .done
            .wait_timeout_while(guard, timeout, |state| state.0 == State::Writing)
            .ok()?;
        (guard.0 == State::Complete).then_some(guard.1)
    }

    /// Whether the file stops short of what was sent to it.
    pub(crate) fn truncated(&self) -> bool {
        self.0.cut.load(Ordering::Acquire)
    }

    fn cut(&self) {
        self.0.cut.store(true, Ordering::Release);
    }

    fn lose(&self) {
        if let Ok(mut state) = self.0.state.lock() {
            if state.0 == State::Writing {
                state.0 = State::Lost;
            }
        }
        self.0.done.notify_all();
    }
}

struct Shared {
    directory: PathBuf,
    maximum_bytes: u64,
    maximum_queued_bytes: usize,
    used: AtomicU64,
    queued: AtomicUsize,
    sequence: AtomicU64,
    /// Parsed request memory held by background part writers until written.
    inflight: AtomicUsize,
}

/// A part a background writer is producing: `None` until it finishes, then
/// the file, or no file when it could not be written.
pub(crate) struct PendingPart {
    state: Mutex<Option<Option<SpillFile>>>,
    done: Condvar,
}

impl fmt::Debug for PendingPart {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PendingPart")
    }
}

impl PendingPart {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(None),
            done: Condvar::new(),
        })
    }

    fn set(&self, file: Option<SpillFile>) {
        if let Ok(mut state) = self.state.lock() {
            *state = Some(file);
        }
        self.done.notify_all();
    }

    /// The finished part; only the destination worker waits.
    pub(crate) fn wait(&self, timeout: Duration) -> Option<SpillFile> {
        let guard = self.state.lock().ok()?;
        let (guard, _) = self
            .done
            .wait_timeout_while(guard, timeout, |state| state.is_none())
            .ok()?;
        guard.clone().flatten()
    }
}

type Job = Box<dyn FnOnce() + Send>;

enum Message {
    Open(SpillFile),
    Chunk(SpillFile, Bytes),
    /// Close a file; `true` when the relay stopped queuing after a refusal,
    /// so the file is cut after every chunk queued before it.
    Finish(SpillFile, bool),
}

/// Shared spill directory, budget and writer threads for one collector.
pub(crate) struct SpillStore {
    shared: Arc<Shared>,
    senders: Vec<mpsc::SyncSender<Message>>,
    jobs: mpsc::SyncSender<Job>,
    pub(crate) part_threshold_bytes: usize,
    /// Parsed request memory background writers may hold at once. Each job
    /// holds the capture's copy of one request's messages only until they are
    /// written (about a tenth of a second for 64 MiB), then frees it. A
    /// request that does not fit falls back to the pending budget.
    maximum_inflight_bytes: usize,
}

impl SpillStore {
    pub(crate) fn new(config: &Configuration) -> Result<Arc<Self>, &'static str> {
        config.validate()?;
        let directory = PathBuf::from(&config.directory);
        fs::create_dir_all(&directory).map_err(|_| "cannot create capture spill directory")?;
        // A previous process's files are unreachable: their records died with it.
        if let Ok(entries) = fs::read_dir(&directory) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.starts_with(FILE_PREFIX))
                {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        let shared = Arc::new(Shared {
            directory,
            maximum_bytes: config.maximum_bytes,
            maximum_queued_bytes: config.maximum_queued_bytes,
            used: AtomicU64::new(0),
            queued: AtomicUsize::new(0),
            sequence: AtomicU64::new(0),
            inflight: AtomicUsize::new(0),
        });
        let mut senders = Vec::with_capacity(WRITERS);
        for index in 0..WRITERS {
            let (sender, receiver) = mpsc::sync_channel(QUEUE_MESSAGES / WRITERS);
            let worker_shared = shared.clone();
            std::thread::Builder::new()
                .name(format!("exp-capture-spill-{index}"))
                .spawn(move || run_writer(receiver, worker_shared))
                .map_err(|_| "cannot start capture spill writer")?;
            senders.push(sender);
        }
        let (jobs, receiver) = mpsc::sync_channel::<Job>(QUEUE_MESSAGES / WRITERS);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..WRITERS {
            let receiver = receiver.clone();
            std::thread::Builder::new()
                .name(format!("exp-capture-part-{index}"))
                .spawn(move || loop {
                    let job = match receiver.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => return,
                    };
                    match job {
                        Ok(job) => job(),
                        Err(_) => return,
                    }
                })
                .map_err(|_| "cannot start capture part writer")?;
        }
        Ok(Arc::new(Self {
            shared,
            senders,
            jobs,
            part_threshold_bytes: config.part_threshold_bytes,
            maximum_inflight_bytes: config.maximum_inflight_bytes,
        }))
    }

    fn new_file(&self) -> SpillFile {
        self.new_file_numbered().1
    }

    fn new_file_numbered(&self) -> (u64, SpillFile) {
        let sequence = self.shared.sequence.fetch_add(1, Ordering::Relaxed);
        let path = self
            .shared
            .directory
            .join(format!("{FILE_PREFIX}{}-{sequence}", std::process::id()));
        (
            sequence,
            SpillFile(Arc::new(FileInner {
                store: self.shared.clone(),
                path,
                state: Mutex::new((State::Writing, 0)),
                done: Condvar::new(),
                cut: AtomicBool::new(false),
            })),
        )
    }

    /// Start a file written by the spill thread from relay chunks.
    pub(crate) fn open_stream(&self) -> Option<SpillStream> {
        let (sequence, file) = self.new_file_numbered();
        let sender = self.senders[(sequence % WRITERS as u64) as usize].clone();
        sender.try_send(Message::Open(file.clone())).ok()?;
        Some(SpillStream {
            store: self.shared.clone(),
            sender,
            file: Some(file),
            failed: false,
        })
    }

    /// Write a part off the caller's thread, charging `charge` bytes of
    /// request memory until done. `None` when the budget or queue is full.
    pub(crate) fn write_in_background(
        self: &Arc<Self>,
        charge: usize,
        write: impl FnOnce(&mut dyn Write) -> std::io::Result<()> + Send + 'static,
    ) -> Option<Arc<PendingPart>> {
        let previous = self.shared.inflight.fetch_add(charge, Ordering::AcqRel);
        if previous.saturating_add(charge) > self.maximum_inflight_bytes {
            self.shared.inflight.fetch_sub(charge, Ordering::AcqRel);
            return None;
        }
        let part = PendingPart::new();
        let (store, pending) = (self.clone(), part.clone());
        let job: Job = Box::new(move || {
            let file = store.write_with(write);
            store.shared.inflight.fetch_sub(charge, Ordering::AcqRel);
            pending.set(file);
        });
        if self.jobs.try_send(job).is_err() {
            self.shared.inflight.fetch_sub(charge, Ordering::AcqRel);
            return None;
        }
        Some(part)
    }

    /// Write a file synchronously; only the destination worker calls this.
    pub(crate) fn write_with(
        &self,
        write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
    ) -> Option<SpillFile> {
        let file = self.new_file();
        let handle = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(file.path())
            .ok()?;
        let mut writer = BudgetWriter {
            inner: BufWriter::with_capacity(WRITE_BUFFER_BYTES, handle),
            file: &file,
        };
        let written = write(&mut writer).and_then(|()| writer.flush());
        drop(writer);
        let mut state = file.0.state.lock().ok()?;
        state.0 = if written.is_ok() {
            State::Complete
        } else {
            State::Lost
        };
        let complete = state.0 == State::Complete;
        drop(state);
        file.0.done.notify_all();
        complete.then_some(file)
    }
}

/// Charge every written byte to the disk budget before it reaches the file.
struct BudgetWriter<'a> {
    inner: BufWriter<File>,
    file: &'a SpillFile,
}

impl Write for BudgetWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        reserve(&self.file.0, bytes.len() as u64)?;
        self.inner.write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn reserve(file: &FileInner, bytes: u64) -> std::io::Result<()> {
    let store = &file.store;
    let previous = store.used.fetch_add(bytes, Ordering::AcqRel);
    if previous.saturating_add(bytes) > store.maximum_bytes {
        store.used.fetch_sub(bytes, Ordering::AcqRel);
        return Err(std::io::Error::other("capture spill budget exhausted"));
    }
    if let Ok(mut state) = file.state.lock() {
        state.1 += bytes;
    }
    Ok(())
}

/// Relay-side handle: queue reference-counted chunks, never block or copy.
/// Dropping it without [`SpillStream::finish`] abandons the file.
pub(crate) struct SpillStream {
    store: Arc<Shared>,
    sender: mpsc::SyncSender<Message>,
    file: Option<SpillFile>,
    failed: bool,
}

impl SpillStream {
    /// Queue one chunk. Returns false (and stops) when a budget refuses it;
    /// the file then keeps the prefix queued before the refusal.
    pub(crate) fn push(&mut self, chunk: Bytes) -> bool {
        let Some(file) = self.file.as_ref().filter(|_| !self.failed) else {
            return false;
        };
        let length = chunk.len();
        let previous = self.store.queued.fetch_add(length, Ordering::AcqRel);
        if previous.saturating_add(length) > self.store.maximum_queued_bytes
            || self
                .sender
                .try_send(Message::Chunk(file.clone(), chunk))
                .is_err()
        {
            self.store.queued.fetch_sub(length, Ordering::AcqRel);
            // Chunks already queued still land; finish marks the cut.
            self.failed = true;
            return false;
        }
        true
    }

    /// Close the stream; the file is complete once the writer drains it.
    pub(crate) fn finish(mut self) -> SpillFile {
        let file = self
            .file
            .take()
            .expect("an unfinished stream holds its file");
        if self
            .sender
            .try_send(Message::Finish(file.clone(), self.failed))
            .is_err()
        {
            file.lose();
        }
        file
    }
}

impl Drop for SpillStream {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            file.lose();
        }
    }
}

fn run_writer(receiver: mpsc::Receiver<Message>, shared: Arc<Shared>) {
    let mut open: std::collections::HashMap<PathBuf, (SpillFile, BufWriter<File>)> =
        std::collections::HashMap::new();
    loop {
        let message = match receiver.recv_timeout(Duration::from_secs(1)) {
            Ok(message) => message,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // A stream that lost bytes on the relay side sends nothing more.
                open.retain(|_, (file, _)| {
                    file.0
                        .state
                        .lock()
                        .is_ok_and(|state| state.0 == State::Writing)
                });
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match message {
            Message::Open(file) => {
                // The relay may already have abandoned the stream.
                if file
                    .0
                    .state
                    .lock()
                    .map_or(true, |state| state.0 != State::Writing)
                {
                    continue;
                }
                match OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(file.path())
                {
                    Ok(handle) => {
                        open.insert(
                            file.path().to_owned(),
                            (file, BufWriter::with_capacity(WRITE_BUFFER_BYTES, handle)),
                        );
                    }
                    Err(_) => file.lose(),
                }
            }
            Message::Chunk(file, chunk) => {
                shared.queued.fetch_sub(chunk.len(), Ordering::AcqRel);
                let writing = file
                    .0
                    .state
                    .lock()
                    .is_ok_and(|state| state.0 == State::Writing);
                if !writing || file.truncated() {
                    // A cut file keeps its prefix; nothing after the cut lands.
                    if !writing {
                        open.remove(file.path());
                    }
                    continue;
                }
                let Some((_, writer)) = open.get_mut(file.path()) else {
                    file.lose();
                    continue;
                };
                if reserve(&file.0, chunk.len() as u64).is_err() {
                    file.cut();
                } else if writer.write_all(&chunk).is_err() {
                    open.remove(file.path());
                    file.lose();
                }
            }
            Message::Finish(file, cut) => {
                if cut {
                    file.cut();
                }
                let flushed = open
                    .remove(file.path())
                    .is_some_and(|(_, mut writer)| writer.flush().is_ok());
                if let Ok(mut state) = file.0.state.lock() {
                    if state.0 == State::Writing {
                        state.0 = if flushed {
                            State::Complete
                        } else {
                            State::Lost
                        };
                    }
                }
                file.0.done.notify_all();
            }
        }
    }
}

#[cfg(test)]
#[path = "spill_test.rs"]
pub(crate) mod tests;
