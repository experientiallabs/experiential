use super::*;

pub(crate) fn temporary_directory(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "exp-capture-spill-test-{name}-{}-{}",
        std::process::id(),
        SystemTimeNanos::now()
    ));
    fs::create_dir_all(&directory).unwrap();
    directory
}

struct SystemTimeNanos;
impl SystemTimeNanos {
    fn now() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }
}

pub(crate) fn store(name: &str, maximum_bytes: u64) -> Arc<SpillStore> {
    SpillStore::new(&Configuration {
        directory: temporary_directory(name).to_string_lossy().into_owned(),
        maximum_bytes,
        maximum_queued_bytes: 4 * 1024 * 1024,
        maximum_inflight_bytes: 256 * 1024 * 1024,
        part_threshold_bytes: 4096,
    })
    .unwrap()
}

#[test]
fn streamed_chunks_land_in_order_and_the_last_handle_removes_the_file() {
    let store = store("order", 64 * 1024 * 1024);
    let mut stream = store.open_stream().unwrap();
    for index in 0..100u8 {
        assert!(stream.push(Bytes::from(vec![index; 1000])));
    }
    let file = stream.finish();
    assert_eq!(file.wait(Duration::from_secs(5)), Some(100_000));
    let written = fs::read(file.path()).unwrap();
    assert_eq!(written.len(), 100_000);
    assert!(written
        .chunks(1000)
        .enumerate()
        .all(|(index, chunk)| chunk.iter().all(|byte| *byte == index as u8)));
    assert_eq!(store.shared.used.load(Ordering::Acquire), 100_000);
    let path = file.path().to_owned();
    // Completion wakes waiters before the writer drops its handle. Wait until
    // this is the last handle so its synchronous deletion is what we test.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while Arc::strong_count(&file.0) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(Arc::strong_count(&file.0), 1);
    drop(file);
    assert!(!path.exists());
    assert_eq!(store.shared.used.load(Ordering::Acquire), 0);
}

#[test]
fn the_disk_budget_cuts_the_file_to_its_prefix_instead_of_growing_past_it() {
    let store = store("budget", 1024 * 1024);
    let mut stream = store.open_stream().unwrap();
    for _ in 0..8 {
        stream.push(Bytes::from(vec![7; 256 * 1024]));
    }
    let file = stream.finish();
    let kept = file.wait(Duration::from_secs(5)).unwrap();
    assert!(file.truncated());
    assert_eq!(kept, 1024 * 1024);
    assert_eq!(fs::read(file.path()).unwrap().len() as u64, kept);
    assert!(store.shared.used.load(Ordering::Acquire) <= 1024 * 1024);
    drop(file);
    // The writer releases a lost file's budget once it drops its handle.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while store.shared.used.load(Ordering::Acquire) != 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(store.shared.used.load(Ordering::Acquire), 0);
}

#[test]
fn the_queue_budget_refuses_bytes_without_blocking_the_caller() {
    let store = SpillStore::new(&Configuration {
        directory: temporary_directory("queue").to_string_lossy().into_owned(),
        maximum_bytes: 1024 * 1024 * 1024,
        maximum_queued_bytes: 64 * 1024,
        maximum_inflight_bytes: 256 * 1024 * 1024,
        part_threshold_bytes: 4096,
    })
    .unwrap();
    let mut stream = store.open_stream().unwrap();
    assert!(stream.push(Bytes::from(vec![1; 1000])));
    // A chunk larger than the queue can hold is refused at once; the file
    // keeps the prefix queued before it.
    assert!(!stream.push(Bytes::from(vec![1; 128 * 1024])));
    assert!(!stream.push(Bytes::from(vec![1; 1])));
    let file = stream.finish();
    assert_eq!(file.wait(Duration::from_secs(5)), Some(1000));
    assert!(file.truncated());
}

#[test]
fn an_abandoned_stream_is_lost_and_its_file_removed() {
    let store = store("abandon", 64 * 1024 * 1024);
    let mut stream = store.open_stream().unwrap();
    assert!(stream.push(Bytes::from_static(b"partial")));
    drop(stream);
    // The writer may create the file after the drop; it must still remove it.
    std::thread::sleep(Duration::from_millis(1500));
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while fs::read_dir(&store.shared.directory).unwrap().count() > 0
        && std::time::Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    let remaining: Vec<_> = fs::read_dir(&store.shared.directory)
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name())
        .collect();
    assert!(remaining.is_empty(), "{remaining:?}");
}

#[test]
fn synchronous_parts_are_charged_and_complete() {
    let store = store("write", 64 * 1024 * 1024);
    let file = store
        .write_with(|out| out.write_all(&vec![b'x'; 10_000]))
        .unwrap();
    assert_eq!(file.wait(Duration::ZERO), Some(10_000));
    assert_eq!(fs::read(file.path()).unwrap().len(), 10_000);
    let refused = SpillStore::new(&Configuration {
        directory: temporary_directory("write-refused")
            .to_string_lossy()
            .into_owned(),
        maximum_bytes: 1024 * 1024,
        maximum_queued_bytes: 64 * 1024,
        maximum_inflight_bytes: 256 * 1024 * 1024,
        part_threshold_bytes: 4096,
    })
    .unwrap();
    assert!(refused
        .write_with(|out| out.write_all(&vec![b'x'; 2 * 1024 * 1024]))
        .is_none());
    assert_eq!(refused.shared.used.load(Ordering::Acquire), 0);
}

#[test]
fn a_new_store_removes_only_its_own_stale_files() {
    let directory = temporary_directory("stale");
    fs::write(directory.join("exp-capture-1-1"), b"stale").unwrap();
    fs::write(directory.join("unrelated"), b"kept").unwrap();
    let _store = SpillStore::new(&Configuration {
        directory: directory.to_string_lossy().into_owned(),
        maximum_bytes: 1024 * 1024,
        maximum_queued_bytes: 64 * 1024,
        maximum_inflight_bytes: 256 * 1024 * 1024,
        part_threshold_bytes: 4096,
    })
    .unwrap();
    assert!(!directory.join("exp-capture-1-1").exists());
    assert!(directory.join("unrelated").exists());
}
