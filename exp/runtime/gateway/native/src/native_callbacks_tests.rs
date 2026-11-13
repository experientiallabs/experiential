use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::Duration;

use super::*;
use crate::bridge::Bridge;

#[derive(Default)]
struct Context {
    mode: AtomicU32,
    calls: AtomicUsize,
    blocked: AtomicBool,
    invalid_sink: AtomicBool,
}

unsafe extern "C" fn retain(context: *mut c_void) {
    unsafe { Arc::increment_strong_count(context.cast::<Context>()) };
}

unsafe extern "C" fn release(context: *mut c_void) {
    unsafe { Arc::decrement_strong_count(context.cast::<Context>()) };
}

unsafe extern "C" fn invoke(
    context: *mut c_void,
    _method: u32,
    input: *const u8,
    len: usize,
    write: WriteResult,
    output: *mut c_void,
) -> u32 {
    let context = unsafe { &*context.cast::<Context>() };
    context.calls.fetch_add(1, Ordering::AcqRel);
    while context.blocked.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let check = |ok: bool| {
        if !ok {
            context.invalid_sink.store(true, Ordering::Release);
        }
    };
    match context.mode.load(Ordering::Acquire) {
        0 => {
            check(unsafe { write(output, input, len) });
            0
        }
        1 => {
            let message =
                br#"{"status_code":429,"code":"quota_exceeded","message":"quota exhausted"}"#;
            check(unsafe { write(output, message.as_ptr(), message.len()) });
            1
        }
        2 => {
            check(unsafe { write(output, [255].as_ptr(), 1) });
            0
        }
        3 => {
            check(unsafe { write(output, input, len) });
            check(!unsafe { write(output, input, len) });
            0
        }
        4 => {
            check(!unsafe { write(output, input, MAXIMUM_RESULT_BYTES + 1) });
            0
        }
        _ => 2,
    }
}

unsafe extern "C" fn destroy(capsule: *mut pyo3::ffi::PyObject) {
    let pointer = unsafe {
        pyo3::ffi::PyCapsule_GetPointer(capsule, c"exp_gateway_native.callbacks.v1".as_ptr())
    };
    let abi = unsafe { Box::from_raw(pointer.cast::<CallbackAbi>()) };
    unsafe { release(abi.context) };
}

fn target(context: &Arc<Context>, version: u32, methods: u64) -> PyResult<NativeCallbacks> {
    Python::initialize();
    Python::attach(|py| {
        let abi = Box::new(CallbackAbi {
            version,
            size: std::mem::size_of::<CallbackAbi>() as u32,
            methods,
            context: Arc::into_raw(context.clone()).cast_mut().cast(),
            retain: Some(retain),
            release: Some(release),
            invoke: Some(invoke),
        });
        let pointer = std::ptr::NonNull::new(Box::into_raw(abi).cast()).unwrap();
        let capsule = unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                pointer,
                c"exp_gateway_native.callbacks.v1",
                Some(destroy),
            )
        }?;
        NativeCallbacks::new(capsule)
    })
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn native_calls_complete_while_python_is_attached_and_target_has_no_methods() {
    let context = Arc::new(Context::default());
    let callbacks = target(&context, 1, ALL_METHODS).unwrap();
    let bridge = Bridge::new(Python::attach(|py| py.None()), 2)
        .unwrap()
        .with_native(Some(callbacks.inner.clone()));
    let runtime = runtime();
    Python::attach(|_py| {
        // Python workers cannot acquire this interpreter attachment. A native
        // accounting call must complete without consulting their target.
        runtime.block_on(async {
            for method in ["start_attempt", "settle", "abandon", "price"] {
                let result = tokio::time::timeout(
                    Duration::from_secs(2),
                    bridge.call(method, "{\"ok\":true}".into()),
                )
                .await
                .expect("native callback must not acquire the GIL")
                .unwrap();
                assert_eq!(result, "{\"ok\":true}");
            }
        });
    });
    assert_eq!(context.calls.load(Ordering::Acquire), 4);
    drop(bridge);
    drop(callbacks);
    Python::attach(|_| {});
    assert_eq!(Arc::strong_count(&context), 1);
}

#[test]
fn invalid_abi_and_results_fail_closed_without_fallback() {
    let context = Arc::new(Context::default());
    assert!(target(&context, 2, ALL_METHODS).is_err());
    assert!(target(&context, 1, 0).is_err());
    assert!(target(&context, 1, 1 << 19).is_err());
    let callbacks = target(&context, 1, 1).unwrap();
    assert!(callbacks.inner.handles("start_attempt"));
    assert!(!callbacks.inner.handles("settle"));
    assert!(!callbacks.inner.handles("authenticate"));
    context.mode.store(1, Ordering::Release);
    assert_eq!(
        callbacks
            .inner
            .call("start_attempt", "{}")
            .unwrap_err()
            .code,
        "quota_exceeded"
    );
    for mode in 2..=5 {
        context.mode.store(mode, Ordering::Release);
        assert_eq!(
            callbacks
                .inner
                .call("start_attempt", "{}")
                .unwrap_err()
                .code,
            "internal_error"
        );
    }
    context.mode.store(0, Ordering::Release);
    assert!(callbacks.inner.call("start_attempt", "not json").is_err());
}

#[test]
fn cancelled_waiters_retain_native_capacity_until_the_call_completes() {
    let context = Arc::new(Context::default());
    context.blocked.store(true, Ordering::Release);
    let callbacks = target(&context, 1, ALL_METHODS).unwrap();
    let bridge = Arc::new(
        Bridge::new(Python::attach(|py| py.None()), 1)
            .unwrap()
            .with_native(Some(callbacks.inner.clone())),
    );
    runtime().block_on(async {
        let owner = bridge.clone();
        let first = tokio::spawn(async move { owner.call("start_attempt", "{}".into()).await });
        tokio::time::timeout(Duration::from_secs(2), async {
            while context.calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        first.abort();
        assert!(tokio::time::timeout(
            Duration::from_millis(30),
            bridge.call("start_attempt", "{}".into())
        )
        .await
        .is_err());
        assert_eq!(context.calls.load(Ordering::Acquire), 1);
        context.blocked.store(false, Ordering::Release);
        assert_eq!(
            bridge.call("start_attempt", "{}".into()).await.unwrap(),
            "{}"
        );
    });
    assert_eq!(context.calls.load(Ordering::Acquire), 2);
}

#[test]
fn concurrent_sink_writers_are_rejected_without_aliasing_mutable_state() {
    let output = Output::default();
    let address = (&output as *const Output) as usize;
    let accepted = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(move || unsafe {
                    write_result(address as *mut c_void, b"{}".as_ptr(), 2)
                })
            })
            .collect();
        writers
            .into_iter()
            .map(|writer| usize::from(writer.join().unwrap()))
            .sum::<usize>()
    });
    assert_eq!(accepted, 1);
    assert!(output.state.into_inner().unwrap().invalid);
}
