//! Explicit native implementations of accounting callbacks.
//!
//! The capsule is inspected only at composition. Calls retain the supplier's
//! context and run on bounded blocking workers without attaching to Python.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::errors::PublicError;

const ABI_VERSION: u32 = 1;
const ALL_METHODS: u64 = (1 << 19) - 1;
const MAXIMUM_RESULT_BYTES: usize = 16 * 1024 * 1024;

type WriteResult = unsafe extern "C" fn(*mut c_void, *const u8, usize) -> bool;
type Invoke =
    unsafe extern "C" fn(*mut c_void, u32, *const u8, usize, WriteResult, *mut c_void) -> u32;
type Ownership = unsafe extern "C" fn(*mut c_void);

/// C layout shared with `include/native_callbacks.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CallbackAbi {
    version: u32,
    size: u32,
    methods: u64,
    context: *mut c_void,
    retain: Option<Ownership>,
    release: Option<Ownership>,
    invoke: Option<Invoke>,
}

/// Native ownership and function pointers, copied out of the capsule at startup.
struct ForeignTarget {
    abi: CallbackAbi,
    // Keep the extension's capsule alive as well as its explicitly retained context.
    _capsule: Py<PyCapsule>,
}

// SAFETY: The named ABI explicitly requires concurrent, interpreter-independent
// calls and thread-safe retain/release. Only trusted in-process native code may
// provide this capsule; it is not an input boundary for untrusted Python code.
unsafe impl Send for ForeignTarget {}
unsafe impl Sync for ForeignTarget {}

/// An interpreter-independent implementation owned by a native execution.
pub(crate) trait CallbackTarget: Send + Sync {
    fn handles(&self, method: &str) -> bool;
    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError>;
    fn requires_request_owner(&self) -> bool {
        false
    }
    fn retain_request(&self, _request_id: &str) -> Option<Arc<dyn Send + Sync>> {
        None
    }
    fn input_resources(&self, _request_id: &str) -> Option<Arc<crate::accounting::InputResources>> {
        None
    }
}

pub(crate) struct NativeTarget {
    target: Arc<dyn CallbackTarget>,
}

impl NativeTarget {
    pub(crate) fn new(target: Arc<dyn CallbackTarget>) -> Arc<Self> {
        Arc::new(Self { target })
    }

    pub(crate) fn handles(&self, method: &str) -> bool {
        self.target.handles(method)
    }

    pub(crate) fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        self.target.call(method, argument)
    }
    pub(crate) fn requires_request_owner(&self) -> bool {
        self.target.requires_request_owner()
    }
    pub(crate) fn retain_request(&self, request_id: &str) -> Option<Arc<dyn Send + Sync>> {
        self.target.retain_request(request_id)
    }
    pub(crate) fn input_resources(
        &self,
        request_id: &str,
    ) -> Option<Arc<crate::accounting::InputResources>> {
        self.target.input_resources(request_id)
    }
}

/// Native accounting target supplied by a compiled extension at composition.
#[pyclass]
pub struct NativeCallbacks {
    pub(crate) inner: Arc<NativeTarget>,
}

#[pymethods]
impl NativeCallbacks {
    #[new]
    fn new(capsule: Bound<'_, PyCapsule>) -> PyResult<Self> {
        let pointer = capsule.pointer_checked(Some(c"exp_gateway_native.callbacks.v1"))?;
        // SAFETY: A supplier of the named capsule promises a live CallbackAbi.
        // Copy its header before taking ownership; no pointer is used on failure.
        let abi = unsafe { *pointer.as_ptr().cast::<CallbackAbi>() };
        if abi.version != ABI_VERSION
            || abi.size as usize != std::mem::size_of::<CallbackAbi>()
            || abi.methods == 0
            || abi.methods & !ALL_METHODS != 0
            || abi.context.is_null()
            || abi.retain.is_none()
            || abi.release.is_none()
            || abi.invoke.is_none()
        {
            return Err(PyValueError::new_err(
                "Invalid native callback capsule. Rebuild the supplier against native_callbacks.h.",
            ));
        }
        // SAFETY: The checked ABI supplies thread-safe ownership operations.
        unsafe { abi.retain.unwrap()(abi.context) };
        Ok(Self {
            inner: NativeTarget::new(Arc::new(ForeignTarget {
                abi,
                _capsule: capsule.unbind(),
            })),
        })
    }
}

impl CallbackTarget for ForeignTarget {
    fn handles(&self, method: &str) -> bool {
        method_id(method).is_some_and(|id| self.abi.methods & (1 << (id - 1)) != 0)
    }

    /// Invoke exactly one selected callback. A failed native call never falls back.
    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        let id = method_id(method).ok_or_else(PublicError::internal)?;
        if !self.handles(method) {
            return Err(PublicError::internal());
        }
        let output = Output::default();
        // SAFETY: The target was validated and retained at composition. Both
        // borrowed buffers live until invoke returns; the ABI forbids retaining
        // them, writing asynchronously, or unwinding through the C boundary.
        let status = unsafe {
            self.abi.invoke.unwrap()(
                self.abi.context,
                id,
                argument.as_ptr(),
                argument.len(),
                write_result,
                (&output as *const Output).cast_mut().cast(),
            )
        };
        let output = output
            .state
            .into_inner()
            .map_err(|_| PublicError::internal())?;
        if output.invalid {
            return Err(PublicError::internal());
        }
        let text = output
            .value
            .and_then(|bytes| String::from_utf8(bytes).ok())
            .ok_or_else(PublicError::internal)?;
        match status {
            0 => {
                crate::lossless_json::from_str::<Box<serde_json::value::RawValue>>(&text)
                    .map_err(|_| PublicError::internal())?;
                Ok(text)
            }
            1 => {
                Err(crate::lossless_json::from_str(&text)
                    .unwrap_or_else(|_| PublicError::internal()))
            }
            _ => Err(PublicError::internal()),
        }
    }
}

impl Drop for ForeignTarget {
    fn drop(&mut self) {
        // SAFETY: Every call owns an Arc, so release cannot race the last call.
        // The capsule remains alive until after this explicit ownership release.
        unsafe { self.abi.release.unwrap()(self.abi.context) };
    }
}

fn method_id(method: &str) -> Option<u32> {
    match method {
        "start_attempt" => Some(1),
        "settle" => Some(2),
        "abandon" => Some(3),
        "price" => Some(4),
        "fleet_acquire" => Some(5),
        "fleet_release" => Some(6),
        "enforce_output_segment" => Some(7),
        "enforce_output" => Some(8),
        "guardrail_input_status" => Some(9),
        "sign_dispatch" => Some(10),
        "seal_reasoning_content" => Some(11),
        "remember" => Some(12),
        "tool_search_round" => Some(13),
        "prepare_explicit_cache" => Some(14),
        "finish_explicit_cache" => Some(15),
        "renew_explicit_cache" => Some(16),
        "cache_transaction" => Some(17),
        "record_late_meter" => Some(18),
        "record_input_moderation" => Some(19),
        _ => None,
    }
}

pub(crate) fn post_admission(method: &str) -> bool {
    method_id(method).is_some_and(|id| id <= 16)
}

#[derive(Default)]
struct Output {
    state: Mutex<OutputState>,
}

#[derive(Default)]
struct OutputState {
    value: Option<Vec<u8>>,
    invalid: bool,
}

unsafe extern "C" fn write_result(context: *mut c_void, bytes: *const u8, len: usize) -> bool {
    // SAFETY: invoke received this pointer to the live call-local Output.
    let output = unsafe { &*context.cast::<Output>() };
    let Ok(mut output) = output.state.lock() else {
        return false;
    };
    if output.invalid || output.value.is_some() || bytes.is_null() || len > MAXIMUM_RESULT_BYTES {
        output.invalid = true;
        return false;
    }
    // SAFETY: The supplier guarantees len readable bytes until this sink returns.
    output.value = Some(unsafe { std::slice::from_raw_parts(bytes, len) }.to_vec());
    true
}

#[cfg(test)]
#[path = "native_callbacks_tests.rs"]
mod tests;
