//! Live no-follow snapshot proofs, including missing leaves and retained ancestors.

use std::fs::{File, Metadata};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use super::db::Result;
use crate::errors::PublicError;

#[derive(PartialEq, Eq)]
struct Stamp {
    identity: (u64, u64),
    length: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

pub(super) struct Snapshot {
    root: PathBuf,
    relative: PathBuf,
    ancestors: Vec<File>,
    identities: Vec<(u64, u64)>,
    leaf: Option<File>,
    stamp: Option<Stamp>,
}

fn components(path: &Path) -> Result<Vec<&std::ffi::OsStr>> {
    // Reject Windows spellings on every host rather than interpreting a drive
    // or backslash as a harmless POSIX filename.
    let text = path.to_str().ok_or_else(PublicError::internal)?;
    if text.is_empty()
        || text.contains(['\\', ':'])
        || text.split('/').any(|p| p == "." || p == "..")
    {
        return Err(PublicError::internal());
    }
    path.components()
        .map(|part| match part {
            Component::Normal(name) => Ok(name),
            _ => Err(PublicError::internal()),
        })
        .collect()
}

#[cfg(unix)]
fn identity(file: &File) -> Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let metadata = file.metadata().map_err(|_| PublicError::internal())?;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(unix)]
fn stamp(file: &File, metadata: &Metadata) -> Result<Stamp> {
    use std::os::unix::fs::MetadataExt;
    Ok(Stamp {
        identity: identity(file)?,
        length: metadata.len(),
        modified: (metadata.mtime(), metadata.mtime_nsec()),
        changed: (metadata.ctime(), metadata.ctime_nsec()),
    })
}

#[cfg(windows)]
fn identity(file: &File) -> Result<(u64, u64)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: The retained File owns a valid handle and info is a writable struct.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
        return Err(PublicError::internal());
    }
    Ok((
        u64::from(info.dwVolumeSerialNumber),
        (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
    ))
}

#[cfg(windows)]
fn stamp(file: &File, metadata: &Metadata) -> Result<Stamp> {
    use std::os::windows::fs::MetadataExt;
    // The leaf handle denies write/delete sharing for the whole operation.
    Ok(Stamp {
        identity: identity(file)?,
        length: metadata.len(),
        modified: (metadata.last_write_time() as i64, 0),
        changed: (metadata.creation_time() as i64, 0),
    })
}

#[cfg(unix)]
fn open_child(
    parent: &File,
    _path: &Path,
    name: &std::ffi::OsStr,
    directory: bool,
) -> std::io::Result<File> {
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(name.as_bytes())?;
    let flags = libc::O_RDONLY
        | libc::O_CLOEXEC
        | libc::O_NOFOLLOW
        | if directory {
            libc::O_DIRECTORY
        } else {
            libc::O_NONBLOCK
        };
    // SAFETY: parent remains open and CString is NUL terminated. The returned
    // descriptor is owned exactly once, including on every error path.
    let descriptor = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

#[cfg(windows)]
fn open_child(
    _parent: &File,
    path: &Path,
    _name: &std::ffi::OsStr,
    directory: bool,
) -> std::io::Result<File> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_SHARE_READ, FILE_SHARE_WRITE,
    };
    let file = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ | if directory { FILE_SHARE_WRITE } else { 0 })
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(std::io::Error::other(
            "snapshot reparse points are not supported",
        ));
    }
    Ok(file)
}

impl Snapshot {
    pub(super) fn open(root: &Path, relative: &Path) -> Result<Self> {
        let parts = components(relative)?;
        let mut path = root.to_path_buf();
        // Resolve only the trusted root. Child spellings always retain every
        // component and are opened relative to already retained ancestors.
        #[cfg(unix)]
        let root_file = File::open(root);
        #[cfg(windows)]
        let root_file = {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::{
                FILE_FLAG_BACKUP_SEMANTICS, FILE_SHARE_READ, FILE_SHARE_WRITE,
            };
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
                .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
                .open(root)
        };
        let root_file = root_file.map_err(|_| PublicError::internal())?;
        if !root_file
            .metadata()
            .map_err(|_| PublicError::internal())?
            .is_dir()
        {
            return Err(PublicError::internal());
        }
        let mut result = Self {
            root: root.into(),
            relative: relative.into(),
            identities: vec![identity(&root_file)?],
            ancestors: vec![root_file],
            leaf: None,
            stamp: None,
        };
        for (index, name) in parts.iter().enumerate() {
            path.push(name);
            let directory = index + 1 < parts.len();
            let file = match open_child(result.ancestors.last().unwrap(), &path, name, directory) {
                Ok(file) => file,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(result),
                Err(_) => return Err(PublicError::internal()),
            };
            let metadata = file.metadata().map_err(|_| PublicError::internal())?;
            if directory {
                if !metadata.is_dir() {
                    return Err(PublicError::internal());
                }
                result.identities.push(identity(&file)?);
                result.ancestors.push(file);
            } else {
                if !metadata.is_file() {
                    return Err(PublicError::internal());
                }
                result.stamp = Some(stamp(&file, &metadata)?);
                result.leaf = Some(file);
            }
        }
        Ok(result)
    }

    pub(super) fn classify(&mut self, maximum: u64, field: &str) -> Result<()> {
        if let Some(file) = &mut self.leaf {
            if self.stamp.as_ref().is_none_or(|s| s.length > maximum) {
                return Err(PublicError::internal());
            }
            let mut content = Vec::new();
            file.take(maximum.saturating_add(1))
                .read_to_end(&mut content)
                .map_err(|_| PublicError::internal())?;
            if content.len() as u64 > maximum {
                return Err(PublicError::internal());
            }
            let value: Value =
                crate::lossless_json::from_slice(&content).map_err(|_| PublicError::internal())?;
            if !value.is_object()
                || value.get(field).is_some_and(|v| match v {
                    Value::Null => false,
                    Value::Bool(b) => *b,
                    Value::Array(items) => !items.is_empty(),
                    Value::Object(fields) => !fields.is_empty(),
                    Value::String(text) => !text.is_empty(),
                    Value::Number(n) => n.as_i64() != Some(0),
                })
            {
                return Err(PublicError::internal());
            }
        }
        self.validate()
    }

    pub(super) fn validate(&self) -> Result<()> {
        let current = Self::open(&self.root, &self.relative)?;
        if self.identities != current.identities || self.stamp != current.stamp {
            return Err(PublicError::internal());
        }
        if let Some(file) = &self.leaf {
            let metadata = file.metadata().map_err(|_| PublicError::internal())?;
            if self.stamp.as_ref() != Some(&stamp(file, &metadata)?) {
                return Err(PublicError::internal());
            }
        }
        Ok(())
    }
}
