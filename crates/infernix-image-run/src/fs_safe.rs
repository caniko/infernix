//! Symlink-safe directory-descriptor filesystem operations, strict path-segment
//! validators, and bounded HTTP response streaming.

use std::{
    ffi::CString,
    fs::{self, File},
    io::{self, Write},
    os::unix::{
        ffi::OsStrExt,
        io::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    },
    path::{Component, Path},
};

use uuid::Uuid;

use crate::error::{RunnerError, RunnerResult};

/// Validate a URL path segment or identifier: ASCII `[A-Za-z0-9._-]`, 1..=128 bytes,
/// never `"."`, `".."`, or containing `".."`.
pub fn valid_id_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment.len() <= 128
        && segment != "."
        && segment != ".."
        && !segment.contains("..")
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Validate a single safe filename (no slashes, no `.` / `..`, no NUL).
pub fn valid_single_filename(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\0')
}

fn cstr_from_bytes(bytes: &[u8]) -> io::Result<CString> {
    CString::new(bytes).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "interior NUL"))
}

/// Open an existing directory without following a symlink at the final component.
pub fn open_dir_nofollow(path: &Path) -> io::Result<OwnedFd> {
    let c_path = cstr_from_bytes(path.as_os_str().as_bytes())?;
    // SAFETY: valid NUL-terminated path, flags do not use mode.
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd was freshly opened by open().
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Walk an absolute path component-by-component creating missing directories if
/// `create` is true, refusing any symlink along the tail once inside `trusted_base`
/// (or along the full path when `trusted_base` is `None`).
pub fn open_dir_strict(path: &Path, create: bool, mode: libc::mode_t) -> io::Result<OwnedFd> {
    if !path.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "directory path must be absolute",
        ));
    }
    let mut cur = open_dir_nofollow(Path::new("/"))?;
    for comp in path.components() {
        match comp {
            Component::RootDir | Component::CurDir => continue,
            Component::ParentDir | Component::Prefix(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "directory path contains disallowed component",
                ));
            }
            Component::Normal(part) => {
                let c_name = cstr_from_bytes(part.as_bytes())?;
                // SAFETY: valid dirfd and NUL-terminated component name.
                let mut fd = unsafe {
                    libc::openat(
                        cur.as_raw_fd(),
                        c_name.as_ptr(),
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                if fd < 0 {
                    let err = io::Error::last_os_error();
                    if create && err.kind() == io::ErrorKind::NotFound {
                        // SAFETY: valid dirfd and component name.
                        let mk = unsafe { libc::mkdirat(cur.as_raw_fd(), c_name.as_ptr(), mode) };
                        if mk < 0 {
                            let mk_err = io::Error::last_os_error();
                            if mk_err.kind() != io::ErrorKind::AlreadyExists {
                                return Err(mk_err);
                            }
                        }
                        // SAFETY: reopen newly created directory strictly with O_NOFOLLOW.
                        fd = unsafe {
                            libc::openat(
                                cur.as_raw_fd(),
                                c_name.as_ptr(),
                                libc::O_RDONLY
                                    | libc::O_DIRECTORY
                                    | libc::O_NOFOLLOW
                                    | libc::O_CLOEXEC,
                            )
                        };
                        if fd < 0 {
                            return Err(io::Error::last_os_error());
                        }
                        // Explicitly apply requested mode so process umask cannot strip group-write
                        // bits needed for cross-UID staging handoff.
                        // SAFETY: fd is a freshly opened directory descriptor.
                        let _ = unsafe { libc::fchmod(fd, mode) };
                    } else {
                        return Err(err);
                    }
                }
                // SAFETY: fd was freshly opened by openat().
                cur = unsafe { OwnedFd::from_raw_fd(fd) };
            }
        }
    }
    Ok(cur)
}

/// Open an existing regular file relative to a trusted directory fd without following symlinks.
pub fn open_read_regular_in_dir(dir_fd: RawFd, name: &str) -> io::Result<File> {
    if !valid_single_filename(name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid filename",
        ));
    }
    let c_name = cstr_from_bytes(name.as_bytes())?;
    // SAFETY: valid dirfd and NUL-terminated single filename.
    let fd = unsafe {
        libc::openat(
            dir_fd,
            c_name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd was freshly opened by openat().
    let file = unsafe { File::from_raw_fd(fd) };
    let meta = file.metadata()?;
    if !meta.file_type().is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "target is not a regular file",
        ));
    }
    Ok(file)
}

/// Create a brand-new regular file relative to a trusted directory fd (`O_CREAT | O_EXCL | O_NOFOLLOW`).
pub fn create_new_file_in_dir(dir_fd: RawFd, name: &str, mode: libc::mode_t) -> io::Result<File> {
    if !valid_single_filename(name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid filename",
        ));
    }
    let c_name = cstr_from_bytes(name.as_bytes())?;
    // SAFETY: valid dirfd and NUL-terminated single filename.
    let fd = unsafe {
        libc::openat(
            dir_fd,
            c_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            mode as libc::c_uint,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd was freshly opened by openat().
    Ok(unsafe { File::from_raw_fd(fd) })
}

pub fn fsync_dir(dir_fd: RawFd) -> io::Result<()> {
    // SAFETY: valid open directory file descriptor.
    let rc = unsafe { libc::fsync(dir_fd) };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn unlink_in_dir(dir_fd: RawFd, name: &str) {
    if let Ok(c_name) = cstr_from_bytes(name.as_bytes()) {
        // SAFETY: valid dirfd and NUL-terminated name.
        unsafe {
            libc::unlinkat(dir_fd, c_name.as_ptr(), 0);
        }
    }
}

/// Safely unlink a regular (non-symlink) file at a validated relative path inside
/// a trusted base directory. Succeeds if the file is already absent (`NotFound`).
pub fn unlink_safe_relative_file(base_dir: &Path, rel_path: &str) -> io::Result<()> {
    if rel_path.is_empty() || rel_path.starts_with('/') || rel_path.contains('\0') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "relative cleanup path is invalid",
        ));
    }
    let parts: Vec<&str> = rel_path.split('/').collect();
    if parts.is_empty() || !parts.iter().all(|p| valid_single_filename(p)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "relative cleanup path contains unsafe segment",
        ));
    }
    let mut cur = open_dir_strict(base_dir, false, 0o750)?;
    for sub in &parts[..parts.len() - 1] {
        let c_sub = cstr_from_bytes(sub.as_bytes())?;
        // SAFETY: valid dirfd and single validated component.
        let fd = unsafe {
            libc::openat(
                cur.as_raw_fd(),
                c_sub.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::NotFound {
                return Ok(());
            }
            return Err(err);
        }
        // SAFETY: freshly opened fd.
        cur = unsafe { OwnedFd::from_raw_fd(fd) };
    }
    let leaf = parts[parts.len() - 1];
    let c_leaf = cstr_from_bytes(leaf.as_bytes())?;
    // Verify target is a regular file (not symlink or dir) via fstatat(AT_SYMLINK_NOFOLLOW).
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let stat_rc = unsafe {
        libc::fstatat(
            cur.as_raw_fd(),
            c_leaf.as_ptr(),
            &mut st,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if stat_rc < 0 {
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(err);
    }
    if (st.st_mode & libc::S_IFMT) != libc::S_IFREG {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing to unlink non-regular file",
        ));
    }
    // SAFETY: validated single filename in trusted dirfd.
    let rc = unsafe { libc::unlinkat(cur.as_raw_fd(), c_leaf.as_ptr(), 0) };
    if rc < 0 {
        let err = io::Error::last_os_error();
        if err.kind() == io::ErrorKind::NotFound {
            return Ok(());
        }
        return Err(err);
    }
    Ok(())
}

/// Atomically publish a brand-new file inside `dir_fd` without ever overwriting an
/// existing file or following symlinks (unique tmp + fsync + `linkat` + `unlinkat` + dir fsync).
pub fn publish_exclusive_in_dir(
    dir_fd: RawFd,
    target_name: &str,
    data: &[u8],
    mode: libc::mode_t,
) -> io::Result<()> {
    if !valid_single_filename(target_name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid target filename",
        ));
    }
    let tmp_name = format!(".tmp-pub-{}", Uuid::new_v4().simple());
    let write_res: io::Result<()> = (|| {
        let mut f = create_new_file_in_dir(dir_fd, &tmp_name, mode)?;
        f.write_all(data)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_res {
        unlink_in_dir(dir_fd, &tmp_name);
        return Err(err);
    }

    let c_tmp = cstr_from_bytes(tmp_name.as_bytes())?;
    let c_dst = cstr_from_bytes(target_name.as_bytes())?;
    // SAFETY: both names are NUL-terminated single filenames inside dir_fd; flags=0 does not follow symlinks.
    let rc = unsafe { libc::linkat(dir_fd, c_tmp.as_ptr(), dir_fd, c_dst.as_ptr(), 0) };
    unlink_in_dir(dir_fd, &tmp_name);
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    fsync_dir(dir_fd)?;
    Ok(())
}

/// Atomically replace `target_name` inside `dir_fd` using a unique `O_CREAT | O_EXCL | O_NOFOLLOW`
/// temp file, `fsync` of the file, `renameat`, and `fsync` of the parent directory.
pub fn atomic_replace_in_dir(
    dir_fd: RawFd,
    target_name: &str,
    data: &[u8],
    mode: libc::mode_t,
) -> io::Result<()> {
    if !valid_single_filename(target_name) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid target filename",
        ));
    }
    let tmp_name = format!(".tmp-rep-{}", Uuid::new_v4().simple());
    let write_res: io::Result<()> = (|| {
        let mut f = create_new_file_in_dir(dir_fd, &tmp_name, mode)?;
        f.write_all(data)?;
        f.sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_res {
        unlink_in_dir(dir_fd, &tmp_name);
        return Err(err);
    }

    let c_tmp = cstr_from_bytes(tmp_name.as_bytes())?;
    let c_dst = cstr_from_bytes(target_name.as_bytes())?;
    // SAFETY: both names are NUL-terminated single filenames inside dir_fd.
    let rc = unsafe { libc::renameat(dir_fd, c_tmp.as_ptr(), dir_fd, c_dst.as_ptr()) };
    if rc < 0 {
        let err = io::Error::last_os_error();
        unlink_in_dir(dir_fd, &tmp_name);
        return Err(err);
    }
    fsync_dir(dir_fd)?;
    Ok(())
}

/// Stream an HTTP response body with a hard ceiling of `max_bytes`, failing before
/// allocating past `max_bytes`.
pub async fn read_bounded_response(
    mut resp: reqwest::Response,
    max_bytes: usize,
    label: &str,
) -> RunnerResult<Vec<u8>> {
    if let Some(len) = resp.content_length() {
        if len > max_bytes as u64 {
            return Err(RunnerError::invalid_input(format!(
                "{label} exceeds {max_bytes} byte limit (content-length {len})"
            )));
        }
    }
    let initial = resp
        .content_length()
        .map(|l| (l as usize).min(max_bytes).min(256 * 1024))
        .unwrap_or(0);
    let mut buf = Vec::with_capacity(initial);
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| RunnerError::internal(format!("read {label} stream: {e}")))?
    {
        if buf.len().saturating_add(chunk.len()) > max_bytes {
            return Err(RunnerError::invalid_input(format!(
                "{label} exceeds {max_bytes} byte limit"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Helper to reject a path whose parent directory cannot be opened strictly (e.g. symlink traversal).
pub fn split_parent_and_name(path: &Path) -> RunnerResult<(&Path, &str)> {
    let parent = path
        .parent()
        .ok_or_else(|| RunnerError::invalid_input("path has no parent directory"))?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .filter(|n| valid_single_filename(n))
        .ok_or_else(|| RunnerError::invalid_input("invalid file name"))?;
    Ok((parent, name))
}

/// Ensure a path is not a symlink using `symlink_metadata`.
pub fn ensure_not_symlink(path: &Path) -> RunnerResult<()> {
    if let Ok(meta) = fs::symlink_metadata(path) {
        if meta.file_type().is_symlink() {
            return Err(RunnerError::invalid_input(format!(
                "refusing symlink at {}",
                path.display()
            )));
        }
    }
    Ok(())
}
