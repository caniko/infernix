//! Strict bounded PNG decoding, metadata-free re-encoding, and symlink-safe
//! atomic non-overwriting publish.
//!
//! Validates PNG signature, chunk framing, IEEE CRC-32 of every chunk, rejects
//! trailing data and animated APNG chunks, performs full bounded pixel decoding
//! via `png::Decoder`, and re-encodes outputs cleanly with `png::Encoder` so zero
//! ancillary metadata chunks (`tEXt`, `zTXt`, `iTXt`, `eXIf`, `iCCP`, `tIME`, …)
//! can ever survive.

use std::{
    io::{Cursor, Read},
    os::unix::io::AsRawFd,
    path::Path,
};

use sha2::{Digest, Sha256};

use crate::{
    error::{RunnerError, RunnerResult},
    fs_safe::{
        open_dir_strict, open_read_regular_in_dir, publish_exclusive_in_dir, split_parent_and_name,
    },
};

pub const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
pub const MAX_PNG_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PNG_DIM: u32 = 4096;
pub const MAX_PNG_PIXELS: u64 = 4 * 1024 * 1024;
/// Upper bound on decoded pixel buffer (4 MP * 8 bytes/pixel + decoder overhead).
const MAX_DECODED_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub size: u64,
    pub sha256: String,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn valid_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// IEEE 802.3 CRC-32 over chunk type + chunk data, as required by ISO/IEC 15948.
fn crc32(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in chunk_type.iter().chain(data.iter()) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Strict container-level framing, CRC, IHDR bounds, critical-chunk ordering,
/// APNG rejection, and trailing-byte check prior to bounded pixel decoding.
fn verify_png_framing(bytes: &[u8]) -> RunnerResult<(u32, u32)> {
    if bytes.is_empty() || bytes.len() > MAX_PNG_BYTES {
        return Err(RunnerError::invalid_input(
            "PNG exceeds size bounds or is empty",
        ));
    }
    if bytes.len() < PNG_SIGNATURE.len() + 12 || bytes[..8] != PNG_SIGNATURE {
        return Err(RunnerError::invalid_input("invalid PNG signature"));
    }

    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut color_type = 0u8;
    let mut saw_ihdr = false;
    let mut saw_plte = false;
    let mut saw_idat = false;
    let mut idat_ended = false;
    let mut saw_iend = false;

    while pos < bytes.len() {
        if saw_iend {
            return Err(RunnerError::invalid_input(
                "trailing bytes after PNG IEND chunk",
            ));
        }
        if bytes.len() - pos < 12 {
            return Err(RunnerError::invalid_input("truncated PNG chunk header"));
        }
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let chunk_type: [u8; 4] = bytes[pos + 4..pos + 8].try_into().unwrap();
        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(len)
            .ok_or_else(|| RunnerError::invalid_input("PNG chunk length overflow"))?;
        let crc_end = data_end
            .checked_add(4)
            .ok_or_else(|| RunnerError::invalid_input("PNG chunk CRC overflow"))?;
        if crc_end > bytes.len() {
            return Err(RunnerError::invalid_input("truncated PNG chunk payload"));
        }

        let data = &bytes[data_start..data_end];
        let expected_crc = u32::from_be_bytes(bytes[data_end..crc_end].try_into().unwrap());
        if crc32(&chunk_type, data) != expected_crc {
            return Err(RunnerError::invalid_input("PNG chunk CRC mismatch"));
        }

        if !saw_ihdr {
            if &chunk_type != b"IHDR" || data.len() != 13 {
                return Err(RunnerError::invalid_input(
                    "PNG must start with a valid 13-byte IHDR chunk",
                ));
            }
            width = u32::from_be_bytes(data[0..4].try_into().unwrap());
            height = u32::from_be_bytes(data[4..8].try_into().unwrap());
            let bit_depth = data[8];
            color_type = data[9];
            let compression = data[10];
            let filter = data[11];
            let interlace = data[12];
            if width == 0 || height == 0 || width > MAX_PNG_DIM || height > MAX_PNG_DIM {
                return Err(RunnerError::invalid_input(
                    "PNG dimensions out of allowed 1..=4096 range",
                ));
            }
            if u64::from(width) * u64::from(height) > MAX_PNG_PIXELS {
                return Err(RunnerError::invalid_input("PNG exceeds 4 megapixel budget"));
            }
            let long = width.max(height);
            let short = width.min(height);
            if long > 4 * short {
                return Err(RunnerError::invalid_input(
                    "PNG aspect ratio must be between 1:4 and 4:1",
                ));
            }
            let valid_depth_color = matches!(
                (color_type, bit_depth),
                (0, 1 | 2 | 4 | 8 | 16)
                    | (2, 8 | 16)
                    | (3, 1 | 2 | 4 | 8)
                    | (4, 8 | 16)
                    | (6, 8 | 16)
            );
            if !valid_depth_color || compression != 0 || filter != 0 || interlace > 1 {
                return Err(RunnerError::invalid_input(
                    "PNG IHDR contains illegal encoding fields",
                ));
            }
            saw_ihdr = true;
        } else if &chunk_type == b"IHDR" {
            return Err(RunnerError::invalid_input("duplicate PNG IHDR chunk"));
        }

        if matches!(&chunk_type, b"acTL" | b"fcTL" | b"fdAT") {
            return Err(RunnerError::invalid_input(
                "animated PNGs (APNG) are not supported",
            ));
        }

        if &chunk_type == b"PLTE" {
            if saw_plte || saw_idat || data.is_empty() || !data.len().is_multiple_of(3) {
                return Err(RunnerError::invalid_input(
                    "invalid or out-of-order PNG PLTE chunk",
                ));
            }
            saw_plte = true;
        }

        if &chunk_type == b"IDAT" {
            if idat_ended {
                return Err(RunnerError::invalid_input(
                    "non-consecutive PNG IDAT chunks",
                ));
            }
            saw_idat = true;
        } else if saw_idat {
            idat_ended = true;
        }

        if &chunk_type == b"IEND" {
            if len != 0 {
                return Err(RunnerError::invalid_input("PNG IEND chunk must be empty"));
            }
            saw_iend = true;
        }

        pos = crc_end;
    }

    if !saw_ihdr || !saw_idat || !saw_iend {
        return Err(RunnerError::invalid_input(
            "PNG missing required IHDR, IDAT, or IEND chunk",
        ));
    }
    if color_type == 3 && !saw_plte {
        return Err(RunnerError::invalid_input(
            "indexed-color PNG missing required PLTE chunk",
        ));
    }
    Ok((width, height))
}

struct DecodedPng {
    width: u32,
    height: u32,
    color_type: png::ColorType,
    bit_depth: png::BitDepth,
    pixels: Vec<u8>,
}

/// Perform full bounded pixel decoding to prove all IDAT scanlines, palettes,
/// and zlib streams decode cleanly.
fn decode_verified_png(bytes: &[u8]) -> RunnerResult<DecodedPng> {
    let (expected_w, expected_h) = verify_png_framing(bytes)?;
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits {
            bytes: MAX_DECODED_BYTES,
        },
    );
    // Expand palette/sub-8-bit/tRNS into clean 8-bit pixels so re-encoding emits
    // only IHDR + IDAT + IEND with zero ancillary chunks.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| RunnerError::invalid_input(format!("PNG decode header failed: {e}")))?;
    if reader.info().is_animated() {
        return Err(RunnerError::invalid_input(
            "animated PNGs (APNG) are not supported",
        ));
    }
    let buf_len = reader.output_buffer_size();
    if buf_len == 0 || buf_len > MAX_DECODED_BYTES {
        return Err(RunnerError::invalid_input(
            "PNG decoded pixel buffer exceeds safety limit",
        ));
    }
    let mut pixels = vec![0u8; buf_len];
    let frame = reader
        .next_frame(&mut pixels)
        .map_err(|e| RunnerError::invalid_input(format!("PNG pixel decode failed: {e}")))?;
    if frame.width != expected_w || frame.height != expected_h {
        return Err(RunnerError::invalid_input(
            "PNG decoded dimensions disagree with IHDR",
        ));
    }
    pixels.truncate(frame.buffer_size());

    Ok(DecodedPng {
        width: frame.width,
        height: frame.height,
        color_type: frame.color_type,
        bit_depth: frame.bit_depth,
        pixels,
    })
}

/// Strictly validate a PNG by checking framing/CRCs AND decoding all pixel data.
pub fn validate_png(bytes: &[u8]) -> RunnerResult<PngInfo> {
    let decoded = decode_verified_png(bytes)?;
    Ok(PngInfo {
        width: decoded.width,
        height: decoded.height,
        size: bytes.len() as u64,
        sha256: sha256_hex(bytes),
    })
}

/// Decode all pixels and re-encode a clean PNG containing ONLY `IHDR`, `IDAT`,
/// and `IEND` (zero text, EXIF, ICC, or other ancillary chunks).
pub fn strip_png_metadata(bytes: &[u8]) -> RunnerResult<(Vec<u8>, PngInfo)> {
    let decoded = decode_verified_png(bytes)?;
    let mut clean = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut clean, decoded.width, decoded.height);
        encoder.set_color(decoded.color_type);
        encoder.set_depth(decoded.bit_depth);
        let mut writer = encoder
            .write_header()
            .map_err(|e| RunnerError::internal(format!("encode clean PNG header: {e}")))?;
        writer
            .write_image_data(&decoded.pixels)
            .map_err(|e| RunnerError::internal(format!("encode clean PNG pixels: {e}")))?;
    }
    if clean.len() > MAX_PNG_BYTES {
        return Err(RunnerError::invalid_input(
            "re-encoded PNG exceeds 16 MiB limit",
        ));
    }
    let info = validate_png(&clean)?;
    Ok((clean, info))
}

/// Read an input PNG without following symlinks (along parent dirs or leaf),
/// verifying size, SHA-256, and full pixel decodability.
pub fn read_verified_input_png(
    path: &Path,
    expected_sha256: &str,
) -> RunnerResult<(Vec<u8>, PngInfo)> {
    if !valid_sha256(expected_sha256) {
        return Err(RunnerError::invalid_input(
            "input SHA-256 must be 64 lowercase hex characters",
        ));
    }
    let (parent, name) = split_parent_and_name(path)?;
    let dir_fd = open_dir_strict(parent, false, 0o750).map_err(|_| {
        RunnerError::invalid_input(format!(
            "input PNG parent directory invalid or symlinked: {}",
            parent.display()
        ))
    })?;
    let mut file = open_read_regular_in_dir(dir_fd.as_raw_fd(), name).map_err(|_| {
        RunnerError::invalid_input(format!(
            "input PNG missing or not regular: {}",
            path.display()
        ))
    })?;
    let meta = file
        .metadata()
        .map_err(|_| RunnerError::invalid_input("failed to stat input PNG"))?;
    if meta.len() as usize > MAX_PNG_BYTES {
        return Err(RunnerError::invalid_input("input PNG exceeds 16 MiB"));
    }

    let mut bytes = Vec::with_capacity(meta.len() as usize);
    (&mut file)
        .take((MAX_PNG_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| RunnerError::invalid_input("failed to read input PNG"))?;
    if bytes.len() > MAX_PNG_BYTES {
        return Err(RunnerError::invalid_input("input PNG exceeds 16 MiB"));
    }

    let actual_sha = sha256_hex(&bytes);
    if actual_sha != expected_sha256 {
        return Err(RunnerError::invalid_input(
            "input PNG SHA-256 does not match request digest",
        ));
    }
    let info = validate_png(&bytes)?;
    Ok((bytes, info))
}

/// Decode, verify optional expected dimensions, re-encode cleanly without metadata,
/// fsync, and atomically publish a PNG without following symlinks or overwriting.
pub fn write_verified_png_checked(
    output_path: &Path,
    raw_bytes: &[u8],
    expected_dims: Option<(u32, u32)>,
) -> RunnerResult<PngInfo> {
    let (clean, info) = strip_png_metadata(raw_bytes)?;
    if let Some((want_w, want_h)) = expected_dims {
        if info.width != want_w || info.height != want_h {
            return Err(RunnerError::invalid_input(format!(
                "output PNG dimensions {}x{} do not match profile {}x{}",
                info.width, info.height, want_w, want_h
            )));
        }
    }
    let (parent, name) = split_parent_and_name(output_path)?;
    let dir_fd = open_dir_strict(parent, true, 0o750)
        .map_err(|e| RunnerError::internal(format!("open output parent dir strictly: {e}")))?;

    publish_exclusive_in_dir(dir_fd.as_raw_fd(), name, &clean, 0o640).map_err(|err| {
        if err.kind() == std::io::ErrorKind::AlreadyExists {
            RunnerError::invalid_input("output PNG path already exists")
        } else {
            RunnerError::internal(format!("publish output PNG: {err}"))
        }
    })?;
    Ok(info)
}

/// Convenience wrapper without expected dimension constraint.
pub fn write_verified_png(output_path: &Path, raw_bytes: &[u8]) -> RunnerResult<PngInfo> {
    write_verified_png_checked(output_path, raw_bytes, None)
}

/// Build a real pixel-decodable synthetic RGB PNG (for tests and fixtures),
/// optionally injecting extra chunks before `IEND` to test metadata stripping,
/// APNG rejection, or chunk validation.
pub fn encode_test_png(width: u32, height: u32, extra_chunks: &[([u8; 4], &[u8])]) -> Vec<u8> {
    fn push_chunk(buf: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
        buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buf.extend_from_slice(&kind);
        buf.extend_from_slice(data);
        buf.extend_from_slice(&crc32(&kind, data).to_be_bytes());
    }

    let pixel_bytes = (width as usize) * (height as usize) * 3;
    let pixels = vec![0x80u8; pixel_bytes];
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("write test PNG header");
        writer
            .write_image_data(&pixels)
            .expect("write test PNG pixels");
    }
    if extra_chunks.is_empty() {
        return encoded;
    }

    // Splice extra chunks immediately before the 12-byte terminal IEND chunk.
    let iend_start = encoded.len() - 12;
    let iend = encoded[iend_start..].to_vec();
    encoded.truncate(iend_start);
    for (kind, data) in extra_chunks {
        push_chunk(&mut encoded, *kind, data);
    }
    encoded.extend_from_slice(&iend);
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_prompt_text_and_iccp_chunks_and_preserves_decodable_pixels() {
        let with_meta = encode_test_png(
            1024,
            1024,
            &[
                (*b"tEXt", b"prompt\0secret user prompt"),
                (*b"iTXt", b"workflow\0\0\0\0\0{}"),
                (*b"iCCP", b"fake-icc\0\0\x78\x9c\x03\x00\x00\x00\x00\x01"),
            ],
        );
        let (clean, info) = strip_png_metadata(&with_meta).expect("strip metadata");
        assert_eq!(info.width, 1024);
        assert_eq!(info.height, 1024);
        assert!(
            !clean.windows(6).any(|w| w == b"secret"),
            "tEXt payload must be stripped"
        );
        assert!(
            !clean
                .windows(4)
                .any(|w| w == b"tEXt" || w == b"iTXt" || w == b"iCCP"),
            "metadata and ancillary chunks must be stripped"
        );
        let rechecked = validate_png(&clean).expect("stripped PNG must decode cleanly");
        assert_eq!(rechecked.sha256, info.sha256);
    }

    #[test]
    fn rejects_dummy_undecodable_idat_corrupt_crc_apng_and_extreme_aspect() {
        // Construct a structurally framed 1024x1024 PNG with a dummy 1-row IDAT: must fail pixel decode!
        let mut fake = Vec::new();
        fake.extend_from_slice(&PNG_SIGNATURE);
        let push = |buf: &mut Vec<u8>, kind: [u8; 4], data: &[u8]| {
            buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
            buf.extend_from_slice(&kind);
            buf.extend_from_slice(data);
            buf.extend_from_slice(&crc32(&kind, data).to_be_bytes());
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1024u32.to_be_bytes());
        ihdr.extend_from_slice(&1024u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        push(&mut fake, *b"IHDR", &ihdr);
        push(
            &mut fake,
            *b"IDAT",
            &[
                0x78, 0x01, 0x01, 0x04, 0x00, 0xFB, 0xFF, 0x00, 0x80, 0x80, 0x80, 0x01, 0x81, 0x01,
                0x81,
            ],
        );
        push(&mut fake, *b"IEND", &[]);
        assert_eq!(
            validate_png(&fake).unwrap_err().code,
            14,
            "dummy truncated IDAT must be rejected by real pixel decode"
        );

        let mut bad_crc = encode_test_png(64, 64, &[]);
        let last = bad_crc.len() - 1;
        bad_crc[last] ^= 0xFF;
        assert_eq!(validate_png(&bad_crc).unwrap_err().code, 14);

        let apng = encode_test_png(64, 64, &[(*b"acTL", &[0, 0, 0, 2, 0, 0, 0, 0])]);
        assert_eq!(validate_png(&apng).unwrap_err().code, 14);

        let skinny = encode_test_png(1000, 100, &[]);
        assert_eq!(validate_png(&skinny).unwrap_err().code, 14);
    }
}
