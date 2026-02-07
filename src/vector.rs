//! # Vector parsing and storage formats
//!
//! This module is the heart of how vectors move in and out of SQLite.
//! Vectors can arrive as **JSON text** (`'[1.0, 2.0, 3.0]'`) or as
//! compact **binary BLOBs** (f32 or f16).  Every path produces the
//! same [`Vector`] struct so the rest of the extension never has to
//! care which format was used on the SQL side.
//!
//! ## Binary BLOB layout
//!
//! All binary vectors share a 4-byte header followed by the raw
//! floating-point data in **little-endian** byte order:
//!
//! ```text
//!  Byte 0     Byte 1     Bytes 2-3          Bytes 4..
//! +----------+----------+-----------------+------------------+
//! |  0xBE    |  format  |  dim (u16 LE)   |  element data …  |
//! |  magic   |  0x01=f32|                 |  (f32 or f16 LE) |
//! |          |  0x02=f16|                 |                  |
//! +----------+----------+-----------------+------------------+
//! ```
//!
//! - **Magic byte** `0xBE` lets us quickly reject non-vector BLOBs.
//! - **Format byte** tells us how wide each element is (4 bytes for f32,
//!   2 bytes for f16).
//! - **Dimension** is stored as a little-endian `u16`, supporting up to
//!   65 535 dimensions.
//! - **Element data** is the raw IEEE 754 floats packed contiguously.
//!
//! ## Why little-endian?
//!
//! x86, ARM (in its default mode), and WASM are all little-endian, so
//! on those platforms the bytes can be reinterpreted in-place with zero
//! conversion cost.  We use `from_le_bytes` / `to_le_bytes` everywhere
//! so the code is still correct on big-endian hosts.

use sqlite_loadable::prelude::*;
use sqlite_loadable::{
    api::{self, ValueType},
    Error, Result,
};

// ── Constants ───────────────────────────────────────────────────────
//
// These constants define the binary wire format.  They are `pub` so
// that external tools (like the `vecgen` CLI) can produce compatible
// BLOBs without importing this crate.

/// Magic byte that starts every vector BLOB.
/// We use 0xBE as a distinctive marker unlikely to collide with
/// other BLOB data stored in SQLite.
pub const BLOB_MAGIC: u8 = 0xBE;

/// Format tag: each element is an IEEE 754 **single-precision** float
/// (4 bytes, `f32`).
pub const FORMAT_F32: u8 = 0x01;

/// Format tag: each element is an IEEE 754 **half-precision** float
/// (2 bytes, `f16`).  Half-precision cuts storage in half at the cost
/// of roughly 3 decimal digits of precision — plenty for embedding
/// vectors that are typically already quantised.
pub const FORMAT_F16: u8 = 0x02;

/// Total header size in bytes: magic (1) + format (1) + dimension (2).
pub const HEADER_LEN: usize = 4;

// ── VectorFormat enum ───────────────────────────────────────────────

/// Tracks which serialization format a [`Vector`] was decoded from.
///
/// This is carried along so that functions like `vec_format()` can
/// report the original storage type back to the user.
///
/// ### Rust note — `#[derive(...)]`
///
/// The `derive` attribute asks the compiler to auto-generate trait
/// implementations.  Here we get:
/// - `Debug`   — lets us use `{:?}` in format strings for printing
/// - `Clone`   — allows `.clone()` to make a copy
/// - `Copy`    — makes the enum implicitly copyable (it's tiny, just
///               one byte under the hood)
/// - `PartialEq` / `Eq` — lets us use `==` to compare variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorFormat {
    /// The vector was parsed from a JSON text string like `'[1,2,3]'`.
    Json,
    /// The vector was parsed from a binary BLOB using 32-bit floats.
    F32,
    /// The vector was parsed from a binary BLOB using 16-bit floats.
    F16,
}

// ── Vector struct ───────────────────────────────────────────────────

/// A parsed, in-memory vector ready for distance calculations.
///
/// Internally we always store elements as `f64` so that distance math
/// (dot products, norms) doesn't lose precision due to intermediate
/// rounding.  The original format is remembered in [`format`] so we
/// can round-trip back if needed.
///
/// ### Rust note — `Vec<f64>`
///
/// `Vec<T>` is Rust's growable, heap-allocated array (like C++'s
/// `std::vector` or Python's `list`).  It owns its data and will free
/// the memory automatically when the `Vector` is dropped.
#[derive(Debug, Clone)]
pub struct Vector {
    /// The vector's elements, always stored in `f64` for precision.
    pub data: Vec<f64>,

    /// Number of elements  (i.e. `data.len()`).  Stored explicitly so
    /// we can validate dimensions without re-counting.
    pub dim: usize,

    /// Which serialization format the vector was decoded from.
    pub format: VectorFormat,
}

impl Vector {
    // ── JSON parsing ────────────────────────────────────────────────

    /// Parse a vector from a JSON array string, e.g. `"[1.0, 2.0, 3.0]"`.
    ///
    /// Under the hood this calls `serde_json::from_str` which handles
    /// whitespace, trailing commas, and numeric coercion automatically.
    ///
    /// ### Rust note — `Result` and the `?` operator
    ///
    /// This function returns `Result<Self>`.  `Self` means "the type
    /// we're inside" — here that's `Vector`.  The `?` operator is
    /// shorthand for "if this is an error, return it immediately".
    /// We don't use `?` here because `serde_json`'s error type is
    /// different from `sqlite_loadable::Error`, so we convert with
    /// `.map_err(...)`.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let data: Vec<f64> = serde_json::from_str(text)
            .map_err(|e| Error::new_message(&format!("Invalid JSON vector: {}", e)))?;
        Ok(Self {
            dim: data.len(),
            data,
            format: VectorFormat::Json,
        })
    }

    // ── f32 BLOB parsing ────────────────────────────────────────────

    /// Decode a vector from a binary BLOB in **f32** format.
    ///
    /// Expected layout (see module-level docs for the diagram):
    ///
    /// | Offset | Length | Content                          |
    /// |--------|--------|----------------------------------|
    /// | 0      | 1      | `BLOB_MAGIC` (`0xVE`)            |
    /// | 1      | 1      | `FORMAT_F32` (`0x01`)            |
    /// | 2      | 2      | dimension as little-endian `u16` |
    /// | 4      | dim×4  | `f32` elements, little-endian    |
    ///
    /// ### Rust note — slices and `chunks_exact`
    ///
    /// The `blob` parameter is a **slice** (`&[u8]`): a borrowed view
    /// into a contiguous block of bytes.  Slices carry their length, so
    /// `blob.len()` is O(1) and bounds-checked at runtime.
    ///
    /// `chunks_exact(4)` splits the slice into non-overlapping 4-byte
    /// windows.  Each chunk `c` is itself a `&[u8]` of length 4, which
    /// we convert to `f32` via `f32::from_le_bytes`.
    pub fn from_f32_blob(blob: &[u8]) -> Result<Self> {
        // ── Step 1: make sure we have at least the 4-byte header ────
        if blob.len() < HEADER_LEN {
            return Err(Error::new_message("f32 BLOB too short for header"));
        }

        // ── Step 2: validate magic and format bytes ─────────────────
        if blob[0] != BLOB_MAGIC || blob[1] != FORMAT_F32 {
            return Err(Error::new_message("Invalid f32 BLOB header"));
        }

        // ── Step 3: read the dimension from bytes 2-3 ───────────────
        //
        // `u16::from_le_bytes` converts two bytes in little-endian
        // order into a Rust `u16`.  We then widen to `usize` because
        // slice lengths and Vec capacities are `usize` in Rust.
        let dim = u16::from_le_bytes([blob[2], blob[3]]) as usize;

        // ── Step 4: slice off the body (everything after the header) ─
        //
        // `&blob[HEADER_LEN..]` creates a sub-slice starting at byte 4.
        // This is a zero-cost operation — no data is copied.
        let body = &blob[HEADER_LEN..];

        // ── Step 5: verify the body is exactly the right length ─────
        let expected = dim * 4; // 4 bytes per f32
        if body.len() != expected {
            return Err(Error::new_message(&format!(
                "f32 BLOB body length {} does not match dimension {} (expected {} bytes)",
                body.len(),
                dim,
                expected
            )));
        }

        // ── Step 6: decode each 4-byte chunk into an f64 ────────────
        //
        // `.chunks_exact(4)` yields an iterator of `&[u8]` slices,
        // each guaranteed to be exactly 4 bytes long.
        //
        // `.map(|c| ...)` transforms each chunk: we reconstruct the
        // `f32` from its little-endian bytes, then widen to `f64` with
        // `as f64` (this conversion is lossless — every f32 value is
        // exactly representable as f64).
        //
        // `.collect()` gathers the iterator into a `Vec<f64>`.
        let data: Vec<f64> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect();

        Ok(Self {
            dim,
            data,
            format: VectorFormat::F32,
        })
    }

    // ── f16 BLOB parsing ────────────────────────────────────────────

    /// Decode a vector from a binary BLOB in **f16** (half-precision)
    /// format.
    ///
    /// The layout is identical to f32 except:
    /// - The format byte is `FORMAT_F16` (`0x02`).
    /// - Each element is 2 bytes instead of 4.
    ///
    /// Half-precision floats have:
    /// - 1 sign bit
    /// - 5 exponent bits  (bias 15)
    /// - 10 mantissa bits
    ///
    /// This gives roughly 3.3 decimal digits of precision and a range
    /// of ±65 504.  That's enough for normalised embedding vectors
    /// (which typically live in [-1, 1]) and cuts storage in half
    /// compared to f32.
    ///
    /// ### Rust note — why we don't use a crate for f16
    ///
    /// The `half` crate is the standard choice, but we hand-roll the
    /// conversion here to avoid adding a dependency for two small
    /// functions.  See [`f16_to_f32`] and [`f32_to_f16`] below.
    pub fn from_f16_blob(blob: &[u8]) -> Result<Self> {
        // Same header validation pattern as from_f32_blob
        if blob.len() < HEADER_LEN {
            return Err(Error::new_message("f16 BLOB too short for header"));
        }
        if blob[0] != BLOB_MAGIC || blob[1] != FORMAT_F16 {
            return Err(Error::new_message("Invalid f16 BLOB header"));
        }

        let dim = u16::from_le_bytes([blob[2], blob[3]]) as usize;
        let body = &blob[HEADER_LEN..];
        let expected = dim * 2; // 2 bytes per f16
        if body.len() != expected {
            return Err(Error::new_message(&format!(
                "f16 BLOB body length {} does not match dimension {} (expected {} bytes)",
                body.len(),
                dim,
                expected
            )));
        }

        // Decode each 2-byte chunk: reconstruct the u16 bit pattern,
        // then promote to f32 via our manual converter, then widen to f64.
        let data: Vec<f64> = body
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
            .collect();

        Ok(Self {
            dim,
            data,
            format: VectorFormat::F16,
        })
    }

    // ── Auto-detect from raw BLOB bytes ─────────────────────────────

    /// Inspect a BLOB's header and dispatch to the correct format-specific
    /// parser.
    ///
    /// This is the main entry point for BLOB values coming from SQLite.
    /// The magic byte tells us "this is definitely a vector BLOB" and
    /// the format byte tells us which decoder to call.
    ///
    /// ### Rust note — `match`
    ///
    /// `match` is Rust's pattern-matching expression.  It's like a
    /// `switch` in C but *exhaustive*: the compiler forces you to
    /// handle every possible case.  The `other` arm with the wildcard
    /// `_` would also work, but naming it `other` lets us include the
    /// byte value in the error message.
    pub fn from_blob(blob: &[u8]) -> Result<Self> {
        if blob.len() < HEADER_LEN {
            return Err(Error::new_message("BLOB too short"));
        }
        if blob[0] != BLOB_MAGIC {
            return Err(Error::new_message(
                "BLOB does not start with vector magic byte 0xBE",
            ));
        }
        match blob[1] {
            FORMAT_F32 => Self::from_f32_blob(blob),
            FORMAT_F16 => Self::from_f16_blob(blob),
            other => Err(Error::new_message(&format!(
                "Unknown vector BLOB format byte: 0x{:02X}",
                other
            ))),
        }
    }

    // ── Encode back to BLOB ─────────────────────────────────────────

    /// Serialize this vector into its **f32 binary BLOB** representation.
    ///
    /// This is the inverse of [`from_f32_blob`].  The returned `Vec<u8>`
    /// can be stored directly in a SQLite BLOB column.
    ///
    /// ### Rust note — `Vec::with_capacity`
    ///
    /// `Vec::with_capacity(n)` pre-allocates room for `n` bytes so that
    /// the subsequent `push` / `extend_from_slice` calls don't trigger
    /// repeated reallocations.  It's a performance optimisation — the
    /// code would be correct without it, just slower for large vectors.
    pub fn to_f32_blob(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_LEN + self.dim * 4);
        buf.push(BLOB_MAGIC);
        buf.push(FORMAT_F32);
        // Write dimension as two little-endian bytes.
        // `as u16` truncates; safe because we enforce dim ≤ 65 535.
        buf.extend_from_slice(&(self.dim as u16).to_le_bytes());
        for &v in &self.data {
            // `v` is f64; cast to f32 then write its 4 LE bytes.
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        buf
    }

    /// Serialize this vector into its **f16 binary BLOB** representation.
    ///
    /// Same structure as [`to_f32_blob`] but each element is compressed
    /// to 2 bytes via [`f32_to_f16`].  Use this when storage or I/O
    /// bandwidth matters more than the last bits of precision.
    pub fn to_f16_blob(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_LEN + self.dim * 2);
        buf.push(BLOB_MAGIC);
        buf.push(FORMAT_F16);
        buf.extend_from_slice(&(self.dim as u16).to_le_bytes());
        for &v in &self.data {
            buf.extend_from_slice(&f32_to_f16(v as f32).to_le_bytes());
        }
        buf
    }

    /// Produce the canonical JSON string, e.g. `"[1.0,2.0,3.0]"`.
    ///
    /// ### Rust note — `unwrap_or_else`
    ///
    /// `serde_json::to_string` returns a `Result`.  In theory it can
    /// fail (e.g. if a value is NaN and the serializer is strict), so
    /// we supply a fallback with `unwrap_or_else`.  The closure `|_|`
    /// ignores the error and returns `"[]"`.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(&self.data).unwrap_or_else(|_| "[]".to_string())
    }
}

// ── Parse from SQLite value (auto-detect TEXT vs BLOB) ──────────────

/// Read a vector from any SQLite value, automatically choosing the
/// right parser based on the value's type.
///
/// SQLite values carry a runtime type tag (TEXT, BLOB, INTEGER, FLOAT,
/// NULL).  We inspect that tag and branch:
///
/// | SQLite type | Parser used                |
/// |-------------|----------------------------|
/// | TEXT        | [`Vector::from_json_str`]   |
/// | BLOB        | [`Vector::from_blob`]      |
/// | NULL        | error                      |
/// | other       | try TEXT as fallback       |
///
/// ### Rust note — raw pointers (`*mut`)
///
/// `*mut sqlite3_value` is a **raw pointer** — Rust's equivalent of
/// C's `sqlite3_value*`.  Raw pointers are "unsafe" to dereference,
/// but the `sqlite-loadable` crate wraps them in safe helper functions
/// like `api::value_type()` and `api::value_text()`.  We pass the
/// raw pointer to those helpers and never dereference it ourselves.
pub fn parse_vector_from_value(value: *mut sqlite3_value) -> Result<Vector> {
    match api::value_type(&value) {
        ValueType::Text => {
            let text = api::value_text(&value)?;
            Vector::from_json_str(text)
        }
        ValueType::Blob => {
            let blob = api::value_blob(&value);
            Vector::from_blob(blob)
        }
        ValueType::Null => Err(Error::new_message("Vector value is NULL")),
        _ => {
            // INTEGER or FLOAT — unlikely for a vector, but try text as
            // a last resort (SQLite will coerce to a string).
            let text = api::value_text(&value)?;
            Vector::from_json_str(text)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  IEEE 754 half-precision (f16) ↔ single-precision (f32) converters
// ═══════════════════════════════════════════════════════════════════
//
// IEEE 754 floating-point numbers are stored as three bit-fields:
//
//   f32 (32 bits):  1 sign  |  8 exponent (bias 127)  | 23 mantissa
//   f16 (16 bits):  1 sign  |  5 exponent (bias  15)  | 10 mantissa
//
// The conversion adjusts the exponent bias and truncates/extends the
// mantissa.  We also handle the special cases: zero, subnormals
// (very small numbers near zero), infinities, and NaNs.
//
// If you're new to floating-point bit layouts, the key insight is:
//
//   value = (-1)^sign  ×  2^(exponent - bias)  ×  (1 + mantissa / 2^mantissa_bits)
//
// For subnormals (exponent == 0) the implicit leading 1 becomes 0.

/// Convert a 16-bit half-precision float (stored as `u16`) to `f32`.
///
/// ### Bit manipulation walkthrough
///
/// ```text
///  f16 bits:  S EEEEE MMMMMMMMMM
///              │  │        │
///              │  │        └─ 10-bit mantissa
///              │  └────────── 5-bit exponent (0..31)
///              └───────────── 1-bit sign
/// ```
///
/// We extract each field with bit shifts and masks, then reassemble
/// them into the wider f32 layout.
pub fn f16_to_f32(half: u16) -> f32 {
    // Extract the three fields using bitwise operations.
    //
    // `>> 15` shifts the sign bit into position 0.
    // `& 0x1` keeps only that one bit.
    let sign = ((half >> 15) & 0x1) as u32;

    // `>> 10` moves the exponent to bits 0..4.
    // `& 0x1F` masks to 5 bits (0x1F = 0b11111).
    let exponent = ((half >> 10) & 0x1F) as u32;

    // `& 0x3FF` masks the lower 10 bits (0x3FF = 0b11_1111_1111).
    let mantissa = (half & 0x3FF) as u32;

    if exponent == 0 {
        // ── Subnormal or zero ───────────────────────────────────
        // When the exponent field is all zeros, the number is either
        // ±0.0 (mantissa == 0) or a "subnormal" — a very tiny number
        // that trades exponent range for gradual underflow.
        if mantissa == 0 {
            // Positive or negative zero.
            // `sign << 31` puts the sign in bit 31 of the f32.
            return f32::from_bits(sign << 31);
        }
        // Subnormal: we need to normalize by shifting the mantissa
        // left until the implicit leading 1 appears, adjusting the
        // exponent down for each shift.
        let mut e = 0i32;
        let mut m = mantissa;
        while (m & 0x400) == 0 {
            // 0x400 = bit 10; once this bit is set, we've found the
            // implicit leading 1.
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF; // strip the leading 1 (it's implicit in f32 normals)
        // Rebias: f16 bias is 15, f32 bias is 127.
        let f32_exp = (127 - 15 + 1 + e) as u32;
        // Reassemble: sign (bit 31), exponent (bits 30..23), mantissa
        // (bits 22..13 — we shift left by 13 to go from 10 bits to 23).
        let bits = (sign << 31) | (f32_exp << 23) | (m << 13);
        f32::from_bits(bits)
    } else if exponent == 31 {
        // ── Infinity or NaN ─────────────────────────────────────
        // Exponent all-ones means Inf (mantissa = 0) or NaN (mantissa ≠ 0).
        // Map to f32's all-ones exponent (0xFF).
        let bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        f32::from_bits(bits)
    } else {
        // ── Normal number ───────────────────────────────────────
        // Rebias the exponent: subtract f16 bias (15), add f32 bias (127).
        let f32_exp = exponent + (127 - 15);
        // Shift the 10-bit mantissa into the upper 10 bits of the f32's
        // 23-bit mantissa field.
        let bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
        f32::from_bits(bits)
    }
}

/// Convert an `f32` to a 16-bit half-precision float (stored as `u16`).
///
/// This is the reverse of [`f16_to_f32`].  We extract the f32 bit
/// fields, rebias the exponent downward, and truncate the mantissa
/// from 23 bits to 10.
///
/// ### Precision loss
///
/// Truncating the mantissa discards the lowest 13 bits.  A more
/// sophisticated implementation could *round* rather than truncate,
/// but for embedding vectors the difference is negligible.
///
/// ### Overflow / underflow
///
/// - Values too large for f16 (|x| > 65 504) become ±Infinity.
/// - Values too small become ±0.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32; // 8-bit, bias 127
    let mantissa = bits & 0x7FFFFF; // 23 bits

    if exponent == 0 {
        // f32 zero or subnormal → too small for f16, flush to zero.
        sign << 15
    } else if exponent == 0xFF {
        // f32 Inf / NaN → f16 Inf / NaN
        let f16_mantissa = (mantissa >> 13) as u16;
        (sign << 15) | (0x1F << 10) | f16_mantissa
    } else {
        // Rebias: subtract f32 bias (127), add f16 bias (15).
        let new_exp = exponent - 127 + 15;
        if new_exp >= 31 {
            // Overflow: the number is too large for f16 → ±Inf
            (sign << 15) | (0x1F << 10)
        } else if new_exp <= 0 {
            // Underflow: the number is too small for f16 → ±0
            sign << 15
        } else {
            // Normal: truncate mantissa from 23 bits to 10 bits.
            let f16_mantissa = (mantissa >> 13) as u16;
            (sign << 15) | ((new_exp as u16) << 10) | f16_mantissa
        }
    }
}
