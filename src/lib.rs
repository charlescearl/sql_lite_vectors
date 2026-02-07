use sqlite_loadable::prelude::*;
use sqlite_loadable::{define_scalar_function, FunctionFlags, Result};

mod distance;
mod functions;
mod vector;

// The entry point that SQLite calls when loading the extension
#[sqlite_entrypoint]
pub fn sqlite3_extension_init(db: *mut sqlite3) -> Result<()> {
    let flags = FunctionFlags::UTF8 | FunctionFlags::DETERMINISTIC;

    define_scalar_function(db, "vec_cosine", 2, functions::vec_cosine, flags)?;
    define_scalar_function(db, "vec_cosine_distance", 2, functions::vec_cosine_distance, flags)?;
    define_scalar_function(db, "vec_l2", 2, functions::vec_l2, flags)?;
    define_scalar_function(db, "vec_l1", 2, functions::vec_l1, flags)?;
    define_scalar_function(db, "vec_ip", 2, functions::vec_ip, flags)?;
    define_scalar_function(db, "vec_ip_neg", 2, functions::vec_ip_neg, flags)?;
    define_scalar_function(db, "vec_dim", 1, functions::vec_dim, flags)?;
    define_scalar_function(db, "vec_assert_dim", 2, functions::vec_assert_dim, flags)?;

    // Parse / normalize (1 or 2 args)
    define_scalar_function(db, "vec_parse", -1, functions::vec_parse, flags)?;

    // Format conversion
    define_scalar_function(db, "vec_to_f32", 1, functions::vec_to_f32, flags)?;
    define_scalar_function(db, "vec_to_f16", 1, functions::vec_to_f16, flags)?;
    define_scalar_function(db, "vec_to_json", 1, functions::vec_to_json, flags)?;
    define_scalar_function(db, "vec_format", 1, functions::vec_format, flags)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::distance;
    use crate::vector::{
        f16_to_f32, f32_to_f16, Vector, VectorFormat, BLOB_MAGIC, FORMAT_F16, FORMAT_F32,
        HEADER_LEN,
    };

    // ── Helper to build a Vector without going through SQLite ────────

    fn vec_from(data: &[f64]) -> Vector {
        Vector {
            data: data.to_vec(),
            dim: data.len(),
            format: VectorFormat::Json,
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  JSON parsing
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn json_parse_basic() {
        let v = Vector::from_json_str("[1.0, 2.0, 3.0]").unwrap();
        assert_eq!(v.dim, 3);
        assert_eq!(v.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(v.format, VectorFormat::Json);
    }

    #[test]
    fn json_parse_integers() {
        // JSON integers should be accepted and promoted to f64
        let v = Vector::from_json_str("[1, 2, 3]").unwrap();
        assert_eq!(v.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn json_parse_empty() {
        let v = Vector::from_json_str("[]").unwrap();
        assert_eq!(v.dim, 0);
        assert!(v.data.is_empty());
    }

    #[test]
    fn json_parse_negative_values() {
        let v = Vector::from_json_str("[-0.5, 0.0, 0.5]").unwrap();
        assert_eq!(v.data, vec![-0.5, 0.0, 0.5]);
    }

    #[test]
    fn json_parse_invalid_string() {
        assert!(Vector::from_json_str("not json").is_err());
    }

    #[test]
    fn json_parse_wrong_type() {
        // A JSON object is not a vector
        assert!(Vector::from_json_str(r#"{"a": 1}"#).is_err());
    }

    #[test]
    fn json_roundtrip() {
        let original = Vector::from_json_str("[1.5, -2.5, 3.0]").unwrap();
        let json = original.to_json_string().unwrap();
        let restored = Vector::from_json_str(&json).unwrap();
        assert_eq!(original.data, restored.data);
    }

    // ═════════════════════════════════════════════════════════════════
    //  f32 BLOB parsing
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn f32_blob_roundtrip() {
        let original = Vector::from_json_str("[1.0, 2.0, 3.0]").unwrap();
        let blob = original.to_f32_blob().unwrap();

        // Verify header
        assert_eq!(blob[0], BLOB_MAGIC);
        assert_eq!(blob[1], FORMAT_F32);
        assert_eq!(u16::from_le_bytes([blob[2], blob[3]]), 3);
        assert_eq!(blob.len(), HEADER_LEN + 3 * 4);

        // Parse back and compare
        let restored = Vector::from_f32_blob(&blob).unwrap();
        assert_eq!(restored.dim, 3);
        assert_eq!(restored.format, VectorFormat::F32);
        for (a, b) in original.data.iter().zip(restored.data.iter()) {
            assert!((a - b).abs() < 1e-6, "f32 roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn f32_blob_too_short() {
        // Only 3 bytes — not even a full header
        assert!(Vector::from_f32_blob(&[BLOB_MAGIC, FORMAT_F32, 0]).is_err());
    }

    #[test]
    fn f32_blob_wrong_magic() {
        let mut blob = vec_from(&[1.0]).to_f32_blob().unwrap();
        blob[0] = 0x00; // corrupt magic
        assert!(Vector::from_f32_blob(&blob).is_err());
    }

    #[test]
    fn f32_blob_wrong_format_byte() {
        let mut blob = vec_from(&[1.0]).to_f32_blob().unwrap();
        blob[1] = FORMAT_F16; // claim it's f16 but it's f32-sized
        assert!(Vector::from_f32_blob(&blob).is_err());
    }

    #[test]
    fn f32_blob_truncated_body() {
        let mut blob = vec_from(&[1.0, 2.0]).to_f32_blob().unwrap();
        blob.truncate(HEADER_LEN + 4); // only 1 float instead of 2
        assert!(Vector::from_f32_blob(&blob).is_err());
    }

    // ═════════════════════════════════════════════════════════════════
    //  f16 BLOB parsing
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn f16_blob_roundtrip() {
        let original = Vector::from_json_str("[1.0, -0.5, 0.25]").unwrap();
        let blob = original.to_f16_blob().unwrap();

        // Verify header
        assert_eq!(blob[0], BLOB_MAGIC);
        assert_eq!(blob[1], FORMAT_F16);
        assert_eq!(u16::from_le_bytes([blob[2], blob[3]]), 3);
        assert_eq!(blob.len(), HEADER_LEN + 3 * 2);

        // Parse back — f16 has lower precision so use a wider epsilon
        let restored = Vector::from_f16_blob(&blob).unwrap();
        assert_eq!(restored.dim, 3);
        assert_eq!(restored.format, VectorFormat::F16);
        for (a, b) in original.data.iter().zip(restored.data.iter()) {
            assert!(
                (a - b).abs() < 0.01,
                "f16 roundtrip mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn f16_blob_too_short() {
        assert!(Vector::from_f16_blob(&[BLOB_MAGIC, FORMAT_F16, 0]).is_err());
    }

    #[test]
    fn f16_blob_wrong_magic() {
        let mut blob = vec_from(&[1.0]).to_f16_blob().unwrap();
        blob[0] = 0xFF;
        assert!(Vector::from_f16_blob(&blob).is_err());
    }

    #[test]
    fn f16_blob_truncated_body() {
        let mut blob = vec_from(&[1.0, 2.0]).to_f16_blob().unwrap();
        blob.truncate(HEADER_LEN + 2); // only 1 half-float instead of 2
        assert!(Vector::from_f16_blob(&blob).is_err());
    }

    // ═════════════════════════════════════════════════════════════════
    //  Auto-detect BLOB format
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn blob_autodetect_f32() {
        let blob = vec_from(&[1.0, 2.0]).to_f32_blob().unwrap();
        let v = Vector::from_blob(&blob).unwrap();
        assert_eq!(v.format, VectorFormat::F32);
    }

    #[test]
    fn blob_autodetect_f16() {
        let blob = vec_from(&[1.0, 2.0]).to_f16_blob().unwrap();
        let v = Vector::from_blob(&blob).unwrap();
        assert_eq!(v.format, VectorFormat::F16);
    }

    #[test]
    fn blob_autodetect_unknown_format() {
        let mut blob = vec_from(&[1.0]).to_f32_blob().unwrap();
        blob[1] = 0xFF; // unknown format byte
        assert!(Vector::from_blob(&blob).is_err());
    }

    #[test]
    fn blob_autodetect_wrong_magic() {
        assert!(Vector::from_blob(&[0x00, FORMAT_F32, 1, 0, 0, 0, 0, 0]).is_err());
    }

    // ═════════════════════════════════════════════════════════════════
    //  f16 ↔ f32 conversion helpers
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn f16_roundtrip_common_values() {
        let test_values: &[f32] = &[0.0, 1.0, -1.0, 0.5, -0.5, 0.25, 100.0, -100.0];
        for &v in test_values {
            let half = f32_to_f16(v);
            let back = f16_to_f32(half);
            assert!(
                (v - back).abs() < 0.01 || (v == 0.0 && back == 0.0),
                "f16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn f16_zero_preserves_sign() {
        let pos = f16_to_f32(f32_to_f16(0.0f32));
        let neg = f16_to_f32(f32_to_f16(-0.0f32));
        assert_eq!(pos, 0.0);
        assert_eq!(neg, 0.0); // both map to 0.0 (sign may differ at bit level)
    }

    #[test]
    fn f16_overflow_becomes_inf() {
        // 100_000.0 is way beyond f16 max (~65504)
        let half = f32_to_f16(100_000.0);
        let back = f16_to_f32(half);
        assert!(back.is_infinite() && back > 0.0);
    }

    #[test]
    fn f16_inf_roundtrip() {
        let half = f32_to_f16(f32::INFINITY);
        assert!(f16_to_f32(half).is_infinite());

        let half_neg = f32_to_f16(f32::NEG_INFINITY);
        let back = f16_to_f32(half_neg);
        assert!(back.is_infinite() && back < 0.0);
    }

    #[test]
    fn f16_nan_roundtrip() {
        let half = f32_to_f16(f32::NAN);
        assert!(f16_to_f32(half).is_nan());
    }

    // ═════════════════════════════════════════════════════════════════
    //  Distance functions
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn l2_identical_vectors() {
        let v = vec_from(&[1.0, 2.0, 3.0]);
        assert_eq!(distance::l2(&v, &v), 0.0);
    }

    #[test]
    fn l2_known_distance() {
        let a = vec_from(&[0.0, 0.0]);
        let b = vec_from(&[3.0, 4.0]);
        assert!((distance::l2(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn l2_is_symmetric() {
        let a = vec_from(&[1.0, 2.0, 3.0]);
        let b = vec_from(&[4.0, 5.0, 6.0]);
        assert!((distance::l2(&a, &b) - distance::l2(&b, &a)).abs() < 1e-10);
    }

    #[test]
    fn l1_identical_vectors() {
        let v = vec_from(&[1.0, 2.0, 3.0]);
        assert_eq!(distance::l1(&v, &v), 0.0);
    }

    #[test]
    fn l1_known_distance() {
        let a = vec_from(&[0.0, 0.0]);
        let b = vec_from(&[3.0, 4.0]);
        assert!((distance::l1(&a, &b) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn l1_is_symmetric() {
        let a = vec_from(&[1.0, -2.0]);
        let b = vec_from(&[-3.0, 4.0]);
        assert!((distance::l1(&a, &b) - distance::l1(&b, &a)).abs() < 1e-10);
    }

    #[test]
    fn inner_product_orthogonal() {
        let a = vec_from(&[1.0, 0.0]);
        let b = vec_from(&[0.0, 1.0]);
        assert_eq!(distance::inner_product(&a, &b), 0.0);
    }

    #[test]
    fn inner_product_parallel() {
        let a = vec_from(&[2.0, 3.0]);
        let b = vec_from(&[4.0, 5.0]);
        // 2*4 + 3*5 = 23
        assert!((distance::inner_product(&a, &b) - 23.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec_from(&[1.0, 2.0, 3.0]);
        assert!((distance::cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec_from(&[1.0, 0.0]);
        let b = vec_from(&[0.0, 1.0]);
        assert!(distance::cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec_from(&[1.0, 0.0]);
        let b = vec_from(&[-1.0, 0.0]);
        assert!((distance::cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec_from(&[0.0, 0.0]);
        let b = vec_from(&[1.0, 2.0]);
        assert_eq!(distance::cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_scaled_vectors() {
        // Cosine similarity is magnitude-independent
        let a = vec_from(&[1.0, 2.0, 3.0]);
        let b = vec_from(&[2.0, 4.0, 6.0]); // 2× a
        assert!((distance::cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_distance_range() {
        let a = vec_from(&[1.0, 0.0]);
        let b = vec_from(&[0.0, 1.0]);
        let d = distance::cosine_distance(&a, &b);
        // Orthogonal → distance should be 1.0
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_distance_identical() {
        let v = vec_from(&[1.0, 2.0, 3.0]);
        assert!(distance::cosine_distance(&v, &v).abs() < 1e-10);
    }

    // ═════════════════════════════════════════════════════════════════
    //  Cross-format distance: JSON vs BLOB should produce same results
    // ═════════════════════════════════════════════════════════════════

    #[test]
    fn distance_across_formats() {
        let a_json = Vector::from_json_str("[1.0, 2.0, 3.0]").unwrap();
        let b_json = Vector::from_json_str("[4.0, 5.0, 6.0]").unwrap();

        // Round-trip through f32 blob
        let a_f32 = Vector::from_f32_blob(&a_json.to_f32_blob().unwrap()).unwrap();
        let b_f32 = Vector::from_f32_blob(&b_json.to_f32_blob().unwrap()).unwrap();

        let json_dist = distance::l2(&a_json, &b_json);
        let f32_dist = distance::l2(&a_f32, &b_f32);
        assert!(
            (json_dist - f32_dist).abs() < 1e-5,
            "L2 mismatch across formats: {} vs {}",
            json_dist,
            f32_dist
        );

        // Round-trip through f16 blob (wider tolerance)
        let a_f16 = Vector::from_f16_blob(&a_json.to_f16_blob().unwrap()).unwrap();
        let b_f16 = Vector::from_f16_blob(&b_json.to_f16_blob().unwrap()).unwrap();

        let f16_dist = distance::l2(&a_f16, &b_f16);
        assert!(
            (json_dist - f16_dist).abs() < 0.1,
            "L2 mismatch JSON vs f16: {} vs {}",
            json_dist,
            f16_dist
        );
    }
}