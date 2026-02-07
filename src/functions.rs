use sqlite_loadable::prelude::*;
use sqlite_loadable::{api, Error, Result};

use crate::distance;
use crate::vector::{parse_vector_from_value, VectorFormat};

fn parse_two_vectors(
    values: &[*mut sqlite3_value],
) -> Result<(crate::vector::Vector, crate::vector::Vector)> {
    let v1_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing 1st argument"))?;
    let v2_val = values
        .get(1)
        .ok_or_else(|| Error::new_message("Missing 2nd argument"))?;

    let v1 = parse_vector_from_value(*v1_val)?;
    let v2 = parse_vector_from_value(*v2_val)?;

    if v1.dim != v2.dim {
        return Err(Error::new_message(&format!(
            "Vector dimension mismatch: {} vs {}",
            v1.dim, v2.dim
        )));
    }

    Ok((v1, v2))
}

// ── Distance functions ──────────────────────────────────────────────

pub fn vec_cosine(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let (v1, v2) = parse_two_vectors(values)?;
    let similarity = distance::cosine_similarity(&v1, &v2);
    api::result_double(context, similarity);
    Ok(())
}

pub fn vec_cosine_distance(
    context: *mut sqlite3_context,
    values: &[*mut sqlite3_value],
) -> Result<()> {
    let (v1, v2) = parse_two_vectors(values)?;
    let dist = distance::cosine_distance(&v1, &v2);
    api::result_double(context, dist);
    Ok(())
}

pub fn vec_l2(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let (v1, v2) = parse_two_vectors(values)?;
    let dist = distance::l2(&v1, &v2);
    api::result_double(context, dist);
    Ok(())
}

pub fn vec_l1(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let (v1, v2) = parse_two_vectors(values)?;
    let dist = distance::l1(&v1, &v2);
    api::result_double(context, dist);
    Ok(())
}

pub fn vec_ip(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let (v1, v2) = parse_two_vectors(values)?;
    let ip = distance::inner_product(&v1, &v2);
    api::result_double(context, ip);
    Ok(())
}

pub fn vec_ip_neg(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let (v1, v2) = parse_two_vectors(values)?;
    let ip = distance::inner_product(&v1, &v2);
    api::result_double(context, -ip);
    Ok(())
}

// ── Dimension helpers ───────────────────────────────────────────────

pub fn vec_dim(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let v1_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing 1st argument"))?;
    let v1 = parse_vector_from_value(*v1_val)?;
    api::result_int(context, v1.dim as i32);
    Ok(())
}

pub fn vec_assert_dim(
    context: *mut sqlite3_context,
    values: &[*mut sqlite3_value],
) -> Result<()> {
    let v1_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing 1st argument"))?;
    let dim_val = values
        .get(1)
        .ok_or_else(|| Error::new_message("Missing 2nd argument"))?;

    let v1 = parse_vector_from_value(*v1_val)?;
    let dim = api::value_int(dim_val) as usize;

    if v1.dim != dim {
        return Err(Error::new_message(&format!(
            "Vector dimension mismatch: {} vs {}",
            v1.dim, dim
        )));
    }

    api::result_int(context, 1);
    Ok(())
}

// ── Parse / normalize helper ────────────────────────────────────────

/// vec_parse(vector_text [, expected_dim])
///
/// Parses a JSON vector string (or BLOB), validates its dimension if
/// the second argument is provided, and returns the canonical JSON
/// representation.  Useful as a CHECK constraint or INSERT wrapper.
pub fn vec_parse(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let v_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing vector argument"))?;
    let v = parse_vector_from_value(*v_val)?;

    // Optional dimension check
    if let Some(dim_val) = values.get(1) {
        let expected = api::value_int(dim_val) as usize;
        if v.dim != expected {
            return Err(Error::new_message(&format!(
                "vec_parse: expected {} dimensions, got {}",
                expected, v.dim
            )));
        }
    }

    let json = v.to_json_string();
    api::result_text(context, json)?;
    Ok(())
}

// ── Format conversion helpers ───────────────────────────────────────

/// vec_to_f32(vector) → BLOB in f32 format with header
pub fn vec_to_f32(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let v_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing vector argument"))?;
    let v = parse_vector_from_value(*v_val)?;
    let blob = v.to_f32_blob();
    api::result_blob(context, &blob);
    Ok(())
}

/// vec_to_f16(vector) → BLOB in f16 format with header
pub fn vec_to_f16(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let v_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing vector argument"))?;
    let v = parse_vector_from_value(*v_val)?;
    let blob = v.to_f16_blob();
    api::result_blob(context, &blob);
    Ok(())
}

/// vec_to_json(vector) → JSON string representation
pub fn vec_to_json(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let v_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing vector argument"))?;
    let v = parse_vector_from_value(*v_val)?;
    let json = v.to_json_string();
    api::result_text(context, json)?;
    Ok(())
}

/// vec_format(vector) → TEXT describing the format ('json', 'f32', 'f16')
pub fn vec_format(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    let v_val = values
        .get(0)
        .ok_or_else(|| Error::new_message("Missing vector argument"))?;
    let v = parse_vector_from_value(*v_val)?;
    let name = match v.format {
        VectorFormat::Json => "json",
        VectorFormat::F32 => "f32",
        VectorFormat::F16 => "f16",
    };
    api::result_text(context, name)?;
    Ok(())
}
