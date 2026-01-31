use sqlite_loadable::prelude::*;
use sqlite_loadable::{api, define_scalar_function, FunctionFlags, Result};
use std::f64;

// Calculate cosine similarity between two vectors
pub fn vec_cosine(context: *mut sqlite3_context, values: &[*mut sqlite3_value]) -> Result<()> {
    // 1. Parse the first argument (vector A)
    let v1_val = values.get(0).ok_or_else(|| sqlite_loadable::Error::new_message("Missing 1st argument"))?;
    let v1_str = api::value_text(v1_val)?;
    let v1: Vec<f64> = serde_json::from_str(v1_str)
        .map_err(|e| sqlite_loadable::Error::new_message(format!("Invalid JSON for vector A: {}", e)))?;

    // 2. Parse the second argument (vector B)
    let v2_val = values.get(1).ok_or_else(|| sqlite_loadable::Error::new_message("Missing 2nd argument"))?;
    let v2_str = api::value_text(v2_val)?;
    let v2: Vec<f64> = serde_json::from_str(v2_str)
        .map_err(|e| sqlite_loadable::Error::new_message(format!("Invalid JSON for vector B: {}", e)))?;

    // 3. Validate dimensions
    if v1.len() != v2.len() {
        return Err(sqlite_loadable::Error::new_message(format!(
            "Vector dimension mismatch: {} vs {}",
            v1.len(),
            v2.len()
        )));
    }

    // 4. Calculate Cosine Similarity: (A . B) / (||A|| * ||B||)
    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        api::result_double(context, 0.0);
        return Ok(());
    }

    let similarity = dot_product / (norm_a * norm_b);
    api::result_double(context, similarity);

    Ok(())
}

// The entry point that SQLite calls when loading the extension
#[sqlite_entrypoint]
pub fn sqlite3_extension_init(db: *mut sqlite3) -> Result<()> {
    // Register the function as 'vec_cosine'
    // deterministic = true allows SQLite to cache results and use it in indexes
    define_scalar_function(db, "vec_cosine", 2, vec_cosine, FunctionFlags::UTF8 | FunctionFlags::DETERMINISTIC)?;
    Ok(())
}