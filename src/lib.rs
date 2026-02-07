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
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(3, 3);
    }
}