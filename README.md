# SQLite Vector Search Starter (Rust)

A starter project for building a SQLite extension in Rust. This extension implements a `vec_cosine` function to calculate the similarity between two vectors stored as JSON strings.

## Prerequisites

- Rust (cargo)
- SQLite3 CLI

## Build

Build the project in release mode to generate the shared library:

```bash
cargo build --release
```

The compiled extension will be located at:
- **Linux:** `target/release/libsqlite_vec_starter.so`
- **macOS:** `target/release/libsqlite_vec_starter.dylib`
- **Windows:** `target/release/sqlite_vec_starter.dll`

## Usage

Start the SQLite CLI and load the extension.

### 1. Load the extension

```sql
.load target/release/libsqlite_vec_starter.dylib
-- Note: On macOS, you might need to omit the 'lib' prefix or extension depending on your sqlite version, 
-- but usually the full path works.
```

### 2. Test the function

```sql
-- Simple test
SELECT vec_cosine('[1.0, 0.0]', '[1.0, 0.0]') as exact_match; -- Should be 1.0
SELECT vec_cosine('[1.0, 0.0]', '[0.0, 1.0]') as orthogonal;  -- Should be 0.0
SELECT vec_cosine('[1.0, 1.0]', '[0.5, 0.5]') as similar;     -- Should be 1.0 (normalized)
```

### 3. "Vector Search" Example

```sql
CREATE TABLE items (id INTEGER PRIMARY KEY, embedding TEXT);
INSERT INTO items (embedding) VALUES ('[0.1, 0.8, 0.1]'), ('[0.9, 0.1, 0.0]'), ('[0.1, 0.1, 0.9]');

-- Find top 1 item most similar to a query vector
SELECT id, embedding, vec_cosine(embedding, '[0.1, 0.9, 0.1]') as score
FROM items
ORDER BY score DESC
LIMIT 1;
```