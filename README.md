# SQLite Vector Search (Rust)

A SQLite extension written in Rust that brings **pgvector-style** vector operations to SQLite.  Store embeddings as JSON strings or compact binary BLOBs (f32 / f16), compute distances with familiar functions, and enforce dimensionality with CHECK constraints — all without leaving SQLite.

## Prerequisites

- Rust (cargo)
- SQLite3 CLI

## Build

```bash
cargo build --release
```

The compiled extension will be located at:
- **Linux:** `target/release/libsqlite_vec_starter.so`
- **macOS:** `target/release/libsqlite_vec_starter.dylib`
- **Windows:** `target/release/sqlite_vec_starter.dll`

The `vecgen` CLI tool is built alongside the extension:
```bash
# It's already compiled by the command above; the binary is at:
target/release/vecgen
```

---

## Quick Start

```sql
.load target/release/libsqlite_vec_starter

-- Create a table with a dimension constraint (pgvector style)
CREATE TABLE items (
  id INTEGER PRIMARY KEY,
  embedding vector(3) CHECK (vec_dim(embedding) = 3)
);

-- Insert vectors as JSON strings
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]');

-- Find nearest neighbours by L2 distance
SELECT * FROM items ORDER BY vec_l2(embedding, '[3,1,2]') LIMIT 5;
```

---

## Function Reference

### Distance functions

All distance functions accept two vectors (JSON text or BLOB) and return a REAL.

| Function | Description | pgvector equiv | ORDER BY |
|---|---|---|---|
| `vec_l2(a, b)` | Euclidean (L2) distance | `<->` | ASC |
| `vec_l1(a, b)` | Manhattan (L1) distance | `<+>` | ASC |
| `vec_cosine_distance(a, b)` | Cosine distance (1 − similarity) | `<=>` | ASC |
| `vec_cosine(a, b)` | Cosine similarity | — | DESC |
| `vec_ip(a, b)` | Inner (dot) product | — | DESC |
| `vec_ip_neg(a, b)` | Negative inner product | `<#>` | ASC |

### Inspection functions

| Function | Args | Returns |
|---|---|---|
| `vec_dim(v)` | 1 | integer dimension of the vector |
| `vec_assert_dim(v, n)` | 2 | 1 if dim = n, error otherwise |
| `vec_format(v)` | 1 | `'json'`, `'f32'`, or `'f16'` |

### Parse / normalize

| Function | Args | Returns |
|---|---|---|
| `vec_parse(v)` | 1 | canonical JSON string |
| `vec_parse(v, dim)` | 2 | canonical JSON string; error if dim ≠ expected |

### Format conversion

| Function | Args | Returns |
|---|---|---|
| `vec_to_f32(v)` | 1 | BLOB in f32 binary format |
| `vec_to_f16(v)` | 1 | BLOB in f16 binary format (half storage) |
| `vec_to_json(v)` | 1 | JSON text string |

---

## Usage Examples

### 1. Load the extension

```sql
.load target/release/libsqlite_vec_starter
```

### 2. Create a table with dimension enforcement

```sql
CREATE TABLE items (
  id INTEGER PRIMARY KEY,
  embedding vector(3) CHECK (vec_dim(embedding) = 3)
);
```

SQLite ignores unknown type names, so `vector(3)` is cosmetic — the real enforcement comes from the CHECK constraint calling `vec_dim()`.

### 3. Insert vectors

```sql
INSERT INTO items (embedding) VALUES
  ('[0.1, 0.8, 0.1]'),
  ('[0.9, 0.1, 0.0]'),
  ('[0.1, 0.1, 0.9]');
```

### 4. Query by distance

```sql
-- L2 distance (Euclidean) — closest vectors first
SELECT id, embedding, vec_l2(embedding, '[0.1, 0.9, 0.1]') AS dist
FROM items
ORDER BY dist ASC
LIMIT 5;

-- Cosine distance — most similar first
SELECT id, embedding, vec_cosine_distance(embedding, '[0.1, 0.9, 0.1]') AS dist
FROM items
ORDER BY dist ASC
LIMIT 5;

-- Inner product (negative, like pgvector <#>)
SELECT id, embedding, vec_ip_neg(embedding, '[0.1, 0.9, 0.1]') AS score
FROM items
ORDER BY score ASC
LIMIT 5;

-- Manhattan (L1) distance
SELECT id, embedding, vec_l1(embedding, '[0.1, 0.9, 0.1]') AS dist
FROM items
ORDER BY dist ASC
LIMIT 5;
```

### 5. Use binary BLOB storage for performance

```sql
-- Store as compact f32 BLOB (4 bytes per element instead of ~8+ for JSON text)
UPDATE items SET embedding = vec_to_f32(embedding);

-- Queries work identically — format is auto-detected
SELECT id, vec_l2(embedding, vec_to_f32('[0.1, 0.9, 0.1]')) AS dist
FROM items
ORDER BY dist ASC
LIMIT 5;

-- Check the format
SELECT id, vec_format(embedding) FROM items;

-- Store as f16 for half the storage (2 bytes per element)
UPDATE items SET embedding = vec_to_f16(embedding);

-- Convert back to readable JSON
SELECT id, vec_to_json(embedding) FROM items;
```

### 6. Validate on insert with `vec_parse`

```sql
-- Use vec_parse to normalise and validate in one step
INSERT INTO items (embedding)
  VALUES (vec_parse('[1, 2, 3]', 3));

-- This will error: wrong dimension
INSERT INTO items (embedding)
  VALUES (vec_parse('[1, 2]', 3));
-- → Error: vec_parse: expected 3 dimensions, got 2
```

### 7. Update and delete

```sql
-- Update a vector
UPDATE items SET embedding = '[1.0, 2.0, 3.0]' WHERE id = 1;

-- Delete a row
DELETE FROM items WHERE id = 1;

-- Upsert
INSERT INTO items (id, embedding) VALUES (1, '[1,2,3]')
  ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
```

---

## `vecgen` CLI — Generate Test Datasets

The `vecgen` binary (located in `target/release/`) generates random vector datasets for experimentation and benchmarking.

### Usage

```bash
vecgen [OPTIONS]
```

### Options

| Flag | Description | Default |
|---|---|---|
| `-n, --count <N>` | Number of vectors | 100 |
| `-d, --dim <D>` | Dimensions per vector | 3 |
| `-f, --format <FMT>` | Output format: `csv`, `jsonl`, `f32`, `f16`, `sql` | csv |
| `-o, --output <FILE>` | Output file (stdout if omitted) | — |
| `-t, --table <NAME>` | Table name (SQL format) | items |
| `-c, --column <NAME>` | Column name (SQL format) | embedding |
| `-s, --seed <SEED>` | Random seed | 42 |
| `--normalize` | L2-normalize each vector | false |

### Examples

```bash
# Generate 1000 vectors of 128 dimensions as CSV
vecgen -n 1000 -d 128 -f csv -o vectors.csv

# Generate JSON lines (one JSON array per line)
vecgen -n 1000 -d 128 -f jsonl -o vectors.jsonl

# Generate ready-to-run SQL INSERT statements
vecgen -n 100 -d 3 -f sql -o seed.sql

# Pipe SQL directly into SQLite
vecgen -n 100 -d 3 -f sql --table items --column embedding | sqlite3 test.db

# Generate normalised vectors (unit length)
vecgen -n 1000 -d 128 -f csv --normalize -o normalised.csv

# Generate raw f32 binary output
vecgen -n 1000 -d 128 -f f32 -o vectors.bin

# Generate raw f16 binary output (half the size)
vecgen -n 1000 -d 128 -f f16 -o vectors_f16.bin
```

### End-to-end benchmarking example

```bash
# 1. Generate 10k 128-dim vectors as SQL
vecgen -n 10000 -d 128 -f sql --table bench --column emb --normalize -o bench.sql

# 2. Create the DB and load the data
sqlite3 bench.db < bench.sql

# 3. Load the extension and run a query
sqlite3 bench.db <<'EOF'
.load target/release/libsqlite_vec_starter
.timer on
SELECT id, vec_l2(emb, '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,
  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]') AS dist
FROM bench
ORDER BY dist ASC
LIMIT 10;
EOF
```

---

## pgvector Syntax Mapping

Since SQLite doesn't support custom operators, here's how pgvector syntax maps to this extension:

| pgvector (PostgreSQL) | sqlite_vec_starter (SQLite) |
|---|---|
| `embedding <-> '[3,1,2]'` | `vec_l2(embedding, '[3,1,2]')` |
| `embedding <=> '[3,1,2]'` | `vec_cosine_distance(embedding, '[3,1,2]')` |
| `embedding <#> '[3,1,2]'` | `vec_ip_neg(embedding, '[3,1,2]')` |
| `embedding <+> '[3,1,2]'` | `vec_l1(embedding, '[3,1,2]')` |
| `vector(3)` type | `vector(3) CHECK (vec_dim(embedding) = 3)` |

## Binary BLOB Format

Vectors can be stored as compact binary BLOBs for better performance and smaller storage. The format uses a 4-byte header:

```
Byte 0:    0xBE (magic byte)
Byte 1:    0x01 (f32) or 0x02 (f16)
Bytes 2-3: dimension as little-endian u16
Bytes 4+:  element data (f32: 4 bytes each, f16: 2 bytes each)
```

| Format | Bytes per element | Storage for 128-dim | Precision |
|---|---|---|---|
| JSON text | ~8-10 | ~1.2 KB | full f64 |
| f32 BLOB | 4 | 516 B | ~7 decimal digits |
| f16 BLOB | 2 | 260 B | ~3 decimal digits |