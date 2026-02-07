//! vecgen – Generate random vector datasets for experimentation and benchmarking.
//!
//! # Usage
//!
//!   # CSV output (default): 1000 vectors of 128 dimensions
//!   vecgen --count 1000 --dim 128 --format csv -o vectors.csv
//!
//!   # Binary f32 output (one BLOB per line, hex-encoded for easy SQLite import)
//!   vecgen --count 1000 --dim 128 --format f32 -o vectors.bin
//!
//!   # Binary f16 output
//!   vecgen --count 1000 --dim 128 --format f16 -o vectors_f16.bin
//!
//!   # JSON lines output (one JSON array per line, ready for INSERT)
//!   vecgen --count 1000 --dim 128 --format jsonl -o vectors.jsonl
//!
//!   # SQL output (INSERT statements ready to run)
//!   vecgen --count 100 --dim 3 --format sql --table items --column embedding -o seed.sql
//!
//!   # Pipe straight into sqlite3
//!   vecgen --count 100 --dim 3 --format sql --table items --column embedding | sqlite3 test.db

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::process;

// ── Argument parsing (no external crate needed) ─────────────────────

struct Args {
    count: usize,
    dim: usize,
    format: OutputFormat,
    output: Option<String>,
    table: String,
    column: String,
    seed: u64,
    normalize: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum OutputFormat {
    Csv,
    Jsonl,
    F32,
    F16,
    Sql,
}

fn print_help() {
    eprintln!(
        r#"vecgen - Generate random vector datasets

USAGE:
    vecgen [OPTIONS]

OPTIONS:
    -n, --count <N>        Number of vectors to generate [default: 100]
    -d, --dim <D>          Dimension of each vector [default: 3]
    -f, --format <FMT>     Output format: csv, jsonl, f32, f16, sql [default: csv]
    -o, --output <FILE>    Output file (stdout if omitted)
    -t, --table <NAME>     Table name for SQL output [default: items]
    -c, --column <NAME>    Column name for SQL output [default: embedding]
    -s, --seed <SEED>      Random seed [default: 42]
    --normalize            L2-normalize each vector
    -h, --help             Print this help message
"#
    );
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut count = 100usize;
    let mut dim = 3usize;
    let mut format = OutputFormat::Csv;
    let mut output = None;
    let mut table = "items".to_string();
    let mut column = "embedding".to_string();
    let mut seed = 42u64;
    let mut normalize = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            }
            "-n" | "--count" => {
                i += 1;
                count = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid count");
                    process::exit(1);
                });
            }
            "-d" | "--dim" => {
                i += 1;
                dim = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid dimension");
                    process::exit(1);
                });
            }
            "-f" | "--format" => {
                i += 1;
                format = match args[i].as_str() {
                    "csv" => OutputFormat::Csv,
                    "jsonl" => OutputFormat::Jsonl,
                    "f32" => OutputFormat::F32,
                    "f16" => OutputFormat::F16,
                    "sql" => OutputFormat::Sql,
                    other => {
                        eprintln!("Unknown format: {}", other);
                        process::exit(1);
                    }
                };
            }
            "-o" | "--output" => {
                i += 1;
                output = Some(args[i].clone());
            }
            "-t" | "--table" => {
                i += 1;
                table = args[i].clone();
            }
            "-c" | "--column" => {
                i += 1;
                column = args[i].clone();
            }
            "-s" | "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid seed");
                    process::exit(1);
                });
            }
            "--normalize" => {
                normalize = true;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_help();
                process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        count,
        dim,
        format,
        output,
        table,
        column,
        seed,
        normalize,
    }
}

// ── Simple xorshift64 PRNG (no external crate) ─────────────────────

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Returns a float in [-1.0, 1.0)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as i64 as f64 / i64::MAX as f64) as f32
    }
}

fn generate_vector(rng: &mut Rng, dim: usize, normalize: bool) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
    if normalize {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
    }
    v
}

// ── f16 conversion (mirrors the extension's vector.rs) ──────────────

const BLOB_MAGIC: u8 = 0xBE;
const FORMAT_F32: u8 = 0x01;
const FORMAT_F16: u8 = 0x02;

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 0 {
        sign << 15
    } else if exponent == 0xFF {
        let f16_mantissa = (mantissa >> 13) as u16;
        (sign << 15) | (0x1F << 10) | f16_mantissa
    } else {
        let new_exp = exponent - 127 + 15;
        if new_exp >= 31 {
            (sign << 15) | (0x1F << 10)
        } else if new_exp <= 0 {
            sign << 15
        } else {
            let f16_mantissa = (mantissa >> 13) as u16;
            (sign << 15) | ((new_exp as u16) << 10) | f16_mantissa
        }
    }
}

fn to_f32_blob(v: &[f32]) -> Vec<u8> {
    let dim = v.len();
    let mut buf = Vec::with_capacity(4 + dim * 4);
    buf.push(BLOB_MAGIC);
    buf.push(FORMAT_F32);
    buf.extend_from_slice(&(dim as u16).to_le_bytes());
    for &val in v {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    buf
}

fn to_f16_blob(v: &[f32]) -> Vec<u8> {
    let dim = v.len();
    let mut buf = Vec::with_capacity(4 + dim * 2);
    buf.push(BLOB_MAGIC);
    buf.push(FORMAT_F16);
    buf.extend_from_slice(&(dim as u16).to_le_bytes());
    for &val in v {
        buf.extend_from_slice(&f32_to_f16(val).to_le_bytes());
    }
    buf
}

fn vec_to_json(v: &[f32]) -> String {
    let inner: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
    format!("[{}]", inner.join(","))
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    let writer: BufWriter<Box<dyn Write>> = match &args.output {
        Some(path) => {
            let f = File::create(path).unwrap_or_else(|e| {
                eprintln!("Cannot create {}: {}", path, e);
                process::exit(1);
            });
            BufWriter::new(Box::new(f))
        }
        None => BufWriter::new(Box::new(io::stdout().lock())),
    };

    let mut w = writer;
    let mut rng = Rng::new(args.seed);

    // CSV header
    if args.format == OutputFormat::Csv {
        let mut header = String::from("id");
        for d in 0..args.dim {
            header.push_str(&format!(",v{}", d));
        }
        writeln!(w, "{}", header).unwrap();
    }

    // SQL preamble
    if args.format == OutputFormat::Sql {
        writeln!(
            w,
            "CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY, {} TEXT);",
            args.table, args.column
        )
        .unwrap();
    }

    for i in 0..args.count {
        let v = generate_vector(&mut rng, args.dim, args.normalize);

        match args.format {
            OutputFormat::Csv => {
                let vals: Vec<String> = v.iter().map(|x| format!("{:.6}", x)).collect();
                writeln!(w, "{},{}", i + 1, vals.join(",")).unwrap();
            }
            OutputFormat::Jsonl => {
                writeln!(w, "{}", vec_to_json(&v)).unwrap();
            }
            OutputFormat::F32 => {
                let blob = to_f32_blob(&v);
                // Write as raw binary
                w.write_all(&blob).unwrap();
            }
            OutputFormat::F16 => {
                let blob = to_f16_blob(&v);
                w.write_all(&blob).unwrap();
            }
            OutputFormat::Sql => {
                writeln!(
                    w,
                    "INSERT INTO {} ({}) VALUES ('{}');",
                    args.table,
                    args.column,
                    vec_to_json(&v)
                )
                .unwrap();
            }
        }
    }

    w.flush().unwrap();

    if let Some(ref path) = args.output {
        eprintln!(
            "Generated {} vectors (dim={}) → {}",
            args.count, args.dim, path
        );
    }
}
