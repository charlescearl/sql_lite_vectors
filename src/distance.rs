//! # Distance metrics for vector comparison
//!
//! This module implements the four standard distance / similarity
//! functions used in vector search.  Each function takes two
//! [`Vector`](crate::vector::Vector) references and returns an `f64`.
//!
//! | Function              | Range       | 0 means…         | Use with `ORDER BY` |
//! |-----------------------|-------------|------------------|---------------------|
//! | `l2`                  | [0, ∞)      | identical        | ASC                 |
//! | `l1`                  | [0, ∞)      | identical        | ASC                 |
//! | `cosine_distance`     | [0, 2]      | identical        | ASC                 |
//! | `cosine_similarity`   | [-1, 1]     | orthogonal       | DESC                |
//! | `inner_product`       | (-∞, ∞)     | orthogonal       | DESC (or negate)    |
//!
//! ### Rust note — references (`&`)
//!
//! All functions take `a: &Vector` — an *immutable reference* (borrow).
//! This means the caller keeps ownership; we only read the data.  In
//! Rust, `&` references are guaranteed to be valid (no null pointers,
//! no dangling pointers), which is enforced at compile time by the
//! borrow checker.

use crate::vector::Vector;

/// **L2 (Euclidean) distance** — the straight-line distance between
/// two points in n-dimensional space.
///
/// $$d = \sqrt{\sum_{i=0}^{n-1} (a_i - b_i)^2}$$
///
/// Equivalent to pgvector's `<->` operator.
///
/// ### Rust note — iterator chains
///
/// `.iter()` creates an iterator over `&f64` references.
/// `.zip(b.data.iter())` pairs elements from both vectors.
/// `.map(|(x, y)| ...)` transforms each pair.
/// `.sum::<f64>()` adds them up — the `::<f64>` is a "turbofish" that
/// tells `.sum()` what type to accumulate into.
/// `.sqrt()` is a method on `f64`.
pub fn l2(a: &Vector, b: &Vector) -> f64 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// **L1 (Manhattan / taxicab) distance** — the sum of absolute
/// differences along each axis.
///
/// $$d = \sum_{i=0}^{n-1} |a_i - b_i|$$
///
/// Equivalent to pgvector's `<+>` operator.
pub fn l1(a: &Vector, b: &Vector) -> f64 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// **Inner (dot) product** — a measure of how much two vectors point
/// in the same direction, scaled by their magnitudes.
///
/// $$\text{ip} = \sum_{i=0}^{n-1} a_i \cdot b_i$$
///
/// This is the raw dot product.  pgvector's `<#>` returns the
/// *negative* inner product so that `ORDER BY … ASC` gives the most
/// similar results; our `vec_ip_neg` function mirrors that behaviour.
pub fn inner_product(a: &Vector, b: &Vector) -> f64 {
    a.data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// **Cosine similarity** — the cosine of the angle between two
/// vectors, independent of their magnitudes.
///
/// $$\text{sim} = \frac{A \cdot B}{\|A\| \, \|B\|}$$
///
/// Returns a value in [-1, 1]:
/// - **1.0** = identical direction
/// - **0.0** = orthogonal (perpendicular)
/// - **-1.0** = opposite direction
///
/// If either vector has zero magnitude we return 0.0 (undefined
/// direction, so no meaningful similarity).
pub fn cosine_similarity(a: &Vector, b: &Vector) -> f64 {
    let dot = inner_product(a, b);
    let norm_a = a.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.data.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// **Cosine distance** — simply `1 - cosine_similarity`.
///
/// Returns a value in [0, 2]:
/// - **0.0** = identical direction
/// - **1.0** = orthogonal
/// - **2.0** = opposite direction
///
/// Equivalent to pgvector's `<=>` operator.
/// Use with `ORDER BY vec_cosine_distance(...) ASC`.
pub fn cosine_distance(a: &Vector, b: &Vector) -> f64 {
    1.0 - cosine_similarity(a, b)
}
