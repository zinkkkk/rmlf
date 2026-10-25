//! Sweep over the **ℚ (exact rationals)** domain — exact reconstruction of
//! integer/rational polynomials and rational functions via the user front door
//! ([`rmlf::exact::rational::fit_poly`] / `fit_rational`), with **byte-level**
//! compression (the metric the earlier abstract "nonzero count" hid).
//!
//! For each catalog case: fit it, verify off-grid recovery, and report the dense
//! vs packed coefficient *bytes* (the packed form pools repeated BigRationals)
//! plus the canonical-form (joint primitive-integer rescale) bit saving.
//!
//! Run: `cargo run --release --example exact_rational > docs/exact_rational_results.txt`.

use rmlf::exact::rational::{self, catalog as qcat};

fn trunc(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n - 1])
    }
}

fn main() {
    println!("=== Exact ℚ sweep: fit_poly / fit_rational, byte-level compression ===\n");
    println!("Front door: rational::fit_poly(f, &degrees) / fit_rational(f, &num, &den).");
    println!("Compression is in BYTES — the packed form pools repeated BigRationals and");
    println!("the canonical rescale clears shared denominators (eval-invariant).\n");

    // Per-lever breakdown: dense → sparse (drop exact zeros) → pooled (intern
    // repeated values), then canonical (clear shared denominators) — all in bytes.
    println!(
        "{:>3}  {:<40} {:>8} {:>8} {:>8} {:>7}  {:<7}",
        "id", "name", "denseB", "sparseB", "pooledB", "canon%", "recover"
    );
    for c in qcat::all() {
        let Ok(model) = c.fit() else {
            println!("{:>3}  {:<40} fit failed", c.id, trunc(c.name, 40));
            continue;
        };
        // Verify off-grid recovery.
        let pt: Vec<i64> = (0..c.ndim).map(|l| 131 + 2 * l as i64).collect();
        let qpt: Vec<_> = pt.iter().map(|&v| rational::Rational::integer(v)).collect();
        let recovered = model.eval_q(&pt) == Some((c.f)(&qpt));
        // Packed (lossless) + verify roundtrip.
        let packed = model.packed();
        assert_eq!(
            packed.unpack().c_num,
            model.inner().c_num,
            "{} roundtrip",
            c.name
        );
        let b = packed.coeff_breakdown();
        let bits_raw = model.bit_size();
        let canon_pct =
            100.0 * (1.0 - model.canonical().bit_size() as f64 / bits_raw.max(1) as f64);
        println!(
            "{:>3}  {:<40} {:>8} {:>8} {:>8} {:>6.1}%  {}",
            c.id,
            trunc(c.name, 40),
            b.dense,
            b.sparse,
            b.pooled,
            canon_pct,
            if c.exact && recovered {
                "EXACT"
            } else if recovered {
                "exact*"
            } else {
                "limit"
            }
        );
    }
    println!("\n  denseB  = dense coefficient bytes (2·∏k entries as full BigRationals)");
    println!(
        "  sparseB = after dropping exact zeros (the SPARSE footprint: nonzero values + index)"
    );
    println!("  pooledB = after also interning repeated values (distinct values once + u32 codes)");
    println!("  canon%  = bit saving from the joint primitive-integer rescale (eval-invariant)");
    println!("\n  The two byte levers are sparsity (denseB→sparseB) and value pooling");
    println!("  (sparseB→pooledB); the canonical rescale then shrinks each value's bits.");
    println!("  Cases 0–9 recover EXACTLY; 10 is a deliberate under-resolved limit;");
    println!("  11/12 are pole / non-coprime edges that still recover off the singular set.");
}
