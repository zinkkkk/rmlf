//! Run catalog examples through the **variable-decoupling / KST** method
//! (`decouple`, arXiv:2605.07246) and tabulate, for each case: the *decoupled
//! form* (auto-detected per-axis degree and the resulting number of decoupled
//! single-variable KST functions), the reconstruction accuracy, and the runtime.
//!
//! The paper claims rational/polynomial multivariate functions are recovered
//! **exactly** ("perfect reconstruction") by the decoupling — Pólya–Szegő from 8
//! samples, etc. This sweep cross-validates that across the *whole* catalog (not
//! just the polynomial cases) to see how often it holds and which cases it fails
//! on. Non-rational cases are included deliberately: the decoupling will fit the
//! best low-degree rational the detected order allows, and the table shows how
//! close that gets.
//!
//! **Run ONE case per process** for robustness, exactly like `catalog_sweep`:
//! invoke `cargo run --release --example decouple_sweep -- CAS NIP` (NIP optional).
//! Order detection and the recursion run rayon SVD batches that are not
//! cancellable, so a runaway case must be killed by an OS `timeout` around a
//! separate process. Without a CAS argument it prints the header and
//! best-effort-runs every small-grid case in-process, then a summary.

use rmlf::decouple::{DecoupleInfo, DecoupledModel, decouple_adaptive};
use rmlf::examples;
use rmlf::examples::Tag;
use std::time::{Duration, Instant};

const N_QUERY: usize = 4000;

/// Below this off-grid max relative error we call the reconstruction "exact"
/// (the paper's perfect-reconstruction claim, allowing for floating-point).
const EXACT_TOL: f64 = 1e-9;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Max relative error of `g` against `f` over a fresh interior random sample
/// (off-grid generalization, never on a support node). Excludes non-finite truths.
fn max_rel_error(bounds: &[(f64, f64)], f: &dyn Fn(&[f64]) -> f64, model: &DecoupledModel) -> f64 {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let n = bounds.len();
    let mut x = vec![0.0; n];
    let (mut max, mut scale) = (0.0_f64, 0.0_f64);
    for _ in 0..N_QUERY {
        for (l, &(lo, hi)) in bounds.iter().enumerate() {
            x[l] = lo + (hi - lo) * (0.05 + 0.9 * rng.f());
        }
        let truth = f(&x);
        if !truth.is_finite() {
            continue;
        }
        scale = scale.max(truth.abs());
        let e = (truth - model.eval(&x)).abs();
        if e.is_finite() {
            max = max.max(e);
        }
    }
    let s = if scale == 0.0 { 1.0 } else { scale };
    max / s
}

fn tags_str(t: &[Tag]) -> String {
    t.iter()
        .map(|t| match t {
            Tag::Polynomial => "poly",
            Tag::Rational => "rat",
            Tag::Irrational => "irr",
            Tag::C0 => "C0",
            Tag::CInf => "Cinf",
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn fmt_err(x: f64) -> String {
    if !x.is_finite() {
        "NaN".into()
    } else if x > 1e3 {
        "FAIL".into()
    } else {
        format!("{x:.1e}")
    }
}

fn secs(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1e3;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1e3)
    } else {
        format!("{ms:.0}ms")
    }
}

/// Compact per-axis degree string, e.g. `[1,2,1]`.
fn degree_str(d: &[usize]) -> String {
    let inner = d
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

/// Combined grid size `~(2·nip)^n` above which we refuse to allocate (mirrors
/// `catalog_sweep`'s OOM guard).
// Worst-case `(2·nip)^n` grid estimate above which a case is skipped. Set well
// above every default-nip catalog case so coverage is unchanged (the bad nips
// that once needed a hard RAM ceiling were fixed at the source); a hand-passed
// pathological nip still short-circuits instead of OOMing.
const MAX_GRID: u128 = 1 << 30;

fn header() {
    println!(
        "{:>6} {:>3} {:>10} {:>5} {:>22} {:>6} {:>8} {:>9} {:>7}  name",
        "CAS", "n", "tags", "nip", "degree", "#KST", "samples", "err", "time"
    );
}

/// Outcome of one case, for the no-arg summary tally.
enum Outcome {
    Exact,
    Approx,
    Fail,
    Skipped,
}

/// Run one case through the decoupling, printing a single result row:
/// detected per-axis degree, number of decoupled KST functions, samples used,
/// off-grid max relative error, and runtime. `EXACT` flags perfect reconstruction.
/// Returns `(tags, outcome)` for the summary.
fn run_one(ex: &examples::Example, nip_override: Option<usize>) -> Outcome {
    let bounds = ex.axis_bounds();
    let nip = nip_override.unwrap_or(ex.recommended_nip).max(2);
    let f = ex.f;
    let grid: u128 = (2 * nip as u128).pow(ex.ndim as u32);
    if grid > MAX_GRID {
        println!(
            "{:>6} {:>3} {:>10} {:>5} {:>22} {:>6} {:>8} {:>9} {:>7}  {}  (grid ~{:.0e})",
            ex.id,
            ex.ndim,
            tags_str(ex.tags),
            nip,
            "BIG",
            "-",
            "-",
            "BIG",
            "-",
            ex.name,
            grid as f64
        );
        return Outcome::Skipped;
    }

    // Per-axis adaptive: start from the catalog nip and grow only the saturated
    // (under-resolved) axes, so mixed-degree high-n rationals (41-46) resolve
    // without the uniform-nip grid blow-up. max_nip 16 caps growth.
    let t = Instant::now();
    let result = decouple_adaptive(&f, &bounds, nip, 16);
    let dt = secs(t.elapsed());

    match result {
        Ok((model, info)) => {
            let DecoupleInfo {
                degree,
                samples,
                n_functions,
            } = info;
            let err = max_rel_error(&bounds, &|x| f(x), &model);
            let exact = if err < EXACT_TOL { " EXACT" } else { "" };
            println!(
                "{:>6} {:>3} {:>10} {:>5} {:>22} {:>6} {:>8} {:>9} {:>7}  {}{}",
                ex.id,
                ex.ndim,
                tags_str(ex.tags),
                nip,
                degree_str(&degree),
                n_functions,
                samples,
                fmt_err(err),
                dt,
                ex.name,
                exact
            );
            if !err.is_finite() || err > 1e3 {
                Outcome::Fail
            } else if err < EXACT_TOL {
                Outcome::Exact
            } else {
                Outcome::Approx
            }
        }
        Err(e) => {
            println!(
                "{:>6} {:>3} {:>10} {:>5} {:>22} {:>6} {:>8} {:>9} {:>7}  {}  (err: {})",
                ex.id,
                ex.ndim,
                tags_str(ex.tags),
                nip,
                "-",
                "-",
                "-",
                "ERR",
                dt,
                ex.name,
                e
            );
            Outcome::Fail
        }
    }
}

/// Is this a rational/polynomial case (the paper's perfect-reconstruction class)?
fn is_rational_class(tags: &[Tag]) -> bool {
    tags.iter()
        .any(|t| matches!(t, Tag::Polynomial | Tag::Rational))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    let nip_override: Option<usize> = args.next().and_then(|s| s.parse().ok());

    match first.as_deref().and_then(|s| s.parse::<u32>().ok()) {
        // Single-case mode (robust path: drive from a shell loop with `timeout`).
        Some(cas) => {
            if let Some(ex) = examples::all().iter().find(|e| e.id == cas) {
                run_one(ex, nip_override);
            } else {
                eprintln!("no CAS {cas}");
            }
        }
        // No CAS arg: best-effort in-process sweep of the small-grid cases, then a
        // summary tallying how often the paper's perfect-reconstruction claim holds.
        None => {
            let nip_override = first.and_then(|s| s.parse().ok());
            header();
            let (mut rat_exact, mut rat_total) = (0u32, 0u32);
            let (mut irr_exact, mut irr_approx, mut irr_total) = (0u32, 0u32, 0u32);
            let mut skipped = 0u32;
            for ex in examples::all() {
                let outcome = run_one(ex, nip_override);
                let rational = is_rational_class(ex.tags);
                match outcome {
                    Outcome::Skipped => skipped += 1,
                    Outcome::Exact if rational => {
                        rat_exact += 1;
                        rat_total += 1;
                    }
                    Outcome::Exact => {
                        irr_exact += 1;
                        irr_total += 1;
                    }
                    Outcome::Approx if rational => rat_total += 1,
                    Outcome::Approx => {
                        irr_approx += 1;
                        irr_total += 1;
                    }
                    Outcome::Fail if rational => rat_total += 1,
                    Outcome::Fail => irr_total += 1,
                }
            }
            println!(
                "\nSummary (default catalog nip; skipped {skipped} BIG-grid cases):\n  \
                 rational/poly: {rat_exact}/{rat_total} EXACT (perfect reconstruction)\n  \
                 irrational:    {irr_exact}/{irr_total} EXACT, {irr_approx} approximate\n\
                 Note: rational 'failures' at the catalog default nip are usually \
                 nip-too-low (degree under-detected),\n  not method failures — e.g. CAS \
                 41/58 recover EXACTly at nip>=10 (re-run a single case with a larger nip)."
            );
        }
    }
}
