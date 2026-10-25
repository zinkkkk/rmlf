//! Run catalog examples through alg1 / alg2 / alg3, collecting accuracy and
//! timings. Used to find where one method wins, where a method fails, and
//! whether results depend fragilely on `nip`.
//!
//! **Run ONE case per process** (`cargo run --release --example catalog_sweep --
//! <CAS> [nip]`). Driving the sweep from a shell loop with an OS-level `timeout`
//! per process is the only robust way to bound runtime: rayon's parallel SVD
//! batch is *not cancellable*, so an in-process thread budget leaks workers that
//! poison later cases (and a single huge `gather`/SVD can commit GBs of RAM before
//! any timer fires). A separate process is killed cleanly. Without a CAS arg it
//! prints the header and runs every *small-grid* case in-process (best-effort).

use rmlf::examples;
use rmlf::examples::Tag;
use rmlf::{Alg1Options, Alg2Options, Alg3Options, alg1, alg2, alg3};
use std::time::{Duration, Instant};

const N_QUERY: usize = 4000;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn eval_errors(
    bounds: &[(f64, f64)],
    f: &dyn Fn(&[f64]) -> f64,
    g: &dyn Fn(&[f64]) -> f64,
) -> (f64, Vec<Vec<f64>>) {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let n = bounds.len();
    let mut x = vec![0.0; n];
    let (mut max, mut scale) = (0.0_f64, 0.0_f64);
    let mut qs = Vec::with_capacity(N_QUERY);
    for _ in 0..N_QUERY {
        for (l, &(lo, hi)) in bounds.iter().enumerate() {
            x[l] = lo + (hi - lo) * (0.05 + 0.9 * rng.f());
        }
        let truth = f(&x);
        scale = scale.max(truth.abs());
        let e = (truth - g(&x)).abs();
        if e.is_finite() {
            max = max.max(e);
        }
        qs.push(x.clone());
    }
    let s = if scale == 0.0 { 1.0 } else { scale };
    (max / s, qs)
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

fn fmt(x: f64) -> String {
    if x > 1e3 {
        "FAIL".into()
    } else {
        format!("{x:.1e}")
    }
}

/// Worst-case `(2·nip)^n` grid estimate above which we skip **alg2 only**: its
/// greedy `Full` candidate scan visits every `prod(k_l+q_l)` grid point each
/// iteration (point count exponential in `n`), so a large grid makes alg2 hang
/// even when alg1/alg3 finish instantly (observed on CAS 27, n=5). alg1 and alg3
/// always run regardless. (The catalog nips are sane, so this rarely fires; the
/// previous `u128::MAX` OOM ceiling is gone — the bad nips that motivated it were
/// fixed at the source.)
const ALG2_GRID_CAP: u128 = 2 << 20; // ~2M grid points

fn header() {
    println!(
        "{:>6} {:>3} {:>10} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7}  name",
        "CAS", "n", "tags", "alg1", "t1", "alg2", "t2", "alg3", "t3"
    );
}

/// Run one case, printing a single result row. Designed to be invoked one-per-
/// process so a runaway (slow / huge-alloc) case is killed cleanly by an external
/// `timeout` rather than leaking rayon workers into later cases.
fn run_one(ex: &examples::Example, nip_override: Option<usize>) {
    let bounds = ex.axis_bounds();
    let nip = nip_override.unwrap_or(ex.recommended_nip).max(2);
    let f = ex.f;
    let grid: u128 = (2 * nip as u128).pow(ex.ndim as u32);

    let pts = ex.points(Some(nip));

    let t = Instant::now();
    let a1 = match alg1(&f, &pts, &Alg1Options::default()) {
        Ok((m, _)) => fmt(eval_errors(&bounds, &|x| f(x), &|x| m.eval(x)).0),
        Err(_) => "ERR".into(),
    };
    let t1 = secs(t.elapsed());

    // alg2's greedy Full scan is the point-count wall (see ALG2_GRID_CAP); skip it
    // on grids large enough to hang, leaving alg1/alg3 to report.
    let t = Instant::now();
    let a2 = if grid > ALG2_GRID_CAP {
        "SKIP".into()
    } else {
        match alg2(&f, &pts, &Alg2Options::default()) {
            Ok((m, _)) => {
                let mx = eval_errors(&bounds, &|x| f(x), &|x| m.eval(x)).0;
                fmt(mx)
            }
            Err(_) => "ERR".into(),
        }
    };
    let t2 = secs(t.elapsed());

    let t = Instant::now();
    let a3 = match alg3(&f, &Alg3Options::new(bounds.clone())) {
        Ok((m, _)) => fmt(eval_errors(&bounds, &|x| f(x), &|x| m.eval(x)).0),
        Err(_) => "ERR".into(),
    };
    let t3 = secs(t.elapsed());

    println!(
        "{:>6} {:>3} {:>10} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7}  {}",
        ex.id,
        ex.ndim,
        tags_str(ex.tags),
        a1,
        t1,
        a2,
        t2,
        a3,
        t3,
        ex.name
    );
}

fn secs(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1e3;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1e3)
    } else {
        format!("{ms:.0}ms")
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let first = args.next();
    let nip_override: Option<usize> = args.next().and_then(|s| s.parse().ok());

    match first.as_deref().and_then(|s| s.parse::<u32>().ok()) {
        // Single-case mode: print just this case's row (no header). The robust
        // path — drive it from a shell loop with `timeout` per invocation.
        Some(cas) => {
            if let Some(ex) = examples::all().iter().find(|e| e.id == cas) {
                run_one(ex, nip_override);
            } else {
                eprintln!("no CAS {cas}");
            }
        }
        // No CAS arg: best-effort in-process sweep of the small-grid cases only
        // (the big ones print BIG and are skipped). Header included.
        None => {
            let nip_override = first.and_then(|s| s.parse().ok());
            header();
            for ex in examples::all() {
                run_one(ex, nip_override);
            }
        }
    }
}
