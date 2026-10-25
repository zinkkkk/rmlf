//! Sweep the complex MATLAB catalog in the same table format as `catalog_sweep`.
//! Run: cargo run --release --example complex_catalog [case-id]
//!
//! Legacy 11–30-variable cases aggregate all 1-D axis sections and a coupled
//! 2-D section, not full exponential grids. Case 59 uses its full original domain.
//! Complex alg3 is not implemented and is explicitly marked SKIP.

use rmlf::examples::{Tag, complex};
use rmlf::prelude::*;
use std::time::{Duration, Instant};
type C = Complex64;

const N_QUERY: usize = 4000;
const SECTION_BOUNDS: (C, C) = (C::new(0.4, 0.1), C::new(1.8, 0.3));

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Same off-grid error normalization and deterministic sample count as the
/// real catalog sweep, using complex magnitudes and segment coordinates.
fn max_rel_error(bounds: &[(C, C)], f: &impl Fn(&[C]) -> C, model: &Barycentric<C>) -> f64 {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    let mut x = vec![C::new(0.0, 0.0); bounds.len()];
    let (mut max, mut scale) = (0.0_f64, 0.0_f64);
    for _ in 0..N_QUERY {
        for (l, &(lo, hi)) in bounds.iter().enumerate() {
            x[l] = lo + (hi - lo) * (0.05 + 0.9 * rng.next());
        }
        let truth = f(&x);
        let value = model.eval(&x);
        if !truth.re.is_finite()
            || !truth.im.is_finite()
            || !value.re.is_finite()
            || !value.im.is_finite()
        {
            return f64::INFINITY;
        }
        scale = scale.max(truth.norm());
        max = max.max((truth - value).norm());
    }
    max / if scale == 0.0 { 1.0 } else { scale }
}

#[derive(Default)]
struct Results {
    alg1_error: f64,
    alg1_time: Duration,
    alg1_failed: bool,
    alg2_error: f64,
    alg2_time: Duration,
    alg2_failed: bool,
}

impl Results {
    fn add_section(
        &mut self,
        bounds: &[(C, C)],
        points: &Points<C>,
        f: &(impl Fn(&[C]) -> C + Sync),
        opt: &Alg1Options,
    ) {
        // Like catalog_sweep, timings include fitting and off-grid checking.
        let start = Instant::now();
        match alg1(f, points, opt) {
            Ok((model, _)) => {
                self.alg1_error = self.alg1_error.max(max_rel_error(bounds, f, &model))
            }
            Err(_) => self.alg1_failed = true,
        }
        self.alg1_time += start.elapsed();

        let start = Instant::now();
        match alg2(f, points, &Alg2Options::default()) {
            Ok((model, _)) => {
                self.alg2_error = self.alg2_error.max(max_rel_error(bounds, f, &model));
            }
            Err(_) => self.alg2_failed = true,
        }
        self.alg2_time += start.elapsed();
    }
}

fn tags_str(tags: &[Tag]) -> String {
    tags.iter()
        .map(|tag| match tag {
            Tag::Polynomial => "poly",
            Tag::Rational => "rat",
            Tag::Irrational => "irr",
            Tag::C0 => "C0",
            Tag::CInf => "Cinf",
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn fmt_error(error: f64, failed: bool) -> String {
    if failed {
        "ERR".into()
    } else if !error.is_finite() || error > 1e3 {
        "FAIL".into()
    } else {
        format!("{error:.1e}")
    }
}

fn secs(duration: Duration) -> String {
    let ms = duration.as_secs_f64() * 1e3;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1e3)
    } else {
        format!("{ms:.0}ms")
    }
}

fn header() {
    println!(
        "{:>6} {:>3} {:>10} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7}  name",
        "CAS", "n", "tags", "alg1", "t1", "alg2", "t2", "alg3", "t3"
    );
}

fn run_one(case: &complex::ComplexExample) -> rmlf::Result<()> {
    let mut results = Results::default();
    let scope = if let Some(degrees) = case.degrees {
        let fixed: Vec<C> = (0..case.ndim)
            .map(|j| C::new(1.05 + 0.009 * j as f64, 0.06 + 0.002 * j as f64))
            .collect();
        let opt = Alg1Options {
            order: rmlf::OrderOptions {
                tol: -1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        for (axis, &degree) in degrees.iter().enumerate() {
            let f = |s: &[C]| {
                let mut x = fixed.clone();
                x[axis] = s[0];
                (case.f)(&x)
            };
            let bounds = [SECTION_BOUNDS];
            let points = Points::chebyshev_segments(&bounds, &[2 * (degree + 1)])?;
            results.add_section(&bounds, &points, &f, &opt);
        }
        let f = |s: &[C]| {
            let mut x = fixed.clone();
            x[..2].copy_from_slice(s);
            (case.f)(&x)
        };
        let bounds = [SECTION_BOUNDS; 2];
        let points =
            Points::chebyshev_segments(&bounds, &[2 * (degrees[0] + 1), 2 * (degrees[1] + 1)])?;
        results.add_section(&bounds, &points, &f, &opt);
        "all-axis + 2D sections"
    } else {
        let points = case.points(None)?;
        results.add_section(
            &case.axis_bounds(),
            &points,
            &case.f,
            &Alg1Options::default(),
        );
        "full original domain"
    };
    println!(
        "{:>6} {:>3} {:>10} {:>9} {:>7} {:>9} {:>7} {:>9} {:>7}  {} [{}]",
        case.id,
        case.ndim,
        tags_str(case.tags),
        fmt_error(results.alg1_error, results.alg1_failed),
        secs(results.alg1_time),
        fmt_error(results.alg2_error, results.alg2_failed),
        secs(results.alg2_time),
        "SKIP",
        "-",
        case.name,
        scope,
    );
    Ok(())
}

fn main() -> rmlf::Result<()> {
    let selected = std::env::args()
        .nth(1)
        .map(|s| s.parse::<u32>().expect("case ID must be an integer"));
    if let Some(id) = selected {
        let case = complex::get(id).expect("unknown complex case ID");
        // Match the real catalog's one-case mode: a row without a header.
        return run_one(&case);
    }
    header();
    for case in complex::all() {
        run_one(case)?;
    }
    println!(
        "\nComplex catalog: errors are max|f-g| / max|f| over 4000 deterministic interior queries per fit."
    );
    println!(
        "Legacy cases: n is the original function dimension; errors are maxima and timings are totals across all 1D axis sections plus the first-two-axis 2D section, not full nD fits."
    );
    println!(
        "Section segments: 0.4+0.1i to 1.8+0.3i; other variables fixed at 1.05+0.009*j + i*(0.06+0.002*j), j=0..n-1."
    );
    println!(
        "Section alg1 uses declared degrees (forced order); alg2 uses default options. Case 59 uses the full original domain and default fitter options."
    );
    println!(
        "SKIP: alg3 is real-only. Timing includes fitting and off-grid evaluation, as in catalog_sweep."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_format_matches_real_catalog_conventions() {
        assert_eq!(tags_str(&[Tag::Irrational, Tag::CInf]), "irr/Cinf");
        assert_eq!(fmt_error(1e-12, false), "1.0e-12");
        assert_eq!(fmt_error(f64::INFINITY, false), "FAIL");
        assert_eq!(fmt_error(0.0, true), "ERR");
        assert_eq!(secs(Duration::from_millis(12)), "12ms");
        assert_eq!(secs(Duration::from_millis(1234)), "1.23s");
    }

    #[test]
    fn error_metric_uses_complex_magnitudes_and_real_sweep_normalization() {
        let value = C::new(3.0, 4.0);
        let model = Barycentric {
            lambda: vec![vec![C::new(10.0, 0.0)]],
            w: vec![C::new(0.0, 0.0)],
            c: vec![C::new(1.0, 0.0)],
        };
        assert_eq!(max_rel_error(&[SECTION_BOUNDS], &|_| value, &model), 1.0);
    }
}
