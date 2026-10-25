//! Alg1's full Loewner construction, with an estimated per-case RAM budget.
//!
//! Run: cargo run --release --example catalog_sweep_full -- [CAS|all] [nip]
//!      [--ram-mib MiB]
//! Default budget: 1024 MiB. No CAS prints the header and sweeps the catalog;
//! a single CAS prints just its row, like catalog_sweep. Only alg1 runs.
//!
//! Check sampling/order-detection storage before sampling, then the reduced
//! matrix, singular vectors, and Faer SVD workspace before selecting/fitting.
//! Estimates include 2x headroom plus 16 MiB, not an OS-enforced RSS limit.
//! Other processes, allocator retention and thread stacks are not bounded.
//! RAM-safe does not mean fast: use an external timeout for large dense SVDs.

use faer::linalg::svd::{ComputeSvdVectors, svd_scratch};
use rmlf::examples::{self, Example, Tag};
use rmlf::select::points_selection_with;
use rmlf::{
    Alg1Options, Barycentric, CoefficientMethod, LoewnerError, alg1_from_tensor, compute_order,
    sample_continued,
};
use std::fmt;
use std::time::{Duration, Instant};

const MIB: u128 = 1 << 20;
const N_QUERY: usize = 4000;
const HEADROOM: u128 = 16 * MIB;
const USAGE: &str = "cargo run --release --example catalog_sweep_full -- [CAS|all] [nip] [--ram-mib MiB]\n\
    Default: all cases, catalog nip, 1024 MiB estimated per-case RAM budget.\n\
    The budget is an allocation estimate, not an OS-enforced RSS or time limit.";

#[derive(Debug, PartialEq, Eq)]
struct Config {
    case: Option<u32>,
    nip: Option<usize>,
    ram_bytes: u128,
}

impl Config {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, String> {
        let mut args = args.into_iter();
        let mut positional = Vec::new();
        let mut ram_bytes = 1024 * MIB;
        let mut ram_given = false;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Ok(None),
                "--ram-mib" if !ram_given => {
                    let value = args.next().ok_or("--ram-mib needs a positive integer")?;
                    let mib = value
                        .parse::<u64>()
                        .map_err(|_| "invalid --ram-mib value")?;
                    ram_bytes = u128::from(mib) * MIB;
                    if mib == 0 || ram_bytes > isize::MAX as u128 {
                        return Err(
                            "RAM budget must be positive and addressable on this platform".into(),
                        );
                    }
                    ram_given = true;
                }
                _ if arg.starts_with('-') => {
                    return Err(format!("unknown or repeated option: {arg}"));
                }
                _ => positional.push(arg),
            }
        }
        if positional.len() > 2 {
            return Err("expected [CAS|all] [nip]".into());
        }
        let case = positional
            .first()
            .filter(|s| s.as_str() != "all")
            .map(|s| {
                s.parse::<u32>()
                    .map_err(|_| "CAS must be an integer or 'all'")
            })
            .transpose()?;
        if case.is_some_and(|id| examples::get(id).is_none()) {
            return Err(format!("no CAS {}", case.unwrap()));
        }
        let nip = positional
            .get(1)
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|_| "nip must be an integer >= 2")
            })
            .transpose()?;
        if nip.is_some_and(|n| n < 2) {
            return Err("nip must be an integer >= 2".into());
        }
        Ok(Some(Self {
            case,
            nip,
            ram_bytes,
        }))
    }
}

#[derive(Debug)]
enum Failure {
    Skip {
        stage: &'static str,
        bytes: Option<u128>,
    },
    Fit(LoewnerError),
}

impl From<LoewnerError> for Failure {
    fn from(error: LoewnerError) -> Self {
        Self::Fit(error)
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Skip {
                stage,
                bytes: Some(bytes),
            } => write!(
                f,
                "{stage}: estimated peak {:.2} MiB exceeds RAM budget",
                *bytes as f64 / MIB as f64
            ),
            Self::Skip { stage, bytes: None } => {
                write!(f, "{stage}: size overflow or unaddressable allocation")
            }
            Self::Fit(error) => write!(f, "{error}"),
        }
    }
}

#[derive(Clone, Copy)]
struct Budget(u128);

impl Budget {
    fn check(self, stage: &'static str, storage: Option<u128>) -> Result<u128, Failure> {
        let bytes = storage.and_then(|n| n.checked_mul(2)?.checked_add(HEADROOM));
        match bytes {
            Some(bytes) if bytes <= self.0 && bytes <= isize::MAX as u128 => Ok(bytes),
            _ => Err(Failure::Skip { stage, bytes }),
        }
    }
}

fn sum(values: &[u128]) -> Option<u128> {
    values.iter().try_fold(0u128, |s, &n| s.checked_add(n))
}

fn product(values: &[usize]) -> Option<u128> {
    values
        .iter()
        .try_fold(1u128, |p, &n| p.checked_mul(n as u128))
}

/// Bound Faer's column padding without allocating any matrix.
fn matrix_bytes(rows: usize, cols: usize) -> Option<u128> {
    let padded_rows = (rows as u128).checked_add(63)? / 64 * 64;
    padded_rows.checked_mul(cols as u128)?.checked_mul(8)
}

/// Catalog inputs have equal column/row counts; default minimal selection keeps
/// them equal. Both order detection and the final SvdNull use a square thin SVD.
fn svd_bytes(
    side: usize,
    resident: u128,
    budget: Budget,
    stage: &'static str,
) -> Result<u128, Failure> {
    let storage = (|| {
        sum(&[
            matrix_bytes(side, side)?.checked_mul(3)?, // input matrix, U, V
            (side as u128).checked_mul(32)?,           // spectrum and null vector
        ])
    })();
    // Reject obviously huge dimensions before asking Faer to size its scratch.
    budget.check(stage, storage.and_then(|n| n.checked_add(resident)))?;
    let scratch = svd_scratch::<f64>(
        side,
        side,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        faer::get_global_parallelism(),
        Default::default(),
    )
    .unaligned_bytes_required();
    let total = storage.and_then(|n| n.checked_add(scratch as u128));
    budget.check(stage, total.and_then(|n| n.checked_add(resident)))?;
    total.ok_or(Failure::Skip { stage, bytes: None })
}

/// Raw sampling tensor, point/axis copies, and per-worker coordinate scratch.
/// All arithmetic is checked before constructing Points or evaluating f.
fn input_storage(ndim: usize, nip: usize) -> Option<u128> {
    let side = nip.checked_mul(2)?;
    let grid = product(&vec![side; ndim])?.checked_mul(8)?;
    let axes = (ndim as u128).checked_mul(nip as u128)?.checked_mul(64)?;
    let scratch = (ndim as u128)
        .checked_mul(rayon::current_num_threads() as u128)?
        .checked_mul(64)?;
    sum(&[grid, axes, scratch])
}

fn input_preflight(ndim: usize, nip: usize, budget: Budget) -> Result<u128, Failure> {
    let resident = input_storage(ndim, nip);
    budget.check("sampling", resident)?;
    let resident = resident.ok_or(Failure::Skip {
        stage: "sampling",
        bytes: None,
    })?;
    let per_context = svd_bytes(nip, resident, budget, "order detection")?;
    // Defaults examine at most five contexts per axis. Count all contexts as
    // live, conservatively allowing nested Rayon tasks to overlap their SVDs.
    let total = per_context
        .checked_mul(ndim as u128)
        .and_then(|n| n.checked_mul(5)?.checked_add(resident));
    budget.check("order detection", total)?;
    Ok(resident)
}

fn full_preflight(
    nip: usize,
    order: &[usize],
    resident: u128,
    budget: Budget,
) -> Result<u128, Failure> {
    let stage = "full Loewner/SVD";
    let counts: Vec<usize> = order
        .iter()
        .map(|&o| o.saturating_add(1).min(nip))
        .collect();
    let columns = product(&counts);
    let combined: Option<Vec<usize>> = counts.iter().map(|&k| k.checked_mul(2)).collect();
    let storage = (|| {
        let columns = columns?;
        // Both selected tensors can coexist: preselection plus Alg1's identity
        // selection on that data. Include original samples even though we drop
        // them before the final solve, plus model/value vectors and the index
        // tables that loewner_full uses while constructing its matrix.
        let selected = product(&combined?)?.checked_mul(16)?;
        let indices = columns.checked_mul(2)?.checked_mul(
            (order.len() as u128)
                .checked_mul(std::mem::size_of::<usize>() as u128)?
                .checked_add(std::mem::size_of::<Vec<usize>>() as u128)?,
        )?;
        let vectors = columns.checked_mul(64)?;
        let cauchy = counts.iter().try_fold(0u128, |s, &k| {
            s.checked_add((k as u128).checked_mul(k as u128)?.checked_mul(8)?)
        })?;
        sum(&[resident, selected, indices, vectors, cauchy])
    })();
    budget.check(stage, storage)?;
    let cols = columns
        .and_then(|n| usize::try_from(n).ok())
        .ok_or(Failure::Skip { stage, bytes: None })?;
    let resident = storage.ok_or(Failure::Skip { stage, bytes: None })?;
    let solve = svd_bytes(cols, resident, budget, stage)?;
    budget.check(stage, resident.checked_add(solve))
}

fn guarded_fit(ex: &Example, nip: usize, budget: Budget) -> Result<Barycentric, Failure> {
    let nip = nip.max(2);
    let resident = input_preflight(ex.ndim, nip, budget)?;
    let points = ex.points(Some(nip));
    let mut options = Alg1Options {
        method: CoefficientMethod::Full,
        ..Default::default()
    };
    let (tab, _) = sample_continued(&ex.f, &points);
    let order = compute_order(&points, tab.view(), &options.order);
    full_preflight(nip, &order, resident, budget)?;

    // Detect once, then pass exactly Alg1's selected data to the ordinary Full
    // implementation. Forced maximal order keeps ALL these preselected nodes;
    // it does not change the original order decision or repeat its SVDs.
    let selected =
        points_selection_with(&points, tab.view(), &order, options.minimal, options.select);
    drop(tab);
    options.order.tol = -1.0;
    let (model, _) = alg1_from_tensor(&selected.tab, &selected.points, &options)?;
    Ok(model)
}

fn eval_error(ex: &Example, model: &Barycentric) -> f64 {
    let bounds = ex.axis_bounds();
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut x = vec![0.0; ex.ndim];
    let (mut error, mut scale) = (0.0f64, 0.0f64);
    for _ in 0..N_QUERY {
        for (xi, &(lo, hi)) in x.iter_mut().zip(&bounds) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = (state >> 11) as f64 / (1u64 << 53) as f64;
            *xi = lo + (hi - lo) * (0.05 + 0.9 * t);
        }
        let truth = (ex.f)(&x);
        scale = scale.max(truth.abs());
        let residual = (truth - model.eval(&x)).abs();
        // Unlike silently ignoring a NaN/Inf, flag a failed evaluation.
        if !truth.is_finite() || !residual.is_finite() {
            return f64::INFINITY;
        }
        error = error.max(residual);
    }
    error / if scale == 0.0 { 1.0 } else { scale }
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

fn secs(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1e3;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1e3)
    } else {
        format!("{ms:.0}ms")
    }
}

fn header() {
    println!(
        "{:>6} {:>3} {:>10} {:>9} {:>7}  name",
        "CAS", "n", "tags", "alg1", "t1"
    );
}

fn run_one(ex: &Example, config: &Config) {
    let start = Instant::now();
    let result = match guarded_fit(
        ex,
        config.nip.unwrap_or(ex.recommended_nip),
        Budget(config.ram_bytes),
    ) {
        Ok(model) => {
            let error = eval_error(ex, &model);
            if error > 1e3 {
                "FAIL".into()
            } else {
                format!("{error:.1e}")
            }
        }
        Err(failure) => match failure {
            Failure::Skip { .. } => "SKIP".into(),
            Failure::Fit(_) => "ERR".into(),
        },
    };
    println!(
        "{:>6} {:>3} {:>10} {:>9} {:>7}  {}",
        ex.id,
        ex.ndim,
        tags_str(ex.tags),
        result,
        secs(start.elapsed()),
        ex.name
    );
}

fn main() -> Result<(), String> {
    let Some(config) = Config::parse(std::env::args().skip(1))? else {
        println!("{USAGE}");
        return Ok(());
    };
    match config.case {
        Some(id) => run_one(&examples::get(id).expect("validated CAS"), &config),
        None => {
            header();
            for ex in examples::all() {
                run_one(ex, &config);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Config>, String> {
        Config::parse(args.iter().map(|s| (*s).into()))
    }

    #[test]
    fn cli_defaults_and_overrides() {
        assert_eq!(
            parse(&[]).unwrap().unwrap(),
            Config {
                case: None,
                nip: None,
                ram_bytes: 1024 * MIB
            }
        );
        assert_eq!(
            parse(&["--ram-mib", "64", "0", "6"]).unwrap().unwrap(),
            Config {
                case: Some(0),
                nip: Some(6),
                ram_bytes: 64 * MIB
            }
        );
        assert_eq!(parse(&["all", "4"]).unwrap().unwrap().nip, Some(4));
        assert!(parse(&["--help"]).unwrap().is_none());
    }

    #[test]
    fn cli_rejects_invalid_input() {
        for args in [
            vec!["--ram-mib"],
            vec!["--ram-mib", "0"],
            vec!["--ram-mib", "NaN"],
            vec!["--ram-mib", "18446744073709551615"],
            vec!["99999"],
            vec!["0", "1"],
            vec!["0", "4", "extra"],
            vec!["--unknown"],
            vec!["--ram-mib", "64", "--ram-mib", "32"],
        ] {
            assert!(parse(&args).is_err(), "{args:?}");
        }
    }

    #[test]
    fn estimates_overflow_safely() {
        assert!(input_storage(30, usize::MAX).is_none());
        assert!(input_preflight(30, usize::MAX, Budget(1024 * MIB)).is_err());
        assert!(full_preflight(usize::MAX, &[usize::MAX; 30], 0, Budget(1024 * MIB)).is_err());
    }

    #[test]
    fn sampling_rejected_before_target_is_called() {
        let mut ex = examples::get(0).unwrap();
        ex.f = |_| panic!("an over-budget case must not be sampled");
        assert!(matches!(
            guarded_fit(&ex, 4, Budget(MIB)),
            Err(Failure::Skip {
                stage: "sampling",
                ..
            })
        ));
    }

    #[test]
    fn grid_and_svd_are_both_budgeted() {
        let budget = Budget(32 * MIB);
        assert!(input_preflight(8, 8, budget).is_err());
        let resident = input_preflight(4, 8, budget).unwrap();
        assert!(full_preflight(8, &[7; 4], resident, budget).is_err());
        // Same input grid, genuinely low detected order: do not assume nip^n
        // coefficients and unnecessarily skip a small full solve.
        assert!(full_preflight(8, &[1; 4], resident, budget).is_ok());
    }

    #[test]
    fn detected_large_full_solve_is_skipped() {
        let mut ex = examples::get(5).unwrap();
        assert_eq!(ex.ndim, 4);
        ex.f = |x| x.iter().map(|v| v.powi(7)).product();
        // Sampling fits, but the degree-seven slices select 8^4 columns;
        // the final dense matrix alone needs 128 MiB. Never build it.
        assert!(matches!(
            guarded_fit(&ex, 8, Budget(32 * MIB)),
            Err(Failure::Skip {
                stage: "full Loewner/SVD",
                ..
            })
        ));
    }

    #[test]
    fn budget_boundary_includes_headroom() {
        assert!(Budget(HEADROOM + 20).check("test", Some(10)).is_ok());
        assert!(Budget(HEADROOM + 19).check("test", Some(10)).is_err());
    }

    #[test]
    fn guarded_pipeline_matches_ordinary_full_alg1() {
        for id in [0, 3, 11, 36, 56] {
            let ex = examples::get(id).unwrap();
            let model = guarded_fit(&ex, 6, Budget(256 * MIB)).unwrap();
            let options = Alg1Options {
                method: CoefficientMethod::Full,
                ..Default::default()
            };
            let (reference, _) = rmlf::alg1(&ex.f, &ex.points(Some(6)), &options).unwrap();
            assert_eq!(model.lambda, reference.lambda, "CAS {id}");
            assert_eq!(model.w, reference.w, "CAS {id}");
            for i in 0..8 {
                let x: Vec<f64> = ex
                    .axis_bounds()
                    .iter()
                    .enumerate()
                    .map(|(axis, &(lo, hi))| lo + (hi - lo) * (0.1 + (i + axis) as f64 * 0.06))
                    .collect();
                let (got, want) = (model.eval(&x), reference.eval(&x));
                assert!(
                    (got - want).abs() <= 1e-10 * (1.0 + want.abs()),
                    "CAS {id}: {got} vs {want}"
                );
            }
        }
    }
}
