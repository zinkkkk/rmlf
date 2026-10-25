//! Standalone timing sweep for the numerical Loewner hot paths.
//!
//! Run: cargo run --release --example loewner_bench [group] [milliseconds]
//! Groups: all (default), recurse, order, fit, eval, dense-vs-tt.
//! Timings are wall-clock averages, not statistical Criterion estimates.
//! The largest dense-vs-tt case samples an 8^8 grid; select a group for a quick run.

use rmlf::internals::loewner_null_rec_svd0;
use rmlf::{
    Alg1Options, CompressedBarycentric, OrderOptions, Points, alg1, compute_order,
    points_selection, sample,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn separable_rational(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(l, &v)| 1.0 / (v + 2.0 + l as f64))
        .product()
}

/// Warm up once, then run for at least the requested measurement duration.
fn measure<T>(budget: Duration, mut run: impl FnMut() -> T) -> (usize, Duration) {
    black_box(run());
    let start = Instant::now();
    let mut iterations = 0;
    loop {
        black_box(run());
        iterations += 1;
        if start.elapsed() >= budget {
            break;
        }
    }
    (iterations, start.elapsed())
}

fn row<T>(operation: &str, size: usize, budget: Duration, run: impl FnMut() -> T) {
    let (iterations, elapsed) = measure(budget, run);
    println!(
        "{operation:<28} {size:>9} {iterations:>12} {:>12.2}",
        elapsed.as_secs_f64() * 1e6 / iterations as f64
    );
}

fn recurse(budget: Duration) -> rmlf::Result<()> {
    for n in [2, 3, 4, 5] {
        let points = Points::alternating(&vec![(-1.0, 1.0); n], &vec![8; n])?;
        let tab = sample(&separable_rational, &points);
        let order = compute_order(&points, tab.view(), &OrderOptions::default());
        let selected = points_selection(&points, tab.view(), &order, true);
        row("recursive null space", n, budget, || {
            loewner_null_rec_svd0(&selected.points, selected.tab.view()).unwrap()
        });
    }
    Ok(())
}

fn order(budget: Duration) -> rmlf::Result<()> {
    let opt = OrderOptions {
        tol: 1e-9,
        max_contexts: None,
        ..Default::default()
    };
    for n in [2, 3, 4] {
        let points = Points::alternating(&vec![(-1.0, 1.0); n], &vec![8; n])?;
        let tab = sample(&separable_rational, &points);
        row("order detection", n, budget, || {
            compute_order(&points, tab.view(), &opt)
        });
    }
    Ok(())
}

fn fit(budget: Duration) -> rmlf::Result<()> {
    for n in [2, 3, 4] {
        let points = Points::alternating(&vec![(-1.0, 1.0); n], &vec![8; n])?;
        row("alg1 end-to-end", n, budget, || {
            alg1(&separable_rational, &points, &Alg1Options::default()).unwrap()
        });
    }
    Ok(())
}

fn eval(budget: Duration) -> rmlf::Result<()> {
    let points = Points::alternating(&[(-1.0, 1.0); 3], &[8; 3])?;
    let (model, _) = alg1(&separable_rational, &points, &Alg1Options::default())?;
    for count in [64, 1024, 16384] {
        let queries: Vec<[f64; 3]> = (0..count)
            .map(|i| {
                let t = i as f64 / count as f64;
                [t - 0.5, 0.3 * t, 0.7 - t]
            })
            .collect();
        let xs: Vec<&[f64]> = queries.iter().map(|x| x.as_slice()).collect();
        row("eval_grid (queries)", count, budget, || {
            model.eval_grid(&xs)
        });
    }
    Ok(())
}

fn dense_vs_tt(budget: Duration) -> rmlf::Result<()> {
    let f = |x: &[f64]| {
        x.iter()
            .map(|&v| 1.0 / ((v + 2.0) * (v + 3.0)))
            .product::<f64>()
    };
    for n in [4, 6, 8] {
        let points = Points::alternating(&vec![(-1.0, 1.0); n], &vec![8; n])?;
        let (model, _) = alg1(&f, &points, &Alg1Options::default())?;
        let tt = CompressedBarycentric::from_model(&model, 1e-10);
        let queries: Vec<Vec<f64>> = (0..1000)
            .map(|i| {
                (0..n)
                    .map(|l| ((i * 7 + l) % 17) as f64 / 17.0 - 0.5)
                    .collect()
            })
            .collect();
        let xs: Vec<&[f64]> = queries.iter().map(|x| x.as_slice()).collect();
        row("dense eval (1000 queries)", n, budget, || {
            model.eval_grid(&xs)
        });
        row("TT eval (1000 queries)", n, budget, || tt.eval_grid(&xs));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let group = args.next().unwrap_or_else(|| "all".into());
    if group == "--help" || group == "-h" {
        println!(
            "cargo run --release --example loewner_bench [all|recurse|order|fit|eval|dense-vs-tt] [milliseconds=100]"
        );
        return Ok(());
    }
    let ms = args
        .next()
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(100);
    if ms == 0 || args.next().is_some() {
        return Err("provide a positive measurement duration in milliseconds".into());
    }
    if !["all", "recurse", "order", "fit", "eval", "dense-vs-tt"].contains(&group.as_str()) {
        return Err(format!("unknown group {group}; use --help for available groups").into());
    }
    let budget = Duration::from_millis(ms);
    println!(
        "{:<28} {:>9} {:>12} {:>12}",
        "operation", "n/queries", "iterations", "us/call"
    );
    for (name, run) in [
        ("recurse", recurse as fn(Duration) -> rmlf::Result<()>),
        ("order", order),
        ("fit", fit),
        ("eval", eval),
        ("dense-vs-tt", dense_vs_tt),
    ] {
        if group == "all" || group == name {
            run(budget)?;
        }
    }
    println!(
        "\nWall-clock averages after one warm-up; at least {ms}ms per row. Setup is excluded except for the end-to-end fit."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurement_runs_once_after_warmup_even_without_a_time_budget() {
        let mut calls = 0;
        let (iterations, _) = measure(Duration::ZERO, || {
            calls += 1;
            calls
        });
        assert_eq!(iterations, 1);
        assert_eq!(calls, 2);
    }
}
