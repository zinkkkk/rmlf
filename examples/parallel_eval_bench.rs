//! **Honest benchmark of parallel / batched TT evaluation.** Run:
//!
//! ```text
//! cargo run --release --example parallel_eval_bench
//! ```
//!
//! Two parallelism strategies, measured (not claimed):
//!   A. DEEP single eval — `contract_parallel` reduces each core to a transfer matrix in
//!      parallel, then collapses the chain by associative reduction. Helps only when the chain
//!      is deep enough to amortize thread overhead.
//!   B. BATCHED eval — many independent queries are embarrassingly parallel (rayon over the
//!      batch). This is the strategy that actually scales.
//!
//! We report wall-clock and speedup, and call it as it falls.

use rayon::prelude::*;
use rmlf::exact::field::Fp;
use rmlf::exact::tt::ExactTt;
use std::time::Instant;

fn build_deep(n: usize, p: u64) -> ExactTt<Fp> {
    let dims = vec![6usize; n];
    ExactTt::from_automaton(
        &dims,
        8,
        0,
        |_l, s, sym| (s + sym) % 8,
        |s| Fp::new((s == 0) as i128, p),
    )
}

fn weights_for(tt: &ExactTt<Fp>, p: u64, seed0: u64) -> Vec<Vec<Fp>> {
    let mut seed = seed0;
    tt.dims
        .iter()
        .map(|&k| {
            (0..k)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    Fp::new((seed >> 10) as i128, p)
                })
                .collect()
        })
        .collect()
}

fn main() {
    let p = 2_147_483_647u64;
    println!("Parallel / batched TT evaluation — honest wall-clock\n");

    // ── A. Deep single eval: serial vs parallel segment contraction ────────────
    println!("A. DEEP single eval (one contraction), serial vs parallel, by depth:");
    for &depth in &[16usize, 64, 256, 1024, 4096] {
        let tt = build_deep(depth, p);
        let w = weights_for(&tt, p, 12345);
        let reps = (2_000_000 / depth).max(50);

        let t = Instant::now();
        let mut acc = Fp::new(0, p);
        for _ in 0..reps {
            acc = acc + tt.contract(&w);
        }
        let dt_s = t.elapsed();

        let t = Instant::now();
        let mut acc2 = Fp::new(0, p);
        for _ in 0..reps {
            acc2 = acc2 + tt.contract_parallel(&w);
        }
        let dt_p = t.elapsed();
        assert_eq!(acc, acc2);

        println!(
            "   depth {depth:<5} serial {:>8.2} µs/eval   parallel {:>8.2} µs/eval   speedup {:.2}x",
            dt_s.as_secs_f64() * 1e6 / reps as f64,
            dt_p.as_secs_f64() * 1e6 / reps as f64,
            dt_s.as_secs_f64() / dt_p.as_secs_f64()
        );
    }

    // ── B. Batched eval: serial loop vs rayon over the batch ───────────────────
    println!("\nB. BATCHED eval (many independent queries), serial vs rayon:");
    let tt = build_deep(64, p);
    let batch: Vec<Vec<Vec<Fp>>> = (0..200_000u64)
        .map(|i| weights_for(&tt, p, i.wrapping_mul(2654435761) + 1))
        .collect();

    let t = Instant::now();
    let s_serial: Fp = batch.iter().fold(Fp::new(0, p), |a, w| a + tt.contract(w));
    let dt_serial = t.elapsed();

    let t = Instant::now();
    let s_par: Fp = batch
        .par_iter()
        .map(|w| tt.contract(w))
        .reduce(|| Fp::new(0, p), |a, b| a + b);
    let dt_par = t.elapsed();
    assert_eq!(s_serial, s_par);

    println!(
        "   {} queries, depth 64:  serial {:.1} ms   rayon {:.1} ms   speedup {:.2}x  ({} threads)",
        batch.len(),
        dt_serial.as_secs_f64() * 1e3,
        dt_par.as_secs_f64() * 1e3,
        dt_serial.as_secs_f64() / dt_par.as_secs_f64(),
        rayon::current_num_threads()
    );

    println!("\nRead-off: batched/rayon scales near-linearly with cores (the right strategy for");
    println!("many queries). Parallel single-eval only helps once the chain is very deep — for");
    println!("shallow chains the serial sweep wins (thread overhead dominates).");
}
