//! Demo of Algorithm 3: AAA-style resampling refinement ([`rmlf::alg3`]).
//!
//! Unlike alg1/alg2 (fixed pre-sampled grid), alg3 starts coarse and, each round,
//! refines its kept support nodes, resamples the target on the finer grid, and
//! refits — growing resolution where the function needs it.
//!
//! ```text
//! cargo run --example alg3_refine
//! ```

use rmlf::{Alg3Options, alg3};

type Case = (&'static str, fn(&[f64]) -> f64);

fn main() {
    let cases: Vec<Case> = vec![
        ("separable rational", |x| {
            1.0 / ((x[0] + 3.0) * (x[1] + 2.0))
        }),
        ("Runge 2-D", |x| {
            1.0 / (1.0 + 25.0 * (x[0] * x[0] + x[1] * x[1]))
        }),
        ("tanh · cos", |x| (x[0] * 4.0).tanh() * (x[1] * 3.0).cos()),
    ];

    for (name, f) in cases {
        let opt = Alg3Options::new(vec![(-1.0, 1.0); 2]);
        let (_, info) = alg3(&f, &opt).unwrap();
        println!(
            "{name:20} : {} rounds, grid {} -> {:?} pts/axis, best rel err {:.2e}",
            info.rounds, opt.start_nip, info.final_nip, info.best_error
        );
        let hist: Vec<String> = info
            .error_history
            .iter()
            .map(|e| format!("{e:.1e}"))
            .collect();
        println!("   error per round: [{}]", hist.join(", "));
    }
}
