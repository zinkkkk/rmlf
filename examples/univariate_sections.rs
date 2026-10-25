//! Extract the set of single-variable barycentric rationals from a multivariate
//! model and analyze each section individually.
//!
//! Two views are demonstrated:
//!
//! 1. **`fibre(axis, pin)`** — one exact 1-D barycentric along a chosen axis with
//!    the other variables pinned. Use it to probe the model along any line.
//! 2. **`fibre_set()`** — the canonical decoupling set (the rational-KST
//!    single-variable functions): every section as a `(nodes, values, weights)`
//!    triple, labelled with its axis and leading multi-index, whose grid-line
//!    products reconstruct the whole model.
//!
//! Run: `cargo run --example univariate_sections`

use rmlf::{Alg1Options, CompressedBarycentric, Fibres, Points, alg1};

fn main() {
    // A coupled 3-variable rational. Build a model with alg1.
    let f = |x: &[f64]| 1.0 / (x[0] + x[1] + x[2] + 5.0);
    let pts = Points::alternating(&[(-1.0, 1.0); 3], &[6, 6, 6]).unwrap();
    let (model, info) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
    println!("model order per axis: {:?}", info.order);

    // --- View 1: a single fibre along axis 0, others pinned ---------------------
    let pin = [0.0, 0.3, -0.2];
    let fib = model.fibre(0, &pin).unwrap();
    println!("\nfibre along axis 0 at pin {pin:?}:");
    println!("  nodes   = {:?}", fib.nodes);
    println!("  values  = {:?}", fib.values);
    println!("  weights = {:?}", fib.weights);
    // It is exact along that line:
    for &t in &[-0.7, 0.15, 0.8] {
        let here = fib.eval(t);
        let full = model.eval(&[t, pin[1], pin[2]]);
        println!(
            "  x0={t:+.2}: fibre={here:+.6}  model={full:+.6}  Δ={:.1e}",
            (here - full).abs()
        );
    }

    // --- View 2: the full decoupling set ---------------------------------------
    let set = model.fibre_set();
    println!("\ndecoupled single-variable sections: {} total", set.len());
    for sec in &set {
        // A cheap per-section structure summary: node count, weight magnitude
        // (how much "mass" the section carries), and value range.
        let wmax = sec
            .bary
            .weights
            .iter()
            .fold(0.0_f64, |m, &w| m.max(w.abs()));
        let (vlo, vhi) = sec
            .bary
            .values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        println!(
            "  axis {} lead {:?}: k={} |w|max={:.3e} values∈[{:+.3},{:+.3}]",
            sec.axis,
            sec.lead,
            sec.bary.nodes.len(),
            wmax,
            vlo,
            vhi
        );
    }

    // --- View 3: the channel decomposition (all axes built in) ------------------
    // A coupled target gives TT-rank > 1, so the per-axis channel matrices are
    // genuinely multivariate and the rank index threads the interactions.
    let g = |x: &[f64]| 1.0 / (x[0] + x[1] + 2.0) + 1.0 / (x[1] + x[2] + 3.0);
    let gpts = Points::alternating(&[(-0.5, 0.5); 3], &[6, 6, 6]).unwrap();
    let (gmodel, _) = alg1(&g, &gpts, &Alg1Options::default()).unwrap();
    let tt = CompressedBarycentric::from_model(&gmodel, 1e-12);
    let cd = tt.channel_decomposition();
    println!(
        "\nchannel decomposition of a coupled target (TT max rank {}):",
        tt.max_rank()
    );
    for ax in &cd.denominator {
        println!(
            "  denom axis {}: {}×{} = {} channel(s), each a 1-D bary in x{}",
            ax.axis,
            ax.rank_in,
            ax.rank_out,
            ax.channels.len(),
            ax.axis
        );
    }
    // Reassemble the model from the channels — proof the 1-D pieces combine back.
    println!("\n  reconstruction from channels vs full eval:");
    for x in [[0.1, -0.2, 0.3], [-0.4, 0.25, -0.1]] {
        let recon = cd.reconstruct(&x);
        let full = tt.eval(&x);
        println!(
            "    x={x:?}: recon={recon:+.8} eval={full:+.8} Δ={:.1e}",
            (recon - full).abs()
        );
    }
}
