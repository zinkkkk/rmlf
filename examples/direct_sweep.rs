//! **Direct-KST sweep over the rational catalog** — a wide correctness/fuzz pass for the
//! `exact::direct` framework plus a coefficient-trend study aimed at the inverse-construction goal.
//!
//! For every catalog case (polynomials & rationals, separable & coupled, 2–5 variables) we, over a
//! prime field 𝔽_p:
//!   1. CONSTRUCT a `DirectKst` by inspection from grid samples (timed),
//!   2. check ACCURACY exactly at many random off-grid points vs the true ℚ→𝔽_p value,
//!   3. report STORAGE: TT stored elements vs the dense ∏k_l a non-decoupled model needs,
//!   4. expose the COEFFICIENT structure: TT bond ranks (the interaction signature) and core
//!      sparsity, so we can see how higher-variable multivariate types are actually built.
//!
//! The closing analysis aggregates the trends: how interaction rank tracks coupling vs dimension,
//! where the stored coefficients are redundant, and what that implies for *placing coefficients
//! directly* to hit a target with no source function (the inverse method).
//!
//! Run: `cargo run --release --example direct_sweep > docs/direct_sweep_results.txt`

use num_bigint::BigInt;
use num_traits::Signed;
use rmlf::exact::direct::{DecoupledPoly, DirectKst};
use rmlf::exact::rational::catalog;
use rmlf::exact::{Field, Fp, Rational};
use std::time::Instant;

const P: u64 = 2_147_483_647; // 2^31−1, a Mersenne prime — big enough to avoid accidental collisions

/// A sweep target: either a single polynomial closure, or an explicit numerator/denominator pair.
/// Paper 2's "by inspection" rational reconstruction needs the numerator and denominator *sampled
/// separately* — you cannot recover the denominator from quotient samples without paper 1's null
/// space. So rationals are supplied as (num, den) closures here; polynomials as one closure.
enum Target {
    Poly(fn(&[Rational]) -> Rational),
    Rat(fn(&[Rational]) -> Rational, fn(&[Rational]) -> Rational),
}

/// Reduce a ℚ value to 𝔽_p: (numer mod p)·(denom mod p)⁻¹. Returns None if the denominator ≡ 0
/// (the rational has a pole mod p at this point — skip it).
fn q_to_fp(q: &Rational) -> Option<Fp> {
    let n = bigint_mod(q.numer(), P);
    let d = bigint_mod(q.denom(), P);
    if d == 0 {
        return None;
    }
    let nf = Fp::new(n as i128, P);
    let df = Fp::new(d as i128, P);
    df.inv().map(|di| nf * di)
}

fn bigint_mod(x: &BigInt, p: u64) -> u64 {
    let m = BigInt::from(p);
    let mut r = x % &m;
    if r.is_negative() {
        r += &m;
    }
    // r ∈ [0, p) and p < 2^31, so it fits in the lowest u64 digit.
    r.to_u64_digits().1.first().copied().unwrap_or(0)
}

/// Build the field grid for a case: per axis, `k_l = degree_l + 1` distinct nodes `1..=k_l`.
fn field_grid(degree: &[usize]) -> Vec<Vec<Fp>> {
    degree
        .iter()
        .map(|&d| (0..=d as i128).map(|i| Fp::new(i + 1, P)).collect())
        .collect()
}

/// Cheap deterministic PRNG for off-grid test points.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn fp_offgrid(&mut self, kmax: usize) -> Fp {
        // pick a value > kmax so it can't land on a node (nodes are 1..=k)
        Fp::new((kmax as i128) + 1 + (self.next() % 10_000) as i128, P)
    }
}

/// A constructed model with a uniform interface, whether it came from the polynomial path
/// (`DecoupledPoly`) or the rational path (`DirectKst`).
struct Built {
    poly: Option<DecoupledPoly<Fp>>,
    rat: Option<DirectKst<Fp>>,
    stored: usize,
    dense: usize,
}
impl Built {
    fn eval(&self, x: &[Fp]) -> Option<Fp> {
        if let Some(p) = &self.poly {
            Some(p.eval(x))
        } else {
            self.rat.as_ref().unwrap().eval(x)
        }
    }
    /// Combined bond-rank signature (numerator then denominator for a rational).
    fn bond_ranks(&self) -> Vec<usize> {
        if let Some(p) = &self.poly {
            p.bond_ranks().to_vec()
        } else {
            let (mut a, b) = self.rat.as_ref().unwrap().bond_ranks();
            a.extend(b);
            a
        }
    }
}

/// The sweep target for a catalog id: polynomial closure, or explicit (num, den) for rationals.
fn target_for(id: u32) -> Target {
    match id {
        6 => Target::Rat(
            |_| Rational::one(),
            |x| (x[0].clone() + Rational::integer(1)) * (x[1].clone() + Rational::integer(2)),
        ),
        7 => Target::Rat(
            |x| x[1].clone() * x[1].clone() + x[0].clone() - Rational::integer(2),
            |x| x[1].clone() * x[1].clone() + Rational::integer(2) * x[0].clone() + Rational::one(),
        ),
        8 => Target::Rat(
            |x| x[0].clone() + x[1].clone() * x[2].clone(),
            |x| Rational::one() + x[0].clone() * x[0].clone() + x[1].clone() * x[1].clone(),
        ),
        11 => Target::Rat(
            |_| Rational::one(),
            |x| x[0].clone() + x[1].clone() + Rational::one(),
        ),
        12 => Target::Rat(
            |x| x[0].clone() * x[0].clone() - x[1].clone() * x[1].clone(),
            |x| x[0].clone() - x[1].clone(),
        ),
        // everything else is a polynomial; reuse the catalog closure
        other => Target::Poly(catalog::get(other).unwrap().f),
    }
}

/// Build a model from a target on `nodes`. None if a grid sample hits a pole mod P.
fn build_target(target: &Target, nodes: &[Vec<Fp>]) -> Option<Built> {
    match target {
        Target::Poly(f) => {
            let values = sample_closure(nodes, |q| q_to_fp(&f(q)))?;
            let poly = DecoupledPoly::from_grid_values(nodes.to_vec(), &values);
            let stored = poly.stored_len();
            let dense = poly.dense_len();
            Some(Built {
                poly: Some(poly),
                rat: None,
                stored,
                dense,
            })
        }
        Target::Rat(num, den) => {
            let numv = sample_closure(nodes, |q| q_to_fp(&num(q)))?;
            let denv = sample_closure(nodes, |q| q_to_fp(&den(q)))?;
            let kst = DirectKst::from_num_den_values(nodes.to_vec(), &numv, &denv);
            let stored = kst.stored_len();
            let dense = kst.dense_len();
            Some(Built {
                poly: None,
                rat: Some(kst),
                stored,
                dense,
            })
        }
    }
}

/// Sample a ℚ→𝔽_p closure on the row-major Kronecker grid (node j on axis l encodes integer j+1).
fn sample_closure(
    nodes: &[Vec<Fp>],
    mut g: impl FnMut(&[Rational]) -> Option<Fp>,
) -> Option<Vec<Fp>> {
    let sizes: Vec<usize> = nodes.iter().map(Vec::len).collect();
    let total: usize = sizes.iter().product();
    let n = nodes.len();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; n];
    for _ in 0..total {
        let qpt: Vec<Rational> = (0..n)
            .map(|l| Rational::integer((idx[l] + 1) as i64))
            .collect();
        out.push(g(&qpt)?);
        for l in (0..n).rev() {
            idx[l] += 1;
            if idx[l] < sizes[l] {
                break;
            }
            idx[l] = 0;
        }
    }
    Some(out)
}

fn main() {
    println!("Direct-KST sweep over the rational catalog   (field 𝔽_{P}, p = 2^31−1)");
    println!("{}", "=".repeat(108));
    println!(
        "{:<3} {:<4} {:<10} {:>11} {:>9} {:>9} {:>7} {:>10} {:>6}  name",
        "id", "n", "tags", "bond-ranks", "stored", "dense", "ratio", "build", "acc"
    );
    println!("{}", "-".repeat(108));

    let mut rng = Rng(0x1234_5678_9abc_def1);
    // aggregate trend buckets
    let mut sep_ranks = Vec::new();
    let mut coupled_ranks = Vec::new();
    let mut total_stored = 0usize;
    let mut total_dense = 0usize;
    let mut pass = 0;
    let mut fail = 0;

    for case in catalog::all() {
        let nodes = field_grid(case.degree);
        let kmax = case.degree.iter().max().copied().unwrap_or(0) + 1;
        let target = target_for(case.id);

        // CONSTRUCT (timed): polynomials via DecoupledPoly; rationals via DirectKst (separate n/d).
        let t = Instant::now();
        let built = build_target(&target, &nodes);
        let build_us = t.elapsed().as_secs_f64() * 1e6;
        let built = match built {
            Some(b) => b,
            None => {
                println!(
                    "{:<3} {:<4} {:<10} {:>11} {:>9} {:>9} {:>7} {:>10} {:>6}  {}",
                    case.id, case.ndim, "(pole)", "-", "-", "-", "-", "-", "skip", case.name
                );
                continue;
            }
        };

        // ACCURACY: many random off-grid points vs the true ℚ→𝔽_p quotient.
        let mut ok = true;
        let mut mism = 0;
        let mut tested = 0;
        for _ in 0..200 {
            let x: Vec<Fp> = (0..case.ndim).map(|_| rng.fp_offgrid(kmax)).collect();
            let qx: Vec<Rational> = x
                .iter()
                .map(|xi| Rational::integer(xi.value() as i64))
                .collect();
            let want = match q_to_fp(&(case.f)(&qx)) {
                Some(w) => w,
                None => continue,
            };
            let got = match built.eval(&x) {
                Some(g) => g,
                None => continue, // model pole at this point
            };
            tested += 1;
            if got != want {
                ok = false;
                mism += 1;
            }
        }
        if tested == 0 {
            ok = false; // couldn't verify anywhere
        }

        let bonds = built.bond_ranks();
        let stored = built.stored;
        let dense = built.dense;
        total_stored += stored;
        total_dense += dense;
        let ratio = dense as f64 / stored.max(1) as f64;

        let separable = bonds.iter().all(|&r| r <= 1);
        if separable {
            sep_ranks.push((case.ndim, bonds.clone()));
        } else {
            coupled_ranks.push((case.ndim, bonds.clone()));
        }

        let acc = if case.exact {
            if ok {
                pass += 1;
                "PASS"
            } else {
                fail += 1;
                "FAIL"
            }
        } else if ok {
            "lim-ok"
        } else {
            "lim-≠"
        };
        let tagstr = tags_str(case.tags);
        let bondstr = format!("{bonds:?}");
        let mismstr = if mism > 0 {
            format!("{acc}({mism})")
        } else {
            acc.to_string()
        };
        println!(
            "{:<3} {:<4} {:<10} {:>11} {:>9} {:>9} {:>6.1}x {:>8.1}us {:>6}  {}",
            case.id, case.ndim, tagstr, bondstr, stored, dense, ratio, build_us, mismstr, case.name
        );
    }

    println!("{}", "-".repeat(108));
    println!(
        "totals: stored {total_stored}  dense {total_dense}  overall compression {:.2}x   |   PASS {pass}  FAIL {fail}",
        total_dense as f64 / total_stored.max(1) as f64
    );

    // ── coefficient-trend analysis (the inverse-construction intel) ──
    println!("\n{}", "=".repeat(108));
    println!("COEFFICIENT-STRUCTURE TRENDS  (how multivariate types are actually built)");
    println!("{}", "-".repeat(108));
    println!("separable cases (all bond ranks ≤ 1 ⇒ storage Σk_l, eval O(Σk_l)):");
    for (n, b) in &sep_ranks {
        println!("    ndim={n}  bonds={b:?}");
    }
    println!("coupled cases (bond rank > 1 = genuine multivariate interaction):");
    for (n, b) in &coupled_ranks {
        println!(
            "    ndim={n}  bonds={b:?}  max-rank={}",
            b.iter().max().copied().unwrap_or(0)
        );
    }
    println!(
        "\nreading: the TT bond rank is the EXACT interaction order along each cut — it is the\n\
         single number that says how far a function is from separable. Separable ⇒ all 1 ⇒ the\n\
         coefficient tensor is a pure outer product of per-axis vectors (place n vectors, done).\n\
         A rank-r bond means r outer-product terms are needed across that cut. For the INVERSE\n\
         method this is the design dial: pick the bond-rank profile you want, fill r·k per core,\n\
         and you have a function of prescribed interaction structure with NO reference function."
    );

    // ── LITERAL ARRAY CONTENTS ── the actual numbers, in order, for a few instructive cases.
    println!("\n{}", "=".repeat(108));
    println!("LITERAL DECOUPLED ARRAYS  (the numbers themselves — how the math composes)");
    println!("note: a small prime p=101 is used here so the values are human-readable mod p.");
    println!("{}", "-".repeat(108));
    for id in [1u32, 0, 2, 7, 5] {
        if let Some(case) = catalog::get(id) {
            dump_case_arrays(case);
        }
    }
}

/// Print the literal nodes, Lagrange coefficients, sample tensor, and every TT core (shape +
/// values, in order) for one case over a tiny readable prime — the raw composition data. For a
/// rational, both the numerator and denominator polynomials are dumped (they decouple).
fn dump_case_arrays(case: &catalog::Case) {
    const PP: u64 = 101; // tiny prime for legibility
    let nodes: Vec<Vec<Fp>> = case
        .degree
        .iter()
        .map(|&d| (0..=d as i128).map(|i| Fp::new(i + 1, PP)).collect())
        .collect();
    println!(
        "\n[case {}] {}   degree={:?}",
        case.id, case.name, case.degree
    );
    match target_for(case.id) {
        Target::Poly(f) => match sample_closure(&nodes, |q| q_to_fp_p(&f(q), PP)) {
            Some(v) => dump_poly("poly", &nodes, &v),
            None => println!("   (poles mod {PP}; skipped)"),
        },
        Target::Rat(num, den) => {
            match (
                sample_closure(&nodes, |q| q_to_fp_p(&num(q), PP)),
                sample_closure(&nodes, |q| q_to_fp_p(&den(q), PP)),
            ) {
                (Some(nv), Some(dv)) => {
                    dump_poly("numerator", &nodes, &nv);
                    dump_poly("denominator", &nodes, &dv);
                }
                _ => println!("   (poles mod {PP}; skipped)"),
            }
        }
    }
}

/// Dump one decoupled polynomial's literal arrays.
fn dump_poly(label: &str, nodes: &[Vec<Fp>], values: &[Fp]) {
    let poly = DecoupledPoly::from_grid_values(nodes.to_vec(), values);
    println!("  ── {label} ──");
    for (l, nd) in poly.nodes.iter().enumerate() {
        let nv: Vec<u64> = nd.iter().map(Fp::value).collect();
        let lg: Vec<u64> = poly.lag[l].iter().map(Fp::value).collect();
        println!("   axis {l}: nodes={nv:?}  lagrange γ={lg:?}");
    }
    let sv: Vec<u64> = values.iter().map(Fp::value).collect();
    println!("   sample tensor (row-major = normalized-Lagrange coeffs): {sv:?}");
    println!(
        "   TT ranks {:?}  (bond profile {:?})",
        poly.tt.ranks,
        poly.bond_ranks()
    );
    for (l, core) in poly.tt.cores.iter().enumerate() {
        let rp = poly.tt.ranks[l];
        let k = poly.tt.dims[l];
        let rn = poly.tt.ranks[l + 1];
        let cv: Vec<u64> = core.iter().map(Fp::value).collect();
        println!("   core {l}  shape (r{rp} × k{k} × r{rn}) = {cv:?}");
    }
}

/// ℚ→𝔽_p reduction at an arbitrary prime (for the legibility dump).
fn q_to_fp_p(q: &Rational, p: u64) -> Option<Fp> {
    let nn = bigint_mod(q.numer(), p);
    let dd = bigint_mod(q.denom(), p);
    if dd == 0 {
        return None;
    }
    Fp::new(dd as i128, p)
        .inv()
        .map(|di| Fp::new(nn as i128, p) * di)
}

fn tags_str(tags: &[catalog::Tag]) -> String {
    use catalog::Tag::*;
    let mut s = String::new();
    for t in tags {
        s.push(match t {
            Polynomial => 'P',
            Rational => 'R',
            Separable => 'S',
            Coupled => 'C',
        });
    }
    s
}
