//! Integration tests for the exact-arithmetic branch — the user front door
//! (`rational::fit_poly/fit_rational`, `field::fit_fp`), bit-exact recovery over
//! ℚ and 𝔽_p, the structural degree ceiling, and lossless packed compression
//! (with the prime hoisted once for finite fields).

use rmlf::exact::Fp;
use rmlf::exact::field;
use rmlf::exact::rational::{self, Rational};

// ── ℚ front door ──

#[test]
fn polya_szego_exact_from_eight_samples() {
    // The headline: xy+xz+yz exact from one fit_poly call (8 column samples).
    let m = rational::fit_poly(
        |x| x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone(),
        &[1, 1, 1],
    )
    .unwrap();
    for &(a, b, c) in &[(5i64, 7, 11), (-13, 4, 100), (1000, -2, 6)] {
        assert_eq!(m.eval_int(&[a, b, c]), Some(a * b + a * c + b * c));
    }
}

#[test]
fn rational_function_exact_from_closure() {
    let m = rational::fit_rational(
        |x| {
            let (s, t) = (x[0].clone(), x[1].clone());
            (t.clone() * t.clone() + s.clone() - Rational::integer(2))
                / (t.clone() * t.clone() + Rational::integer(2) * s + Rational::integer(1))
        },
        &[1, 2],
        &[1, 2],
    )
    .unwrap();
    // (81+7-2)/(81+14+1) = 86/96 = 43/48.
    assert_eq!(m.eval_q(&[7, 9]), Some(Rational::new(43, 48)));
}

#[test]
fn every_well_posed_q_case_recovers() {
    for c in rational::catalog::well_posed() {
        let m = c.fit().unwrap_or_else(|e| panic!("{}: {e:?}", c.name));
        for seed in [11i64, -23, 77] {
            let pt: Vec<i64> = (0..c.ndim).map(|l| seed + 2 * l as i64 + 1).collect();
            let qpt: Vec<Rational> = pt.iter().map(|&v| Rational::integer(v)).collect();
            assert_eq!(
                m.eval_q(&pt),
                Some((c.f)(&qpt)),
                "case {} ({})",
                c.id,
                c.name
            );
        }
    }
}

// ── 𝔽_p front door + the degree ceiling ──

#[test]
fn fit_fp_recovers_over_mersenne_2_61() {
    let p = (1u64 << 61) - 1;
    let m = field::fit_fp(|x| x[0] * x[1] + x[0] + x[1], p, &[1, 1]).unwrap();
    for &(a, b) in &[(123i64, 4567), (10_000, 7)] {
        assert_eq!(
            m.eval_int(&[a, b]),
            Some(((a * b + a + b) as u128 % p as u128) as u64)
        );
    }
}

#[test]
fn fit_fp_rejects_degree_above_ceiling() {
    // Over GF(7), degree 3 needs 8 distinct nodes > 7 → the ⌊p/2⌋−1 ceiling.
    assert!(field::fit_fp(|x| x[0], 7, &[3]).is_err());
}

// ── packed compression is lossless and hoists the prime ──

#[test]
fn packed_fp_model_hoists_prime_and_roundtrips() {
    let p = 257;
    let m = field::fit_fp(|x| x[0] * x[0] * x[0] + x[0], p, &[3]).unwrap();
    let packed = m.packed();
    // The prime is stored once, in the context.
    assert_eq!(packed.ctx, p);
    // Coefficient bytes beat the dense working footprint (no per-element prime).
    assert!(
        packed.coeff_bytes() < packed.dense_coeff_bytes(),
        "packed {} should beat dense {}",
        packed.coeff_bytes(),
        packed.dense_coeff_bytes()
    );
    // Lossless + eval parity.
    assert_eq!(packed.unpack().c_num, m.inner().c_num);
    for t in [10i64, 99, 200] {
        assert_eq!(
            packed.eval(&[Fp::new(t as i128, p)]),
            m.eval(&[Fp::new(t as i128, p)])
        );
    }
}

#[test]
fn coeff_breakdown_separates_the_levers() {
    // The breakdown stages are monotone non-increasing in the "real" levers and
    // expose each contribution. For Pólya–Szegő over ℚ: dense ≫ sparse (zeros
    // dropped) ≥ pooled (repeats interned).
    let m = rational::fit_poly(
        |x| x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone(),
        &[1, 1, 1],
    )
    .unwrap();
    let b = m.packed().coeff_breakdown();
    assert!(
        b.sparse < b.dense,
        "dropping zeros must shrink: {} < {}",
        b.sparse,
        b.dense
    );
    assert!(
        b.pooled <= b.sparse,
        "pooling must not grow: {} <= {}",
        b.pooled,
        b.sparse
    );
    // For ℚ, dense == hoisted (BigRational packs to itself, no shared context).
    assert_eq!(b.dense, b.hoisted, "ℚ has no per-element context to hoist");
}

#[test]
fn fp_breakdown_shows_prime_hoist_halving() {
    // Over 𝔽_p the prime-hoist alone halves the dense footprint (Fp 16 B → u64 8 B).
    let m = field::fit_fp(|x| x[0] * x[1], 257, &[1, 1]).unwrap();
    let b = m.packed().coeff_breakdown();
    assert_eq!(
        b.hoisted * 2,
        b.dense,
        "prime-hoist halves: {} ·2 == {}",
        b.hoisted,
        b.dense
    );
}

#[test]
fn packed_q_model_pools_repeated_values() {
    let m = rational::fit_poly(
        |x| x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone(),
        &[1, 1, 1],
    )
    .unwrap();
    let packed = m.packed();
    // Symmetry pools coefficients: fewer distinct values than nonzero terms.
    assert!(packed.dict_len() < packed.nnz());
    assert_eq!(packed.unpack().c_num, m.inner().c_num);
}
