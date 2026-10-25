//! Tensor-train compression of **raw data** (a given matrix or n-D tensor), as
//! opposed to a barycentric model. Exercises `TtTensor::from_dense` /
//! `to_dense` directly: low-rank data compresses with bounded reconstruction
//! error, incompressible data falls back to full rank but still honors the
//! tolerance, and the round-trip is exact (to `tol`) in every case.

use rmlf::TtTensor;

fn frobenius(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Relative Frobenius reconstruction error `‖a − b‖_F / ‖a‖_F`.
fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let diff: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    frobenius(&diff) / frobenius(a).max(1e-300)
}

/// Deterministic LCG noise in `[-0.5, 0.5)` (no rng dependency).
fn noise(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

/// Row-major multi-index of flat `i` for `dims`.
fn multi(i: usize, dims: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; dims.len()];
    let mut r = i;
    for ax in (0..dims.len()).rev() {
        idx[ax] = r % dims[ax];
        r /= dims[ax];
    }
    idx
}

#[test]
fn rank1_outer_product_matrix_compresses_to_rank_1() {
    // A rank-1 matrix u ⊗ v: TT (here just an SVD) must recover rank 1 and
    // reconstruct it to machine precision, storing O(m + n) instead of m·n.
    let (m, n) = (7, 9);
    let u: Vec<f64> = (0..m).map(|i| 1.0 + i as f64).collect();
    let v: Vec<f64> = (0..n).map(|j| (2.0 + j as f64).sin()).collect();
    let data: Vec<f64> = (0..m * n).map(|k| u[k / n] * v[k % n]).collect();

    let tt = TtTensor::from_dense(&data, &[m, n], 1e-12);
    assert_eq!(
        tt.max_rank(),
        1,
        "rank-1 matrix should compress to TT-rank 1"
    );
    assert!(
        tt.storage() < m * n,
        "storage {} not < {}",
        tt.storage(),
        m * n
    );
    assert!(rel_err(&data, &tt.to_dense()) < 1e-12);
}

#[test]
fn separable_4d_tensor_is_rank_1() {
    // A fully separable tensor T[i,j,k,l] = a_i b_j c_k d_l is TT-rank 1; dense
    // storage (here 360) collapses to O(Σ dims).
    let dims = [4, 5, 3, 6];
    let total: usize = dims.iter().product();
    let data: Vec<f64> = (0..total)
        .map(|i| {
            let x = multi(i, &dims);
            (1.0 + x[0] as f64)
                * (2.0 + x[1] as f64).ln()
                * (0.5 + x[2] as f64)
                * (1.0 + x[3] as f64).sqrt()
        })
        .collect();

    let tt = TtTensor::from_dense(&data, &dims, 1e-12);
    assert_eq!(tt.max_rank(), 1, "separable tensor should be TT-rank 1");
    assert!(
        tt.storage() * 10 < total,
        "storage {} not << {}",
        tt.storage(),
        total
    );
    assert!(rel_err(&data, &tt.to_dense()) < 1e-13);
}

#[test]
fn low_rank_sum_of_two_separable_terms() {
    // Sum of two separable terms ⇒ TT-rank 2 (each axis unfolding has rank ≤ 2).
    // Verifies the compressor finds the true low rank, not 1 and not full.
    let dims = [6, 5, 7];
    let total: usize = dims.iter().product();
    let data: Vec<f64> = (0..total)
        .map(|i| {
            let x = multi(i, &dims);
            let (a, b, c) = (x[0] as f64, x[1] as f64, x[2] as f64);
            // term1 = a·b·c (separable), term2 = sin(a)·sin(b)·sin(c) (separable)
            (1.0 + a) * (1.0 + b) * (1.0 + c) + a.sin() * b.sin() * c.sin()
        })
        .collect();

    let tt = TtTensor::from_dense(&data, &dims, 1e-12);
    assert!(
        tt.max_rank() <= 2,
        "sum of 2 separable terms should be rank ≤ 2, got {}",
        tt.max_rank()
    );
    assert!(rel_err(&data, &tt.to_dense()) < 1e-12);
}

#[test]
fn incompressible_tensor_falls_back_to_full_rank_but_honors_tol() {
    // Random data has no low-rank structure: the compressor cannot shrink it (rank
    // saturates), but the reconstruction must still satisfy the requested relative
    // tolerance — TT compression never silently exceeds its error budget.
    let dims = [4, 5, 3, 6];
    let total: usize = dims.iter().product();
    let data = noise(total, 0x1234_5678);

    // A loose tolerance truncates aggressively; the error stays under it.
    let tt = TtTensor::from_dense(&data, &dims, 1e-1);
    let err = rel_err(&data, &tt.to_dense());
    assert!(err <= 1e-1 + 1e-9, "relerr {err:e} exceeds tol 1e-1");
    // It genuinely could not compress to rank 1.
    assert!(tt.max_rank() > 1, "random data unexpectedly low-rank");
}

#[test]
fn tighter_tolerance_keeps_more_rank_and_less_error() {
    // The tol knob trades storage for accuracy monotonically: a tighter tol keeps
    // at least as much rank and yields at most as much error.
    let dims = [8, 8, 8];
    let total: usize = dims.iter().product();
    // Smoothly decaying spectrum: a Hilbert-like tensor, low-rank-ish but not exact.
    let data: Vec<f64> = (0..total)
        .map(|i| {
            let x = multi(i, &dims);
            1.0 / (1.0 + x[0] as f64 + x[1] as f64 + x[2] as f64)
        })
        .collect();

    let loose = TtTensor::from_dense(&data, &dims, 1e-2);
    let tight = TtTensor::from_dense(&data, &dims, 1e-10);

    assert!(
        tight.max_rank() >= loose.max_rank(),
        "tighter tol kept less rank ({} < {})",
        tight.max_rank(),
        loose.max_rank()
    );
    assert!(rel_err(&data, &tight.to_dense()) <= rel_err(&data, &loose.to_dense()) + 1e-12);
    assert!(rel_err(&data, &loose.to_dense()) <= 1e-2 + 1e-9);
}

#[test]
fn single_axis_tensor_round_trips() {
    // Degenerate n = 1 case (a plain vector): from_dense/to_dense must be exact.
    let data: Vec<f64> = (0..10).map(|i| (i as f64 * 0.7).cos()).collect();
    let tt = TtTensor::from_dense(&data, &[10], 0.0);
    assert!(rel_err(&data, &tt.to_dense()) < 1e-14);
}
