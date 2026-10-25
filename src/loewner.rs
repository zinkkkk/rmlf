//! Loewner-matrix construction.
//!
//! For column abscissae `lambda` (length `k`) with values `w`, and row abscissae
//! `mu` (length `q`) with values `v`, the 1-D Loewner matrix is `q x k` with
//!
//! ```text
//! L[i, j] = (v[i] - w[j]) / (mu[i] - lambda[j]).
//! ```
//!
//! The recursive method is built from these tiny matrices.  The full method uses
//! the multivariate analogue implemented by [`loewner_full`], whose entries are
//! divided differences over Cartesian row and column grids.  This replaces
//! `mlf.loewnerMatrix` without materializing MATLAB-style combination tables.

use crate::grid::{flat_to_multi, grid_len};
use crate::points::Points;
use crate::scalar::Scalar;
use crate::tensor::{View, scalar_at};
use faer::Mat;

/// Build the `q x k` single-variable Loewner matrix.
///
/// `lambda`/`w` are the column abscissae/values (length `k`); `mu`/`v` are the
/// row abscissae/values (length `q`). The caller guarantees `lambda` and `mu`
/// are disjoint (see [`crate::points::Points::check_disjoint`]).
pub fn loewner_1d<S: Scalar>(lambda: &[S], w: &[S], mu: &[S], v: &[S]) -> Mat<S> {
    debug_assert_eq!(lambda.len(), w.len());
    debug_assert_eq!(mu.len(), v.len());
    Mat::from_fn(mu.len(), lambda.len(), |i, j| {
        (v[i] - w[j]) / (mu[i] - lambda[j])
    })
}

/// Build the full multivariate Loewner matrix.
///
/// If `K = prod(k_l)` and `Q = prod(q_l)`, the result is `Q x K`.  For row
/// multi-index `I` and column multi-index `J`,
///
/// ```text
/// L[I, J] = (V[I] - W[J]) / prod_l (mu_l[I_l] - lambda_l[J_l]).
/// ```
///
/// `tab` is sampled on the per-axis `column ++ row` grids described by
/// [`Points`].  Only the all-column (`W`) and all-row (`V`) blocks are read.
/// Unlike the recursive construction this matrix is exponential in the number
/// of variables; callers should select it deliberately (or use the bounded
/// automatic policy in the high-level algorithms).
pub fn loewner_full<S: Scalar>(points: &Points<S>, tab: View<'_, S>) -> Mat<S> {
    let n = points.ndim();
    let k = points.column_sizes();
    let q = points.row_sizes();
    let ncols = grid_len(&k);
    let nrows = grid_len(&q);

    // Decode each Cartesian index once.  The MATLAB implementation rebuilds
    // these combinations inside nested loops; keeping the compact index tables
    // avoids that runaround while preserving canonical row-major ordering.
    let mut column_multi = vec![vec![0usize; n]; ncols];
    for (flat, multi) in column_multi.iter_mut().enumerate() {
        flat_to_multi(flat, &k, multi);
    }
    let mut row_multi = vec![vec![0usize; n]; nrows];
    for (flat, multi) in row_multi.iter_mut().enumerate() {
        flat_to_multi(flat, &q, multi);
    }

    let w: Vec<S> = column_multi
        .iter()
        .map(|idx| scalar_at(&tab, idx))
        .collect();
    let mut tab_idx = vec![0usize; n];
    let v: Vec<S> = row_multi
        .iter()
        .map(|idx| {
            for axis in 0..n {
                tab_idx[axis] = k[axis] + idx[axis];
            }
            scalar_at(&tab, &tab_idx)
        })
        .collect();

    // Cache all one-dimensional reciprocal differences.  A matrix entry then
    // costs only `n` multiplications plus the value subtraction.
    let cauchy: Vec<Vec<S>> = (0..n)
        .map(|axis| {
            let ka = k[axis];
            let mut factor = vec![S::zero(); q[axis] * ka];
            for (ri, &mu) in points.row[axis].iter().enumerate() {
                for (ci, &lambda) in points.column[axis].iter().enumerate() {
                    factor[ri * ka + ci] = S::one() / (mu - lambda);
                }
            }
            factor
        })
        .collect();

    Mat::from_fn(nrows, ncols, |row, col| {
        let mut inv_den = S::one();
        for axis in 0..n {
            inv_den *= cauchy[axis][row_multi[row][axis] * k[axis] + column_multi[col][axis]];
        }
        (v[row] - w[col]) * inv_den
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::sample;
    use approx::assert_relative_eq;

    #[test]
    fn entries_match_definition() {
        let lambda = [1.0, 2.0];
        let w = [10.0, 20.0];
        let mu = [3.0, 4.0, 5.0];
        let v = [30.0, 40.0, 50.0];
        let l = loewner_1d(&lambda, &w, &mu, &v);
        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 2);
        for i in 0..3 {
            for j in 0..2 {
                assert_relative_eq!(l[(i, j)], (v[i] - w[j]) / (mu[i] - lambda[j]));
            }
        }
    }

    #[test]
    fn full_entries_match_multivariate_definition() {
        let points = Points::new(
            vec![vec![0.0, 2.0], vec![-1.0, 1.0]],
            vec![vec![0.5, 3.0], vec![-0.5, 2.0]],
        )
        .unwrap();
        let f = |x: &[f64]| x[0] * x[1] + 2.0 * x[0] - x[1];
        let tab = sample(&f, &points);
        let l = loewner_full(&points, tab.view());

        assert_eq!((l.nrows(), l.ncols()), (4, 4));
        let rows = [[0usize, 0usize], [0, 1], [1, 0], [1, 1]];
        let cols = [[0usize, 0usize], [0, 1], [1, 0], [1, 1]];
        for (ri, r) in rows.iter().enumerate() {
            let mu = [points.row[0][r[0]], points.row[1][r[1]]];
            for (ci, c) in cols.iter().enumerate() {
                let lambda = [points.column[0][c[0]], points.column[1][c[1]]];
                let expected = (f(&mu) - f(&lambda)) / ((mu[0] - lambda[0]) * (mu[1] - lambda[1]));
                assert_relative_eq!(l[(ri, ci)], expected, epsilon = 1e-14);
            }
        }
    }
}
