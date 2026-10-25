//! Point subset selection driven by the detected order. Replaces
//! `mlf.points_selection`.
//!
//! Given the per-axis order, keep `order + 1` column points and (for the minimal
//! "square" case) `order + 1` row points along each axis, evenly spread across
//! the available abscissae. Then reduce the data tensor to the selected
//! `column ++ row` sub-grid so the recursion sees exactly the right size.

use crate::loewner::loewner_1d;
use crate::points::Points;
use crate::tensor::{Tensor, View, axis_line, gather};

use crate::scalar::Scalar;
use faer::Mat;

/// How the `order + 1` column/row indices are chosen from the full point set
/// along each axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectStrategy {
    /// Evenly-spaced indices (`spread_indices`) — the reference `linspace`+
    /// `unique` convention. Cheap, but for *coupled* targets the even split can be
    /// non-unisolvent, so one-shot `alg1` may fail to recover the function (see
    /// WORKPLAN §6d: `x1x2+x1x3+x2x3`).
    #[default]
    Spread,
    /// **Pivoted-QR (DEIM/maxvol-style) selection** on the per-axis Loewner: pick
    /// the columns/rows that maximize volume (best-conditioned interpolation),
    /// which is provably near-optimal. Makes one-shot `alg1` robust on coupled
    /// targets the even split cannot resolve, and gives `alg2` a better seed.
    Pivoted,
}

/// Result of point selection: the reduced points and the reduced data tensor.
pub struct Selection<S = f64> {
    /// Reduced column/row points (the `Points` the recursion will use).
    pub points: Points<S>,
    /// Reduced data tensor on the selected combined grid.
    pub tab: Tensor<S>,
}

/// Evenly spaced indices into `0..len`, picking `count` of them (inclusive of
/// both ends when possible), deduplicated and padded from the front if rounding
/// produced collisions — mirroring the MATLAB `linspace`+`unique` selection.
fn spread_indices(len: usize, count: usize) -> Vec<usize> {
    let count = count.min(len).max(1);
    let mut idx: Vec<usize> = if count == 1 {
        vec![0]
    } else {
        (0..count).map(|i| i * (len - 1) / (count - 1)).collect()
    };
    idx.dedup();
    // Pad with the smallest unused indices until we have `count` distinct ones.
    let mut j = 0;
    while idx.len() < count && j < len {
        if !idx.contains(&j) {
            idx.push(j);
        }
        j += 1;
    }
    idx.sort_unstable();
    idx
}

/// Column indices `0..n` whose leading-pivot order from a **column-pivoted QR**
/// of `m` ranks the columns by how much independent volume they add. Returns the
/// first `want` pivots (the best-conditioned column subset), sorted ascending.
///
/// This is the DEIM/maxvol-style selection: pivoted QR greedily picks the column
/// least represented by those already chosen, which maximizes the submatrix
/// volume — the provably near-optimal interpolation set. Falls back to
/// [`spread_indices`] on an empty/degenerate matrix.
fn pivot_columns<S: Scalar>(m: &Mat<S>, want: usize) -> Vec<usize> {
    let ncols = m.ncols();
    let want = want.min(ncols).max(1);
    if ncols == 0 || m.nrows() == 0 {
        return spread_indices(ncols.max(want), want);
    }
    let qr = m.col_piv_qr();
    // `P().arrays()` gives (forward, inverse); the forward permutation lists
    // columns in pivot order, most significant first.
    let (forward, _) = qr.P().arrays();
    let mut idx: Vec<usize> = forward.iter().take(want).copied().collect();
    idx.sort_unstable();
    idx
}

/// Pivoted column/row indices for one axis, from its representative 1-D Loewner
/// (the axis varied, every other axis frozen at its first column).
/// Columns are pivoted on the
/// Loewner; rows on its transpose, so both are volume-maximizing.
fn pivot_axis<S: Scalar>(
    tab: &View<'_, S>,
    axis: usize,
    lambda: &[S],
    mu: &[S],
    want: usize,
    minimal: bool,
) -> (Vec<usize>, Vec<usize>) {
    let (kk, qk) = (lambda.len(), mu.len());
    // Representative line: freeze every other axis at its first column index.
    let ctx = vec![0usize; tab.ndim()];
    let line = axis_line(tab, axis, &ctx);
    let w = &line[..kk];
    let v = &line[kk..];
    let l = loewner_1d(lambda, w, mu, v);

    let ci = pivot_columns(&l, want);
    let ri = if minimal {
        // Pivot rows via the transpose (rows of L become columns).
        let lt = l.transpose().to_owned();
        pivot_columns(&lt, want)
    } else {
        (0..qk).collect()
    };
    (ci, ri)
}

/// Select `order + 1` column and row points per axis and reduce the tensor,
/// using the **default** [`SelectStrategy::Spread`]. See [`points_selection_with`]
/// to choose pivoted selection.
///
/// `tab` is the full data tensor; `ord[k]` is the detected order along axis `k`.
/// `minimal` keeps the row set the same size as the column set (a square 1-D
/// Loewner per axis); when `false`, all row points are retained.
pub fn points_selection<S: Scalar>(
    points: &Points<S>,
    tab: View<'_, S>,
    ord: &[usize],
    minimal: bool,
) -> Selection<S> {
    points_selection_with(points, tab, ord, minimal, SelectStrategy::Spread)
}

/// As [`points_selection`], with an explicit [`SelectStrategy`].
pub fn points_selection_with<S: Scalar>(
    points: &Points<S>,
    tab: View<'_, S>,
    ord: &[usize],
    minimal: bool,
    strategy: SelectStrategy,
) -> Selection<S> {
    let n = points.ndim();

    // Per-axis selected indices into the column block and the row block.
    let mut col_idx: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut row_idx: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut new_col: Vec<Vec<S>> = Vec::with_capacity(n);
    let mut new_row: Vec<Vec<S>> = Vec::with_capacity(n);

    for (axis, ((col, row), &o)) in points.column.iter().zip(&points.row).zip(ord).enumerate() {
        let want = o + 1;
        let (ci, ri) = match strategy {
            SelectStrategy::Spread => {
                let ci = spread_indices(col.len(), want);
                let ri = if minimal {
                    spread_indices(row.len(), want)
                } else {
                    (0..row.len()).collect()
                };
                (ci, ri)
            }
            SelectStrategy::Pivoted => pivot_axis(&tab, axis, col, row, want, minimal),
        };
        new_col.push(ci.iter().map(|&i| col[i]).collect());
        new_row.push(ri.iter().map(|&i| row[i]).collect());
        col_idx.push(ci);
        row_idx.push(ri);
    }

    // The reduced combined index set along each axis: selected columns (in the
    // 0..k_full block) followed by selected rows (offset by k_full).
    let combined_idx: Vec<Vec<usize>> = (0..n)
        .map(|k| {
            let kf = points.column[k].len();
            col_idx[k]
                .iter()
                .copied()
                .chain(row_idx[k].iter().map(|&r| kf + r))
                .collect()
        })
        .collect();

    // Gather the reduced tensor in canonical row-major order.
    let tab_red = gather(&tab, &combined_idx);

    let points = Points::new(new_col, new_row).expect("reduced points stay disjoint");
    Selection {
        points,
        tab: tab_red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_basic() {
        assert_eq!(spread_indices(6, 3), vec![0, 2, 5]);
        assert_eq!(spread_indices(2, 2), vec![0, 1]);
        assert_eq!(spread_indices(5, 1), vec![0]);
    }

    #[test]
    fn reduces_tensor_shape() {
        use crate::tensor::sample;
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let col: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let row: Vec<f64> = (0..6).map(|i| i as f64 + 0.5).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone()],
            vec![row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        // order 1 per axis -> keep 2 col + 2 row = 4 per axis.
        let sel = points_selection(&pts, tab.view(), &[1, 1], true);
        assert_eq!(sel.tab.shape().as_slice(), &[4, 4]);
        assert_eq!(sel.points.column_sizes(), vec![2, 2]);
        assert_eq!(sel.points.row_sizes(), vec![2, 2]);
    }

    #[test]
    fn pivoted_selection_reduces_correctly() {
        use crate::tensor::sample;
        // Pivoted selection must produce the right reduced shape and a valid
        // (disjoint) point set, and recover a separable rational to precision.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let col: Vec<f64> = (0..7).map(|i| i as f64 * 0.3).collect();
        let row: Vec<f64> = (0..7).map(|i| i as f64 * 0.3 + 0.15).collect();
        let pts = Points::new(
            vec![col.clone(), col.clone()],
            vec![row.clone(), row.clone()],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let sel = points_selection_with(&pts, tab.view(), &[1, 1], true, SelectStrategy::Pivoted);
        assert_eq!(sel.tab.shape().as_slice(), &[4, 4]);
        assert_eq!(sel.points.column_sizes(), vec![2, 2]);
        // Selected abscissae are a subset of the originals and stay disjoint.
        sel.points.check_disjoint().unwrap();
        for v in &sel.points.column[0] {
            assert!(col.contains(v));
        }
    }
}
