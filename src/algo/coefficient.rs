//! Coefficient-construction policy shared by Algorithms 1 and 2.

use crate::error::Result;
use crate::loewner::loewner_full;
use crate::nullspace::NullSpace;
use crate::points::Points;
use crate::recurse::{RecResult, loewner_null_rec};
use crate::scalar::Scalar;
use crate::tensor::View;

/// How an `alg1`/`alg2` fit constructs its barycentric coefficient tensor.
///
/// This is independent of [`super::Candidates`]: `Candidates::Full` controls
/// which residual-grid points Algorithm 2 scans, whereas
/// `CoefficientMethod::Full` controls the Loewner null-space construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoefficientMethod {
    /// Recursive sequence of small one-dimensional null-space solves.  This is
    /// the scalable method and remains the default.
    #[default]
    Recursive,
    /// Form the complete `prod(q_l) x prod(k_l)` multivariate Loewner matrix and
    /// take its minimum right-singular vector.  This can better capture coupled
    /// data, but its storage and SVD cost grow exponentially with dimension.
    Full,
    /// Prefer the full construction while its matrix and SVD remain modest;
    /// otherwise use recursion.  The bounds are deliberately conservative so
    /// selecting `Auto` cannot accidentally launch a giant dense SVD.
    Auto,
}

// The dense matrix itself is at most 2 MiB at this bound.  More importantly, the
// column cap limits the cubic SVD work and its additional workspaces.
const AUTO_FULL_MAX_ENTRIES: u128 = 1 << 18;
const AUTO_FULL_MAX_COLUMNS: u128 = 256;

#[inline]
fn auto_uses_full<S: Scalar>(points: &Points<S>) -> bool {
    let columns = points
        .column_sizes()
        .into_iter()
        .map(|x| x as u128)
        .product::<u128>();
    let rows = points
        .row_sizes()
        .into_iter()
        .map(|x| x as u128)
        .product::<u128>();
    columns <= AUTO_FULL_MAX_COLUMNS
        && rows
            .saturating_mul(columns)
            .saturating_mul(std::mem::size_of::<S>() as u128)
            <= AUTO_FULL_MAX_ENTRIES * 8
}

/// Construct coefficients using `method`, returning the same compact result
/// shape as the recursive implementation.
pub(super) fn coefficients<S: Scalar, N: NullSpace<S>>(
    method: CoefficientMethod,
    points: &Points<S>,
    tab: View<'_, S>,
    null: &N,
) -> Result<RecResult<S>> {
    let use_full = match method {
        CoefficientMethod::Recursive => false,
        CoefficientMethod::Full => true,
        CoefficientMethod::Auto => auto_uses_full(points),
    };
    if !use_full {
        return loewner_null_rec(points, tab, null);
    }

    let matrix = loewner_full(points, tab);
    let flop = (matrix.nrows() as u128).saturating_mul((matrix.ncols() as u128).pow(2));
    let c = null.null(&matrix)?.vector;
    Ok(RecResult { c, flop })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_policy_is_bounded() {
        let small = Points::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.5, 1.5], vec![0.5, 1.5]],
        )
        .unwrap();
        assert!(auto_uses_full(&small));

        let axis: Vec<f64> = (0..17).map(|i| i as f64).collect();
        let row: Vec<f64> = (0..17).map(|i| i as f64 + 0.5).collect();
        let large = Points::new(vec![axis.clone(); 2], vec![row; 2]).unwrap();
        assert!(!auto_uses_full(&large)); // 17^2 columns exceeds the SVD cap.
    }
}
