//! Exact dense linear algebra over a [`Field`]: row-reduction, rank, and kernel.
//!
//! This is the exact-arithmetic replacement for the floating-point SVD used in
//! [`crate::recurse`] / [`crate::order`]. Over a field, rank is the number of
//! pivots (a *theorem*, not a `σ > tol` threshold), and the right null space is
//! read directly off the reduced row-echelon form — see `docs/EXACT_FIELDS.md`
//! §1. No tolerances, no conditioning concerns, no rank inflation.

use super::core::Field;

/// A dense `rows × cols` matrix, row-major, over a field `F`.
#[derive(Clone, Debug)]
pub struct Matrix<F: Field> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<F>, // row-major
}

impl<F: Field> Matrix<F> {
    /// Build from a flat row-major buffer.
    pub fn from_rows(rows: usize, cols: usize, data: Vec<F>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    /// Build from a closure `f(i, j)`.
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> &F {
        &self.data[i * self.cols + j]
    }

    /// Reduce to reduced row-echelon form in place, returning the list of pivot
    /// columns (in increasing order). The rank is `pivots.len()`.
    pub fn rref(&mut self) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut pr = 0; // current pivot row
        for col in 0..self.cols {
            if pr >= self.rows {
                break;
            }
            // Find a nonzero entry in this column at or below pr.
            let mut sel = None;
            for r in pr..self.rows {
                if !self.data[r * self.cols + col].is_zero() {
                    sel = Some(r);
                    break;
                }
            }
            let Some(sr) = sel else { continue };
            self.swap_rows(pr, sr);
            // Scale pivot row so the pivot is 1.
            let inv = self.data[pr * self.cols + col]
                .inv()
                .expect("pivot is nonzero");
            for j in 0..self.cols {
                let v = self.data[pr * self.cols + j].clone() * inv.clone();
                self.data[pr * self.cols + j] = v;
            }
            // Eliminate the pivot column from every other row.
            for r in 0..self.rows {
                if r == pr {
                    continue;
                }
                let factor = self.data[r * self.cols + col].clone();
                if factor.is_zero() {
                    continue;
                }
                for j in 0..self.cols {
                    let sub = factor.clone() * self.data[pr * self.cols + j].clone();
                    let v = self.data[r * self.cols + j].clone() - sub;
                    self.data[r * self.cols + j] = v;
                }
            }
            pivots.push(col);
            pr += 1;
        }
        pivots
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for j in 0..self.cols {
            self.data.swap(a * self.cols + j, b * self.cols + j);
        }
    }

    /// The rank (number of pivots) — exact.
    pub fn rank(&self) -> usize {
        self.clone().rref().len()
    }

    /// A basis for the right null space `{ x : A x = 0 }`, one vector per free
    /// column, each in the original column order. Exact and tolerance-free.
    ///
    /// For each free column `c`, the basis vector has `1` at `c`, `0` at the other
    /// free columns, and `−RREF[pivot_row(p), c]` at each pivot column `p`.
    pub fn null_space(&self) -> Vec<Vec<F>> {
        let mut m = self.clone();
        let pivots = m.rref();
        let is_pivot = {
            let mut v = vec![false; self.cols];
            for &p in &pivots {
                v[p] = true;
            }
            v
        };
        let mut basis = Vec::new();
        for free in 0..self.cols {
            if is_pivot[free] {
                continue;
            }
            let mut x = vec![F::zero(); self.cols];
            x[free] = F::one();
            for (prow, &pcol) in pivots.iter().enumerate() {
                // x[pcol] = − RREF[prow, free]
                x[pcol] = -m.data[prow * m.cols + free].clone();
            }
            basis.push(x);
        }
        basis
    }
}

#[cfg(test)]
mod tests {
    use super::super::field::Fp;
    use super::super::rational::Rational;
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::integer(n)
    }

    #[test]
    fn rank_and_kernel_over_q() {
        // [[1,2,3],[2,4,6]] has rank 1, 2-dim kernel.
        let m = Matrix::from_rows(2, 3, vec![r(1), r(2), r(3), r(2), r(4), r(6)]);
        assert_eq!(m.rank(), 1);
        let ns = m.null_space();
        assert_eq!(ns.len(), 2);
        // Every basis vector is annihilated.
        for v in &ns {
            for i in 0..m.rows {
                let mut acc = Rational::zero();
                for (j, vj) in v.iter().enumerate() {
                    acc = acc + m.at(i, j).clone() * vj.clone();
                }
                assert!(acc.is_zero());
            }
        }
    }

    #[test]
    fn one_dim_kernel_is_the_lagrange_direction() {
        // A rank-2 3-column matrix → 1-dim kernel.
        let m = Matrix::from_rows(2, 3, vec![r(1), r(0), r(-1), r(0), r(1), r(-2)]);
        assert_eq!(m.rank(), 2);
        let ns = m.null_space();
        assert_eq!(ns.len(), 1);
    }

    #[test]
    fn kernel_over_fp() {
        let p = 7;
        let e = |v: i128| Fp::new(v, p);
        // Rows [1,1,1],[1,2,4] over F_7, rank 2 → 1-dim kernel.
        let m = Matrix::from_rows(2, 3, vec![e(1), e(1), e(1), e(1), e(2), e(4)]);
        assert_eq!(m.rank(), 2);
        let ns = m.null_space();
        assert_eq!(ns.len(), 1);
        for i in 0..2 {
            let mut acc = Fp::new(0, p);
            for (j, nj) in ns[0].iter().enumerate() {
                acc = acc + *m.at(i, j) * *nj;
            }
            assert!(acc.is_zero());
        }
    }
}
