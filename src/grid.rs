//! Multi-index <-> flat-index ordering: the one canonical convention the whole
//! crate agrees on.
//!
//! The MATLAB reference flattens an `n`-dimensional grid with **row-major**
//! (C-order) ordering: the *last* axis varies fastest, the *first* axis slowest.
//! This is exactly what `mlf.combinations_dim(dims)` returns (verified against
//! Octave). We adopt the same order so that the coefficient vector `c`, the
//! value vector `w`, and the evaluation grid all line up index-for-index.
//!
//! Crucially, choosing row-major storage for the data tensor makes this order
//! identical to the tensor's natural memory layout. That removes the need for
//! MATLAB's `mat2vec` / `vec2mat` round-trips and lets the recursion slice the
//! leading axis as a contiguous window with zero copies.

/// Total number of grid points for the given per-axis sizes.
#[inline]
pub fn grid_len(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Convert a row-major flat index into its multi-index, writing into `out`.
///
/// `out.len()` must equal `dims.len()`.
#[inline]
pub fn flat_to_multi(mut flat: usize, dims: &[usize], out: &mut [usize]) {
    debug_assert_eq!(out.len(), dims.len());
    for (slot, &d) in out.iter_mut().zip(dims).rev() {
        *slot = flat % d;
        flat /= d;
    }
}

/// Convert a multi-index into its row-major flat index.
#[inline]
pub fn multi_to_flat(multi: &[usize], dims: &[usize]) -> usize {
    debug_assert_eq!(multi.len(), dims.len());
    let mut flat = 0;
    for (&i, &d) in multi.iter().zip(dims) {
        flat = flat * d + i;
    }
    flat
}

/// Iterator over all multi-indices in row-major order.
///
/// Yields `&[usize]` of length `dims.len()`; the slice is reused between
/// iterations (zero allocation per step). This replaces MATLAB's materialized
/// `combinations_dim` matrix in every hot loop.
pub struct MultiIndices<'a> {
    dims: &'a [usize],
    cur: Vec<usize>,
    remaining: usize,
    first: bool,
}

impl<'a> MultiIndices<'a> {
    /// Create an iterator over the row-major multi-indices of `dims`.
    pub fn new(dims: &'a [usize]) -> Self {
        MultiIndices {
            dims,
            cur: vec![0; dims.len()],
            remaining: grid_len(dims),
            first: true,
        }
    }

    /// Advance the internal counter; returns `false` when exhausted.
    fn step(&mut self) -> bool {
        if self.remaining == 0 {
            return false;
        }
        if self.first {
            self.first = false;
            self.remaining -= 1;
            return true;
        }
        // Increment the last axis, carrying toward the first (row-major).
        for k in (0..self.dims.len()).rev() {
            self.cur[k] += 1;
            if self.cur[k] < self.dims[k] {
                self.remaining -= 1;
                return true;
            }
            self.cur[k] = 0;
        }
        self.remaining = 0;
        false
    }
}

impl Iterator for MultiIndices<'_> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step() {
            Some(self.cur.clone())
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

/// Materialize the full multi-index table in row-major order: row `i` is the
/// multi-index of flat index `i`. Mirrors `mlf.combinations_dim(dims)` (0-based).
///
/// Prefer [`MultiIndices`] in hot paths; this exists mainly for parity tests.
pub fn combinations(dims: &[usize]) -> Vec<Vec<usize>> {
    MultiIndices::new(dims).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_flat_multi() {
        let dims = [3, 2, 4];
        let n = grid_len(&dims);
        let mut multi = [0usize; 3];
        for flat in 0..n {
            flat_to_multi(flat, &dims, &mut multi);
            assert_eq!(multi_to_flat(&multi, &dims), flat);
        }
    }

    #[test]
    fn combinations_is_row_major() {
        // Matches Octave `mlf.combinations_dim([2 3])` (shifted to 0-based):
        // (0,0)(0,1)(0,2)(1,0)(1,1)(1,2)
        let got = combinations(&[2, 3]);
        let expect = vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
        ];
        assert_eq!(got, expect);
    }

    #[test]
    fn combinations_three_axes() {
        // Matches Octave `mlf.combinations_dim([2 2 2])` (0-based).
        let got = combinations(&[2, 2, 2]);
        let expect = vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
            vec![1, 0, 0],
            vec![1, 0, 1],
            vec![1, 1, 0],
            vec![1, 1, 1],
        ];
        assert_eq!(got, expect);
    }

    #[test]
    fn iterator_len_matches() {
        let dims = [4, 1, 3];
        assert_eq!(MultiIndices::new(&dims).count(), grid_len(&dims));
    }
}
