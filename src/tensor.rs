//! Data tensor: an rstsr C-order (row-major) tensor plus the sampling that
//! builds it.
//!
//! Storage, slicing and the faer bridge are delegated to rstsr — we do not
//! reimplement them. We keep row-major (`.c()`) layout so that the flat memory
//! order equals the canonical [`crate::grid`] order: the coefficient vector,
//! the value vector `w`, and the evaluation grid then all line up, and fixing
//! the leading axis is a zero-copy contiguous slice.
//!
//! Along each axis the sampling abscissae are `column ++ row` (first `k_l` are
//! column points, next `q_l` are row points), matching
//! [`crate::points::Points::combined_axis`].

use crate::grid::{flat_to_multi, grid_len};
use crate::points::Points;
use crate::scalar::Scalar;
use crate::source::TargetFn;

/// Check shape and finite samples before numerical factorizations. Data entry
/// points are strict; closure entry points may explicitly use continuation fill.
pub(crate) fn validate<S: Scalar>(tab: &Tensor<S>, points: &Points<S>) -> crate::Result<()> {
    let shape: Vec<usize> = points
        .column
        .iter()
        .zip(&points.row)
        .map(|(c, r)| c.len() + r.len())
        .collect();
    if tab.shape().as_slice() != shape.as_slice() {
        return Err(crate::LoewnerError::Degenerate(
            "tensor shape does not match the column ++ row grid".into(),
        ));
    }
    let view = tab.view();
    let data = contig_slice(&view);
    let count = data.iter().filter(|&&v| !v.finite()).count();
    if count > 0 {
        return Err(crate::LoewnerError::NonFinite {
            count,
            total: data.len(),
        });
    }
    Ok(())
}

use rstsr::prelude::rt::{self, DeviceFaer, IxD, RowMajor, asarray};
use rstsr::prelude::*;

use rayon::prelude::*;

/// Backing device. `DeviceFaer` gives us the zero-copy `into_faer()` bridge for
/// the 2-D Loewner SVDs and a parallel CPU backend.
pub type Device = DeviceFaer;

/// An owned row-major n-D tensor (rstsr-backed), defaulting to `f64`.
pub type Tensor<S = f64> = rt::Tensor<S, Device, IxD>;
/// A borrowing n-D view (always dynamic-dim: rstsr erases the static rank on
/// every slice).
pub type View<'a, S = f64> = rt::TensorView<'a, S, Device, IxD>;

/// Sample a target function on the full `column ++ row` grid of `points`,
/// producing the data tensor `tab_n` as a row-major rstsr tensor. Replaces
/// `mlf.make_tab`.
pub fn sample<S: Scalar, F: TargetFn<S> + Sync>(f: &F, points: &Points<S>) -> Tensor<S> {
    let axes: Vec<Vec<S>> = (0..points.ndim())
        .map(|l| points.combined_axis(l))
        .collect();
    let dims: Vec<usize> = axes.iter().map(Vec::len).collect();
    let total = grid_len(&dims);

    let data = sample_grid_par(f, &axes, &dims, total);
    asarray((data, dims, &row_major_device())).into_dim()
}

/// Evaluate `f` on the full row-major grid `axes` (shape `dims`, `total` points)
/// **in parallel**. Each grid point's coordinates are recovered from its flat
/// index independently, so the map is embarrassingly parallel; per-thread scratch
/// (`map_init`) avoids allocating an `x` buffer per point. For a large grid (the
/// `O(prod(k+q))` sampling that every method needs up front) this turns the
/// sample-fill into a parallel pass instead of a serial bottleneck.
fn sample_grid_par<S: Scalar, F: TargetFn<S> + Sync>(
    f: &F,
    axes: &[Vec<S>],
    dims: &[usize],
    total: usize,
) -> Vec<S> {
    let n = dims.len();
    (0..total)
        .into_par_iter()
        .map_init(
            || vec![S::zero(); n],
            |x, flat| {
                let mut rem = flat;
                for l in (0..n).rev() {
                    let d = dims[l];
                    x[l] = axes[l][rem % d];
                    rem /= d;
                }
                f.eval(x)
            },
        )
        .collect()
}

/// Sample like [`sample`], but **replace every non-finite sample** (NaN / ±Inf)
/// with the nearest finite neighbour (mean of the finite ones as fallback),
/// returning the sanitized tensor and the count replaced.
///
/// **Legacy / sanitization helper.** For *fitting*, prefer the per-slice
/// [`crate::decouple::NonFinite`] policies (`Excise` / `Pole`), which do not
/// fabricate interior values or destroy genuine poles. This fill is still used
/// internally to produce a finite tensor for **order detection / point selection**
/// (which only need a degree estimate, robust to a fill); the coefficient build
/// then re-samples raw and applies the chosen policy.
///
/// *Why:* a function can be non-finite at isolated points yet still admit a good
/// rational approximation by **analytic continuation** — `sinc(x) = sin(x)/x` is
/// `0/0 = NaN` at `0` but equals `1` there in the limit; `1/x` is `Inf` at `0` but
/// the barycentric form represents that pole natively. Killing the whole fit
/// because one grid point landed on the singularity throws away all the other
/// (perfectly good) data. Because the method requires a *dense* grid (no holes —
/// see WORKPLAN), we cannot drop the bad entries; instead we fill them with a
/// neutral finite value and let the rational interpolant continue through the
/// gap. Empirically the fit recovers the continued value at the singularity
/// (e.g. `sinc` → `1` at `0` to ~1e-14) because the surrounding samples pin the
/// function and the few filled points are a vanishing fraction of the constraint.
///
/// This is a heuristic, not a guarantee: if the non-finite region is large or
/// carries real structure, the fill biases the fit. Prefer choosing abscissae
/// that *avoid* the singularity (the alternating/Chebyshev builders usually do);
/// use this when the grid unavoidably hits one.
pub fn sample_continued<S: Scalar, F: TargetFn<S> + Sync>(
    f: &F,
    points: &Points<S>,
) -> (Tensor<S>, usize) {
    let axes: Vec<Vec<S>> = (0..points.ndim())
        .map(|l| points.combined_axis(l))
        .collect();
    let dims: Vec<usize> = axes.iter().map(Vec::len).collect();
    let total = grid_len(&dims);

    // Parallel sampling pass (the bottleneck for large grids), then a parallel
    // reduction for the finite mean. The neighbor-fill below stays sequential (its
    // outward search has data dependencies and only touches the few non-finite
    // entries — negligible unless almost everything is non-finite).
    let mut data = sample_grid_par(f, &axes, &dims, total);
    let (finite_sum, finite_cnt) = data
        .par_iter()
        .filter(|v| v.finite())
        .map(|&v| (v, 1usize))
        .reduce(|| (S::zero(), 0), |(s0, c0), (s1, c1)| (s0 + s1, c0 + c1));

    let mean = if finite_cnt > 0 {
        finite_sum / S::from_real(finite_cnt as f64)
    } else {
        S::zero()
    };
    let mut replaced = 0usize;
    for i in 0..total {
        if data[i].finite() {
            continue;
        }
        replaced += 1;
        // Search outward for the closest finite entry.
        let mut fill = mean;
        for d in 1..total {
            let lo = i.checked_sub(d).map(|j| data[j]);
            let hi = (i + d < total).then(|| data[i + d]);
            if let Some(v) = lo.filter(|v| v.finite()) {
                fill = v;
                break;
            }
            if let Some(v) = hi.filter(|v| v.finite()) {
                fill = v;
                break;
            }
            if lo.is_none() && hi.is_none() {
                break;
            }
        }
        data[i] = fill;
    }

    let tab = asarray((data, dims, &row_major_device())).into_dim();
    (tab, replaced)
}

/// A faer device pinned to row-major (C-order) default layout, so runtime-length
/// shapes build C-contiguous without the array-only `.c()` helper.
fn row_major_device() -> Device {
    let mut device = Device::default();
    device.set_default_order(RowMajor);
    device
}

/// Borrow a contiguous view's underlying data as a flat scalar slice.
///
/// The recursion only ever produces C-contiguous views (the whole tensor, or
/// leading-axis sub-blocks of it), so the data we need is the single contiguous
/// window `raw[offset .. offset + size]`. This is the bridge rstsr does not
/// expose as a one-liner but which the recursion leans on.
#[inline]
pub fn contig_slice<'a, S: Scalar>(v: &'a View<'_, S>) -> &'a [S] {
    debug_assert!(
        v.c_contig() || v.f_contig() || v.size() <= 1,
        "contig_slice called on a non-contiguous view"
    );
    let off = v.offset();
    &v.raw()[off..off + v.size()]
}

/// Fix the leading axis of `v` at index `i`, returning the contiguous
/// `(n-1)`-D sub-view. Thin wrapper over rstsr `.i(i)` naming the operation the
/// recursion relies on (`tab(i, :, ..., :)`).
#[inline]
pub fn index_leading<'a, S: Scalar>(v: &'a View<'_, S>, i: usize) -> View<'a, S> {
    v.i(i)
}

/// Fix several **leading** axes at once: `tab(idx[0], idx[1], ..., :, ..., :)`.
///
/// Reducing `idx.len()` leading axes in a single borrow (rstsr `slice([..])`)
/// avoids the borrow-rebinding that iterated single-axis slicing would require,
/// and stays zero-copy in row-major. `idx.len()` must be `< v.ndim()` to leave
/// at least one trailing axis (use [`scalar_at`] to read a full element).
#[inline]
pub fn index_leading_prefix<'a, S: Scalar>(v: &'a View<'_, S>, idx: &[usize]) -> View<'a, S> {
    v.slice(idx.to_vec())
}

/// Read a single element `tab(idx[0], ..., idx[n-1])` from a contiguous,
/// row-major view by direct offset arithmetic (no per-axis view machinery).
///
/// This is the efficient path the rstsr book recommends over elementwise `[]`
/// indexing in hot loops.
#[inline]
pub fn scalar_at<S: Scalar>(v: &View<'_, S>, idx: &[usize]) -> S {
    let dims: &[usize] = v.shape();
    debug_assert_eq!(idx.len(), dims.len());
    let data = contig_slice(v);
    let mut flat = 0usize;
    for (&i, &d) in idx.iter().zip(dims) {
        flat = flat * d + i;
    }
    data[flat]
}

/// Gather a sub-tensor of `v` selecting, along each axis `k`, the source indices
/// `pick[k]` (in the given order), into a fresh contiguous row-major [`Tensor`].
///
/// This is the one place index-driven tensor reduction lives: both order-based
/// point selection ([`crate::select`]) and the greedy iteration
/// ([`crate::algo::alg2()`]) reduce the data tensor to an arbitrary per-axis index
/// subset, so they share this. The result's shape is `pick.map(len)`, filled in
/// canonical row-major order.
pub fn gather<S: Scalar>(v: &View<'_, S>, pick: &[Vec<usize>]) -> Tensor<S> {
    let n = pick.len();
    debug_assert_eq!(n, v.ndim());
    let new_dims: Vec<usize> = pick.iter().map(Vec::len).collect();
    let total: usize = new_dims.iter().product();

    let mut data = Vec::with_capacity(total);
    let mut sub = vec![0usize; n];
    let mut src = vec![0usize; n];
    for flat in 0..total {
        flat_to_multi(flat, &new_dims, &mut sub);
        for k in 0..n {
            src[k] = pick[k][sub[k]];
        }
        data.push(scalar_at(v, &src));
    }
    asarray((data, new_dims, &row_major_device())).into_dim()
}

/// Gather the 1-D line along `axis` with every other axis fixed by `ctx`.
///
/// `ctx` has length `n` and its `axis` entry is ignored; the returned vector
/// has length `dims[axis]`. In row-major this line is strided (stride = product
/// of the dims after `axis`), so we read it by direct offset arithmetic rather
/// than constructing a strided rstsr view per call — the hot path of order
/// detection.
pub fn axis_line<S: Scalar>(v: &View<'_, S>, axis: usize, ctx: &[usize]) -> Vec<S> {
    let dims: &[usize] = v.shape();
    debug_assert_eq!(ctx.len(), dims.len());
    let data = contig_slice(v);
    let stride: usize = dims[axis + 1..].iter().product();
    // Base flat offset with the axis-coordinate set to 0.
    let mut base = 0usize;
    for (l, &d) in dims.iter().enumerate() {
        let coord = if l == axis { 0 } else { ctx[l] };
        base = base * d + coord;
    }
    (0..dims[axis]).map(|i| data[base + i * stride]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_matches_manual_grid() {
        // f(x1,x2) = x1*10 + x2; verify against the row-major order.
        let pts = Points::new(vec![vec![1.0], vec![2.0, 3.0]], vec![vec![9.0], vec![8.0]]).unwrap();
        let tab = sample(&|x: &[f64]| x[0] * 10.0 + x[1], &pts);
        assert_eq!(tab.shape().as_slice(), &[2, 3]);
        let v = tab.view();
        let flat = contig_slice(&v);
        let expect = [
            1.0 * 10.0 + 2.0,
            1.0 * 10.0 + 3.0,
            1.0 * 10.0 + 8.0,
            9.0 * 10.0 + 2.0,
            9.0 * 10.0 + 3.0,
            9.0 * 10.0 + 8.0,
        ];
        assert_eq!(flat, &expect);
    }

    #[test]
    fn index_leading_is_contiguous_subblock() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let device = Device::default();
        let t: Tensor = asarray((data, [2, 3, 4].c(), &device)).into_dim();
        let v = t.view();
        let s = index_leading(&v, 1); // tab(1,:,:) -> 12..24
        assert_eq!(s.shape().as_slice(), &[3, 4]);
        assert_eq!(
            contig_slice(&s),
            &(12..24).map(|x| x as f64).collect::<Vec<_>>()[..]
        );
        let s2 = index_leading(&s, 2); // tab(1,2,:) -> 20..24
        assert_eq!(s2.shape().as_slice(), &[4]);
        assert_eq!(contig_slice(&s2), &[20.0, 21.0, 22.0, 23.0]);
    }
}
