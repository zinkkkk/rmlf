//! Variable decoupling for rational functions — the explicit, *computation-free*
//! Kolmogorov-Superposition-Theorem (KST) representation.
//!
//! Implements Antoulas, Gosea & Poussot-Vassal, *"Variable decoupling and the
//! Kolmogorov Superposition Theorem for rational functions"* (arXiv:2605.07246,
//! 2026) — the follow-up to the first paper (which this crate's [`crate::recurse`]
//! already realizes).
//!
//! ## What is new versus [`crate::recurse`]
//!
//! The recursion in [`crate::recurse`] builds **one** coefficient tensor `c` (the
//! barycentric *denominator* weights) and evaluates `N/D` with `N = c ⊙ w`
//! reconstructed from the sampled values `w`. The follow-up paper sharpens this in
//! two ways, both realized here:
//!
//! 1. **By inspection, no null-space solve.** For a 1-D rational `H = n/d` the
//!    Loewner null space is `Bary_den = [γ_1 d(λ_1), …, γ_k d(λ_k)]` (Theorem 2.1)
//!    and the numerator weights are `Bary_num = [γ_1 n(λ_1), …, γ_k n(λ_k)]`
//!    (Corollary 2.1), where `γ_j = 1 / ∏_{i≠j}(λ_j − λ_i)` are the Lagrange
//!    coefficients. Both are pure evaluations — no SVD. We compute them directly
//!    (see [`crate::nullspace::InspectNull`]).
//! 2. **Numerator and denominator decouple separately.** Instead of `N = c ⊙ w`,
//!    the method builds **two independent** Kronecker-structured coefficient
//!    tensors, `C_num` and `C_den` (Eq. 18–19), each a Hadamard/Kronecker product
//!    of per-axis weight vectors. `H(x) = (Σ C_num · S) / (Σ C_den · S)` with the
//!    Cauchy kernel `S` (Eq. 19). On the grid both agree with `c ⊙ w`, but the
//!    separated form is what exposes the **decoupled single-variable functions**
//!    `Φ_l`, `Φ̂_l` (the rational analogue of the KST inner/outer functions, which
//!    here are *explicit and non-fractal* — affine for affine-per-axis targets).
//!
//! ## Data-driven recipe (Appendix C)
//!
//! We only ever sample `H`, never `n` and `d` separately. The per-axis weight
//! vectors are recovered from **1-D slices** of `H`: along a coordinate line the
//! slice is a 1-D rational, whose denominator weights are the slice's Loewner null
//! space and whose numerator weights are that null space times the slice's column
//! values (`Bary_num = Bary_den ⊙ w_slice`, valid because `H(λ_j) = n(λ_j)/d(λ_j)`).
//! Because the null space is determined by `d` alone, we obtain it *by inspection*
//! by row elimination without an SVD — see [`crate::nullspace::InspectNull`]. The
//! coefficient tensors are then assembled by the same recursive Kronecker cascade
//! as [`crate::recurse`], tracking the numerator and denominator products in
//! lockstep.

use crate::error::{LoewnerError, Result};
use crate::eval::axis_weights;
use crate::order::OrderOptions;
use crate::points::{AxisSampling, AxisSpec, Points};
use crate::select::SelectStrategy;
use crate::source::TargetFn;
use crate::tensor::{Tensor, View, contig_slice, index_leading, sample_continued, scalar_at};

/// How [`decouple_opt`] places the support abscissae.
///
/// `Adaptive` is the recommended default: it starts from `start_nip` points per
/// axis and **grows only the under-resolved axes** (see [`decouple_adaptive`]),
/// keeping mixed-degree high-`n` targets tractable. The fixed modes place a chosen
/// number of points and do not grow.
///
/// **Picking a variant:**
/// * one rule everywhere → [`Sampling::Uniform`], e.g.
///   `Sampling::Uniform(AxisSampling::Integers)` or
///   `Sampling::Uniform(AxisSampling::Chebyshev(8))` — the everyday case;
/// * a per-axis *count* of a common rule → [`Sampling::Linspace`] /
///   [`Sampling::Chebyshev`] / [`Sampling::Logspace`] (each takes `Vec<usize>`);
/// * genuinely *different methods or column/row placement per axis* →
///   [`Sampling::PerAxis`] with [`AxisSpec`] (the advanced escape hatch).
// Not `PartialEq`: the `PerAxis` variant can carry custom-closure placements
// (`AxisSpec` / `Placement::Custom`), which are not comparable.
#[derive(Debug, Clone)]
pub enum Sampling {
    /// Per-axis adaptive growth from `start_nip`, capped at `max_nip` (the default).
    Adaptive { start_nip: usize, max_nip: usize },
    /// **The same rule on every axis** — the easy path. Pass one
    /// [`AxisSampling`] (e.g. `Sampling::Uniform(AxisSampling::Integers)` or
    /// `Sampling::Uniform(AxisSampling::Chebyshev(8))`) and it is applied to all
    /// `n` axes. This is the one-liner for the common "I just want integers /
    /// Chebyshev / logspace everywhere" case — no [`AxisSpec`] needed.
    Uniform(AxisSampling),
    /// Fixed linear (`linspace`) placement, `nip` points per axis.
    Linspace(Vec<usize>),
    /// Fixed Chebyshev placement, `nip` points per axis.
    Chebyshev(Vec<usize>),
    /// Fixed **logarithmic** (`logspace`) placement, `nip` points per axis. Each
    /// axis must avoid `0` / a sign change (see [`AxisSampling::Logspace`]).
    Logspace(Vec<usize>),
    /// The **original MATLAB reference** placement, `nip` points per axis:
    /// `column = linspace`, `row = column + step/2` (shifted rows). **Samples
    /// slightly outside the bounds** (the last row point is `hi + step/2`); see
    /// [`AxisSampling::Reference`]. For exact parity with the MATLAB results.
    Reference(Vec<usize>),
    /// **Different rule per axis** — full control. One [`AxisSpec`] per axis. Build
    /// it the easy way with [`Sampling::per_axis`], which takes bare
    /// [`AxisSampling`] rules (or [`AxisSpec`]s) without wrapping:
    /// `Sampling::per_axis([AxisSampling::Integers, AxisSampling::Chebyshev(5)])`.
    /// Use a full [`AxisSpec`] (via [`AxisSpec::split`] / [`AxisSpec::custom`]) only
    /// on the axes that genuinely need independent column/row placement. For one
    /// rule on *every* axis use [`Sampling::Uniform`].
    PerAxis(Vec<AxisSpec>),
}

impl Sampling {
    /// Build [`Sampling::PerAxis`] from a list of **per-axis rules**, accepting
    /// either bare [`AxisSampling`] rules or full [`AxisSpec`]s (anything
    /// `Into<AxisSpec>`) — so the simple "a different method per axis" case needs no
    /// `AxisSpec::new` wrapping:
    ///
    /// ```
    /// use rmlf::decouple::Sampling;
    /// use rmlf::points::{AxisSampling, AxisSpec};
    ///
    /// // Different built-in method per axis — the easy form.
    /// let s = Sampling::per_axis([AxisSampling::Integers, AxisSampling::Chebyshev(5)]);
    ///
    /// // Mix in a column/row split on one axis where you actually want it.
    /// let mixed = Sampling::per_axis([
    ///     AxisSpec::from(AxisSampling::Integers),
    ///     AxisSpec::split(AxisSampling::Chebyshev(8), AxisSampling::Linspace(7)),
    /// ]);
    /// # let _ = (s, mixed);
    /// ```
    pub fn per_axis<I>(rules: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<AxisSpec>,
    {
        Sampling::PerAxis(rules.into_iter().map(Into::into).collect())
    }
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling::Adaptive {
            start_nip: 6,
            max_nip: 16,
        }
    }
}

/// Configuration for [`decouple_opt`] — the configurable KST-decoupling front door,
/// mirroring [`crate::Alg1Options`] in style.
///
/// All fields have sensible defaults ([`Default`]), so `DecoupleOptions::default()`
/// reproduces the recommended adaptive path. Override only what you need:
///
/// ```
/// use rmlf::decouple::{decouple_opt, DecoupleOptions, Sampling};
/// use rmlf::points::AxisSampling;
///
/// let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
/// // One rule on every axis — the easy path (here: 8 Chebyshev points per axis).
/// let opts = DecoupleOptions {
///     sampling: Sampling::Uniform(AxisSampling::Chebyshev(8)),
///     ..Default::default()
/// };
/// let (model, _info) = decouple_opt(&f, &[(-1.0, 1.0); 3], &opts).unwrap();
/// ```
///
/// Only reach for [`Sampling::PerAxis`] (with [`AxisSpec`]) when axes need
/// *different* methods or independent column/row placement.
// Note: deliberately NOT `#[non_exhaustive]` — like `Alg1Options`, callers build
// it with struct-literal + `..Default::default()`. Adding a field with a default
// keeps `..Default::default()` callers compiling, which covers the common case.
#[derive(Debug, Clone, Default)]
pub struct DecoupleOptions {
    /// How the support abscissae are placed. Default: [`Sampling::Adaptive`].
    pub sampling: Sampling,
    /// Order-detection options (tolerance, `gap_stop` non-rational stabilization,
    /// per-axis caps). Default [`OrderOptions::default`].
    pub order: OrderOptions,
    /// How the `order + 1` support points are selected per axis. Default
    /// [`SelectStrategy::Spread`].
    pub select: SelectStrategy,
    /// How each 1-D slice's null vector (the barycentric denominator weights) is
    /// recovered. Default [`NullMethod::ByInspection`] — the paper's exact,
    /// SVD-free method. See [`NullMethod`] for when to switch.
    pub null_method: NullMethod,
    /// How non-finite samples (NaN / ±Inf) are handled when building the
    /// coefficient tensors. Default [`NonFinite::Raw`] — error out and tell the
    /// user to pick a handling policy, fabricating nothing by default. See
    /// [`NonFinite`].
    pub non_finite: NonFinite,
}

/// How [`decouple_opt`] handles **non-finite samples** (`NaN` / `±Inf`) when
/// building the coefficient tensors.
///
/// A target can be non-finite at isolated grid points yet still admit a good
/// rational fit — `sinc(0) = 0/0` is `NaN` but equals `1` in the limit; `1/x` is
/// `±Inf` at `0` but the barycentric form represents that pole natively. The mLF
/// needs a *dense* grid, so a bad point cannot simply be deleted from the tensor;
/// instead the active policies act **per 1-D Loewner slice**, leaving the tensor
/// dense. No policy ever fabricates a value except the explicit, legacy
/// [`NonFinite::Fill`].
///
/// - [`NonFinite::Raw`] (default) — do nothing special: if any sample is
///   non-finite the fit **errors** ([`LoewnerError::NonFinite`]) and asks the user
///   to choose `Excise` or `Pole`. The honest default — it never silently changes
///   your data.
/// - [`NonFinite::Excise`] — every non-finite sample (NaN *or* Inf) is dropped
///   from its 1-D Loewner row/column; the corresponding weight is set to `0`.
///   Treats a singular point as missing data.
/// - [`NonFinite::Pole`] — a `±Inf` sample at node `λ_j` means the denominator
///   vanishes there, so its denominator weight is set to `0` (a genuine pole,
///   exact for `1/x`-type behaviour) and the numerator value `n(λ_j)` is recovered
///   by **barycentric interpolation** through the finite nodes (stable, no extra
///   sampling); a `NaN` sample is excised as in `Excise`. The most faithful policy
///   for functions with real poles.
/// - [`NonFinite::Fill`] — the legacy heuristic: replace each non-finite sample
///   with the nearest finite neighbour (global finite mean as fallback) *before*
///   fitting. Fabricates interior values and destroys genuine poles; kept only for
///   backwards compatibility and the occasional removable-singularity case where
///   the neighbour happens to be the right limit.
///
/// For `Excise`/`Pole`, if excision would shrink a slice below the order it needs,
/// the affected entries are kept (not dropped) so the fit still has enough data —
/// the slice degrades gracefully rather than erroring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NonFinite {
    /// Error on any non-finite sample (the honest default).
    #[default]
    Raw,
    /// Any non-finite sample ⇒ excise from the slice (weight 0).
    Excise,
    /// `±Inf` ⇒ denominator pole (weight 0) + barycentric numerator recovery;
    /// `NaN` ⇒ excise.
    Pole,
    /// Replace non-finite samples with a neighbour/mean before fitting (legacy).
    Fill,
}

/// How [`decouple_opt`] recovers each 1-D slice's null vector — the barycentric
/// denominator weights of Theorem 2.1. **Neither option uses an SVD** (that is
/// reserved for the `alg*` algorithms); both are pure-arithmetic and always
/// produce a result regardless of the data.
///
/// The two methods agree exactly on genuinely rational/polynomial targets (where
/// the slice is rank-deficient and the null line is exact). They differ only in
/// the **approximate** regime — a non-rational target, where the slice is
/// numerically full rank and there is no true null:
///
/// - [`NullMethod::ByInspection`] is the paper's headline result: the null space
///   "written down by inspection" via a single Gaussian elimination. Exact and
///   fastest for rational data, and field-generic (only `+ − × ÷`). On a
///   non-rational slice it forces a null by elimination, which can be badly
///   ill-conditioned (huge coefficients, occasional non-finite eval).
/// - [`NullMethod::MinNorm`] returns the *minimum-norm* approximate null via
///   inverse iteration on `LᵀL` — the same direction an SVD would give, but with
///   no SVD. Unit-norm, so coefficients stay O(1); markedly more robust on
///   non-rational targets, at a small extra cost. Choose this for approximating
///   smooth-but-non-rational functions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NullMethod {
    /// Exact, SVD-free Gaussian elimination (Theorem 2.1). The default.
    #[default]
    ByInspection,
    /// Minimum-norm approximate null via inverse iteration (no SVD). More robust
    /// for non-rational approximation.
    MinNorm,
}

/// Build a decoupled rational model of `f`, **configurably** — the front door that
/// lets the user pick the sampling mode, order-detection options, and selection
/// strategy, with defaults that reproduce the recommended adaptive path.
///
/// This unifies [`decouple`] (fixed nip) and [`decouple_adaptive`] (per-axis
/// growth) under one [`DecoupleOptions`], matching [`crate::alg1()`]'s ergonomics.
/// `DecoupleOptions::default()` ⇒ adaptive sampling, standard order detection,
/// spread selection.
pub fn decouple_opt<F: TargetFn + Sync>(
    f: &F,
    bounds: &[(f64, f64)],
    opts: &DecoupleOptions,
) -> Result<(DecoupledModel, DecoupleInfo)> {
    use crate::order::compute_order_report;
    use crate::select::points_selection_with;

    let to_even = |v: usize| if v.is_multiple_of(2) { v } else { v + 1 };

    // Resolve sampling → a final (points, tab, detected order). The adaptive mode
    // grows under-resolved axes; the fixed modes build once.
    let (points, tab, order) = match &opts.sampling {
        Sampling::Adaptive { start_nip, max_nip } => {
            let n = bounds.len();
            let mut nip: Vec<usize> = vec![to_even((*start_nip).max(4)); n];
            loop {
                let points = Points::alternating(bounds, &nip)?;
                let (tab, _) = sample_continued(f, &points);
                let report = compute_order_report(&points, tab.view(), &opts.order);
                let mut grew = false;
                for (nl, &ol) in nip.iter_mut().zip(&report.order) {
                    let budget = *nl / 2;
                    if ol + 1 >= budget && *nl < *max_nip {
                        *nl = to_even((*nl * 2).min(*max_nip));
                        grew = true;
                    }
                }
                if !grew {
                    break (points, tab, report.order);
                }
            }
        }
        fixed => {
            let points = match fixed {
                Sampling::Uniform(rule) => {
                    let rules = vec![*rule; bounds.len()];
                    Points::sampled_uniform(bounds, &rules)?
                }
                Sampling::Linspace(nip) => Points::alternating(bounds, nip)?,
                Sampling::Chebyshev(nip) => Points::chebyshev(bounds, nip)?,
                Sampling::Logspace(nip) => {
                    let rules: Vec<AxisSampling> =
                        nip.iter().map(|&m| AxisSampling::Logspace(m)).collect();
                    Points::sampled_uniform(bounds, &rules)?
                }
                Sampling::Reference(nip) => Points::reference(bounds, nip)?,
                Sampling::PerAxis(specs) => Points::sampled(bounds, specs)?,
                Sampling::Adaptive { .. } => unreachable!(),
            };
            let (tab, _) = sample_continued(f, &points);
            let order = crate::order::compute_order(&points, tab.view(), &opts.order);
            (points, tab, order)
        }
    };

    let sel = points_selection_with(&points, tab.view(), &order, true, opts.select);

    // Order detection and point selection run on the sanitized `tab` (robust to a
    // fill). The coefficient build, however, must see the *raw* values so the
    // chosen `NonFinite` policy acts on genuine NaN/±Inf — except for `Fill`, which
    // by definition wants the already-filled samples. Re-sample raw on the small,
    // selected point set (cheap — it is `order+1` points per axis).
    if opts.non_finite == NonFinite::Fill {
        return decouple_from_tensor_with(&sel.tab, &sel.points, opts.null_method, NonFinite::Fill);
    }
    let raw = crate::tensor::sample(f, &sel.points);
    decouple_from_tensor_with(&raw, &sel.points, opts.null_method, opts.non_finite)
}

/// Lagrange coefficients `γ_j = 1 / ∏_{i≠j}(λ_j − λ_i)` for the abscissae `lambda`.
///
/// These depend only on the column support points; they are the partition-of-unity
/// scaling of the normalized Lagrange basis (`ℓ_j = γ_j ∏_{i≠j}(s − λ_i)`) and the
/// `γ` of Theorem 2.1. `lambda` must have distinct entries (caller-guaranteed by
/// [`Points::check_disjoint`] within each set).
pub fn lagrange_coeffs(lambda: &[f64]) -> Vec<f64> {
    lambda
        .iter()
        .enumerate()
        .map(|(j, &lj)| {
            let prod: f64 = lambda
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != j)
                .map(|(_, &li)| lj - li)
                .product();
            1.0 / prod
        })
        .collect()
}

/// Normalize a weight vector in place by a chosen pivot entry.
///
/// `pivot` is `Some(i)` to divide by entry `i` (the paper's last-entry convention
/// is `Some(k−1)`), or `None` to leave the vector un-normalized (the leading
/// variable, per Eq. 18).
fn normalize_pivot(v: &mut [f64], pivot: Option<usize>) {
    if let Some(p) = pivot {
        let d = v[p];
        if d != 0.0 {
            for x in v.iter_mut() {
                *x /= d;
            }
        }
    }
}

/// A built decoupled rational model: the column support points and the two
/// **independent** coefficient tensors (numerator and denominator), both in the
/// canonical [`crate::grid`] row-major order.
///
/// Evaluation is `N(x)/D(x)` with `N = Σ_J C_num[J] ∏_l 1/(x_l−λ_l)` and
/// `D = Σ_J C_den[J] ∏_l 1/(x_l−λ_l)` (Eq. 19). On node coincidence the removable
/// singularity is handled exactly as in [`crate::eval::Barycentric`] (the limit
/// keeps only the surviving slice).
#[derive(Debug, Clone)]
pub struct DecoupledModel {
    /// Column support points per axis.
    pub lambda: Vec<Vec<f64>>,
    /// Numerator coefficient tensor `C_num` (canonical order).
    pub c_num: Vec<f64>,
    /// Denominator coefficient tensor `C_den` (canonical order).
    pub c_den: Vec<f64>,
}

impl DecoupledModel {
    /// Number of variables `n`.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.lambda.len()
    }

    /// Per-axis number of support nodes `k_l`.
    #[inline]
    pub fn sizes(&self) -> Vec<usize> {
        self.lambda.iter().map(Vec::len).collect()
    }

    /// Evaluate the decoupled rational `H(x) = N(x)/D(x)` at `x` (length `n`).
    ///
    /// Guards node coincidence exactly as [`crate::eval::Barycentric::eval`] does
    /// (WORKPLAN §6g): when `x` lands on a support node along **every** axis the
    /// barycentric limit is the sampled value `C_num[J]/C_den[J] = H(λ_J)` —
    /// returned directly. This is essential because a degenerate fit (e.g. of a
    /// piecewise-constant target like `round`-ed modular multiplication) can have a
    /// **zero coefficient** `C_den[J]` at most nodes, making the contraction's `0/0`
    /// produce `NaN`; the fast path (and the trailing `0/0 → 0` guard for partial
    /// node coincidence) keep eval finite on every in-domain query. A genuine pole
    /// (`num ≠ 0`, `den = 0`) still yields `±Inf`. (Previously only the
    /// `Barycentric` / `CompressedBarycentric` paths were guarded, which is why
    /// `to_barycentric().eval(..)` returned a value where `DecoupledModel::eval`
    /// returned `NaN`.)
    pub fn eval(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), self.ndim());
        if let Some(j) = self.full_node_index(x) {
            let (cn, cd) = (self.c_num[j], self.c_den[j]);
            if cd != 0.0 {
                return cn / cd;
            }
            // Zero-coefficient node: no interpolation mass here. Fall through to the
            // contraction, whose `0/0` is caught below (degenerate fit).
        }
        let dims = self.sizes();
        let weights = axis_weights(&self.lambda, x);
        let mut num = self.c_num.clone();
        let mut den = self.c_den.clone();
        let n = contract_rank1(&mut num, &dims, &weights);
        let d = contract_rank1(&mut den, &dims, &weights);
        if d == 0.0 && n == 0.0 {
            return 0.0;
        }
        n / d
    }

    /// If `x` coincides with a support node along **every** axis, return the flat
    /// index `J` of that grid point (canonical row-major order); else `None`.
    /// Mirrors [`crate::eval::Barycentric`]'s helper.
    fn full_node_index(&self, x: &[f64]) -> Option<usize> {
        let mut flat = 0usize;
        for (lam, &xl) in self.lambda.iter().zip(x) {
            let j = lam.iter().position(|&v| v == xl)?;
            flat = flat * lam.len() + j;
        }
        Some(flat)
    }

    /// Evaluate at many points (one value per row of `xs`).
    pub fn eval_grid(&self, xs: &[&[f64]]) -> Vec<f64> {
        xs.iter().map(|x| self.eval(x)).collect()
    }

    /// Convert to the standard [`crate::eval::Barycentric`] model.
    ///
    /// The denominator coefficients are `c = C_den`; the per-node values are
    /// `w_J = C_num[J] / C_den[J]` (which equals `H(λ_J)` exactly on the grid). This
    /// lets a decoupled fit reuse every [`crate::eval::Barycentric`] facility
    /// (batched/parallel `eval_grid`, the TT-compression path, the denominator
    /// diagnostic).
    pub fn to_barycentric(&self) -> crate::eval::Barycentric {
        let w: Vec<f64> = self
            .c_num
            .iter()
            .zip(&self.c_den)
            .map(|(&n, &d)| if d != 0.0 { n / d } else { 0.0 })
            .collect();
        crate::eval::Barycentric {
            lambda: self.lambda.clone(),
            w,
            c: self.c_den.clone(),
        }
    }

    /// The **decoupled single-variable functions** — the rational analogue of the
    /// KST inner/outer functions, made explicit (Section 3 / Eq. 18).
    ///
    /// The paper's central qualitative claim is that an `n`-variable rational
    /// decouples into a sum of products of *genuinely single-variable* functions,
    /// and that — unlike classical KST's fractal inner functions — these are
    /// **explicit, smooth, low-degree** (affine for an affine-per-axis target). For
    /// degrees `(k_1−1, …, k_n−1)` there are exactly
    /// `1 + k_1 + k_1 k_2 + … + k_1⋯k_{n−1}` of them ([`DecoupleInfo::n_functions`];
    /// 7 for Pólya–Szegő).
    ///
    /// Each such function is a **grid-line slice** of the model: axis `l`'s
    /// functions are `x_l ↦ H(λ_{0,j_0}, …, λ_{l−1,j_{l−1}}, x_l, λ_{l+1,last}, …,
    /// λ_{n−1,last})`, indexed by the leading multi-index `(j_0,…,j_{l−1})` (Pólya–
    /// Szegő: `ϕ(x)=p(x,y₂,z₂)`, `ψ_{j}(y)=p(x_j,y,z₂)`, `ω_{j,k}(z)=p(x_j,y_k,z)`).
    /// This returns, for every axis `l`, the list of those slices as
    /// [`DecoupledFn`]s — each a 1-D barycentric rational evaluable in closed form,
    /// carrying its leading index. Summing the Hadamard/Kronecker products of these
    /// (per Eq. 17/19) reproduces `H`; the structure is exposed for inspection,
    /// symbolic export, or feeding a downstream 1-D approximator.
    pub fn decoupled_functions(&self) -> Vec<Vec<DecoupledFn>> {
        let n = self.ndim();
        let k = self.sizes();
        let bary = self.to_barycentric();
        let mut out = Vec::with_capacity(n);
        for l in 0..n {
            // Leading multi-index ranges over k_0 × … × k_{l-1}.
            let lead_dims = &k[..l];
            let count: usize = lead_dims.iter().product::<usize>().max(1);
            let mut fns = Vec::with_capacity(count);
            let mut lead = vec![0usize; l];
            for flat in 0..count {
                crate::grid::flat_to_multi(flat, lead_dims, &mut lead);
                fns.push(self.single_var_fn(&bary, l, &lead));
            }
            out.push(fns);
        }
        out
    }

    /// One decoupled single-variable function: the slice along `axis` with the
    /// leading axes pinned to `lead` (column indices) and the trailing axes pinned
    /// to their last column index. The returned [`DecoupledFn`] is a 1-D
    /// barycentric rational in `x_axis`.
    fn single_var_fn(
        &self,
        bary: &crate::eval::Barycentric,
        axis: usize,
        lead: &[usize],
    ) -> DecoupledFn {
        let lam_axis = self.lambda[axis].clone();
        // Sample the slice at this axis's own column nodes; the 1-D barycentric
        // weights of that data ARE the decoupled function's weights, by inspection.
        let mut x = vec![0.0f64; self.ndim()];
        for (j, &li) in lead.iter().enumerate() {
            x[j] = self.lambda[j][li];
        }
        for (l, xl) in x.iter_mut().enumerate().take(self.ndim()).skip(axis + 1) {
            *xl = *self.lambda[l].last().unwrap();
        }
        let values: Vec<f64> = lam_axis
            .iter()
            .map(|&xl| {
                x[axis] = xl;
                bary.eval(&x)
            })
            .collect();
        DecoupledFn {
            axis,
            lead: lead.to_vec(),
            lambda: lam_axis,
            values,
        }
    }
}

impl crate::model::Model for DecoupledModel {
    fn ndim(&self) -> usize {
        DecoupledModel::ndim(self)
    }
    fn eval(&self, x: &[f64]) -> f64 {
        DecoupledModel::eval(self, x)
    }
    fn eval_grid(&self, xs: &[&[f64]]) -> Vec<f64> {
        DecoupledModel::eval_grid(self, xs)
    }
}

/// One **decoupled factor** `Φ_l` from [`DecoupledModel::decoupled_functions`] —
/// the rational-KST single-variable function of the papers (Paper 1 §6 Eq. 23;
/// Paper 2 §4 Eq. 17). Explicit and smooth (degree ≤ the axis degree), never
/// fractal, unlike the classical KST inner functions.
///
/// **This is a single factor, not the whole function.** The full multivariate
/// model is the **superposition of Hadamard/Kronecker products** of these factors
/// across axes — `H = Σ_rows ∏_l Φ_l` (Eq. 17/19) — exactly the way the
/// Kolmogorov Superposition Theorem composes a multivariate function out of
/// single-variable pieces. A lone `DecoupledFn` evaluates only its own factor, not
/// `H` on a line.
///
/// (Contrast [`crate::Fibres::fibre`], which *does* evaluate the model restricted
/// to a line — a slice, not a factor. See `docs/MODELS.md`.)
#[derive(Debug, Clone)]
pub struct DecoupledFn {
    /// Which variable this function is a function of.
    pub axis: usize,
    /// The pinned column indices of the leading axes `0..axis` (empty for axis 0).
    pub lead: Vec<usize>,
    /// The axis's column support nodes.
    pub lambda: Vec<f64>,
    /// The function's values at those nodes (it interpolates these). Off-node
    /// values come from the [`Self::eval`] barycentric/Lagrange interpolant.
    pub values: Vec<f64>,
}

impl DecoupledFn {
    /// Evaluate the single-variable function at `xl` via the barycentric/Lagrange
    /// interpolant through `(lambda, values)`.
    pub fn eval(&self, xl: f64) -> f64 {
        // Lagrange interpolation through the nodes (exact: the slice is a low-degree
        // rational the model already represents, and on these nodes the values are
        // exact samples of it).
        if let Some(j) = self.lambda.iter().position(|&v| v == xl) {
            return self.values[j];
        }
        let k = self.lambda.len();
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..k {
            let gamma: f64 = (0..k)
                .filter(|&i| i != j)
                .map(|i| 1.0 / (self.lambda[j] - self.lambda[i]))
                .product();
            let w = gamma / (xl - self.lambda[j]);
            num += w * self.values[j];
            den += w;
        }
        num / den
    }
}

/// Contract a dense row-major tensor in `buf` (shape `dims`) against the rank-1
/// weight structure `⊗_l weights[l]`, in place, to the scalar
/// `Σ_J buf[J] ∏_l weights[l][j_l]`. (Dense special case shared with
/// [`crate::eval`]; duplicated here to keep the modules decoupled.)
fn contract_rank1(buf: &mut [f64], dims: &[usize], weights: &[Vec<f64>]) -> f64 {
    let mut block = buf.len();
    for (l, wl) in weights.iter().enumerate().rev() {
        let kl = dims[l];
        let outer = block / kl;
        for o in 0..outer {
            let base = o * kl;
            let mut s = 0.0;
            for (j, &w) in wl.iter().enumerate() {
                s += buf[base + j] * w;
            }
            buf[o] = s;
        }
        block = outer;
    }
    buf[0]
}

/// Per-slice barycentric weights, by inspection — the data-driven core.
///
/// Given a 1-D slice of `H` sampled at the combined `column ++ row` abscissae of
/// axis `l`, return `(num, den)`: the length-`k_l` numerator and denominator
/// barycentric weight vectors. The denominator weights are the slice's Loewner null
/// space, which by Theorem 2.1 is `Bary_den = Δ_λ γ = [γ_j d(λ_j)]`. We recover the
/// `d(λ_j)` ratios *from the data* without ever forming `d`: the unique (up to
/// scale) null vector of the `q×k` Loewner matrix `L_{ij} = (v_i − w_j)/(μ_i − λ_j)`
/// is `[γ_j d(λ_j)]`, and we obtain it **by inspection** — a single Gaussian
/// elimination ([`crate::nullspace::InspectNull`]), never an SVD. This is the
/// paper's central finding ("no computation required"). It always runs and always
/// returns weights regardless of the data: exact for a rational slice, the
/// approximate decoupling otherwise. The numerator weights are then `den ⊙ w`
/// (Corollary 2.1).
///
/// `pivot` normalizes both vectors by the same pivot entry (so the
/// numerator/denominator ratio at that node is `H(λ_pivot)`), keeping the Kronecker
/// cascade's per-axis scalars consistent. `None` leaves them un-normalized (the
/// leading axis).
fn slice_weights(
    lambda: &[f64],
    mu: &[f64],
    w: &[f64],
    v: &[f64],
    pivot: Option<usize>,
    method: NullMethod,
    non_finite: NonFinite,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // Identify non-finite samples in this slice.
    let bad_col = |j: usize| !w[j].is_finite();
    let bad_row = |i: usize| !v[i].is_finite();
    let any_bad = (0..w.len()).any(bad_col) || (0..v.len()).any(bad_row);

    // The common (clean) path: no policy work needed.
    if !any_bad {
        return slice_weights_clean(lambda, mu, w, v, pivot, method);
    }

    match non_finite {
        NonFinite::Raw => {
            let total = w.len() + v.len();
            let count = (0..w.len()).filter(|&j| bad_col(j)).count()
                + (0..v.len()).filter(|&i| bad_row(i)).count();
            Err(LoewnerError::NonFinite { count, total })
        }
        NonFinite::Fill => {
            // Legacy: the caller already filled via `sample_continued`, so a bad
            // sample reaching here is unexpected. Treat as excise to stay finite.
            slice_weights_excise(lambda, mu, w, v, pivot, method, false)
        }
        NonFinite::Excise => slice_weights_excise(lambda, mu, w, v, pivot, method, false),
        NonFinite::Pole => slice_weights_excise(lambda, mu, w, v, pivot, method, true),
    }
}

/// Per-slice weights with **no** non-finite handling (the hot, common path).
fn slice_weights_clean(
    lambda: &[f64],
    mu: &[f64],
    w: &[f64],
    v: &[f64],
    pivot: Option<usize>,
    method: NullMethod,
) -> Result<(Vec<f64>, Vec<f64>)> {
    use crate::loewner::loewner_1d;
    use crate::nullspace::{InspectNull, MinNormNull, Normalize, NullSpace};

    // Denominator weights = Loewner null space (Theorem 2.1): `[γ_j d(λ_j)]`. This
    // is the paper's headline result — the null space is recovered *by inspection*
    // (Gaussian elimination on the tiny slice matrix), never by SVD. It always
    // runs and always returns a vector regardless of the data: exact for a
    // rational slice, the approximate decoupling (§5(c)) otherwise. `MinNorm` swaps
    // the forced-Gaussian null for the minimum-norm inverse-iteration null (still
    // no SVD) — more robust when approximating non-rational targets. SVD belongs to
    // the `alg*` algorithms, not here.
    let l = loewner_1d(lambda, w, mu, v);
    let nrm = match pivot {
        Some(p) => Normalize::Index(p),
        None => Normalize::Last,
    };
    let mut den = match method {
        NullMethod::ByInspection => InspectNull { normalize: nrm }.null(&l)?.vector,
        NullMethod::MinNorm => MinNormNull::new(nrm).null(&l)?.vector,
    };

    if den.len() != lambda.len() {
        return Err(LoewnerError::Degenerate(
            "slice null vector length mismatch".into(),
        ));
    }

    // Numerator weights = den ⊙ w (Corollary 2.1: n(λ_j) = d(λ_j)·H(λ_j)). Both are
    // re-normalized by the same pivot so the num/den ratio is preserved and the
    // pivot entry of `den` is 1 when `pivot` is `Some`.
    let mut num: Vec<f64> = den.iter().zip(w).map(|(&d, &wj)| d * wj).collect();
    normalize_pivot(&mut den, pivot);
    normalize_pivot(&mut num, pivot);
    Ok((num, den))
}

/// Per-slice weights with non-finite samples **excised** at the 1-D Loewner
/// level: drop the offending columns (`w_j` non-finite) and rows (`v_i`
/// non-finite), solve the reduced null system, then re-insert `den = 0` at the
/// excised columns. The dense tensor is untouched — only this slice's matrix
/// shrinks, so the requirement for a dense grid still holds.
///
/// When `recover_pole` is set (the [`NonFinite::Pole`] policy), an excised column
/// whose sample was `±Inf` (a genuine pole, `d(λ_j) = 0`) keeps `den_j = 0` but its
/// numerator value `n(λ_j)` is recovered by **barycentric interpolation** through
/// the finite (kept) nodes — stable, exact for rational slices, no extra sampling.
/// A `NaN` column is treated as missing data (`num_j = den_j = 0`).
///
/// Guard: if excision would leave fewer than two kept columns (too little to fit a
/// null line), we keep the slice intact and fall back to the clean path so the fit
/// degrades gracefully rather than collapsing.
fn slice_weights_excise(
    lambda: &[f64],
    mu: &[f64],
    w: &[f64],
    v: &[f64],
    pivot: Option<usize>,
    method: NullMethod,
    recover_pole: bool,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let k = lambda.len();
    let keep_col: Vec<usize> = (0..k).filter(|&j| w[j].is_finite()).collect();
    let keep_row: Vec<usize> = (0..mu.len()).filter(|&i| v[i].is_finite()).collect();

    // Need at least 2 finite columns and 1 finite row to recover a null line.
    if keep_col.len() < 2 || keep_row.is_empty() {
        return slice_weights_clean(lambda, mu, w, v, pivot, method);
    }

    let lam_k: Vec<f64> = keep_col.iter().map(|&j| lambda[j]).collect();
    let w_k: Vec<f64> = keep_col.iter().map(|&j| w[j]).collect();
    let mu_k: Vec<f64> = keep_row.iter().map(|&i| mu[i]).collect();
    let v_k: Vec<f64> = keep_row.iter().map(|&i| v[i]).collect();

    // Solve the reduced (un-normalized) slice, then re-expand to full length with
    // zeros at excised columns. We normalize *after* re-expansion so the pivot
    // index still refers to the original full-length position.
    let (num_k, den_k) = slice_weights_clean(&lam_k, &mu_k, &w_k, &v_k, None, method)?;

    let mut den = vec![0.0f64; k];
    let mut num = vec![0.0f64; k];
    for (slot, &j) in keep_col.iter().enumerate() {
        den[j] = den_k[slot];
        num[j] = num_k[slot];
    }

    // Pole numerator recovery: for an excised ±Inf column, d(λ_j)=0 (den already 0)
    // but n(λ_j) is finite. We have the numerator *values* at the kept nodes —
    // n(λ_i) = num_k[slot]/γ_slot (since num_k[slot] = γ_i·n(λ_i)) — and the
    // numerator is a polynomial of degree < (kept count), so we recover n(λ_j) by
    // **second-kind barycentric interpolation** of those values (stable). The slice
    // numerator weight at the pole is then γ_j·n(λ_j) with γ_j over the FULL node
    // set (so it composes with the rest of the slice consistently).
    if recover_pole {
        let any_pole = (0..k).any(|j| w[j].is_infinite());
        if any_pole {
            let gamma_k = lagrange_coeffs(&lam_k);
            let n_vals: Vec<f64> = num_k.iter().zip(&gamma_k).map(|(&nk, &g)| nk / g).collect();
            let gamma_full = lagrange_coeffs(lambda);
            for j in 0..k {
                if !w[j].is_infinite() {
                    continue; // only ±Inf columns are poles; NaN stays excised
                }
                let n_at_pole = bary_eval_1d(&lam_k, &n_vals, lambda[j]);
                num[j] = gamma_full[j] * n_at_pole;
                den[j] = 0.0;
            }
        }
    }

    // Choose a pivot that survived excision (the requested one if kept, else the
    // last kept finite column) so normalization never divides by an excised 0.
    let eff_pivot = pivot.map(|p| {
        if keep_col.contains(&p) {
            p
        } else {
            *keep_col.last().unwrap()
        }
    });
    normalize_pivot(&mut den, eff_pivot);
    normalize_pivot(&mut num, eff_pivot);
    Ok((num, den))
}

/// Evaluate the degree-`<k` barycentric rational with weights `c` at nodes `nodes`
/// at point `x` (first-kind barycentric / Lagrange numerator form: `Σ c_j ℓ_j(x)`
/// via the standard barycentric weights). Used to recover a pole's numerator value
/// from the finite nodes. Takes the exact limit if `x` lands on a node.
fn bary_eval_1d(nodes: &[f64], c: &[f64], x: f64) -> f64 {
    // Standard barycentric weights β_j = 1 / ∏_{i≠j}(x_j − x_i).
    let n = nodes.len();
    for (j, &xj) in nodes.iter().enumerate() {
        if x == xj {
            return c[j];
        }
    }
    let mut num = 0.0;
    let mut den = 0.0;
    for j in 0..n {
        let beta: f64 = (0..n)
            .filter(|&i| i != j)
            .map(|i| nodes[j] - nodes[i])
            .product::<f64>()
            .recip();
        let t = beta / (x - nodes[j]);
        num += t * c[j];
        den += t;
    }
    num / den
}

/// Build the decoupled (separated numerator/denominator) coefficient tensors via
/// the recursive Kronecker cascade — the data-driven realization of Eq. 18–19.
///
/// `points` gives the per-axis column/row split (reduced to `order+1` points per
/// axis, exactly as the recursion requires); `tab` is `H` sampled on the combined
/// grid. Returns `(c_num, c_den)`, both length `∏ k_l` in canonical order.
///
/// The cascade mirrors [`crate::recurse::loewner_null_rec`] but tracks the
/// numerator and denominator weight products in lockstep: at each axis it computes
/// the slice's `(num, den)` weight vectors by inspection ([`slice_weights`]) and
/// multiplies them into the running per-axis products. The leading axis is left
/// un-normalized (paper Eq. 18: "the first variable is not normalized"); every
/// nested axis is normalized by its last entry.
fn build_tensors(
    points: &Points,
    tab: View<'_>,
    method: NullMethod,
    non_finite: NonFinite,
) -> Result<(Vec<f64>, Vec<f64>)> {
    use crate::eval::column_values;
    let k: Vec<usize> = points.column_sizes();
    let total: usize = k.iter().product();

    // Fast path when every sample is finite: build the denominator recursively and
    // reconstruct the numerator by the exact pointwise C_num[J] = C_den[J]·H(λ_J)
    // (the cheaper, well-tested route — `column_values` matches `c_den`'s order).
    let w = column_values(points, tab.view());
    let all_finite = w.iter().all(|x| x.is_finite());
    if all_finite {
        let mut c_den = Vec::with_capacity(total);
        recurse(points, &tab, 0, None, &mut c_den, method, non_finite)?;
        let c_num: Vec<f64> = c_den.iter().zip(&w).map(|(&d, &wj)| d * wj).collect();
        return Ok((c_num, c_den));
    }

    // Non-finite present: the pointwise C_num = C_den·H trick breaks at poles
    // (0·Inf = NaN), so build BOTH tensors recursively in lockstep, with each
    // slice's policy-aware `(num, den)` weights.
    let mut c_num = Vec::with_capacity(total);
    let mut c_den = Vec::with_capacity(total);
    recurse_pair(
        points, &tab, 0, None, &mut c_num, &mut c_den, method, non_finite,
    )?;
    Ok((c_num, c_den))
}

/// Recurse over axes `axis..n`, appending the **denominator** coefficients
/// `C_den[J] = (∏_l γ_{l,j_l}) · d(λ_J)` in canonical order. `pivot` is the
/// normalization of *this* axis: `None` for the leading (top) axis, `Some(last)`
/// for every nested axis.
///
/// Only the denominator tensor is built recursively: the numerator tensor is the
/// pointwise `C_num[J] = C_den[J] · w_J` where `w_J = H(λ_J)` is the *actual*
/// sampled grid value (the caller forms it). This is exact — `C_num[J]/C_den[J] =
/// n(λ_J)/d(λ_J) = H(λ_J)` — and avoids the trap of multiplying per-slice
/// numerator *ratios*, which only reconstructs `n` when `n` happens to be
/// separable. The genuine per-axis decoupling of `n` and `d` (the explicit KST
/// single-variable functions) is recovered separately via [`slice_weights`] and
/// [`DecoupledModel::axis_weight_vectors`]; the *evaluation* tensors are the exact
/// pair above.
fn recurse(
    points: &Points,
    tab: &View<'_>,
    axis: usize,
    pivot: Option<usize>,
    c_den: &mut Vec<f64>,
    method: NullMethod,
    non_finite: NonFinite,
) -> Result<()> {
    let n = points.ndim();
    let k0 = points.column[axis].len();

    // Slice along the leading axis, trailing axes frozen at their last column.
    let slab = freeze_trailing_last_column(points, tab, axis);
    let (lambda, mu) = (&points.column[axis], &points.row[axis]);
    let w = &slab[..k0];
    let vv = &slab[k0..];
    let (_num, den) = slice_weights(lambda, mu, w, vv, pivot, method, non_finite)?;

    if axis + 1 == n {
        c_den.extend_from_slice(&den);
        return Ok(());
    }

    let nested_pivot = Some(points.column[axis + 1].len() - 1);
    for (p, &sden) in den.iter().enumerate().take(k0) {
        let sub = index_leading(tab, p);
        let dstart = c_den.len();
        recurse(
            points,
            &sub,
            axis + 1,
            nested_pivot,
            c_den,
            method,
            non_finite,
        )?;
        for x in &mut c_den[dstart..] {
            *x *= sden;
        }
    }
    Ok(())
}

/// As [`recurse`], but builds the **numerator and denominator** tensors in
/// lockstep using each slice's policy-aware `(num, den)` weights. Used only when a
/// non-finite sample is present (the pointwise `C_num = C_den·H` reconstruction
/// breaks at poles); otherwise [`build_tensors`] takes the cheaper pointwise route.
#[allow(clippy::too_many_arguments)]
fn recurse_pair(
    points: &Points,
    tab: &View<'_>,
    axis: usize,
    pivot: Option<usize>,
    c_num: &mut Vec<f64>,
    c_den: &mut Vec<f64>,
    method: NullMethod,
    non_finite: NonFinite,
) -> Result<()> {
    let n = points.ndim();
    let k0 = points.column[axis].len();

    let slab = freeze_trailing_last_column(points, tab, axis);
    let (lambda, mu) = (&points.column[axis], &points.row[axis]);
    let w = &slab[..k0];
    let vv = &slab[k0..];
    let (num, den) = slice_weights(lambda, mu, w, vv, pivot, method, non_finite)?;

    if axis + 1 == n {
        c_num.extend_from_slice(&num);
        c_den.extend_from_slice(&den);
        return Ok(());
    }

    let nested_pivot = Some(points.column[axis + 1].len() - 1);
    // Size of one nested sub-block (∏ of the trailing axes' column counts).
    let sub_block: usize = ((axis + 1)..n).map(|l| points.column[l].len()).product();
    for (p, (&snum, &sden)) in num.iter().zip(&den).enumerate().take(k0) {
        // If this leading-axis column was excised (non-finite sample → weight 0),
        // its entire sub-block is multiplied by 0; recursing into it would hit an
        // all-non-finite sub-tensor and produce `0·NaN = NaN`. Emit zeros directly.
        if !w[p].is_finite() {
            c_num.resize(c_num.len() + sub_block, 0.0);
            c_den.resize(c_den.len() + sub_block, 0.0);
            continue;
        }
        let sub = index_leading(tab, p);
        let nstart = c_num.len();
        let dstart = c_den.len();
        recurse_pair(
            points,
            &sub,
            axis + 1,
            nested_pivot,
            c_num,
            c_den,
            method,
            non_finite,
        )?;
        for x in &mut c_num[nstart..] {
            *x *= snum;
        }
        for x in &mut c_den[dstart..] {
            *x *= sden;
        }
    }
    Ok(())
}

/// Gather the leading-axis line of `tab` with every trailing axis frozen at its
/// last column index `k_l − 1` (the representative slice; mirrors
/// [`crate::recurse`]'s helper).
fn freeze_trailing_last_column(points: &Points, tab: &View<'_>, axis: usize) -> Vec<f64> {
    if tab.ndim() == 1 {
        return contig_slice(tab).to_vec();
    }
    let lead_dim = tab.shape()[0];
    let n = tab.ndim();
    let mut idx = vec![0usize; n];
    for (j, l) in ((axis + 1)..points.ndim()).enumerate() {
        idx[j + 1] = points.column[l].len() - 1;
    }
    (0..lead_dim)
        .map(|i| {
            idx[0] = i;
            scalar_at(tab, &idx)
        })
        .collect()
}

/// Information returned alongside a decoupled model.
#[derive(Debug, Clone)]
pub struct DecoupleInfo {
    /// Per-axis degree used (`k_l − 1`).
    pub degree: Vec<usize>,
    /// Number of grid samples taken (`∏ (k_l + q_l)`).
    pub samples: usize,
    /// Number of decoupled single-variable functions, `Σ_{l} ∏_{m<l} k_m`
    /// (`1 + k_1 + k_1 k_2 + …`) — the count the paper reports (e.g. 7 for
    /// Pólya–Szegő).
    pub n_functions: usize,
}

/// Build a decoupled rational model of `f` with **explicit per-axis degrees**.
///
/// This is the paper's worked-example entry point: the user supplies the degree
/// `degree[l]` along each axis (so `k_l = degree[l] + 1` column points), and the
/// method evaluates `f` on the Kronecker grid and writes the numerator/denominator
/// weights down **by inspection** — a single Gaussian elimination per 1-D slice
/// (Theorem 2.1), never an SVD, at any degree. The support points are built from
/// `bounds` via the well-conditioned [`Points::alternating`] convention.
///
/// Exact for any rational `f` whose per-axis degree is `≤ degree[l]`.
pub fn decouple_with_degrees<F: TargetFn + Sync>(
    f: &F,
    bounds: &[(f64, f64)],
    degree: &[usize],
) -> Result<(DecoupledModel, DecoupleInfo)> {
    if bounds.len() != degree.len() {
        return Err(LoewnerError::DimMismatch {
            what: "bounds vs degree length",
            expected: bounds.len(),
            got: degree.len(),
        });
    }
    // k_l = degree+1 column points; we need a disjoint row set per axis to form the
    // Loewner matrices. `nip = 2*k_l` evenly spaced, split alternating, gives k_l
    // column + k_l row points (q_l = k_l), comfortably ≥ the order.
    let nip: Vec<usize> = degree.iter().map(|&d| 2 * (d + 1)).collect();
    let points = Points::alternating(bounds, &nip)?;
    decouple_from_closure(f, &points, degree)
}

/// Build a decoupled rational model of `f` on the given support `points`, with
/// known per-axis degrees. The column set of `points` must have exactly
/// `degree[l] + 1` points along axis `l`.
pub fn decouple_from_closure<F: TargetFn + Sync>(
    f: &F,
    points: &Points,
    degree: &[usize],
) -> Result<(DecoupledModel, DecoupleInfo)> {
    for (l, &d) in degree.iter().enumerate() {
        if points.column[l].len() != d + 1 {
            return Err(LoewnerError::DimMismatch {
                what: "column points along axis (need degree+1)",
                expected: d + 1,
                got: points.column[l].len(),
            });
        }
    }
    let (tab, _) = sample_continued(f, points);
    decouple_from_tensor(&tab, points)
}

/// Build a decoupled model from an already-sampled tensor on the combined grid of
/// `points` (the data-driven entry point). The column set of `points` is taken to
/// have exactly `k_l = degree_l + 1` points along each axis.
///
/// Uses the default [`NullMethod::ByInspection`]; for the minimum-norm null on
/// non-rational data, use [`decouple_from_tensor_with`].
pub fn decouple_from_tensor(
    tab: &Tensor,
    points: &Points,
) -> Result<(DecoupledModel, DecoupleInfo)> {
    // The bare data-driven door errors on non-finite samples (`Raw`), so the user
    // is told to choose a policy rather than getting silently altered data.
    decouple_from_tensor_with(tab, points, NullMethod::ByInspection, NonFinite::Raw)
}

/// As [`decouple_from_tensor`], with an explicit [`NullMethod`] for the per-slice
/// null-vector recovery and a [`NonFinite`] policy for non-finite samples.
pub fn decouple_from_tensor_with(
    tab: &Tensor,
    points: &Points,
    null_method: NullMethod,
    non_finite: NonFinite,
) -> Result<(DecoupledModel, DecoupleInfo)> {
    points.check_disjoint()?;
    let k = points.column_sizes();
    let (c_num, c_den) = build_tensors(points, tab.view(), null_method, non_finite)?;

    let n_functions = {
        let mut sum = 0usize;
        let mut prod = 1usize;
        for &kl in &k {
            sum += prod;
            prod *= kl;
        }
        sum
    };
    let samples: usize = (0..points.ndim())
        .map(|l| points.column[l].len() + points.row[l].len())
        .product();

    let model = DecoupledModel {
        lambda: points.column.clone(),
        c_num,
        c_den,
    };
    let info = DecoupleInfo {
        degree: k.iter().map(|&kl| kl - 1).collect(),
        samples,
        n_functions,
    };
    Ok((model, info))
}

/// Build a decoupled model of `f`, **auto-detecting** the per-axis degree.
///
/// Convenience wrapper mirroring [`crate::alg1()`]: it samples `f` on a generous grid
/// over `bounds`, runs the same per-axis order detection as the rest of the crate
/// ([`crate::order`]) to find each variable's rational degree, reduces to
/// `order + 1` points per axis, and then writes the decoupled representation down
/// by inspection. Exact for rational `f`; for non-rational `f` it produces the
/// best low-degree rational decoupling the detected order allows.
///
/// `nip` is the per-axis interpolation-point count of the *initial* grid; if an
/// axis's detected order saturates that budget the fit will under-resolve it — see
/// [`decouple_adaptive`], which grows the saturated axes automatically and is the
/// recommended entry point when the per-axis degrees are unknown and may differ.
pub fn decouple<F: TargetFn + Sync>(
    f: &F,
    bounds: &[(f64, f64)],
    nip: &[usize],
) -> Result<(DecoupledModel, DecoupleInfo)> {
    decouple_opt(
        f,
        bounds,
        &DecoupleOptions {
            sampling: Sampling::Linspace(nip.to_vec()),
            ..Default::default()
        },
    )
}

/// Build a decoupled model of `f` with **per-axis adaptive** point counts.
///
/// The papers decompose an `n`-D map into independent **1-D** problems — one per
/// axis — so each axis carries its *own* degree and deserves its *own* number of
/// interpolation points. A single uniform `nip` is the wrong shape for that: it
/// either under-resolves the high-degree axes (too few points) or, when raised to
/// fix them, blows the grid up combinatorially (`(2·nip)^n`) over the *low*-degree
/// axes that never needed it. The fix is per-axis growth.
///
/// Starting from `start_nip` points per axis, this:
/// 1. samples, detects the per-axis order;
/// 2. finds any axis whose detected order **saturates** its column budget
///    (`order[l] == nip[l]/2 - 1` — a sign it is capped, not resolved);
/// 3. grows *only those axes'* `nip` (×`grow`, capped at `max_nip`) and repeats;
/// 4. stops when no axis is saturated (every axis resolved) or the cap is hit,
///    then reduces each axis to `order+1` and writes the decoupling by inspection.
///
/// Because only the saturated axes grow, a mixed-degree target like case 42
/// (degrees `[4,1,1,3,2,2]`) lands on a grid of `~10·4·4·8·6·6 ≈ 46 k` points
/// rather than the `20^6 ≈ 64 M` a uniform `nip=10` would force — resolving the
/// degree-4 axis without paying for it on the five low-degree axes.
pub fn decouple_adaptive<F: TargetFn + Sync>(
    f: &F,
    bounds: &[(f64, f64)],
    start_nip: usize,
    max_nip: usize,
) -> Result<(DecoupledModel, DecoupleInfo)> {
    decouple_opt(
        f,
        bounds,
        &DecoupleOptions {
            sampling: Sampling::Adaptive { start_nip, max_nip },
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::sample;
    use approx::assert_relative_eq;

    #[test]
    fn lagrange_coeffs_partition_of_unity() {
        // Σ_j γ_j ∏_{i≠j}(s − λ_i) = 1 for all s (the normalized basis sums to 1).
        let lam = [0.0, 1.0, 2.5];
        let g = lagrange_coeffs(&lam);
        for &s in &[0.3, -1.2, 4.0] {
            let sum: f64 = (0..lam.len())
                .map(|j| {
                    g[j] * lam
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != j)
                        .map(|(_, &li)| s - li)
                        .product::<f64>()
                })
                .sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn adaptive_recovers_high_n_mixed_degree_rational() {
        // A 6-var rational with DIFFERENT degrees per axis: denominator
        // x0^4 + x1 x2 + x3^3 + x4^2 + x5 ⇒ per-axis degrees [4,1,1,3,2,2].
        // A uniform nip cannot supply 5 columns on axis 0 and 2 on axis 1 at once
        // without blowing the grid up; `decouple_adaptive` grows only the saturated
        // axes and recovers it exactly. (MATLAB's `compute_order` errors on this
        // case — the combinations machinery overflows at n=6 — so this is a genuine
        // library improvement, not a port of reference behavior.)
        let f = |x: &[f64]| {
            (x[0] + x[2] - 2f64.sqrt() * x[5] * x[5])
                / (x[0].powi(4) + x[1] * x[2] + x[3].powi(3) + x[4] * x[4] + x[5])
        };
        let bounds = vec![(0.1, 1.0); 6];
        let (model, info) = decouple_adaptive(&f, &bounds, 6, 16).unwrap();
        assert_eq!(
            info.degree,
            vec![4, 1, 1, 3, 2, 2],
            "per-axis degrees resolved"
        );
        let mut err = 0f64;
        let mut seed = 4242u64;
        for _ in 0..200 {
            let mut x = [0.0; 6];
            for (xi, &(lo, hi)) in x.iter_mut().zip(&bounds) {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (seed >> 33) as f64 / (1u64 << 31) as f64;
                *xi = lo + (hi - lo) * u;
            }
            err = err.max((model.eval(&x) - f(&x)).abs());
        }
        assert!(
            err < 1e-9,
            "adaptive max err {err:.2e} should be ~machine precision"
        );
    }

    #[test]
    fn decouples_polya_szego_from_8_samples() {
        // The headline example: p = x1 x2 + x1 x3 + x2 x3, degree 1 per axis, k=2,
        // 8 grid points → exact recovery (Appendix A / Remark A.3).
        let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
        let (model, info) = decouple_with_degrees(&f, &[(-1.0, 1.0); 3], &[1, 1, 1]).unwrap();
        assert_eq!(info.degree, vec![1, 1, 1]);
        // 1 + k1 + k1 k2 = 1 + 2 + 4 = 7 decoupled single-variable functions.
        assert_eq!(info.n_functions, 7);
        for x in [[0.1, -0.3, 0.5], [-0.7, 0.2, 0.9], [0.33, 0.33, -0.21]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-10);
        }
    }

    #[test]
    fn decouples_2d_rational_appendix_c() {
        // Appendix C worked example: H(s,t) = (t² + s − 2)/(t² + 2s + 1), a rational
        // with a *coupled* denominator, degrees (1, 2) → k = (2, 3). With a
        // unisolvent interpolation grid (the paper chooses its points explicitly;
        // the evenly-spread `alternating` builder is not unisolvent for this coupled
        // case — see WORKPLAN §6d) the decoupling reconstructs it to machine
        // precision. Uses `decouple_from_closure` with explicit points.
        let f = |x: &[f64]| (x[1] * x[1] + x[0] - 2.0) / (x[1] * x[1] + 2.0 * x[0] + 1.0);
        let pts = Points::new(
            vec![vec![0.2, 0.6], vec![-0.5, 0.0, 0.5]],
            vec![vec![0.9, -0.3], vec![0.25, -0.25, 0.75]],
        )
        .unwrap();
        let (model, info) = decouple_from_closure(&f, &pts, &[1, 2]).unwrap();
        assert_eq!(info.degree, vec![1, 2]);
        for x in [[0.3, 0.7], [-0.4, 1.2], [0.9, -0.5]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn separable_numerator_denominator_decouple() {
        // A separable rational with non-trivial numerator AND denominator per axis.
        let f = |x: &[f64]| ((x[0] + 1.0) * (x[1] - 2.0)) / ((x[0] + 3.0) * (x[1] + 4.0));
        let (model, _) = decouple_with_degrees(&f, &[(-0.9, 0.9); 2], &[1, 1]).unwrap();
        for x in [[0.2, 0.5], [-0.3, -0.6], [0.8, 0.1]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn raw_policy_errors_on_nonfinite() {
        // 1/x has a pole at 0; a grid that lands on 0 produces ±Inf. The default
        // Raw policy must error (no fabrication) and name the alternatives.
        let f = |x: &[f64]| 1.0 / x[0] + x[1];
        // Force a grid that includes 0 along axis 0.
        let pts = Points::new(
            vec![vec![-1.0, 0.0], vec![-0.5, 0.5]],
            vec![vec![-0.3, 0.3], vec![0.2, -0.2]],
        )
        .unwrap();
        let tab = crate::tensor::sample(&f, &pts);
        let err = decouple_from_tensor_with(&tab, &pts, NullMethod::ByInspection, NonFinite::Raw)
            .unwrap_err();
        assert!(matches!(err, LoewnerError::NonFinite { .. }));
    }

    #[test]
    fn pole_policy_recovers_simple_pole() {
        // f(x,y) = 1/x + y has a genuine pole at x = 0. Build a grid that lands a
        // column exactly on x = 0 (→ ±Inf sample); the Pole policy must set the
        // denominator weight to 0 there (the pole) and recover the numerator by
        // barycentric interpolation, giving an accurate model away from the pole —
        // WITHOUT any fabricated fill.
        let f = |x: &[f64]| 1.0 / x[0] + x[1];
        // Columns along axis 0 include 0.0; degrees (1,1) → 2 cols/axis. Use enough
        // columns that excising the pole node still leaves ≥ 2 finite columns.
        let pts = Points::new(
            vec![vec![-1.0, 0.0, 1.0], vec![-0.5, 0.5]],
            vec![vec![-0.3, 0.3, 0.7], vec![0.2, -0.2]],
        )
        .unwrap();
        let tab = crate::tensor::sample(&f, &pts);
        let (model, _) =
            decouple_from_tensor_with(&tab, &pts, NullMethod::ByInspection, NonFinite::Pole)
                .unwrap();
        for x in [[0.7, 0.3], [-0.8, -0.5], [1.0, 0.9]] {
            let y = model.eval(&x);
            assert!(y.is_finite(), "pole policy must keep eval finite, got {y}");
            assert_relative_eq!(y, f(&x), epsilon = 1e-6);
        }
    }

    #[test]
    fn excise_policy_stays_finite_through_singularity() {
        // sinc(x) = sin(x)/x is NaN (0/0) at x = 0 but equals 1 in the limit. A grid
        // hitting 0 makes the sample NaN; Excise must drop that node and still
        // produce a finite model that approximates sinc elsewhere.
        let f = |x: &[f64]| {
            let s = x[0];
            let sinc = if s == 0.0 { 1.0 } else { s.sin() / s };
            sinc + x[1]
        };
        // Build a grid that explicitly includes 0 along axis 0 so a NaN appears
        // (use raw f without the s==0 guard to force the NaN path).
        let raw = |x: &[f64]| x[0].sin() / x[0] + x[1];
        let pts = Points::new(
            vec![vec![-0.8, 0.0, 0.8], vec![-0.5, 0.5]],
            vec![vec![-0.4, 0.4, 1.2], vec![0.2, -0.2]],
        )
        .unwrap();
        let tab = crate::tensor::sample(&raw, &pts);
        let (model, _) =
            decouple_from_tensor_with(&tab, &pts, NullMethod::ByInspection, NonFinite::Excise)
                .unwrap();
        // Model must be finite everywhere and roughly track sinc away from 0.
        for x in [[0.3, 0.1], [-0.6, -0.2], [0.7, 0.5]] {
            let y = model.eval(&x);
            assert!(y.is_finite(), "excise must keep eval finite, got {y}");
            assert!(
                (y - f(&x)).abs() < 0.5,
                "excise sinc approx off: {y} vs {}",
                f(&x)
            );
        }
    }

    #[test]
    fn to_barycentric_agrees() {
        let f = |x: &[f64]| (x[1] * x[1] + x[0] - 2.0) / (x[1] * x[1] + 2.0 * x[0] + 1.0);
        let (model, _) = decouple_with_degrees(&f, &[(-1.0, 1.0); 2], &[1, 2]).unwrap();
        let bary = model.to_barycentric();
        for x in [[0.3, 0.7], [-0.4, 1.2]] {
            assert_relative_eq!(bary.eval(&x), model.eval(&x), epsilon = 1e-12);
        }
    }

    #[test]
    fn auto_degree_detection_matches() {
        // The auto-detecting wrapper finds order 1 per axis and recovers exactly.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let (model, info) = decouple(&f, &[(-1.0, 1.0); 2], &[8, 8]).unwrap();
        assert_eq!(info.degree, vec![1, 1]);
        for x in [[0.3, 0.7], [-0.4, 0.9]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn uniform_sampling_one_rule_all_axes() {
        // The easy path: one AxisSampling on every axis, no AxisSpec wrapping.
        // Integers on [0, 6] gives a small discrete grid; the affine target is
        // recovered exactly.
        let f = |x: &[f64]| 2.0 * x[0] + 3.0 * x[1] - 1.0;
        let opts = DecoupleOptions {
            sampling: Sampling::Uniform(AxisSampling::Integers),
            non_finite: NonFinite::Excise,
            ..Default::default()
        };
        let (model, _) = decouple_opt(&f, &[(0.0, 6.0); 2], &opts).unwrap();
        for x in [[1.5, 2.5], [4.0, 0.5]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn per_axis_helper_takes_bare_rules_and_mixed_specs() {
        // `Sampling::per_axis` accepts bare AxisSampling rules (the old simple form)
        // AND full AxisSpecs, mixed — different method per axis with no `.into()`.
        let f = |x: &[f64]| x[0] + 2.0 * x[1];
        let bare = Sampling::per_axis([AxisSampling::Integers, AxisSampling::Chebyshev(6)]);
        let opts = DecoupleOptions {
            sampling: bare,
            non_finite: NonFinite::Excise,
            ..Default::default()
        };
        let (model, _) = decouple_opt(&f, &[(0.0, 6.0), (-1.0, 1.0)], &opts).unwrap();
        assert_relative_eq!(model.eval(&[3.0, 0.4]), f(&[3.0, 0.4]), epsilon = 1e-9);

        // Mixed: bare integer axis + an explicit column/row split on axis 1
        // (Chebyshev columns + inset linspace rows — disjoint within bounds).
        let mixed = Sampling::per_axis([
            AxisSpec::from(AxisSampling::Integers),
            AxisSpec::custom(
                |lo, hi| Ok(crate::points::chebyshev_lobatto(lo, hi, 7)),
                |lo, hi| {
                    let (a, b) = (lo + 0.05, hi - 0.05);
                    Ok((0..6).map(|i| a + (b - a) * i as f64 / 5.0).collect())
                },
            ),
        ]);
        let opts2 = DecoupleOptions {
            sampling: mixed,
            non_finite: NonFinite::Excise,
            ..Default::default()
        };
        let (m2, _) = decouple_opt(&f, &[(0.0, 6.0), (-1.0, 1.0)], &opts2).unwrap();
        assert_relative_eq!(m2.eval(&[2.0, -0.3]), f(&[2.0, -0.3]), epsilon = 1e-9);
    }

    #[test]
    fn decoupled_functions_count_and_reconstruct_polya_szego() {
        // Pólya–Szegő: 7 decoupled single-variable functions (1 + 2 + 4), each
        // affine. They are grid-line slices of the model and each interpolates its
        // node values; verify the count and that they evaluate the right slices.
        let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
        let (model, info) = decouple_with_degrees(&f, &[(-1.0, 1.0); 3], &[1, 1, 1]).unwrap();
        let funcs = model.decoupled_functions();
        let total: usize = funcs.iter().map(Vec::len).sum();
        assert_eq!(total, info.n_functions);
        assert_eq!(funcs[0].len(), 1); // ϕ(x)
        assert_eq!(funcs[1].len(), 2); // ψ_j(y)
        assert_eq!(funcs[2].len(), 4); // ω_{j,k}(z)
        // ϕ(x) = p(x, y_last, z_last); check it matches a direct slice.
        let phi = &funcs[0][0];
        let ylast = *model.lambda[1].last().unwrap();
        let zlast = *model.lambda[2].last().unwrap();
        for &xq in &[-0.5, 0.0, 0.7] {
            let direct = f(&[xq, ylast, zlast]);
            assert_relative_eq!(phi.eval(xq), direct, epsilon = 1e-9);
        }
    }

    #[test]
    fn matches_recurse_on_grid_for_pure_denominator() {
        // When the numerator is constant the decoupled model must agree with the
        // existing recursive Barycentric fit everywhere (both represent the same H).
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.5, 1.5], vec![0.5, 1.5]],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let (model, _) = decouple_from_tensor(&tab, &pts).unwrap();
        for x in [[0.3, 0.7], [-0.4, 1.2], [0.9, 0.1]] {
            assert_relative_eq!(model.eval(&x), f(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn eval_is_finite_on_degenerate_zero_coefficient_model() {
        // Reproduces the user-reported `DecoupledModel::eval -> NaN` on a degenerate
        // fit (a piecewise-constant / `round`-ed target collapses to a model whose
        // coefficient tensor is zero at most nodes). `DecoupledModel::eval` must
        // match the guarded `Barycentric` path — finite everywhere, returning the
        // sampled value on a full node and a finite `0` on a `0/0` degeneracy.
        let lambda = vec![vec![1.0, 2.0], vec![10.0, 20.0]];
        // c_den zero except the (1,1) corner; c_num = c_den ⊙ w on the grid.
        let mut c_den = vec![0.0; 4];
        c_den[3] = 1.0;
        let w = [5.0, 6.0, 7.0, 8.0];
        let c_num: Vec<f64> = c_den.iter().zip(w).map(|(&d, wj)| d * wj).collect();
        let model = DecoupledModel {
            lambda,
            c_num,
            c_den,
        };

        // On the zero-coefficient (0,0) node: finite (not NaN). The model carries no
        // mass there, so the guarded result is a finite 0 (degenerate fit).
        let on_zero = model.eval(&[1.0, 10.0]);
        assert!(on_zero.is_finite(), "got {on_zero}");

        // On the nonzero-coefficient node: the exact sampled value.
        assert_relative_eq!(model.eval(&[2.0, 20.0]), 8.0, epsilon = 1e-12);

        // Off-grid (and near-node, like the user's 64.0001 bounds): finite.
        assert!(model.eval(&[1.5, 15.0]).is_finite());
        assert!(model.eval(&[1.0000001, 10.0000001]).is_finite());
    }
}
