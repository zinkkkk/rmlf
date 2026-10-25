//! **Per-axis contribution analysis** — collapse the `n`-D model to *one* 1-D
//! [`BarycentricRational1D`] per variable, each summarising how that axis acts in
//! the full model, so the traditional 1-D toolkit (pole/zero, phase portraits,
//! plotting) applies to a single variable at a time.
//!
//! [`Fibres::fibre`] already gives the model *along a line*
//! (every other coordinate pinned), and [`Fibres::fibre_set`]
//! gives the full canonical decoupling — but those are pin-dependent or numerous.
//! A *contribution* is the opposite: **one** 1-D function per axis, the other
//! variables summarised rather than pinned. There are three distinct, genuinely
//! different ways to summarise the rest, each answering a different question; all
//! three are provided and all three return a usable [`BarycentricRational1D`].
//!
//! | Method | Summarises the other axes by… | Answers |
//! |--------|------------------------------|---------|
//! | [`Contributions::rank_contracted_factor`] | contracting their TT cores against a canonical boundary weight | "what is this axis's role *inside the coupling*" |
//! | [`Contributions::marginal`] | averaging `g` over their node grid | "what is the *average* behaviour along this axis" |
//! | [`Contributions::sobol_main_effect`] | the marginal **plus** its variance share | "*how much* does this axis matter" |
//!
//! All three are exposed on any [`Model`] through the [`Contributions`] trait, and
//! bundled per axis by [`Contributions::contributions`].
//!
//! ## Rank-contracted factor — the framework-native one
//!
//! The TT factorisation writes each coefficient tensor as a matrix product across
//! axes (see [`crate::ChannelDecomposition`]):
//!
//! ```text
//! D(x) = M^den_0(x_0) ⋯ M^den_{n−1}(x_{n−1}),   g = N/D.
//! ```
//!
//! Pick a single **canonical coordinate** `c_m` on every *other* axis `m ≠ l`
//! (rmlf uses the node centroid — representative, pin-free, axis-symmetric) and
//! collapse those factors to their scalar/boundary values, leaving axis `l`'s core
//! open:
//!
//! ```text
//! factor_l(x_l) = [L_l · open-core_l(x_l) · R_l]_num
//!                 ────────────────────────────────── ,
//!                 [L_l · open-core_l(x_l) · R_l]_den
//! ```
//!
//! where `L_l`, `R_l` are the contracted boundary vectors of the surrounding axes.
//! This is a genuine 1-D barycentric in `x_l` that carries the axis's behaviour
//! *as it participates in the rank coupling* — high TT rank widens `L_l`/`R_l` and
//! mixes more of the structure in. It is computed straight from the TT cores
//! (`O(Σ_m k_m r²)`), never forming the dense tensor.
//!
//! ## Marginal & Sobol — the statistical ones
//!
//! The **marginal** integrates the other axes out:
//! `m_l(x_l) = mean over the other axes' node grid of g(…, x_l, …)`. Evaluated at
//! axis `l`'s own nodes it gives node values, and interpolating those with the
//! axis's Lagrange weights produces an exact 1-D barycentric through them — the
//! conditional mean `E[g | x_l]` under the uniform-on-nodes measure. The **Sobol
//! main-effect index** is that curve's variance share,
//! `Var_i m_l(λ_l(i)) / Var(g)`, attached to the same barycentric so you get the
//! curve *and* the importance scalar together.
//!
//! ## Usage — same workflow as fibres (manual axis, or automatic full set)
//!
//! ```
//! use rmlf::prelude::*;
//!
//! let f = |x: &[f64]| (1.0 + x[0] * x[1] + 2.0 * x[2]) / (3.0 + x[0] + x[1] * x[2]);
//! let (model, _) = decouple_adaptive(&f, &[(0.2, 1.2); 3], 3, 8).unwrap();
//!
//! // Manual: one axis, one definition → a classic 1-D barycentric.
//! let c = model.rank_contracted_factor(0).unwrap();
//! let _y = c.function.eval(0.7);              // ready for 1-D analysis
//!
//! // Automatic: the whole set + decomposition metadata.
//! let set = model.contribution_set(ContributionKind::Sobol);
//! let _degrees = set.degrees();              // per-axis rational degree
//! let _ranking = set.ranked();              // axes by importance (Sobol order)
//! assert_eq!(set.ndim(), 3);
//! ```

use crate::eval::axis_weight;
use crate::grid::flat_to_multi;
use crate::model::Model;
use crate::tt::{CompressedBarycentric, TtTensor};
use crate::univariate::{BarycentricRational1D, Fibres};

/// One axis's contribution to the model: a 1-D [`BarycentricRational1D`] plus the
/// dimensional/decomposition metadata describing where it came from. Produced by
/// the three [`Contributions`] methods (the `function` is whichever summary you
/// asked for; the `magnitude`'s meaning is documented per method).
///
/// This is the contribution analogue of [`crate::LabelledFibre`]:
/// where a labelled fibre carries the *pinned* leading indices, a contribution
/// carries the axis's *decomposition properties* (degree and representation rank),
/// since the other axes are summarised rather than pinned.
#[derive(Debug, Clone)]
pub struct AxisContribution {
    /// The variable this contribution is a function of.
    pub axis: usize,
    /// Which of the three definitions built this (see [`ContributionKind`]).
    pub kind: ContributionKind,
    /// The summarised 1-D rational along this axis — feed it to the 1-D toolkit.
    pub function: BarycentricRational1D,
    /// A scalar weight for ranking axes. For [`Contributions::sobol_main_effect`]
    /// it is the Sobol main-effect index in `[0, 1]`; for the others it is the
    /// function's RMS variation over its nodes (a magnitude proxy, comparable
    /// across axes built the same way).
    pub magnitude: f64,
    /// This axis's rational **degree** along its own variable (`k_l − 1`, the
    /// number of support nodes minus one).
    pub degree: usize,
    /// The axis's **representation rank** — `max(r_l, r_{l+1})` of the denominator
    /// tensor-train around this axis. This is the size of the TT bond the axis sits
    /// on: it sets the axis's share of storage/eval cost and how compressibly it
    /// folds into the rest of the model.
    ///
    /// **Note:** this is the rank of the *barycentric coefficient* tensor, not a
    /// separability test of `f` — even a pure product `∏(1+x_l)` has rank > 1 here,
    /// because the barycentric weights mix the constant term across nodes. For an
    /// *interaction*-based importance use the Sobol main-effect
    /// ([`Contributions::sobol_main_effect`]); for raw cost/compressibility use
    /// this.
    pub representation_rank: usize,
}

/// **Per-axis contribution views** of a multivariate model — one summary 1-D
/// rational per variable. Three independent definitions
/// ([`Contributions::rank_contracted_factor`],
/// [`Contributions::marginal`],
/// [`Contributions::sobol_main_effect`]); see the
/// [module docs](crate::contribution).
///
/// Blanket-implemented for every [`Model`] that converts to the exact tensor-train
/// form (all three core model types), so you never convert by hand.
pub trait Contributions: Model {
    /// Axis `l`'s **role inside the coupling**: contract every other axis's TT
    /// cores against their canonical boundary weight (node centroid), leaving axis
    /// `l` as one 1-D barycentric. The framework-native contribution — exact, TT
    /// cheap, deterministic. `magnitude` = the function's RMS node variation.
    fn rank_contracted_factor(&self, axis: usize) -> Option<AxisContribution>;

    /// Axis `l`'s **average behaviour**: the marginal `mean_{others} g`, sampled at
    /// axis `l`'s nodes and interpolated as a 1-D barycentric. Interaction with the
    /// other axes is averaged out by design. `magnitude` = RMS node variation.
    fn marginal(&self, axis: usize) -> Option<AxisContribution>;

    /// Axis `l`'s **importance**: the same marginal curve as [`Self::marginal`],
    /// but `magnitude` is the **Sobol main-effect index** `Var(E[g|x_l]) / Var(g)`
    /// (fraction of output variance explained by `x_l` alone, in `[0, 1]`).
    fn sobol_main_effect(&self, axis: usize) -> Option<AxisContribution>;

    /// One axis's contribution under a chosen definition — the unified manual
    /// entry point (dispatches to the three methods above). `None` if `axis` is out
    /// of range.
    fn contribution(&self, axis: usize, method: ContributionKind) -> Option<AxisContribution> {
        match method {
            ContributionKind::RankContracted => self.rank_contracted_factor(axis),
            ContributionKind::Marginal => self.marginal(axis),
            ContributionKind::Sobol => self.sobol_main_effect(axis),
        }
    }

    /// **All axes'** contributions under one definition, in axis order — the
    /// automatic full-set retrieval (the contribution analogue of
    /// [`Fibres::fibre_set`]). `method` selects which of the three to build.
    fn contributions(&self, method: ContributionKind) -> Vec<AxisContribution> {
        (0..self.ndim())
            .map(|l| self.contribution(l, method).expect("axis in range"))
            .collect()
    }

    /// The full per-axis contribution set **with aggregate decomposition
    /// metadata** ([`ContributionSet`]) — the richest automatic view: every axis's
    /// 1-D summary plus the dimensional properties (per-axis degree, coupling rank,
    /// which axes are separable) of the whole decomposition.
    fn contribution_set(&self, method: ContributionKind) -> ContributionSet {
        ContributionSet {
            kind: method,
            axes: self.contributions(method),
        }
    }
}

/// The full per-axis contribution decomposition of a model under one definition:
/// every axis's 1-D summary [`AxisContribution`] plus aggregate accessors for the
/// decomposition's dimensional properties. The automatic, metadata-carrying
/// counterpart to a bare `Vec<AxisContribution>` — analogous to how
/// [`Fibres::fibre_set`] returns labelled fibres rather than raw triples.
#[derive(Debug, Clone)]
pub struct ContributionSet {
    /// Which definition built every axis in this set.
    pub kind: ContributionKind,
    /// One contribution per axis, in axis order (`axes[l].axis == l`).
    pub axes: Vec<AxisContribution>,
}

impl ContributionSet {
    /// Number of variables `n`.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.axes.len()
    }

    /// Per-axis rational degrees `k_l − 1`.
    pub fn degrees(&self) -> Vec<usize> {
        self.axes.iter().map(|c| c.degree).collect()
    }

    /// Per-axis representation ranks (the TT bond size around each axis; see
    /// [`AxisContribution::representation_rank`]). Higher ⇒ that axis is more
    /// expensive to store/evaluate and folds less compressibly into the rest.
    pub fn representation_ranks(&self) -> Vec<usize> {
        self.axes.iter().map(|c| c.representation_rank).collect()
    }

    /// The largest per-axis [`AxisContribution::representation_rank`]
    /// — the model's peak TT bond, the headline complexity number of the
    /// decomposition.
    pub fn max_representation_rank(&self) -> usize {
        self.representation_ranks().into_iter().max().unwrap_or(1)
    }

    /// Axes ordered by descending [`AxisContribution::magnitude`] — the importance
    /// ranking (for [`ContributionKind::Sobol`] this is the Sobol main-effect
    /// order). Returns `(axis, magnitude)` pairs.
    pub fn ranked(&self) -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = self.axes.iter().map(|c| (c.axis, c.magnitude)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v
    }
}

/// Which per-axis contribution definition to build (see [`Contributions`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContributionKind {
    /// [`Contributions::rank_contracted_factor`] — role within the coupling.
    RankContracted,
    /// [`Contributions::marginal`] — average behaviour along the axis.
    Marginal,
    /// [`Contributions::sobol_main_effect`] — variance-based importance.
    Sobol,
}

// Blanket impl: any model that converts to the exact TT form gets all three.
impl<M> Contributions for M
where
    M: Model + Fibres + Clone,
    CompressedBarycentric: for<'a> From<&'a M>,
{
    fn rank_contracted_factor(&self, axis: usize) -> Option<AxisContribution> {
        let n = self.ndim();
        if axis >= n {
            return None;
        }
        let tt: CompressedBarycentric = self.into();
        // Canonical boundary weight on every other axis: the eval-weight at that
        // axis's node centroid (mean of its support nodes) — representative and
        // pin-free. The `axis` slot is ignored by the contraction.
        let pin_w: Vec<Vec<f64>> = tt
            .lambda
            .iter()
            .map(|lam| axis_weight(lam, centroid(lam)))
            .collect();
        let den = tt_contract_except(&tt.denom, axis, &pin_w);
        let num = tt_contract_except(&tt.numer, axis, &pin_w);
        let function = assemble(tt.lambda[axis].clone(), num, den);
        let magnitude = rms_variation(&function);
        let (degree, representation_rank) = axis_meta(&tt, axis);
        Some(AxisContribution {
            axis,
            kind: ContributionKind::RankContracted,
            function,
            magnitude,
            degree,
            representation_rank,
        })
    }

    fn marginal(&self, axis: usize) -> Option<AxisContribution> {
        let (function, _, _) = self.marginal_curve(axis)?;
        let magnitude = rms_variation(&function);
        let (degree, representation_rank) = axis_meta(&self.into(), axis);
        Some(AxisContribution {
            axis,
            kind: ContributionKind::Marginal,
            function,
            magnitude,
            degree,
            representation_rank,
        })
    }

    fn sobol_main_effect(&self, axis: usize) -> Option<AxisContribution> {
        let (function, node_means, grand) = self.marginal_curve(axis)?;
        // Main-effect variance: variance of the conditional mean E[g|x_l] across
        // axis l's nodes, over the total variance of g across the full node grid.
        let var_cond = variance(&node_means, grand);
        let total = self.total_variance(grand);
        let magnitude = if total > 0.0 {
            (var_cond / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (degree, representation_rank) = axis_meta(&self.into(), axis);
        Some(AxisContribution {
            axis,
            kind: ContributionKind::Sobol,
            function,
            magnitude,
            degree,
            representation_rank,
        })
    }
}

/// Per-axis decomposition metadata read off the exact TT form: the axis's rational
/// `degree = k_l − 1` and its representation rank `max(r_l, r_{l+1})` of the
/// denominator tensor-train (the TT bond size around the axis).
fn axis_meta(tt: &CompressedBarycentric, axis: usize) -> (usize, usize) {
    let degree = tt.lambda[axis].len().saturating_sub(1);
    let ranks = &tt.denom.ranks;
    let representation_rank = ranks[axis].max(ranks[axis + 1]);
    (degree, representation_rank)
}

/// Shared marginal builder (used by both `marginal` and `sobol_main_effect`):
/// returns the interpolating barycentric, the per-node conditional means
/// `E[g | x_l = λ_l(i)]`, and the grand mean of `g` over the full node grid.
trait MarginalCurve: Model + Fibres {
    fn marginal_curve(&self, axis: usize) -> Option<(BarycentricRational1D, Vec<f64>, f64)>;
    /// Variance of `g` over the full node grid (about its grand mean `grand`).
    fn total_variance(&self, grand: f64) -> f64;
}

impl<M: Model + Fibres> MarginalCurve for M {
    fn marginal_curve(&self, axis: usize) -> Option<(BarycentricRational1D, Vec<f64>, f64)> {
        let n = self.ndim();
        let nodes = self.nodes();
        if axis >= n {
            return None;
        }
        let lam_axis = nodes[axis].clone();
        let ka = lam_axis.len();
        // Other axes' node grids; iterate their Cartesian product once.
        let other: Vec<&Vec<f64>> = (0..n).filter(|&m| m != axis).map(|m| &nodes[m]).collect();
        let other_dims: Vec<usize> = other.iter().map(|v| v.len()).collect();
        let other_count: usize = other_dims.iter().product::<usize>().max(1);

        let mut sums = vec![0.0; ka];
        let mut x = vec![0.0; n];
        let mut idx = vec![0usize; other.len()];
        for flat in 0..other_count {
            flat_to_multi(flat, &other_dims, &mut idx);
            // Place the fixed other-axis coordinates.
            let mut oi = 0;
            for (m, slot) in x.iter_mut().enumerate() {
                if m != axis {
                    *slot = other[oi][idx[oi]];
                    oi += 1;
                }
            }
            for (i, &xl) in lam_axis.iter().enumerate() {
                x[axis] = xl;
                sums[i] += self.eval(&x);
            }
        }
        let inv = 1.0 / other_count as f64;
        let node_means: Vec<f64> = sums.iter().map(|&s| s * inv).collect();
        let grand = node_means.iter().sum::<f64>() / ka as f64;
        // Interpolate the marginal through axis l's nodes with the Lagrange
        // weights — an exact barycentric passing through (λ_l(i), node_means[i]).
        let weights = crate::decouple::lagrange_coeffs(&lam_axis);
        let function = BarycentricRational1D {
            nodes: lam_axis,
            values: node_means.clone(),
            weights,
        };
        Some((function, node_means, grand))
    }

    fn total_variance(&self, grand: f64) -> f64 {
        let nodes = self.nodes();
        let dims: Vec<usize> = nodes.iter().map(Vec::len).collect();
        let count: usize = dims.iter().product::<usize>().max(1);
        let mut x = vec![0.0; nodes.len()];
        let mut idx = vec![0usize; nodes.len()];
        let mut acc = 0.0;
        for flat in 0..count {
            flat_to_multi(flat, &dims, &mut idx);
            for (m, slot) in x.iter_mut().enumerate() {
                *slot = nodes[m][idx[m]];
            }
            let d = self.eval(&x) - grand;
            acc += d * d;
        }
        acc / count as f64
    }
}

/// Mean of a node set (the centroid coordinate used as the canonical boundary).
fn centroid(lam: &[f64]) -> f64 {
    lam.iter().sum::<f64>() / lam.len() as f64
}

/// Population variance of `vals` about a supplied mean `mean`.
fn variance(vals: &[f64], mean: f64) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64
}

/// RMS variation of a 1-D barycentric across its own nodes (about their mean) — a
/// magnitude proxy for ranking same-kind contributions.
fn rms_variation(f: &BarycentricRational1D) -> f64 {
    let m = f.values.iter().sum::<f64>() / f.values.len() as f64;
    variance(&f.values, m).sqrt()
}

/// Assemble a [`BarycentricRational1D`] from the reduced numerator/denominator
/// weight vectors along an axis: `weights = den`, `values = num/den` (dead node
/// `0/0 → 0`). Mirrors the private assembler in [`crate::univariate`].
fn assemble(nodes: Vec<f64>, num: Vec<f64>, den: Vec<f64>) -> BarycentricRational1D {
    let values = num
        .iter()
        .zip(&den)
        .map(|(&n, &d)| if d != 0.0 { n / d } else { 0.0 })
        .collect();
    BarycentricRational1D {
        nodes,
        values,
        weights: den,
    }
}

/// Contract a [`TtTensor`] over every axis except `axis` against the canonical
/// per-axis boundary weights, returning the length-`k_axis` open slice. Same
/// left/right partial-product sandwich as the fibre contraction in
/// [`crate::univariate`], but with canonical (not pinned) boundary weights.
fn tt_contract_except(tt: &TtTensor, axis: usize, pin_w: &[Vec<f64>]) -> Vec<f64> {
    let n = tt.dims.len();
    let mut left = vec![1.0]; // 1 × r_0
    for (l, w) in pin_w.iter().enumerate().take(axis) {
        left = vec_times_core_weighted(&left, tt, l, w);
    }
    let mut right = vec![1.0]; // r_n × 1
    for l in (axis + 1..n).rev() {
        right = core_weighted_times_vec(tt, l, &pin_w[l], &right);
    }
    let kl = tt.dims[axis];
    let r0 = tt.ranks[axis];
    let r1 = tt.ranks[axis + 1];
    let core = &tt.cores[axis];
    let mut out = vec![0.0; kl];
    for (i, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (a, &la) in left.iter().enumerate().take(r0) {
            if la == 0.0 {
                continue;
            }
            let base = (a * kl + i) * r1;
            let mut t = 0.0;
            for (b, &rb) in right.iter().enumerate().take(r1) {
                t += core[base + b] * rb;
            }
            s += la * t;
        }
        *o = s;
    }
    out
}

/// `left (1×r_l) · M_l → 1×r_{l+1}` where `M_l = Σ_i w[i] G_l[i]`.
fn vec_times_core_weighted(left: &[f64], tt: &TtTensor, l: usize, w: &[f64]) -> Vec<f64> {
    let kl = tt.dims[l];
    let r0 = tt.ranks[l];
    let r1 = tt.ranks[l + 1];
    let core = &tt.cores[l];
    let mut out = vec![0.0; r1];
    for (a, &la) in left.iter().enumerate().take(r0) {
        if la == 0.0 {
            continue;
        }
        for (i, &wi) in w.iter().enumerate().take(kl) {
            if wi == 0.0 {
                continue;
            }
            let base = (a * kl + i) * r1;
            let coef = la * wi;
            for (b, o) in out.iter_mut().enumerate() {
                *o += coef * core[base + b];
            }
        }
    }
    out
}

/// `M_l · right (r_{l+1}×1) → r_l×1` where `M_l = Σ_i w[i] G_l[i]`.
fn core_weighted_times_vec(tt: &TtTensor, l: usize, w: &[f64], right: &[f64]) -> Vec<f64> {
    let kl = tt.dims[l];
    let r0 = tt.ranks[l];
    let r1 = tt.ranks[l + 1];
    let core = &tt.cores[l];
    let mut out = vec![0.0; r0];
    for (a, o) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for (i, &wi) in w.iter().enumerate().take(kl) {
            if wi == 0.0 {
                continue;
            }
            let base = (a * kl + i) * r1;
            let mut t = 0.0;
            for (b, &rb) in right.iter().enumerate().take(r1) {
                t += core[base + b] * rb;
            }
            s += wi * t;
        }
        *o = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decouple::decouple_with_degrees;

    // A genuinely coupled rational in 3 variables (cross terms ⇒ TT rank > 1).
    fn coupled() -> crate::decouple::DecoupledModel {
        let f = |x: &[f64]| (1.0 + x[0] * x[1] + 2.0 * x[2]) / (3.0 + x[0] + x[1] * x[2]);
        decouple_with_degrees(&f, &[(0.2, 1.2); 3], &[1, 1, 1])
            .unwrap()
            .0
    }

    #[test]
    fn all_three_return_one_function_per_axis() {
        let m = coupled();
        for kind in [
            ContributionKind::RankContracted,
            ContributionKind::Marginal,
            ContributionKind::Sobol,
        ] {
            let cs = m.contributions(kind);
            assert_eq!(cs.len(), 3);
            for (l, c) in cs.iter().enumerate() {
                assert_eq!(c.axis, l);
                assert_eq!(c.function.len(), m.lambda[l].len());
                // Each is a runnable, finite 1-D barycentric off-node.
                let v = c.function.eval(0.55);
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn rank_factor_interpolates_its_own_nodes() {
        // The rank-contracted factor is a barycentric through (λ_l(i), value_i);
        // it must interpolate those node values exactly.
        let m = coupled();
        for l in 0..3 {
            let c = m.rank_contracted_factor(l).unwrap();
            for (i, &node) in c.function.nodes.iter().enumerate() {
                assert!((c.function.eval(node) - c.function.values[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn rank_factor_matches_full_eval_at_canonical_point() {
        // With every other axis fixed at its node centroid, the rank-contracted
        // factor must reproduce the full model along the remaining axis (it is the
        // exact fibre through the canonical point).
        let m = coupled();
        let cents: Vec<f64> = m.lambda.iter().map(|l| centroid(l)).collect();
        for l in 0..3 {
            let c = m.rank_contracted_factor(l).unwrap();
            for &t in &[0.3, 0.6, 0.9, 1.1] {
                let mut x = cents.clone();
                x[l] = t;
                assert!((c.function.eval(t) - m.eval(&x)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn marginal_node_values_equal_average_over_other_axes() {
        // Spot-check the marginal definition by recomputing the average by hand.
        let m = coupled();
        let c = m.marginal(0).unwrap();
        let (l1, l2) = (m.lambda[1].clone(), m.lambda[2].clone());
        for (i, &x0) in m.lambda[0].clone().iter().enumerate() {
            let mut s = 0.0;
            for &y in &l1 {
                for &z in &l2 {
                    s += m.eval(&[x0, y, z]);
                }
            }
            let avg = s / (l1.len() * l2.len()) as f64;
            assert!((c.function.values[i] - avg).abs() < 1e-10);
        }
    }

    #[test]
    fn sobol_indices_are_fractions_and_sum_sensibly() {
        let m = coupled();
        let mut total = 0.0;
        for l in 0..3 {
            let s = m.sobol_main_effect(l).unwrap().magnitude;
            assert!((0.0..=1.0).contains(&s), "S_{l} = {s} out of range");
            total += s;
        }
        // Main effects (no interactions) sum to ≤ 1 + slack; for a mildly coupled
        // function they explain most of the variance.
        assert!(total <= 1.0 + 1e-9, "Σ main effects = {total} > 1");
        assert!(total > 0.0);
    }

    #[test]
    fn contribution_set_carries_dimensional_metadata() {
        // The full set reports per-axis degree and representation rank; the manual
        // single-axis call agrees with the set entry.
        let m = coupled();
        let set = m.contribution_set(ContributionKind::RankContracted);
        assert_eq!(set.ndim(), 3);
        assert_eq!(set.degrees(), vec![1, 1, 1]); // degree-1 per axis
        // Each set entry equals the manual `contribution(axis, kind)`.
        for (l, axis) in set.axes.iter().enumerate() {
            let manual = m.contribution(l, ContributionKind::RankContracted).unwrap();
            assert_eq!(manual.degree, axis.degree);
            assert_eq!(manual.representation_rank, axis.representation_rank);
            assert_eq!(manual.kind, ContributionKind::RankContracted);
        }
        // Representation ranks are ≥ 1 and the max matches the model's TT bond.
        assert!(set.representation_ranks().iter().all(|&r| r >= 1));
        assert_eq!(
            set.max_representation_rank(),
            m.to_compressed().denom.max_rank()
        );
    }

    #[test]
    fn ranked_orders_axes_by_magnitude() {
        let m = coupled();
        let set = m.contribution_set(ContributionKind::Sobol);
        let ranked = set.ranked();
        assert_eq!(ranked.len(), 3);
        for w in ranked.windows(2) {
            assert!(w[0].1 >= w[1].1, "not descending: {ranked:?}");
        }
    }

    #[test]
    fn agrees_across_model_representations() {
        // Same contributions whether computed on the decoupled, dense, or TT model.
        let dm = coupled();
        let bary = dm.to_barycentric();
        let tt = bary.to_compressed();
        for l in 0..3 {
            let a = dm.rank_contracted_factor(l).unwrap();
            let b = bary.rank_contracted_factor(l).unwrap();
            let c = tt.rank_contracted_factor(l).unwrap();
            for &t in &[0.3, 0.7, 1.0] {
                let (va, vb, vc) = (a.function.eval(t), b.function.eval(t), c.function.eval(t));
                assert!((va - vb).abs() < 1e-9 && (vb - vc).abs() < 1e-9);
            }
        }
    }
}
