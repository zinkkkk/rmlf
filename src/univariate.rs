//! Convert a multivariate model into **traditional single-variable barycentric
//! rational functions** — the `(nodes, values, weights)` triple every classic
//! 1-D barycentric routine expects.
//!
//! ## What "convert to single-variable barycentric" means here
//!
//! A 1-D barycentric rational is
//!
//! ```text
//!          Σ_j weights[j] · values[j] / (x − nodes[j])
//! r(x) =  ---------------------------------------------
//!          Σ_j weights[j]            / (x − nodes[j])
//! ```
//!
//! The multivariate model this crate builds is the *same* form with the rank-1
//! product weight `∏_l 1/(x_l − λ_l(j_l))`:
//!
//! ```text
//!          Σ_J c(J) w(J) ∏_l 1/(x_l − λ_l(j_l))
//! g(x) =  -------------------------------------
//!          Σ_J c(J)      ∏_l 1/(x_l − λ_l(j_l))
//! ```
//!
//! Because that product **factorizes per axis**, fixing every coordinate except
//! one (`axis = l`, the others pinned to a point `pin`) collapses `g` *exactly*
//! to a 1-D barycentric rational in `x_l`: contract the coefficient tensors over
//! the pinned axes and what survives is a length-`k_l` triple
//!
//! ```text
//! weights[j] = Σ_{J_{−l}} c(J)      ∏_{m≠l} 1/(pin_m − λ_m(j_m))   (= the
//!                                                      denominator coefficients)
//! values[j]  = (Σ_{J_{−l}} c(J) w(J) ∏_{m≠l} 1/(pin_m − λ_m(j_m))) / weights[j]
//! nodes[j]   = λ_l(j)
//! ```
//!
//! This is **exact**, not an approximation: `r(x_l) = g(pin with x_l in slot l)`
//! for all `x_l`. So [`Barycentric::fibre`] / [`crate::DecoupledModel::fibre`] /
//! [`CompressedBarycentric::fibre`] turn the n-D model into a genuine 1-D
//! barycentric rational along any chosen axis through any chosen point.
//!
//! ## How many, and "does it depend on the degree?"
//!
//! A *single* fibre per axis only captures the model along that one pinned line.
//! To represent the **whole** model as a set of 1-D functions you need the
//! variable-decoupling (KST) structure of the follow-up paper (see
//! [`crate::decouple()`]): the model is a sum of Kronecker products of grid-line
//! slices, and there are
//!
//! ```text
//! 1 + k_1 + k_1 k_2 + … + k_1 ⋯ k_{n−1}
//! ```
//!
//! of them (e.g. 7 for Pólya–Szegő at degree 1 in 3 variables) — so the count
//! *does* depend on the per-axis degree `k_l − 1`, not just on `n`.
//! [`crate::DecoupledModel::fibre_set`] returns exactly that canonical set, each entry a
//! `(nodes, values, weights)` triple, pinned at the grid points whose Kronecker
//! products reconstruct the model (the rational analogue of the KST inner/outer
//! functions). For the dense [`Barycentric`] / [`CompressedBarycentric`] the same
//! set is available via [`Fibres::fibre_set`], which pin at the canonical
//! grid lines and contract directly.

use crate::eval::{Barycentric, axis_weight};
use crate::grid::flat_to_multi;
use crate::model::Model;
use crate::tt::CompressedBarycentric;

/// A traditional single-variable barycentric rational, the `(nodes, values,
/// weights)` triple. Evaluates as
/// `Σ_j weights[j] values[j]/(x−nodes[j]) / Σ_j weights[j]/(x−nodes[j])`.
///
/// This is the format every classic 1-D barycentric / AAA routine consumes; it
/// is produced from a multivariate model by pinning all but one variable (see the
/// module docs).
#[derive(Debug, Clone)]
pub struct BarycentricRational1D {
    /// The support abscissae `λ_j` (distinct).
    pub nodes: Vec<f64>,
    /// The value at each node — the function interpolates these
    /// (`r(nodes[j]) = values[j]`).
    pub values: Vec<f64>,
    /// The barycentric weights `w_j`.
    pub weights: Vec<f64>,
}

impl BarycentricRational1D {
    /// Build from the three arrays, checking they share one length and the nodes
    /// are distinct. Returns `None` on a length mismatch or a repeated node.
    pub fn new(nodes: Vec<f64>, values: Vec<f64>, weights: Vec<f64>) -> Option<Self> {
        if nodes.len() != values.len() || nodes.len() != weights.len() || nodes.is_empty() {
            return None;
        }
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if nodes[i] == nodes[j] {
                    return None;
                }
            }
        }
        Some(Self {
            nodes,
            values,
            weights,
        })
    }

    /// Number of support nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the function has no nodes (never true for a `new`-built one).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The precomputed `weights[j] · values[j]` (the barycentric numerator
    /// coefficients), so an evaluation loop can use `nodes`, `weights`, and this
    /// product directly without re-multiplying each call.
    pub fn weighted_values(&self) -> Vec<f64> {
        self.weights
            .iter()
            .zip(&self.values)
            .map(|(&w, &v)| w * v)
            .collect()
    }

    /// The **raw barycentric node-sum** `Σ_j weights[j] / (x − nodes[j])` — the
    /// un-normalized partial sum, *not* the ratio [`Self::eval`] returns.
    ///
    /// This is the quantity that composes multiplicatively in the
    /// [`ChannelDecomposition`]: a channel function contributes its node-sum to a
    /// matrix product across axes, and only the final numerator/denominator pair is
    /// divided. (For a standalone 1-D rational you want [`Self::eval`]; this is for
    /// inspecting/reassembling channel contributions.) Returns `+∞`/`NaN` on a node
    /// — channel sums are only combined off-node.
    pub fn barycentric_sum(&self, x: f64) -> f64 {
        self.nodes
            .iter()
            .zip(&self.weights)
            .map(|(&n, &w)| w / (x - n))
            .sum()
    }

    /// Evaluate `r(x)` via the barycentric formula. On a node it returns that
    /// node's value (the removable-singularity limit); off all nodes it is the
    /// ratio of the two sums. A degenerate `0/0` (zero weight mass at a node)
    /// returns a finite `0`, matching the multivariate eval guards.
    pub fn eval(&self, x: f64) -> f64 {
        if let Some(j) = self.nodes.iter().position(|&v| v == x) {
            return self.values[j];
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..self.nodes.len() {
            let t = self.weights[j] / (x - self.nodes[j]);
            num += t * self.values[j];
            den += t;
        }
        if den == 0.0 {
            return if num == 0.0 { 0.0 } else { num / den };
        }
        num / den
    }
}

/// Contract a dense row-major coefficient tensor `coeff` (shape `dims`) over
/// **every axis except `axis`**, against the per-axis weight vectors `pin_w`
/// (`pin_w[m]` the length-`dims[m]` weight vector for the pinned coordinate on
/// axis `m`; the entry for `axis` is ignored). Returns the length-`dims[axis]`
/// reduced vector
/// `out[p] = Σ_{J: j_axis = p} coeff[J] ∏_{m≠axis} pin_w[m][j_m]`.
fn contract_except(coeff: &[f64], dims: &[usize], axis: usize, pin_w: &[Vec<f64>]) -> Vec<f64> {
    let n = dims.len();
    let kl = dims[axis];
    let mut out = vec![0.0; kl];
    let mut idx = vec![0usize; n];
    for (flat, &cf) in coeff.iter().enumerate() {
        flat_to_multi(flat, dims, &mut idx);
        let mut wprod = 1.0;
        for (m, w) in pin_w.iter().enumerate() {
            if m != axis {
                wprod *= w[idx[m]];
            }
        }
        out[idx[axis]] += cf * wprod;
    }
    out
}

/// Assemble a [`BarycentricRational1D`] along `axis` from the reduced denominator
/// and numerator weight vectors and the axis's nodes. `weights = den`,
/// `values = num/den` (the limit value `0/0 → 0` for a dead node).
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

/// **1-D slice views** of a multivariate model — `fibre`, `fibre_set`,
/// `fibres_by_axis`.
///
/// A *fibre* is the model restricted to one axis with the other coordinates pinned
/// (see the module docs): `m.fibre(l, pin).eval(t) == m.eval(pin with t in slot l)`,
/// exactly. This is an rmlf utility (not paper terminology) for extracting the
/// model along a line as a classic 1-D [`BarycentricRational1D`].
///
/// Implemented by [`Barycentric`], [`crate::DecoupledModel`], and
/// [`CompressedBarycentric`]. Each provides its own `fibre` (the per-axis
/// contraction differs by representation); `fibre_set` / `fibres_by_axis` are
/// shared defaults built from it.
///
/// > **Not to be confused with** [`crate::DecoupledModel::decoupled_functions`], which
/// > returns the **KST decoupled factors** (Φ) — multiplicative building blocks,
/// > *not* model slices. A fibre evaluates the model; a Φ factor does not.
pub trait Fibres {
    /// Per-axis column support nodes `λ` (needed to enumerate the canonical set).
    fn nodes(&self) -> &[Vec<f64>];

    /// The 1-D barycentric rational obtained by pinning every variable except
    /// `axis` to the coordinates of `pin` (the `axis` slot is ignored). Exact
    /// along `axis`. `None` if `axis` is out of range or `pin` has the wrong
    /// length. Each model implements its own contraction.
    fn fibre(&self, axis: usize, pin: &[f64]) -> Option<BarycentricRational1D>;

    /// The **canonical set** of fibres pinned at the grid lines whose Kronecker
    /// products reconstruct the model (leading axes over their column nodes,
    /// trailing axes at their last node). `1 + k_0 + k_0 k_1 + …` entries, each a
    /// labelled 1-D barycentric. Shared default built from [`Fibres::fibre`].
    fn fibre_set(&self) -> Vec<LabelledFibre> {
        let nodes = self.nodes();
        fibre_set_via(nodes.len(), nodes, |axis, pin| {
            self.fibre(axis, pin).unwrap()
        })
    }

    /// [`Fibres::fibre_set`] grouped one bucket per axis (labels dropped):
    /// `out[l]` is axis `l`'s sections as a `Vec<BarycentricRational1D>`.
    fn fibres_by_axis(&self) -> Vec<Vec<BarycentricRational1D>> {
        let n = self.nodes().len();
        group_by_axis(n, self.fibre_set())
    }
}

impl Fibres for Barycentric {
    fn nodes(&self) -> &[Vec<f64>] {
        &self.lambda
    }
    fn fibre(&self, axis: usize, pin: &[f64]) -> Option<BarycentricRational1D> {
        let n = Model::ndim(self);
        if axis >= n || pin.len() != n {
            return None;
        }
        let dims: Vec<usize> = self.lambda.iter().map(Vec::len).collect();
        let pin_w: Vec<Vec<f64>> = self
            .lambda
            .iter()
            .zip(pin)
            .map(|(lam, &p)| axis_weight(lam, p))
            .collect();
        let den = contract_except(&self.c, &dims, axis, &pin_w);
        let cw: Vec<f64> = self.c.iter().zip(&self.w).map(|(&c, &w)| c * w).collect();
        let num = contract_except(&cw, &dims, axis, &pin_w);
        Some(assemble(self.lambda[axis].clone(), num, den))
    }
}

impl Fibres for crate::decouple::DecoupledModel {
    fn nodes(&self) -> &[Vec<f64>] {
        &self.lambda
    }
    fn fibre(&self, axis: usize, pin: &[f64]) -> Option<BarycentricRational1D> {
        let n = Model::ndim(self);
        if axis >= n || pin.len() != n {
            return None;
        }
        let dims = self.sizes();
        let pin_w: Vec<Vec<f64>> = self
            .lambda
            .iter()
            .zip(pin)
            .map(|(lam, &p)| axis_weight(lam, p))
            .collect();
        // Numerator and denominator tensors are contracted independently (the
        // decoupling keeps them separate).
        let den = contract_except(&self.c_den, &dims, axis, &pin_w);
        let num = contract_except(&self.c_num, &dims, axis, &pin_w);
        Some(assemble(self.lambda[axis].clone(), num, den))
    }
}

impl Fibres for CompressedBarycentric {
    fn nodes(&self) -> &[Vec<f64>] {
        &self.lambda
    }
    fn fibre(&self, axis: usize, pin: &[f64]) -> Option<BarycentricRational1D> {
        let n = Model::ndim(self);
        if axis >= n || pin.len() != n {
            return None;
        }
        // The pinned axes are contracted out of the TT cores directly —
        // `O(Σ_l k_l r²)`, never materializing the dense tensor.
        let pin_w: Vec<Vec<f64>> = self
            .lambda
            .iter()
            .zip(pin)
            .map(|(lam, &p)| axis_weight(lam, p))
            .collect();
        let den = tt_contract_except(&self.denom, axis, &pin_w);
        let num = tt_contract_except(&self.numer, axis, &pin_w);
        Some(assemble(self.lambda[axis].clone(), num, den))
    }
}

impl CompressedBarycentric {
    /// The **channel decomposition** — the format where the contributions of
    /// *every* axis are built into a single set of 1-D barycentrics that
    /// **reassemble the whole model**.
    ///
    /// The TT structure factorizes both coefficient tensors as a matrix product
    /// across axes:
    ///
    /// ```text
    /// D(x) = M^den_0(x_0) · M^den_1(x_1) ⋯ M^den_{n−1}(x_{n−1})
    /// N(x) = M^num_0(x_0) · M^num_1(x_1) ⋯ M^num_{n−1}(x_{n−1})
    /// g(x) = N(x) / D(x)
    /// ```
    ///
    /// where each `M_l(x_l)` is an `r_l × r_{l+1}` matrix whose `(a,b)` entry is
    /// the single-variable barycentric node-sum
    /// `Σ_i G_l[a,i,b] / (x_l − λ_l(i))` ([`BarycentricRational1D::barycentric_sum`]).
    /// Each such entry is a genuine 1-D function of `x_l` alone — a **channel** —
    /// and the rank index `(a → b)` threads the channels across axes (channel `a`
    /// out of axis `l−1` feeds channel `a` into axis `l`). So the full set is the
    /// per-axis channel matrices for the numerator and the denominator; multiplying
    /// them back ([`ChannelDecomposition::reconstruct`]) recovers `g(x)` exactly.
    /// This is what lets you investigate **individual contributions and
    /// interactions**: a channel's magnitude/pole pattern is its own 1-D function,
    /// and which channels couple is the rank structure.
    ///
    /// (Rank-1 / separable models have all `r_l = 1`, so this collapses to *one*
    /// channel per axis and `g = ∏ num_l / ∏ den_l` — the classic separable
    /// product. Higher TT-rank is the genuine multivariate coupling.)
    pub fn channel_decomposition(&self) -> ChannelDecomposition {
        ChannelDecomposition {
            numerator: axis_channels(&self.numer, &self.lambda),
            denominator: axis_channels(&self.denom, &self.lambda),
        }
    }
}

/// One axis's matrix of single-variable **channels** from one TT tensor (the
/// numerator or denominator). The channels form an `rank_in × rank_out` matrix
/// `M_l(x_l)` whose `(a,b)` entry is `channels[a * rank_out + b]`, a 1-D
/// barycentric node-sum in `x_l`. See [`ChannelDecomposition`].
#[derive(Debug, Clone)]
pub struct AxisChannels {
    /// The variable `x_l` these channels are functions of.
    pub axis: usize,
    /// Incoming TT rank `r_l` (matrix rows; `1` for the first axis).
    pub rank_in: usize,
    /// Outgoing TT rank `r_{l+1}` (matrix columns; `1` for the last axis).
    pub rank_out: usize,
    /// The `rank_in × rank_out` channel barycentrics, row-major
    /// (`channels[a * rank_out + b]`). Evaluate a channel's contribution with
    /// [`BarycentricRational1D::barycentric_sum`] (the entries combine as a matrix
    /// product, not as normalized ratios).
    pub channels: Vec<BarycentricRational1D>,
}

impl AxisChannels {
    /// The contracted channel matrix `M_l(x_l)` at coordinate `x_l`, row-major
    /// (`rank_in × rank_out`): entry `(a,b)` is `channels[a*rank_out+b]
    /// .barycentric_sum(x_l)`.
    pub fn matrix(&self, xl: f64) -> Vec<f64> {
        self.channels
            .iter()
            .map(|c| c.barycentric_sum(xl))
            .collect()
    }
}

/// The full **channel decomposition** of a [`CompressedBarycentric`]: the per-axis
/// channel matrices for the numerator and the denominator. The model is
/// `g(x) = (∏_l M^num_l(x_l)) / (∏_l M^den_l(x_l))` (matrix products collapsing to
/// scalars at the rank-1 boundaries). See
/// [`CompressedBarycentric::channel_decomposition`] for the math.
#[derive(Debug, Clone)]
pub struct ChannelDecomposition {
    /// Per-axis channel matrices of the **numerator** tensor `N = C ⊙ W`.
    pub numerator: Vec<AxisChannels>,
    /// Per-axis channel matrices of the **denominator** tensor `C`.
    pub denominator: Vec<AxisChannels>,
}

impl ChannelDecomposition {
    /// Number of variables `n`.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.denominator.len()
    }

    /// Reassemble the model `g(x) = N(x)/D(x)` from the channels by multiplying
    /// the per-axis channel matrices across axes — the proof that the per-axis 1-D
    /// pieces combine back into the whole. Equals
    /// [`CompressedBarycentric::eval`] to machine precision off-node.
    pub fn reconstruct(&self, x: &[f64]) -> f64 {
        let num = chain_matrices(&self.numerator, x);
        let den = chain_matrices(&self.denominator, x);
        if den == 0.0 && num == 0.0 {
            return 0.0;
        }
        num / den
    }
}

/// Multiply the per-axis channel matrices `M_0(x_0) · M_1(x_1) ⋯` left-to-right,
/// returning the resulting `1×1` scalar (the boundary ranks are 1).
fn chain_matrices(axes: &[AxisChannels], x: &[f64]) -> f64 {
    // Running row vector `acc` of length `rank_in` of the current axis (starts
    // 1×1 = [1]). At each axis, `acc ← acc · M_l(x_l)`.
    let mut acc = vec![1.0_f64];
    for ax in axes {
        let m = ax.matrix(x[ax.axis]); // rank_in × rank_out, row-major
        let (ri, ro) = (ax.rank_in, ax.rank_out);
        debug_assert_eq!(acc.len(), ri);
        let mut next = vec![0.0; ro];
        for (a, &av) in acc.iter().enumerate() {
            if av == 0.0 {
                continue;
            }
            for (b, nb) in next.iter_mut().enumerate() {
                *nb += av * m[a * ro + b];
            }
        }
        acc = next;
    }
    debug_assert_eq!(acc.len(), 1);
    acc[0]
}

/// Build one TT tensor's per-axis channel barycentrics. Channel `(a,b)` of axis
/// `l` has nodes `λ_l`, weights `G_l[a, ·, b]` (the core slice), and node values
/// `1.0` (a channel is its weight/pole structure; combine via
/// [`BarycentricRational1D::barycentric_sum`], not [`BarycentricRational1D::eval`]).
fn axis_channels(tt: &crate::tt::TtTensor, lambda: &[Vec<f64>]) -> Vec<AxisChannels> {
    let n = tt.dims.len();
    let mut out = Vec::with_capacity(n);
    // `l` indexes several parallel arrays (dims, ranks, cores, lambda).
    #[allow(clippy::needless_range_loop)]
    for l in 0..n {
        let kl = tt.dims[l];
        let ri = tt.ranks[l];
        let ro = tt.ranks[l + 1];
        let core = &tt.cores[l];
        let nodes = &lambda[l];
        let mut channels = Vec::with_capacity(ri * ro);
        for a in 0..ri {
            for b in 0..ro {
                // weights[i] = G_l[a, i, b] at core[(a*kl + i)*ro + b].
                let weights: Vec<f64> = (0..kl).map(|i| core[(a * kl + i) * ro + b]).collect();
                channels.push(BarycentricRational1D {
                    nodes: nodes.clone(),
                    values: vec![1.0; kl],
                    weights,
                });
            }
        }
        out.push(AxisChannels {
            axis: l,
            rank_in: ri,
            rank_out: ro,
            channels,
        });
    }
    out
}

/// Contract a [`crate::tt::TtTensor`] over every axis except `axis` against the
/// pinned per-axis weight vectors, returning the length-`k_axis` reduced vector
/// (the axis-`axis` "open" slice of the TT). Left and right partial products are
/// formed as `1×r` / `r×1` row vectors and combined through the open core.
fn tt_contract_except(tt: &crate::tt::TtTensor, axis: usize, pin_w: &[Vec<f64>]) -> Vec<f64> {
    let n = tt.dims.len();
    // Per-axis contracted core matrix M_l = Σ_i pin_w[l][i] G_l[i], shape
    // r_l × r_{l+1}, for every pinned axis. We never form M_axis; instead we keep
    // the open core and sandwich it between the left/right partial products.

    // Left product L: 1 × r_axis (product of M_0 … M_{axis-1}).
    let mut left = vec![1.0]; // 1 × r_0 (= 1)
    for (l, w) in pin_w.iter().enumerate().take(axis) {
        left = vec_times_core_weighted(&left, tt, l, w);
    }
    // Right product R: r_{axis+1} × 1 (product of M_{axis+1} … M_{n-1}).
    let mut right = vec![1.0]; // r_n × 1 (= 1)
    for l in (axis + 1..n).rev() {
        right = core_weighted_times_vec(tt, l, &pin_w[l], &right);
    }

    // Open core at `axis`: for each node i, value = left · G_axis[i] · right.
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
fn vec_times_core_weighted(
    left: &[f64],
    tt: &crate::tt::TtTensor,
    l: usize,
    w: &[f64],
) -> Vec<f64> {
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
fn core_weighted_times_vec(
    tt: &crate::tt::TtTensor,
    l: usize,
    w: &[f64],
    right: &[f64],
) -> Vec<f64> {
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

/// One labelled section of a [`Fibres::fibre_set`]: the 1-D
/// barycentric together with the axis it varies and the leading multi-index (the
/// pinned column indices of axes `0..axis`) that places it on its grid line.
/// Carries enough context to analyze each single-variable section's structure and
/// its contribution to the overall approximation.
#[derive(Debug, Clone)]
pub struct LabelledFibre {
    /// The variable this section is a function of.
    pub axis: usize,
    /// Pinned column indices of the leading axes `0..axis` (empty for axis 0); the
    /// trailing axes `axis+1..n` are pinned to their last node.
    pub lead: Vec<usize>,
    /// The single-variable barycentric rational `(nodes, values, weights)`.
    pub bary: BarycentricRational1D,
}

/// Group a flat [`Fibres::fibre_set`] into one bucket per axis,
/// dropping the labels. `out[l]` is the `Vec<BarycentricRational1D>` of axis `l`'s
/// sections (in canonical leading-multi-index order) — ready to index and
/// [`BarycentricRational1D::eval`] directly. `out.len() == n`.
fn group_by_axis(n: usize, set: Vec<LabelledFibre>) -> Vec<Vec<BarycentricRational1D>> {
    let mut out: Vec<Vec<BarycentricRational1D>> = vec![Vec::new(); n];
    for lf in set {
        out[lf.axis].push(lf.bary);
    }
    out
}

/// Shared driver for the canonical decoupling set: for each axis `l`, range the
/// leading axes `0..l` over their column nodes (trailing axes pinned to their
/// last node) and call `fibre(l, pin)` for each leading multi-index. Mirrors
/// [`crate::decouple::DecoupledModel::decoupled_functions`]'s pinning convention,
/// so the two agree term-for-term. Returns the flat labelled list (axis-major,
/// then canonical leading-multi-index order).
fn fibre_set_via(
    n: usize,
    lambda: &[Vec<f64>],
    fibre: impl Fn(usize, &[f64]) -> BarycentricRational1D,
) -> Vec<LabelledFibre> {
    let k: Vec<usize> = lambda.iter().map(Vec::len).collect();
    let mut out = Vec::new();
    for l in 0..n {
        let lead_dims = &k[..l];
        let count: usize = lead_dims.iter().product::<usize>().max(1);
        let mut lead = vec![0usize; l];
        for flat in 0..count {
            flat_to_multi(flat, lead_dims, &mut lead);
            // Pin: leading axes at their column index, trailing at last node.
            let mut pin = vec![0.0; n];
            for (j, &li) in lead.iter().enumerate() {
                pin[j] = lambda[j][li];
            }
            for (m, p) in pin.iter_mut().enumerate().take(n).skip(l + 1) {
                *p = *lambda[m].last().unwrap();
            }
            out.push(LabelledFibre {
                axis: l,
                lead: lead.clone(),
                bary: fibre(l, &pin),
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decouple::decouple_with_degrees;
    use crate::points::Points;
    use crate::recurse::loewner_null_rec_svd0;
    use crate::tensor::sample;
    use approx::assert_relative_eq;

    fn polya_model() -> Barycentric {
        // p = x1 x2 + x1 x3 + x2 x3 via the decoupling, as a Barycentric.
        let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
        let (dm, _) = decouple_with_degrees(&f, &[(-1.0, 1.0); 3], &[1, 1, 1]).unwrap();
        dm.to_barycentric()
    }

    #[test]
    fn bary1d_eval_matches_definition() {
        // r interpolates its node values and matches the explicit barycentric sum.
        let r = BarycentricRational1D::new(
            vec![0.0, 1.0, 2.0],
            vec![3.0, 5.0, 11.0],
            vec![1.0, -2.0, 1.0],
        )
        .unwrap();
        // On nodes: returns the values.
        for (j, &node) in r.nodes.iter().enumerate() {
            assert_relative_eq!(r.eval(node), r.values[j], epsilon = 1e-14);
        }
        // Off node: matches the formula via weighted_values.
        let wv = r.weighted_values();
        let x = 0.37;
        let mut num = 0.0;
        let mut den = 0.0;
        for (j, &node) in r.nodes.iter().enumerate() {
            num += wv[j] / (x - node);
            den += r.weights[j] / (x - node);
        }
        assert_relative_eq!(r.eval(x), num / den, epsilon = 1e-14);
    }

    #[test]
    fn fibre_is_exact_along_axis() {
        // f = 1/((x1+3)(x2+2)) — a clean rational; the fibre along each axis must
        // reproduce the full model on that line, off-grid.
        let f = |x: &[f64]| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0));
        let pts = Points::new(
            vec![vec![0.0, 1.0], vec![0.0, 1.0]],
            vec![vec![0.5, 1.5], vec![0.5, 1.5]],
        )
        .unwrap();
        let tab = sample(&f, &pts);
        let c = loewner_null_rec_svd0(&pts, tab.view()).unwrap().c;
        let w = crate::eval::column_values(&pts, tab.view());
        let model = Barycentric {
            lambda: pts.column.clone(),
            w,
            c,
        };
        let pin = [0.42, -0.17];
        // Axis 0 free.
        let r0 = model.fibre(0, &pin).unwrap();
        for &t in &[-0.6, 0.3, 0.95, 2.1] {
            assert_relative_eq!(r0.eval(t), model.eval(&[t, pin[1]]), epsilon = 1e-12);
        }
        // Axis 1 free.
        let r1 = model.fibre(1, &pin).unwrap();
        for &t in &[-0.5, 0.2, 1.3] {
            assert_relative_eq!(r1.eval(t), model.eval(&[pin[0], t]), epsilon = 1e-12);
        }
    }

    #[test]
    fn fibre_set_count_matches_decoupled_functions() {
        let model = polya_model();
        let set = model.fibre_set();
        // 1 + 2 + 4 = 7 (k = [2,2,2]).
        assert_eq!(set.len(), 7);
        assert_eq!(set.iter().filter(|f| f.axis == 0).count(), 1);
        assert_eq!(set.iter().filter(|f| f.axis == 1).count(), 2);
        assert_eq!(set.iter().filter(|f| f.axis == 2).count(), 4);
    }

    #[test]
    fn fibre_set_slices_reconstruct_the_model() {
        // Each fibre in the set is a grid-line slice; it must match the model on
        // its line. Check the leading φ(x) = p(x, y_last, z_last).
        let model = polya_model();
        let set = model.fibre_set();
        let ylast = *model.lambda[1].last().unwrap();
        let zlast = *model.lambda[2].last().unwrap();
        let phi = &set.iter().find(|f| f.axis == 0).unwrap().bary;
        for &t in &[-0.5, 0.0, 0.7] {
            assert_relative_eq!(phi.eval(t), model.eval(&[t, ylast, zlast]), epsilon = 1e-10);
        }
    }

    #[test]
    fn decoupled_and_dense_fibres_agree() {
        // The decoupled-model fibre and the to_barycentric() fibre are the same
        // 1-D rational (they represent the same H), so they evaluate identically.
        let f = |x: &[f64]| (x[1] * x[1] + x[0] - 2.0) / (x[1] * x[1] + 2.0 * x[0] + 1.0);
        let (dm, _) = decouple_with_degrees(&f, &[(-1.0, 1.0); 2], &[1, 2]).unwrap();
        let bary = dm.to_barycentric();
        let pin = [0.3, 0.7];
        let a = dm.fibre(1, &pin).unwrap();
        let b = bary.fibre(1, &pin).unwrap();
        for &t in &[-0.4, 0.0, 0.55, 1.2] {
            assert_relative_eq!(a.eval(t), b.eval(t), epsilon = 1e-10);
        }
    }

    #[test]
    fn compressed_fibre_matches_dense() {
        // The TT-compressed fibre must equal the dense fibre (same model).
        let model = polya_model();
        let tt = CompressedBarycentric::from_model(&model, 0.0);
        let pin = [0.21, -0.4, 0.6];
        for axis in 0..3 {
            let rd = model.fibre(axis, &pin).unwrap();
            let rc = tt.fibre(axis, &pin).unwrap();
            for &t in &[-0.7, 0.1, 0.45, 0.9] {
                assert_relative_eq!(rd.eval(t), rc.eval(t), epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn fibres_by_axis_groups_and_matches_flat_set() {
        // `fibres_by_axis` is the per-axis-bucketed `fibre_set`; check the shape
        // (n buckets, right counts) and that every section evals identically to its
        // flat-set counterpart — for both the dense and TT-compressed models.
        let model = polya_model();
        let tt = CompressedBarycentric::from_model(&model, 0.0);

        for by_axis in [model.fibres_by_axis(), tt.fibres_by_axis()] {
            assert_eq!(by_axis.len(), 3); // one bucket per variable
            assert_eq!(by_axis[0].len(), 1);
            assert_eq!(by_axis[1].len(), 2);
            assert_eq!(by_axis[2].len(), 4);
        }

        // Each grouped section equals the corresponding flat-set section.
        let flat = model.fibre_set();
        let grouped = model.fibres_by_axis();
        for (axis, gbucket) in grouped.iter().enumerate() {
            let flat_axis: Vec<_> = flat.iter().filter(|f| f.axis == axis).collect();
            assert_eq!(flat_axis.len(), gbucket.len());
            for (lf, g) in flat_axis.iter().zip(gbucket) {
                for &t in &[-0.6, 0.2, 0.8] {
                    assert_relative_eq!(lf.bary.eval(t), g.eval(t), epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn channel_decomposition_reconstructs_the_model() {
        // The per-axis channel matrices, multiplied back across axes, must recover
        // the full model exactly off-node — the "all axes built in" reassembly.
        let model = polya_model();
        let tt = CompressedBarycentric::from_model(&model, 0.0);
        let cd = tt.channel_decomposition();
        assert_eq!(cd.ndim(), 3);
        // Channel matrix shapes are consistent across the chain (r_0 = r_n = 1).
        assert_eq!(cd.denominator[0].rank_in, 1);
        assert_eq!(cd.denominator[2].rank_out, 1);
        for l in 0..2 {
            assert_eq!(cd.denominator[l].rank_out, cd.denominator[l + 1].rank_in);
        }
        for x in [[0.1, -0.3, 0.5], [-0.7, 0.2, 0.9], [0.33, -0.21, 0.4]] {
            assert_relative_eq!(cd.reconstruct(&x), tt.eval(&x), epsilon = 1e-9);
        }
    }

    #[test]
    fn channel_is_a_runnable_1d_function() {
        // Each channel is a genuine 1-D barycentric: its node-sum is a finite
        // function of one variable, evaluable in isolation.
        let model = polya_model();
        let tt = CompressedBarycentric::from_model(&model, 0.0);
        let cd = tt.channel_decomposition();
        // Pick axis 1's denominator channels and check the matrix entries equal the
        // individual channels' barycentric_sum.
        let ax = &cd.denominator[1];
        let xl = 0.37;
        let mat = ax.matrix(xl);
        for (idx, ch) in ax.channels.iter().enumerate() {
            assert_relative_eq!(mat[idx], ch.barycentric_sum(xl), epsilon = 1e-14);
            assert!(ch.barycentric_sum(xl).is_finite());
        }
    }

    #[test]
    fn decoupled_fibres_by_axis_eval_individually() {
        // The DecoupledModel exposes the same per-axis buckets; each is a runnable
        // 1-D barycentric.
        let f = |x: &[f64]| (x[1] * x[1] + x[0] - 2.0) / (x[1] * x[1] + 2.0 * x[0] + 1.0);
        let (dm, _) = decouple_with_degrees(&f, &[(-1.0, 1.0); 2], &[1, 2]).unwrap();
        let by_axis = dm.fibres_by_axis();
        assert_eq!(by_axis.len(), 2);
        assert_eq!(by_axis[0].len(), 1); // φ(x): one leading section
        assert_eq!(by_axis[1].len(), 2); // ψ_j(y): one per axis-0 column node
        // The leading φ(x) is the slice at (x, y_last); it must match the model.
        let ylast = *dm.lambda[1].last().unwrap();
        let phi = &by_axis[0][0];
        for &t in &[-0.4, 0.0, 0.55] {
            assert_relative_eq!(phi.eval(t), dm.eval(&[t, ylast]), epsilon = 1e-10);
        }
    }
}
