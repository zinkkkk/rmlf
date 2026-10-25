//! Interpolation (support) points and the column/row split.
//!
//! Per variable `l`, the Loewner framework splits the sample abscissae into a
//! *column* set `lambda_l` (the support points kept in the barycentric form) and
//! a disjoint *row* set `mu_l`. Variables are referred to by their **axis index**
//! `0..n`, never by name — the MATLAB `'s1','s2',...` string arrays exist only
//! for the symbolic KST path and are dropped here.

use crate::error::{LoewnerError, Result};
use crate::scalar::Scalar;
use num_complex::Complex64;

/// Column and row interpolation points for an `n`-variable problem.
///
/// `column[l]` and `row[l]` hold the support abscissae along axis `l`.
#[derive(Debug, Clone)]
pub struct Points<S = f64> {
    pub column: Vec<Vec<S>>,
    pub row: Vec<Vec<S>>,
}

/// How to place a set of abscissae along a single axis. Each variant is a
/// self-contained placement rule producing a point set on `[lo, hi]`.
///
/// This is the placement of **one** set (a column set *or* a row set). To form the
/// disjoint column/row pair the Loewner framework needs, an [`AxisSpec`] pairs two
/// placements — by default the same rule alternating-split, but the column and row
/// can be set **independently** (different method *and* count). For arbitrary
/// user-supplied points, see [`AxisSpec::custom`].
///
/// [`Integers`]: Self::Integers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisSampling {
    /// `nip` evenly (linearly) spaced points — a `linspace`.
    Linspace(usize),
    /// `nip` Chebyshev points of the second kind (endpoints included, clustered
    /// toward the ends — well-conditioned for rational fits).
    Chebyshev(usize),
    /// `nip` points **logarithmically spaced** on `[lo, hi]` (a `logspace`).
    /// Clusters toward the small-magnitude end — the right placement when the axis
    /// spans several orders of magnitude (frequency sweeps, decay rates) and the
    /// interesting behaviour is near the low end.
    ///
    /// Requires the interval to be sign-consistent and not cross zero: either
    /// `0 < lo < hi` (clusters near `lo`) or `lo < hi < 0` (clusters near `hi`, by
    /// symmetry). A bound of exactly `0`, or `lo·hi < 0`, is rejected (log spacing
    /// is undefined through `0`).
    Logspace(usize),
    /// **Every integer** in `[ceil(lo), floor(hi)]`, endpoints included. For a
    /// discrete input axis.
    ///
    /// Unlike [`Self::Linspace`]/[`Self::Chebyshev`], the point count is **fixed by
    /// the range** (`floor(hi) − ceil(lo) + 1` integers) — there is no `nip` knob,
    /// because you cannot place more nodes than there are integers in the interval.
    /// Two consequences:
    /// * **Hard degree ceiling.** After the disjoint column/row split the
    ///   recoverable per-axis degree is ≈ `⌊count/2⌋ − 1` — the same scarcity limit
    ///   as the finite-field grid. Distinct nodes are not a soundness issue, just a
    ///   degree cap; if the target needs higher degree on this axis the only lever
    ///   is to **widen the integer range** (more nodes), not raise a `nip`.
    /// * **Not adaptively growable.** The per-axis adaptive paths
    ///   ([`crate::decouple_adaptive`]) grow `nip` on under-resolved axes; an
    ///   integer axis cannot grow (its nodes are pinned to the range). Use the
    ///   fixed-grid entry points ([`crate::alg1()`], [`crate::decouple_from_closure`])
    ///   with [`Points::sampled`] for integer axes.
    Integers,
    /// The **original MATLAB reference** placement (`mLF` demos / `examples.m`):
    /// `nip` `linspace` points shifted up by **half a step**, `x_i = lo + step·i +
    /// step/2`.
    ///
    /// ⚠️ **Not recommended — kept only for MATLAB parity.** It **deliberately steps
    /// outside `[lo, hi]`**: the last point lands at `hi + step/2`, past the upper
    /// bound, so `f` is sampled *outside the domain you specified*. Breaking the
    /// stated bounds is a poor way to obtain disjointness — it can evaluate `f` where
    /// it is undefined or meaningless. No default or automatic path ever selects this;
    /// it is purely opt-in. Prefer a within-bounds placement
    /// ([`Self::Linspace`]/[`Self::Chebyshev`]/[`Self::Logspace`]) with the
    /// alternating split ([`AxisSpec::new`]), which is disjoint *and* stays in domain.
    /// Use `Reference` only to reproduce the exact MATLAB results (e.g. via
    /// [`Points::reference`], where it is the row set paired with a `Linspace`
    /// column — the MATLAB `p_r = p_c + dx`).
    Reference(usize),
}

/// A placement that is either a built-in [`AxisSampling`] rule or a **custom**
/// user closure producing the abscissae for a given `[lo, hi]`.
///
/// A `Placement` fills **one side** (column or row), so a custom closure works the
/// same in either role — with [`AxisSpec::new`] (one rule, alternating-split: the
/// closure provides the single point set that gets split) or [`AxisSpec::split`] /
/// [`AxisSpec::custom`] (the closure provides that side in full). `AxisSampling`
/// converts into `Placement` via `Into`, so built-in and custom mix freely.
#[derive(Clone)]
pub enum Placement {
    /// A built-in rule.
    Builtin(AxisSampling),
    /// A user-supplied placement closure `|lo, hi| -> Result<Vec<f64>>`.
    ///
    /// **Contract** — the closure must:
    /// * return **≥ 2** points (a 1-D Loewner slice needs at least two abscissae);
    /// * return **distinct** values within the side it provides (no internal dupes);
    /// * stay **disjoint from the other side** of the same axis — the column and row
    ///   sets must not share any value (validated when the [`Points`] is built, which
    ///   errors with [`crate::LoewnerError::NonDisjoint`] otherwise). When pairing two
    ///   placements that both naturally include the interval endpoints, keep them
    ///   disjoint *within bounds* — e.g. interleave the two sets, or place one on a
    ///   slightly inset sub-interval. (Stepping outside the bounds, as
    ///   [`AxisSampling::Reference`] does, also disjoins them but is discouraged.)
    ///
    /// It is **strongly preferred** to keep all points inside `[lo, hi]`. Sampling
    /// outside is *allowed* (it may ignore `lo`/`hi` and return a fixed set) only
    /// when `f` is genuinely defined there — breaking the stated domain is a poor
    /// default. Return `Err(..)` to abort the fit.
    Custom(std::sync::Arc<dyn Fn(f64, f64) -> Result<Vec<f64>> + Send + Sync>),
}

impl std::fmt::Debug for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Placement::Builtin(s) => write!(f, "Builtin({s:?})"),
            Placement::Custom(_) => write!(f, "Custom(<closure>)"),
        }
    }
}

impl Placement {
    /// The abscissae this placement puts on `[lo, hi]`.
    pub fn abscissae(&self, lo: f64, hi: f64) -> Result<Vec<f64>> {
        match self {
            Placement::Builtin(s) => s.abscissae(lo, hi),
            Placement::Custom(f) => {
                let v = f(lo, hi)?;
                if v.len() < 2 {
                    return Err(LoewnerError::Degenerate(
                        "custom placement must yield >= 2 points".into(),
                    ));
                }
                Ok(v)
            }
        }
    }
}

impl From<AxisSampling> for Placement {
    fn from(s: AxisSampling) -> Self {
        Placement::Builtin(s)
    }
}

/// A per-axis sampling spec: the **column** and **row** placements, chosen
/// independently. Each side can use a different method *and* a different count
/// (e.g. column `Chebyshev(8)`, row `Linspace(10)`), or a [`Placement::Custom`]
/// closure of arbitrary points.
///
/// `AxisSampling` `impl Into<AxisSpec>` via [`AxisSpec::new`] — a single rule used
/// for **both** sides and alternating-split — so the common case stays terse: pass
/// a bare `AxisSampling` to [`Points::sampled`].
#[derive(Debug, Clone)]
pub struct AxisSpec {
    /// Placement of the column (support) set.
    pub column: Placement,
    /// Placement of the row set. Must stay disjoint from the column set (checked
    /// when the [`Points`] is built).
    pub row: Placement,
    /// When `column` and `row` are the *same* built-in rule, this flags the classic
    /// "place one set, alternating-split" behaviour ([`AxisSpec::new`]); when set,
    /// the single placement is split even→row / odd→column instead of placing two
    /// independent full sets. `false` for independent or custom placements.
    alternating_split: bool,
}

impl AxisSpec {
    /// A spec that places **one** set by `placement` and **alternating-splits** it
    /// into the disjoint column/row pair (even indices → row, odd → column). This is
    /// the well-conditioned default and what a bare `AxisSampling` converts to.
    pub fn new(placement: impl Into<Placement>) -> Self {
        let p = placement.into();
        AxisSpec {
            column: p.clone(),
            row: p,
            alternating_split: true,
        }
    }

    /// A spec with **independent** column and row placements — different method
    /// and/or count on each side. The two sets must be **disjoint** (validated when
    /// the [`Points`] is built); two placements that both include the interval
    /// endpoints (e.g. `Linspace` and `Chebyshev`) collide there, so keep them
    /// disjoint *within bounds* — e.g. place each on a slightly inset, non-
    /// overlapping sub-interval, or use a [`Placement::Custom`] closure that
    /// interleaves the two sets. (Reaching for the bound-breaking
    /// [`AxisSampling::Reference`] row is discouraged; see its docs.)
    ///
    /// ```
    /// use rmlf::points::{AxisSampling, AxisSpec};
    /// // Preferred — independent and WITHIN bounds: Chebyshev-8 columns on [0,1],
    /// // linspace-9 rows on the inset [0.05, 0.95] so the two never share a point.
    /// let inset = 0.05;
    /// let within_bounds = AxisSpec::custom(
    ///     move |lo, hi| Ok(rmlf::points::chebyshev_lobatto(lo, hi, 8)),
    ///     move |lo, hi| {
    ///         let (a, b) = (lo + inset, hi - inset);
    ///         Ok((0..9).map(|i| a + (b - a) * i as f64 / 8.0).collect())
    ///     },
    /// );
    ///
    /// // MATLAB-parity alternative (rows step OUTSIDE the bounds — see
    /// // AxisSampling::Reference; only when f is defined past the domain):
    /// let matlab_style = AxisSpec::split(AxisSampling::Chebyshev(8), AxisSampling::Reference(8));
    /// # let _ = (within_bounds, matlab_style);
    /// ```
    pub fn split(column: impl Into<Placement>, row: impl Into<Placement>) -> Self {
        AxisSpec {
            column: column.into(),
            row: row.into(),
            alternating_split: false,
        }
    }

    /// A spec with a **custom** column and row placement (arbitrary user points).
    /// Each closure maps `(lo, hi) -> Vec<f64>`; the two results must be disjoint.
    ///
    /// ```
    /// use rmlf::points::AxisSpec;
    /// let spec = AxisSpec::custom(
    ///     |lo, hi| Ok(vec![lo, 0.5 * (lo + hi), hi]),   // columns
    ///     |lo, hi| Ok(vec![0.25 * (lo + hi), 0.75 * (lo + hi)]), // rows
    /// );
    /// ```
    pub fn custom(
        column: impl Fn(f64, f64) -> Result<Vec<f64>> + Send + Sync + 'static,
        row: impl Fn(f64, f64) -> Result<Vec<f64>> + Send + Sync + 'static,
    ) -> Self {
        AxisSpec {
            column: Placement::Custom(std::sync::Arc::new(column)),
            row: Placement::Custom(std::sync::Arc::new(row)),
            alternating_split: false,
        }
    }

    /// The `(column, row)` disjoint pair this spec produces on `[lo, hi]`.
    pub fn column_row(&self, lo: f64, hi: f64) -> Result<(Vec<f64>, Vec<f64>)> {
        if self.alternating_split {
            // Classic: place one set, even → row, odd → column.
            let ip = self.column.abscissae(lo, hi)?;
            let row = (0..ip.len()).step_by(2).map(|i| ip[i]).collect();
            let column = (1..ip.len()).step_by(2).map(|i| ip[i]).collect();
            return Ok((column, row));
        }
        // Independent: each side placed in full from its own rule.
        Ok((self.column.abscissae(lo, hi)?, self.row.abscissae(lo, hi)?))
    }
}

impl From<AxisSampling> for AxisSpec {
    fn from(placement: AxisSampling) -> Self {
        AxisSpec::new(placement)
    }
}

impl AxisSampling {
    /// The distinct abscissae this placement puts on `[lo, hi]`. At least 2 are
    /// required to form a disjoint pair.
    pub fn abscissae(&self, lo: f64, hi: f64) -> Result<Vec<f64>> {
        let ip: Vec<f64> = match *self {
            AxisSampling::Linspace(m) => {
                if m < 2 {
                    return Err(LoewnerError::Degenerate("Linspace needs nip >= 2".into()));
                }
                (0..m)
                    .map(|i| lo + (hi - lo) * i as f64 / (m - 1) as f64)
                    .collect()
            }
            AxisSampling::Chebyshev(m) => {
                if m < 2 {
                    return Err(LoewnerError::Degenerate("Chebyshev needs nip >= 2".into()));
                }
                chebyshev_lobatto(lo, hi, m)
            }
            AxisSampling::Logspace(m) => logspace(lo, hi, m)?,
            AxisSampling::Integers => {
                let (a, b) = (lo.ceil() as i64, hi.floor() as i64);
                if b - a < 1 {
                    return Err(LoewnerError::Degenerate(format!(
                        "Integers on [{lo}, {hi}] yields < 2 integer nodes (need ≥ 2 for a disjoint split)"
                    )));
                }
                (a..=b).map(|v| v as f64).collect()
            }
            AxisSampling::Reference(m) => {
                if m < 2 {
                    return Err(LoewnerError::Degenerate("Reference needs nip >= 2".into()));
                }
                // linspace shifted up by half a step → last point is hi + step/2,
                // past the upper bound (the deliberate reference escape).
                let step = (hi - lo) / (m - 1) as f64;
                (0..m).map(|i| lo + step * i as f64 + step / 2.0).collect()
            }
        };
        Ok(ip)
    }
}

impl<S: Scalar> Points<S> {
    /// Build from per-axis column and row vectors, validating shape + disjointness.
    pub fn new(column: Vec<Vec<S>>, row: Vec<Vec<S>>) -> Result<Self> {
        if column.len() != row.len() {
            return Err(LoewnerError::DimMismatch {
                what: "number of axes (column vs row)",
                expected: column.len(),
                got: row.len(),
            });
        }
        let p = Points { column, row };
        p.check_disjoint()?;
        Ok(p)
    }
}

impl Points<f64> {
    /// Build a well-conditioned column/row split from per-axis bounds.
    ///
    /// Along each axis `l`, take `nip[l]` evenly spaced abscissae on
    /// `[lo_l, hi_l]`, then split them by the **alternating** convention: odd
    /// (1-based) indices become row points, even indices become column points
    /// (`ip(2:2:end)` / `ip(1:2:end)` in MATLAB). Alternating keeps the selected
    /// sub-points well-conditioned — picking the two endpoints can be a
    /// degenerate selection for coupled functions (see WORKPLAN §6b/§6c).
    ///
    /// This is the one-call point builder the example catalog and benchmarks use
    /// to go from `(bounds, nip)` straight to a [`Points`]. `bounds` and `nip`
    /// must have the same length `n`; each `nip[l]` must be `>= 2`.
    pub fn alternating(bounds: &[(f64, f64)], nip: &[usize]) -> Result<Self> {
        if bounds.len() != nip.len() {
            return Err(LoewnerError::DimMismatch {
                what: "bounds vs nip length",
                expected: bounds.len(),
                got: nip.len(),
            });
        }
        let mut column = Vec::with_capacity(bounds.len());
        let mut row = Vec::with_capacity(bounds.len());
        for (&(lo, hi), &m) in bounds.iter().zip(nip) {
            if m < 2 {
                return Err(LoewnerError::Degenerate(
                    "alternating points need nip >= 2 per axis".into(),
                ));
            }
            let ip: Vec<f64> = (0..m)
                .map(|i| lo + (hi - lo) * i as f64 / (m - 1) as f64)
                .collect();
            column.push((1..m).step_by(2).map(|i| ip[i]).collect());
            row.push((0..m).step_by(2).map(|i| ip[i]).collect());
        }
        Points::new(column, row)
    }

    /// Build a column/row split from **Chebyshev** abscissae per axis.
    ///
    /// Along each axis take `nip[l]` Chebyshev points of the second kind on
    /// `[lo_l, hi_l]` — `x_i = mid - half · cos(π i / (m-1))`, which includes both
    /// endpoints and **clusters toward the ends**. This is the well-conditioned
    /// placement AAA-style rational fitting wants: dense sampling where rational
    /// targets vary fastest (near the domain edges / poles just outside), and far
    /// better conditioned than uniform `linspace` for high order. Every point
    /// stays *inside* `[lo, hi]` (unlike the reference `p_r = p_c + dx`, which
    /// pushes the last row point past `hi`).
    ///
    /// The `m` points are split by the same **alternating** convention as
    /// [`Self::alternating`] (even indices → row, odd → column), so both sets
    /// inherit the Chebyshev clustering and are disjoint by construction.
    /// `bounds` and `nip` must share length `n`; each `nip[l]` must be `>= 2`.
    pub fn chebyshev(bounds: &[(f64, f64)], nip: &[usize]) -> Result<Self> {
        if bounds.len() != nip.len() {
            return Err(LoewnerError::DimMismatch {
                what: "bounds vs nip length",
                expected: bounds.len(),
                got: nip.len(),
            });
        }
        let mut column = Vec::with_capacity(bounds.len());
        let mut row = Vec::with_capacity(bounds.len());
        for (&(lo, hi), &m) in bounds.iter().zip(nip) {
            if m < 2 {
                return Err(LoewnerError::Degenerate(
                    "chebyshev points need nip >= 2 per axis".into(),
                ));
            }
            let ip = chebyshev_lobatto(lo, hi, m);
            row.push((0..m).step_by(2).map(|i| ip[i]).collect());
            column.push((1..m).step_by(2).map(|i| ip[i]).collect());
        }
        Points::new(column, row)
    }

    /// Build a column/row split with a **per-axis sampling spec**.
    ///
    /// Each axis carries an [`AxisSpec`] choosing its column and row placements —
    /// either one rule alternating-split ([`AxisSpec::new`]), two independent
    /// placements ([`AxisSpec::split`]), or custom closures ([`AxisSpec::custom`]).
    /// A single problem can thus mix, e.g., integer sampling on a discrete input,
    /// Chebyshev on a smooth one, and independent column/row placements on another.
    /// For bare per-axis rules (one rule each, alternating-split) use the terser
    /// [`Self::sampled_uniform`].
    ///
    /// `bounds` and `specs` must share length `n`.
    ///
    /// ```
    /// use rmlf::points::{AxisSampling, AxisSpec, Points};
    /// // axis 0 = integers in [0,8] (one rule, alternating-split);
    /// // axis 1 = independent, WITHIN-bounds column/row via a custom interleave.
    /// let pts = Points::sampled(
    ///     &[(0.0, 8.0), (-1.0, 1.0)],
    ///     &[
    ///         AxisSpec::new(AxisSampling::Integers),
    ///         AxisSpec::custom(
    ///             |lo, hi| Ok((0..5).map(|i| lo + (hi - lo) * i as f64 / 4.0).collect()), // cols
    ///             |lo, hi| Ok((0..4).map(|i| lo + (hi - lo) * (i as f64 + 0.5) / 4.0).collect()), // rows
    ///         ),
    ///     ],
    /// ).unwrap();
    /// ```
    pub fn sampled(bounds: &[(f64, f64)], specs: &[AxisSpec]) -> Result<Self> {
        if bounds.len() != specs.len() {
            return Err(LoewnerError::DimMismatch {
                what: "bounds vs specs length",
                expected: bounds.len(),
                got: specs.len(),
            });
        }
        let mut column = Vec::with_capacity(bounds.len());
        let mut row = Vec::with_capacity(bounds.len());
        for (&(lo, hi), spec) in bounds.iter().zip(specs) {
            let (col, rw) = spec.column_row(lo, hi)?;
            column.push(col);
            row.push(rw);
        }
        Points::new(column, row)
    }

    /// As [`Self::sampled`], but from bare per-axis [`AxisSampling`] rules (each used
    /// for both sides, alternating-split). Terser when no axis needs independent or
    /// custom column/row placement.
    pub fn sampled_uniform(bounds: &[(f64, f64)], rules: &[AxisSampling]) -> Result<Self> {
        let specs: Vec<AxisSpec> = rules.iter().map(|&r| AxisSpec::new(r)).collect();
        Points::sampled(bounds, &specs)
    }

    /// Build the **original MATLAB reference** column/row split from bounds.
    ///
    /// Per axis: `Linspace(nip)` columns paired with `Reference(nip)` rows — i.e. the
    /// columns shifted up by half a step (`p_r = p_c + dx`). This is the placement the
    /// `mLF` MATLAB demos and `examples.m` use, reproduced for exact parity. **Note it
    /// samples slightly outside the bounds:** the last row point is `hi + step/2`,
    /// past the upper bound — see [`AxisSampling::Reference`]. Prefer
    /// [`Self::alternating`] / [`Self::chebyshev`] when staying strictly in-domain
    /// matters.
    pub fn reference(bounds: &[(f64, f64)], nip: &[usize]) -> Result<Self> {
        if bounds.len() != nip.len() {
            return Err(LoewnerError::DimMismatch {
                what: "bounds vs nip length",
                expected: bounds.len(),
                got: nip.len(),
            });
        }
        let specs: Vec<AxisSpec> = nip
            .iter()
            .map(|&m| AxisSpec::split(AxisSampling::Linspace(m), AxisSampling::Reference(m)))
            .collect();
        Points::sampled(bounds, &specs)
    }
}

impl<S: Scalar> Points<S> {
    /// Number of variables `n`.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.column.len()
    }

    /// Number of column support points per axis (`k_l`).
    pub fn column_sizes(&self) -> Vec<usize> {
        self.column.iter().map(Vec::len).collect()
    }

    /// Number of row support points per axis (`q_l`).
    pub fn row_sizes(&self) -> Vec<usize> {
        self.row.iter().map(Vec::len).collect()
    }

    /// The combined `column ++ row` abscissae along an axis, in that order.
    ///
    /// This is the sampling order the data tensor uses along that axis: the
    /// first `k_l` entries are column points, the remaining `q_l` are row points.
    pub fn combined_axis(&self, axis: usize) -> Vec<S> {
        let mut v = self.column[axis].clone();
        v.extend_from_slice(&self.row[axis]);
        v
    }

    /// Verify that column and row sets are disjoint along every axis.
    ///
    /// Replaces `mlf.check`. Uses an exact-equality test (matching MATLAB's
    /// `unique`); points generated by the half-step offset convention never
    /// coincide, so exact comparison is appropriate here.
    pub fn check_disjoint(&self) -> Result<()> {
        for (axis, (col, row)) in self.column.iter().zip(&self.row).enumerate() {
            for &c in col {
                for &r in row {
                    if c == r {
                        return Err(S::duplicate_node(axis, c));
                    }
                }
            }
        }
        Ok(())
    }
}

/// The `m` Chebyshev points of the second kind (Chebyshev–Lobatto) on `[lo, hi]`,
/// in ascending order: `x_i = mid - half·cos(π i / (m-1))` for `i in 0..m`. The
/// endpoints `lo`, `hi` are included and the points cluster toward both ends.
/// `m >= 2` (caller-guaranteed); endpoints are pinned exactly to avoid the tiny
/// floating-point drift `cos(0)`/`cos(π)` would leave at the bounds.
pub fn chebyshev_lobatto(lo: f64, hi: f64, m: usize) -> Vec<f64> {
    let mid = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let last = (m - 1) as f64;
    (0..m)
        .map(|i| match i {
            0 => lo,
            i if i == m - 1 => hi,
            i => mid - half * (std::f64::consts::PI * i as f64 / last).cos(),
        })
        .collect()
}

/// The `m` **logarithmically spaced** points on `[lo, hi]`, ascending, endpoints
/// included. Spacing is uniform in `log|x|`, so points cluster toward the
/// small-magnitude end. The interval must not cross or touch zero (log of `0` /
/// sign change is undefined): either `0 < lo < hi`, or `lo < hi < 0` (handled by
/// reflecting through the origin so clustering lands near the smaller magnitude,
/// `hi`). `m >= 2` (caller-guaranteed); endpoints are pinned exactly.
pub fn logspace(lo: f64, hi: f64, m: usize) -> Result<Vec<f64>> {
    if m < 2 {
        return Err(LoewnerError::Degenerate("Logspace needs nip >= 2".into()));
    }
    if lo == 0.0 || hi == 0.0 || lo * hi < 0.0 {
        return Err(LoewnerError::Degenerate(format!(
            "Logspace on [{lo}, {hi}] crosses or touches 0 (log spacing is undefined through 0)"
        )));
    }
    // Work in positive magnitudes; restore the common sign at the end.
    let sign = if lo < 0.0 { -1.0 } else { 1.0 };
    let (a, b) = (lo.abs().min(hi.abs()), lo.abs().max(hi.abs()));
    let (la, lb) = (a.ln(), b.ln());
    let last = (m - 1) as f64;
    let mut mags: Vec<f64> = (0..m)
        .map(|i| match i {
            0 => a,
            i if i == m - 1 => b,
            i => (la + (lb - la) * i as f64 / last).exp(),
        })
        .collect();
    // For a negative interval, magnitudes ascend from |hi| to |lo|; negate and
    // reverse so the result is ascending in the original (signed) coordinate.
    if sign < 0.0 {
        mags.iter_mut().for_each(|x| *x = -*x);
        mags.reverse();
    }
    Ok(mags)
}

/// Insert `p` interior points into every gap of the sorted abscissae `s`,
/// returning **only the new points** (ascending). Each `[s[i], s[i+1]]` gap gets
/// `p` evenly spaced interior points at fractions `j/(p+1)`, `j in 1..=p`, so they
/// land strictly *inside* the gap (never on an existing node or outside the
/// range). This is the refinement step an iterative AAA-style pass uses to densify
/// a coarse grid where the residual is largest.
///
/// (Cleaned-up form of the common AAA `refine`: the usual `j/p + 1` offset lands
/// the points in the *next* gap, outside `[s[i], s[i+1]]`; `j/(p+1)` keeps them
/// interior, which is the intent.)
pub fn refine(s: &[f64], p: usize) -> Vec<f64> {
    if s.len() < 2 || p == 0 {
        return Vec::new();
    }
    let mut s = s.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("refine: NaN abscissa"));
    let denom = (p + 1) as f64;
    let mut out = Vec::with_capacity((s.len() - 1) * p);
    for w in s.windows(2) {
        let (a, gap) = (w[0], w[1] - w[0]);
        for j in 1..=p {
            out.push(a + gap * j as f64 / denom);
        }
    }
    out
}

impl Points<f64> {
    /// Embed a real grid in complex coordinates without changing its ordering.
    pub fn to_complex(&self) -> Points<Complex64> {
        Points {
            column: self
                .column
                .iter()
                .map(|a| a.iter().map(|&x| Complex64::new(x, 0.0)).collect())
                .collect(),
            row: self
                .row
                .iter()
                .map(|a| a.iter().map(|&x| Complex64::new(x, 0.0)).collect())
                .collect(),
        }
    }
}

impl Points<Complex64> {
    /// Alternate column/row nodes along straight complex segments. Real and
    /// imaginary axes may be mixed by using real endpoints for the real axes.
    pub fn alternating_segments(bounds: &[(Complex64, Complex64)], nip: &[usize]) -> Result<Self> {
        Self::segments(bounds, nip, false, false)
    }

    /// Chebyshev placement in the real parameter of each complex segment.
    pub fn chebyshev_segments(bounds: &[(Complex64, Complex64)], nip: &[usize]) -> Result<Self> {
        Self::segments(bounds, nip, true, false)
    }

    /// MATLAB catalog placement: column linspace on each segment, rows shifted
    /// by half the absolute node gap in the REAL direction, as in examples.m.
    /// Imaginary-axis columns therefore have rows on a parallel line.
    pub fn reference_segments(bounds: &[(Complex64, Complex64)], nip: &[usize]) -> Result<Self> {
        Self::segments(bounds, nip, false, true)
    }

    fn segments(
        bounds: &[(Complex64, Complex64)],
        nip: &[usize],
        chebyshev: bool,
        reference: bool,
    ) -> Result<Self> {
        if bounds.len() != nip.len() {
            return Err(LoewnerError::DimMismatch {
                what: "bounds vs nip length",
                expected: bounds.len(),
                got: nip.len(),
            });
        }
        let mut column = Vec::with_capacity(bounds.len());
        let mut row = Vec::with_capacity(bounds.len());
        for (&(lo, hi), &m) in bounds.iter().zip(nip) {
            if m < 2 || !lo.finite() || !hi.finite() || lo == hi {
                return Err(LoewnerError::Degenerate(
                    "complex segment needs finite distinct endpoints and nip >= 2".into(),
                ));
            }
            let t = if chebyshev {
                chebyshev_lobatto(0.0, 1.0, m)
            } else {
                (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
            };
            let nodes: Vec<_> = t.iter().map(|&t| lo + (hi - lo) * t).collect();
            if reference {
                let shift = (nodes[m - 1] - nodes[m - 2]).norm() / 2.0;
                row.push(nodes.iter().map(|&x| x + shift).collect());
                column.push(nodes);
            } else {
                column.push(nodes.iter().skip(1).step_by(2).copied().collect());
                row.push(nodes.iter().step_by(2).copied().collect());
            }
        }
        Self::new(column, row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn disjoint_ok() {
        let p = Points::new(vec![vec![0.0, 1.0]], vec![vec![0.5, 1.5]]);
        assert!(p.is_ok());
    }

    #[test]
    fn non_disjoint_detected() {
        let err = Points::new(vec![vec![0.0, 1.0]], vec![vec![1.0, 1.5]]).unwrap_err();
        matches!(err, LoewnerError::NonDisjoint { axis: 0, .. });
    }

    #[test]
    fn combined_axis_order() {
        let p = Points::new(vec![vec![0.0, 1.0]], vec![vec![0.5, 1.5]]).unwrap();
        assert_eq!(p.combined_axis(0), vec![0.0, 1.0, 0.5, 1.5]);
    }

    #[test]
    fn integer_axis_uses_distinct_integers_and_is_disjoint() {
        // Integers on [0, 8] → 0..=8 (9 nodes), alternating-split into disjoint
        // column/row, every node an integer.
        let p = Points::sampled_uniform(&[(0.0, 8.0)], &[AxisSampling::Integers]).unwrap();
        let combined = p.combined_axis(0);
        assert_eq!(combined.len(), 9);
        for v in &combined {
            assert_eq!(*v, v.round(), "node {v} is not an integer");
        }
        // Disjointness held (Points::new validated it).
        for c in &p.column[0] {
            assert!(!p.row[0].contains(c), "column node {c} also a row node");
        }
    }

    #[test]
    fn integer_axis_too_few_integers_errors() {
        // [0.2, 0.9] contains no integers in [ceil, floor] → < 2 nodes → Err.
        assert!(Points::sampled_uniform(&[(0.2, 0.9)], &[AxisSampling::Integers]).is_err());
    }

    #[test]
    fn mixed_per_axis_sampling_builds() {
        let p = Points::sampled(
            &[(0.0, 8.0), (-1.0, 1.0), (0.0, 1.0)],
            &[
                AxisSpec::new(AxisSampling::Integers),
                // independent column/row, within bounds: cheb columns + interleaved
                // rows on an inset interval (disjoint, no escape).
                AxisSpec::custom(
                    |lo, hi| Ok(chebyshev_lobatto(lo, hi, 9)),
                    |lo, hi| {
                        let (a, b) = (lo + 0.1, hi - 0.1);
                        Ok((0..8).map(|i| a + (b - a) * i as f64 / 7.0).collect())
                    },
                ),
                AxisSpec::new(AxisSampling::Linspace(6)),
            ],
        )
        .unwrap();
        assert_eq!(p.ndim(), 3);
    }

    #[test]
    fn chebyshev_endpoints_and_clustering() {
        let x = chebyshev_lobatto(-1.0, 1.0, 5);
        // Exact endpoints, ascending, symmetric about the midpoint.
        assert_relative_eq!(x[0], -1.0);
        assert_relative_eq!(x[4], 1.0);
        assert_relative_eq!(x[2], 0.0, epsilon = 1e-15);
        assert!(x.windows(2).all(|w| w[0] < w[1]));
        // Clusters at the ends: the end gap is strictly smaller than the mid gap.
        let end_gap = x[1] - x[0];
        let mid_gap = x[2] - x[1];
        assert!(end_gap < mid_gap, "end {end_gap} not < mid {mid_gap}");
    }

    #[test]
    fn chebyshev_stays_within_bounds_and_disjoint() {
        let p = Points::chebyshev(&[(80.0, 100.0), (-1.0, 1.0)], &[7, 6]).unwrap();
        for axis in 0..2 {
            let (lo, hi) = [(80.0, 100.0), (-1.0, 1.0)][axis];
            for &v in p.column[axis].iter().chain(&p.row[axis]) {
                assert!(v >= lo && v <= hi, "{v} outside [{lo},{hi}]");
            }
        }
        // Disjointness is checked by the constructor; reaching here proves it.
    }

    #[test]
    fn reference_matches_matlab_and_escapes_bounds() {
        // MATLAB: p_c = linspace(lo,hi,nip); dx = step/2; p_r = p_c + dx.
        // For [0,6], nip=4: step=2, dx=1, columns 0,2,4,6; rows 1,3,5,7.
        let p = Points::reference(&[(0.0, 6.0)], &[4]).unwrap();
        assert_eq!(p.column[0], vec![0.0, 2.0, 4.0, 6.0]);
        assert_eq!(p.row[0], vec![1.0, 3.0, 5.0, 7.0]);
        // The defining feature: the last row point (7.0) is OUTSIDE [0, 6].
        assert!(
            p.row[0].iter().any(|&r| r > 6.0),
            "reference must escape the upper bound"
        );
        // Column/row counts are both exactly nip (no alternating halving).
        assert_eq!(p.column[0].len(), 4);
        assert_eq!(p.row[0].len(), 4);
    }

    #[test]
    fn reference_equals_split_linspace_reference() {
        // Points::reference is Linspace columns + Reference (shifted) rows per axis.
        let a = Points::reference(&[(-1.0, 1.0)], &[5]).unwrap();
        let b = Points::sampled(
            &[(-1.0, 1.0)],
            &[AxisSpec::split(
                AxisSampling::Linspace(5),
                AxisSampling::Reference(5),
            )],
        )
        .unwrap();
        assert_eq!(a.column, b.column);
        assert_eq!(a.row, b.row);
    }

    #[test]
    fn independent_column_row_placements_within_bounds() {
        // The headline capability: column and row use DIFFERENT method AND count,
        // staying WITHIN bounds (the preferred way) — Chebyshev-6 columns, and rows
        // on a slightly inset interval so the two are disjoint without escaping.
        let p = Points::sampled(
            &[(0.0, 10.0)],
            &[AxisSpec::custom(
                |lo, hi| Ok(chebyshev_lobatto(lo, hi, 6)),
                |lo, hi| {
                    let (a, b) = (lo + 0.3, hi - 0.3);
                    Ok((0..9).map(|i| a + (b - a) * i as f64 / 8.0).collect())
                },
            )],
        )
        .unwrap();
        // Each side placed IN FULL: 6 column points, 9 row points (counts differ).
        assert_eq!(p.column[0].len(), 6);
        assert_eq!(p.row[0].len(), 9);
        // Everything stays inside [0, 10] — no bound breaking.
        for &v in p.column[0].iter().chain(&p.row[0]) {
            assert!((0.0..=10.0).contains(&v), "{v} escaped [0,10]");
        }
        // Disjointness validated by the constructor (reaching here proves it).
    }

    #[test]
    fn independent_column_row_matlab_reference_escapes() {
        // The MATLAB-parity alternative: a Reference (shifted) row set, which
        // DELIBERATELY steps past the upper bound. Kept working for parity, but it is
        // the discouraged path (only when f is defined outside the domain).
        let p = Points::sampled(
            &[(0.0, 10.0)],
            &[AxisSpec::split(
                AxisSampling::Chebyshev(6),
                AxisSampling::Reference(9),
            )],
        )
        .unwrap();
        assert_eq!(p.column[0].len(), 6);
        assert_eq!(p.row[0].len(), 9);
        assert!(
            p.row[0].iter().any(|&r| r > 10.0),
            "reference rows escape the bound"
        );
    }

    #[test]
    fn reference_row_placement_escapes_bounds() {
        // AxisSampling::Reference as a standalone row placement: linspace + half step,
        // last point past hi.
        let r = AxisSampling::Reference(4).abscissae(0.0, 6.0).unwrap();
        assert_eq!(r, vec![1.0, 3.0, 5.0, 7.0]);
        assert!(
            *r.last().unwrap() > 6.0,
            "Reference escapes the upper bound"
        );
    }

    #[test]
    fn custom_placement_arbitrary_points() {
        // A custom closure supplies exact points for each side.
        let p = Points::sampled(
            &[(0.0, 1.0)],
            &[AxisSpec::custom(
                |lo, hi| Ok(vec![lo, 0.5 * (lo + hi), hi]),
                |lo, hi| Ok(vec![0.25 * (lo + hi), 0.75 * (lo + hi)]),
            )],
        )
        .unwrap();
        assert_eq!(p.column[0], vec![0.0, 0.5, 1.0]);
        assert_eq!(p.row[0], vec![0.25, 0.75]);
    }

    #[test]
    fn custom_placement_single_rule_alternating_split() {
        // Custom also works with AxisSpec::new: the closure provides ONE set that
        // gets alternating-split into column/row.
        let spec = AxisSpec::new(Placement::Custom(std::sync::Arc::new(|_, _| {
            Ok(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        })));
        let p = Points::sampled(&[(0.0, 4.0)], &[spec]).unwrap();
        // even idx → row (0,2,4), odd idx → column (1,3).
        assert_eq!(p.row[0], vec![0.0, 2.0, 4.0]);
        assert_eq!(p.column[0], vec![1.0, 3.0]);
    }

    #[test]
    fn custom_placement_too_few_points_errors() {
        // A custom side yielding < 2 points is rejected.
        let err = Points::sampled(
            &[(0.0, 1.0)],
            &[AxisSpec::custom(
                |_, _| Ok(vec![0.5]),
                |_, _| Ok(vec![0.2, 0.8]),
            )],
        );
        assert!(err.is_err());
    }

    #[test]
    fn logspace_clusters_low_and_hits_endpoints() {
        let x = logspace(1.0, 1000.0, 4).unwrap();
        // Endpoints exact, ascending, and the classic decade spacing 1,10,100,1000.
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(x[1], 10.0, epsilon = 1e-9);
        assert_relative_eq!(x[2], 100.0, epsilon = 1e-9);
        assert_relative_eq!(x[3], 1000.0, epsilon = 1e-9);
        // Clusters toward the small end: first gap << last gap.
        assert!((x[1] - x[0]) < (x[3] - x[2]));
    }

    #[test]
    fn logspace_negative_interval_is_ascending() {
        // [-1000, -1] should give ascending -1000, -100, -10, -1 (cluster near -1).
        let x = logspace(-1000.0, -1.0, 4).unwrap();
        assert!(x.windows(2).all(|w| w[0] < w[1]), "not ascending: {x:?}");
        assert_relative_eq!(x[0], -1000.0, epsilon = 1e-9);
        assert_relative_eq!(x[3], -1.0, epsilon = 1e-12);
        // Clusters near the small magnitude (-1): last gap << first gap.
        assert!((x[3] - x[2]) < (x[1] - x[0]));
    }

    #[test]
    fn logspace_rejects_zero_crossing() {
        assert!(logspace(-1.0, 1.0, 5).is_err()); // crosses 0
        assert!(logspace(0.0, 10.0, 5).is_err()); // touches 0
        assert!(logspace(1.0, 10.0, 1).is_err()); // nip < 2
    }

    #[test]
    fn logspace_axis_splits_disjoint() {
        let p = Points::sampled_uniform(&[(1.0, 1e4)], &[AxisSampling::Logspace(8)]).unwrap();
        // Constructor validated disjointness; all points within [1, 1e4].
        for &v in p.column[0].iter().chain(&p.row[0]) {
            assert!((1.0..=1e4).contains(&v), "{v} outside [1, 1e4]");
        }
    }

    #[test]
    fn refine_lands_interior_points() {
        // Two gaps, p = 2 interior points each, all strictly inside the gap.
        let got = refine(&[0.0, 1.0, 3.0], 2);
        assert_relative_eq!(got[0], 1.0 / 3.0, epsilon = 1e-15);
        assert_relative_eq!(got[1], 2.0 / 3.0, epsilon = 1e-15);
        assert_relative_eq!(got[2], 1.0 + 2.0 / 3.0, epsilon = 1e-15);
        assert_relative_eq!(got[3], 1.0 + 4.0 / 3.0, epsilon = 1e-15);
        // Every refined point is strictly between the original extremes.
        assert!(got.iter().all(|&v| v > 0.0 && v < 3.0));
        // Degenerate inputs yield nothing.
        assert!(refine(&[1.0], 3).is_empty());
        assert!(refine(&[0.0, 1.0], 0).is_empty());
    }
}
