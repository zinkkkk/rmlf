//! The [`Model`] trait — the shared interface of every fitted multivariate model.
//!
//! `alg1`/`alg2`/`alg3` produce a [`Barycentric`];
//! `decouple_opt`/`decouple_adaptive` produce a
//! [`DecoupledModel`]; both can be stored/evaluated in
//! tensor-train form as a [`CompressedBarycentric`].
//! All three represent the *same kind of object* — an `n`-variable rational
//! interpolant you evaluate at a point — so they share this trait. Write
//! `fn use_model(m: &impl Model)` and accept any of them.
//!
//! See `docs/MODELS.md` for how the three representations relate and convert.

/// A fitted `n`-variable rational model that can be evaluated at a point.
///
/// Implemented by [`Barycentric`],
/// [`DecoupledModel`], and
/// [`CompressedBarycentric`]. The conversions
/// between them are `From`/`Into` impls (see `docs/MODELS.md`); the 1-D
/// slice views are the [`crate::Fibres`] trait.
pub trait Model<S: crate::scalar::Scalar = f64> {
    /// Number of input variables `n`.
    fn ndim(&self) -> usize;

    /// Evaluate the model at `x` (length `n`).
    fn eval(&self, x: &[S]) -> S;

    /// Evaluate at many points at once (each `xs[i]` has length `n`), one value
    /// per row. Defaults to mapping [`Model::eval`]; implementors may override
    /// with a batched/parallel version.
    fn eval_grid(&self, xs: &[&[S]]) -> Vec<S> {
        xs.iter().map(|x| self.eval(x)).collect()
    }

    /// Convert to the dense [`Barycentric`] form (the hub
    /// representation). Exact. Convenience wrapper over the `From`/`Into`
    /// conversions so it is discoverable as a method: `model.to_barycentric()`.
    fn to_barycentric(&self) -> Barycentric<S>
    where
        for<'a> Barycentric<S>: From<&'a Self>,
    {
        self.into()
    }

    /// Convert to the tensor-train [`CompressedBarycentric`]
    /// storage form (eval/storage linear in `n`). **Exact / lossless** (`tol = 0`);
    /// for lossy truncation call
    /// [`CompressedBarycentric::from_model`]
    /// with `tol > 0`. Convenience wrapper: `model.to_compressed()`.
    fn to_compressed(&self) -> CompressedBarycentric
    where
        for<'a> CompressedBarycentric: From<&'a Self>,
    {
        self.into()
    }

    /// Convert to the [`DecoupledModel`] form (separate
    /// numerator / denominator coefficient tensors). Exact for non-pole models.
    /// Convenience wrapper: `model.to_decoupled()`.
    fn to_decoupled(&self) -> crate::decouple::DecoupledModel
    where
        for<'a> crate::decouple::DecoupledModel: From<&'a Self>,
    {
        self.into()
    }
}

/// Structural-analysis views available on **any** [`Model`] (auto-routing through
/// the lossless tensor-train form). Separated from [`Model`] so the core eval
/// interface stays minimal, and blanket-implemented so you never have to convert
/// to [`CompressedBarycentric`] by hand first.
pub trait Analyze: Model {
    /// The **channel decomposition** — the model's internal tensor-train structure
    /// laid bare: `g = (∏_l M^num_l) / (∏_l M^den_l)`, each `M_l` a matrix of 1-D
    /// "channel" functions that combine by **matrix product** across axes. Exposes
    /// the coupling/rank structure (separable ⇒ all `1×1`; high rank ⇒ wide
    /// matrices = strong interaction). This is an *analysis* view, not a conversion
    /// step. See [`crate::ChannelDecomposition`].
    fn channels(&self) -> crate::univariate::ChannelDecomposition;
}

// Blanket impl: anything that converts to the exact TT form gets channel analysis.
impl<M> Analyze for M
where
    M: Model + Clone,
    CompressedBarycentric: for<'a> From<&'a M>,
{
    fn channels(&self) -> crate::univariate::ChannelDecomposition {
        let tt: CompressedBarycentric = self.into();
        tt.channel_decomposition()
    }
}

// ─── Conversions between the three core models ───────────────────────────────
//
// All three represent the same rational; these `From` impls let you move between
// representations with `.into()`. The dense `Barycentric` is the hub. Every
// conversion here is **exact** (lossless) — TT compression uses `tol = 0`. To
// trade exactness for smaller TT rank, call
// [`CompressedBarycentric::from_model`]
// with `tol > 0` directly. See `docs/MODELS.md`.

use crate::decouple::DecoupledModel;
use crate::eval::Barycentric;
use crate::tt::CompressedBarycentric;

impl From<&DecoupledModel> for Barycentric {
    /// Exact: collapse the separate `C_num`/`C_den` tensors to `(w = C_num/C_den,
    /// c = C_den)`.
    fn from(m: &DecoupledModel) -> Self {
        m.to_barycentric()
    }
}
impl From<DecoupledModel> for Barycentric {
    fn from(m: DecoupledModel) -> Self {
        (&m).into()
    }
}

impl From<&Barycentric> for CompressedBarycentric {
    /// **Exact / lossless** TT compression (`tol = 0`, full rank kept). For lossy
    /// truncation use [`CompressedBarycentric::from_model`] with `tol > 0`.
    fn from(m: &Barycentric) -> Self {
        CompressedBarycentric::from_model(m, 0.0)
    }
}
impl From<Barycentric> for CompressedBarycentric {
    fn from(m: Barycentric) -> Self {
        (&m).into()
    }
}

impl From<&CompressedBarycentric> for Barycentric {
    /// Exact: expand the TT cores back to the dense `(λ, w, c)` form. `c` is the
    /// denominator tensor and `w = numer/denom` recovered elementwise.
    fn from(m: &CompressedBarycentric) -> Self {
        let c = m.denom.to_dense();
        let cw = m.numer.to_dense();
        let w: Vec<f64> = cw
            .iter()
            .zip(&c)
            .map(|(&n, &d)| if d != 0.0 { n / d } else { 0.0 })
            .collect();
        Barycentric {
            lambda: m.lambda.clone(),
            w,
            c,
        }
    }
}
impl From<CompressedBarycentric> for Barycentric {
    fn from(m: CompressedBarycentric) -> Self {
        (&m).into()
    }
}

impl From<&DecoupledModel> for CompressedBarycentric {
    /// Exact, via the dense hub (`DecoupledModel → Barycentric → TT`).
    fn from(m: &DecoupledModel) -> Self {
        let b: Barycentric = m.into();
        (&b).into()
    }
}
impl From<DecoupledModel> for CompressedBarycentric {
    fn from(m: DecoupledModel) -> Self {
        (&m).into()
    }
}

impl From<&Barycentric> for DecoupledModel {
    /// Exact: split the folded weights back into separate numerator/denominator
    /// tensors — `c_den = c`, `c_num = w · c` (the inverse of
    /// [`DecoupledModel::to_barycentric`]). A `Barycentric` carries no distinct
    /// numerator *at a pole* (`c = 0` there), so the recovered `c_num` is `0` at
    /// such nodes; for non-pole models this round-trips losslessly.
    fn from(m: &Barycentric) -> Self {
        let c_den = m.c.clone();
        let c_num: Vec<f64> = m.w.iter().zip(&m.c).map(|(&w, &c)| w * c).collect();
        DecoupledModel {
            lambda: m.lambda.clone(),
            c_num,
            c_den,
        }
    }
}
impl From<Barycentric> for DecoupledModel {
    fn from(m: Barycentric) -> Self {
        (&m).into()
    }
}

impl From<&CompressedBarycentric> for DecoupledModel {
    /// Exact, via the dense hub (`TT → Barycentric → DecoupledModel`).
    fn from(m: &CompressedBarycentric) -> Self {
        let b: Barycentric = m.into();
        (&b).into()
    }
}
impl From<CompressedBarycentric> for DecoupledModel {
    fn from(m: CompressedBarycentric) -> Self {
        (&m).into()
    }
}

// Identity (by-reference) conversion: makes `to_compressed()` and the
// TT-routed analysis traits ([`Analyze`], [`crate::Contributions`]) total over
// **all** model types, including `CompressedBarycentric` itself. (`std` already
// provides `From<T> for T` by value; this is the borrowing counterpart it does
// not.) A cheap clone — the model is already in TT form.
impl From<&CompressedBarycentric> for CompressedBarycentric {
    fn from(m: &CompressedBarycentric) -> Self {
        m.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decouple::{DecoupleOptions, NonFinite, Sampling, decouple_opt};

    #[test]
    fn conversions_are_exact_and_agree_on_eval() {
        // Build a DecoupledModel, convert it every which way, and check all four
        // representations evaluate identically (conversions are lossless).
        // This target has a genuine pole on the grid (t²+2s+1 = 0), so use the Pole
        // non-finite policy rather than the erroring `Raw` default.
        let f = |x: &[f64]| (x[1] * x[1] + x[0] - 2.0) / (x[1] * x[1] + 2.0 * x[0] + 1.0);
        let opts = DecoupleOptions {
            sampling: Sampling::Adaptive {
                start_nip: 6,
                max_nip: 16,
            },
            non_finite: NonFinite::Pole,
            ..Default::default()
        };
        let (dec, _) = decouple_opt(&f, &[(-1.0, 1.0), (-1.0, 1.0)], &opts).unwrap();

        let bary: Barycentric = (&dec).into();
        let comp: CompressedBarycentric = (&dec).into(); // direct Dec -> TT
        let comp_via_bary: CompressedBarycentric = (&bary).into();
        let bary_back: Barycentric = (&comp).into(); // TT -> dense round trip

        for x in [[0.3, 0.7], [-0.4, 0.2], [0.9, -0.5]] {
            let a = Model::eval(&dec, &x);
            assert!((Model::eval(&bary, &x) - a).abs() < 1e-12, "Dec->Bary");
            assert!((Model::eval(&comp, &x) - a).abs() < 1e-12, "Dec->TT");
            assert!(
                (Model::eval(&comp_via_bary, &x) - a).abs() < 1e-12,
                "Bary->TT"
            );
            assert!((Model::eval(&bary_back, &x) - a).abs() < 1e-12, "TT->Bary");
        }
    }

    #[test]
    fn named_conversion_methods_cover_all_edges() {
        // The ergonomic discoverable methods `to_barycentric` / `to_compressed` /
        // `to_decoupled` must exist on every model and agree on eval — including the
        // two previously-missing edges (anything → DecoupledModel).
        let f = |x: &[f64]| (x[0] + 1.0) / (x[0] + 3.0) + x[1] * x[1];
        let opts = DecoupleOptions {
            sampling: Sampling::Linspace(vec![4, 4]),
            ..Default::default()
        };
        let (dec, _) = decouple_opt(&f, &[(-0.9, 0.9), (-0.9, 0.9)], &opts).unwrap();

        // From a Barycentric: to_compressed / to_decoupled (the common asks).
        let bary = dec.to_barycentric();
        let from_bary_comp = bary.to_compressed();
        let from_bary_dec = bary.to_decoupled(); // previously missing edge

        // From a CompressedBarycentric: to_barycentric / to_decoupled.
        let comp = dec.to_compressed();
        let from_comp_bary = comp.to_barycentric();
        let from_comp_dec = comp.to_decoupled(); // previously missing edge

        // From a DecoupledModel: to_compressed (trait) + to_barycentric (inherent).
        let from_dec_comp = dec.to_compressed();

        for x in [[0.2, 0.5], [-0.6, -0.3], [0.8, 0.1]] {
            let a = Model::eval(&dec, &x);
            for (label, y) in [
                ("bary.to_compressed", Model::eval(&from_bary_comp, &x)),
                ("bary.to_decoupled", Model::eval(&from_bary_dec, &x)),
                ("comp.to_barycentric", Model::eval(&from_comp_bary, &x)),
                ("comp.to_decoupled", Model::eval(&from_comp_dec, &x)),
                ("dec.to_compressed", Model::eval(&from_dec_comp, &x)),
            ] {
                assert!((y - a).abs() < 1e-10, "{label}: {y} vs {a}");
            }
        }
    }
}
