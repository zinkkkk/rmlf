//! # rmlf — multivariate Loewner Framework
//!
//! Given an unknown function `f(x₁, …, xₙ)` (a Rust closure) or a precomputed
//! `n`-dimensional data tensor, rmlf builds an `n`-variable **rational
//! interpolant** in barycentric form — taming the curse of dimensionality so
//! construction, storage, and evaluation all stay tractable as `n` grows.
//!
//! It is a Rust port of the multivariate Loewner Framework of Antoulas, Gosea &
//! Poussot-Vassal (SIAM Review 2025) plus the rational-KST variable decoupling
//! (arXiv 2026), with measured improvements over the reference (per-axis adaptive
//! sampling, non-rational stabilization).
//!
//! ## Quick start
//!
//! ```
//! use rmlf::prelude::*;
//!
//! // Any function of n inputs → one output. (Pólya–Szegő, n = 3.)
//! let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
//!
//! // Pick support points on per-axis bounds, then fit.
//! let pts = Points::alternating(&[(-1.0, 1.0); 3], &[4, 4, 4]).unwrap();
//! let (model, _info) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
//!
//! let y = model.eval(&[0.1, -0.3, 0.5]);   // barycentric evaluation, off-grid
//! assert!((y - f(&[0.1, -0.3, 0.5])).abs() < 1e-12);
//! ```
//!
//! Algorithms 1 and 2 also accept [`Complex64`]. The scalar defaults to `f64`;
//! real and complex fits share monomorphized code. For complex segments:
//!
//! ```
//! use rmlf::prelude::*;
//! let f = |z: &[Complex64]| Complex64::new(1.0, 2.0) / (z[0] + 3.0);
//! let pts = Points::chebyshev_segments(
//!     &[(Complex64::new(-1.0, 0.2), Complex64::new(1.0, 0.8))], &[10]
//! ).unwrap();
//! let (model, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
//! let x = [Complex64::new(0.3, -0.4)];
//! assert!((model.eval(&x) - f(&x)).norm() < 1e-12);
//! ```
//!
//! Complex models support recursive/full/automatic coefficient construction,
//! tensor input, batched evaluation and Algorithm 2's candidate scan modes.
//! Compression, decoupling, grid refinement, AAA coefficient fitting, and the
//! fibre/contribution addons remain real-only. Complex Algorithm 2 ignores
//! `compress` and evaluates densely. [`Barycentric::to_complex`] embeds a real
//! fit for complex evaluation without refitting.
//! Complex fits do not impose a real-valued or conjugate-symmetric target.
//!
//! ## Choosing an entry point
//!
//! | I want… | Use |
//! |---------|-----|
//! | a one-shot fit on a fixed grid | [`alg1()`] (closure) / [`alg1_from_tensor`] (data) |
//! | adaptive support selection (greedy) | [`alg2()`] |
//! | adaptive **resolution** (grid refinement) | [`alg3()`] |
//! | an exact, SVD-free **decoupled** rational (KST) | [`decouple_adaptive`] |
//! | a 1-D near-minimax fit (AAA + Lawson) | [`aaa`] |
//! | exact arithmetic over ℚ / 𝔽_p | [`exact`] |
//!
//! All four `alg*`/`decouple*` paths return an evaluable [`Barycentric`] /
//! [`DecoupledModel`]; [`CompressedBarycentric`] ([`tt`]) is the tensor-train
//! storage form for high-`n` models (eval linear in `n`).
//!
//! Move between the three representations with one idiomatic method each — every
//! model has [`Model::to_barycentric`],
//! [`Model::to_compressed`], and
//! [`Model::to_decoupled`] (all exact; `From`/`Into` also work):
//!
//! ```
//! use rmlf::prelude::*;
//! let f = |x: &[f64]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
//! let pts = Points::alternating(&[(-1.0, 1.0); 3], &[4, 4, 4]).unwrap();
//! let (model, _) = alg1(&f, &pts, &Alg1Options::default()).unwrap();
//! let tt = model.to_compressed();   // → tensor-train storage, eval linear in n
//! assert!((tt.eval(&[0.1, -0.3, 0.5]) - model.eval(&[0.1, -0.3, 0.5])).abs() < 1e-12);
//! ```
//!
//! ## How it's organized
//!
//! - **Entry points** (re-exported at the crate root, in [`prelude`]): the `alg*`
//!   and `decouple*` fitters, [`Points`], the model types, and their option/info
//!   structs.
//! - **Building blocks** (the per-module API): [`order`] (degree detection),
//!   [`select`] (support reduction), [`eval`] (the barycentric evaluator),
//!   [`tensor`] (sampling/storage), [`tt`]/[`ttcross`] (compression).
//! - **Low-level internals** ([`internals`]): the recursive null-space solver and
//!   1-D Loewner-matrix builder. Most users never need these — they are the
//!   machinery `alg*` runs underneath. Reach for [`alg1()`], not `loewner_null_rec`.

pub mod algo;
pub mod contribution;
pub mod decouple;
pub mod error;
pub mod eval;
pub mod exact;
pub mod examples;
pub mod grid;
pub mod lawson;
pub mod loewner;
pub mod model;
pub mod nullspace;
pub mod order;
pub mod points;
pub mod recurse;
pub mod scalar;
pub mod select;
pub mod source;
pub mod tensor;
pub mod tt;
pub mod ttcross;
pub mod univariate;

// ─── Entry points (the recommended path) ─────────────────────────────────────

/// The three orchestration algorithms and their option/info types.
pub use algo::{
    Alg1Info, Alg1Options, Alg2Info, Alg2Options, Alg3Info, Alg3Options, Candidates,
    CoefficientMethod, alg1, alg1_from_tensor, alg2, alg2_from_tensor, alg3,
};
/// KST variable decoupling — the SVD-free, by-inspection rational path.
pub use decouple::{
    DecoupleInfo, DecoupleOptions, DecoupledFn, DecoupledModel, NonFinite, NullMethod, Sampling,
    decouple, decouple_adaptive, decouple_from_closure, decouple_from_tensor,
    decouple_from_tensor_with, decouple_opt, decouple_with_degrees, lagrange_coeffs,
};
/// 1-D AAA + Lawson near-minimax fit (a best-approximation complement).
pub use lawson::{AaaInfo, AaaOptions, aaa};

// ─── Models, points, errors (the common vocabulary) ──────────────────────────

pub use algo::scan::FitScalar;
pub use contribution::{AxisContribution, ContributionKind, ContributionSet, Contributions};
pub use error::{LoewnerError, Result};
pub use eval::{Barycentric, column_values};
pub use model::{Analyze, Model};
pub use num_complex::Complex64;
pub use points::{AxisSampling, AxisSpec, Placement, Points, chebyshev_lobatto, logspace, refine};
pub use scalar::Scalar;
pub use source::TargetFn;
pub use tt::{CompressedBarycentric, TtTensor};
pub use ttcross::{TtCrossInfo, TtCrossModel, TtCrossOptions, tt_cross, tt_cross_rational};
pub use univariate::{
    AxisChannels, BarycentricRational1D, ChannelDecomposition, Fibres, LabelledFibre,
};

// ─── Building blocks (used by the above; available for custom pipelines) ──────

pub use order::{OrderOptions, compute_order};
pub use select::{SelectStrategy, Selection, points_selection, points_selection_with};
pub use tensor::{Tensor, View, sample, sample_continued};

/// Low-level solver internals — the recursive null space ([`recurse`]) and 1-D
/// Loewner-matrix builder ([`loewner`]) that the `alg*` entry points run
/// underneath, plus the null-space strategy traits ([`nullspace`]).
///
/// **You almost never need these directly.** To fit a model, call [`alg1()`],
/// [`alg2()`], [`alg3()`], or [`decouple_adaptive`] — not `loewner_null_rec`. These
/// are exposed for building a custom construction pipeline.
pub mod internals {
    pub use crate::loewner::{loewner_1d, loewner_full};
    pub use crate::nullspace::{Normalize, NullSpace, SvdNull};
    pub use crate::recurse::{RecResult, loewner_null_rec, loewner_null_rec_svd0};
}

/// The common imports for everyday use: the entry-point fitters, model types,
/// [`Points`], and the option/info structs. `use rmlf::prelude::*;` is the
/// one-line way to get going.
pub mod prelude {
    pub use crate::algo::{
        Alg1Info, Alg1Options, Alg2Info, Alg2Options, Alg3Info, Alg3Options, CoefficientMethod,
        alg1, alg1_from_tensor, alg2, alg2_from_tensor, alg3,
    };
    pub use crate::contribution::{
        AxisContribution, ContributionKind, ContributionSet, Contributions,
    };
    pub use crate::decouple::{
        DecoupleInfo, DecoupleOptions, DecoupledModel, NonFinite, NullMethod, Sampling,
        decouple_adaptive, decouple_opt,
    };
    pub use crate::error::{LoewnerError, Result};
    pub use crate::eval::Barycentric;
    pub use crate::lawson::{AaaOptions, aaa};
    pub use crate::model::{Analyze, Model};
    pub use crate::points::{AxisSampling, AxisSpec, Points};
    pub use crate::source::TargetFn;
    pub use crate::tt::CompressedBarycentric;
    pub use crate::univariate::Fibres;
    pub use crate::{Complex64, FitScalar, Scalar};
}
