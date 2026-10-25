//! Exact-arithmetic Loewner / rational-KST decoupling — the integer, rational (ℚ),
//! and prime-field (𝔽_p) branch of the crate.
//!
//! Always available (the BigInt stack is an ordinary dependency, no feature flag).
//! The floating-point core (`alg1`/`alg2`/`decouple` over `f64`) is untouched; this
//! is a parallel, SVD-free pipeline yielding **bit-exact** reconstruction of
//! integer/rational polynomials and rational functions, and exact finite-field
//! table compression. See `docs/EXACT_FIELDS.md` for the full assessment.
//!
//! # Layout
//!
//! The two domains are genuinely different mathematical objects, so they are
//! separated; the SVD-free construction and compression are field-generic and
//! shared:
//!
//! - [`core`] — the [`Field`] trait both domains implement.
//! - [`rational`] — ℚ ([`Rational`]) + its catalog and one-call fitters.
//! - [`field`] — prime fields 𝔽_p ([`Fp`]), their catalog and one-call fitters.
//!   Prime-power extension fields such as GF(256) are not implemented.
//! - generic core (shared): [`decouple`] (the Kronecker cascade + models),
//!   [`linalg`] (exact RREF null space), [`tt`] / [`packed`]
//!   (lossless compression), [`store`] (the packed-scalar layer).
//!
//! # Quick start (the user front door)
//!
//! ```
//! use rmlf::exact::rational;
//! // A polynomial in 3 variables, per-axis degree 1 — fit it from a closure.
//! let model = rational::fit_poly(
//!     |x| &x[0] * &x[1] + &x[0] * &x[2] + &x[1] * &x[2],
//!     &[1, 1, 1],
//! ).unwrap();
//! assert_eq!(model.eval_int(&[5, 7, 11]).unwrap(), 5 * 7 + 5 * 11 + 7 * 11);
//! ```
//!
//! ```
//! use rmlf::exact::field;
//! // A finite-field map 𝔽₉₇² → 𝔽₉₇ — fit from a closure, packed storage is compact.
//! let model = field::fit_fp(|x| x[0] * x[1], 97, &[1, 1]).unwrap();
//! assert_eq!(model.eval_int(&[5, 7]).unwrap(), (5 * 7) % 97);
//! ```

pub mod core;
pub mod decouple;
pub mod direct;
pub mod field;
pub mod linalg;
pub mod packed;
pub mod rational;
pub mod store;
pub mod tt;
pub mod ttcross;

// ── core trait ──
pub use core::Field;

// ── generic construction + models ──
pub use decouple::{
    ExactDecoupled, ExactGrid, MimoDecoupled, decouple_from_closure, decouple_from_tensor,
    decouple_mimo, lagrange_coeffs, poly_decouple_from_values, sample,
};

// ── direct KST-decoupled rationals (Σ-sizes storage, by-inspection construction) ──
pub use direct::{DecoupledPoly, DirectKst};

// ── lossless compression ──
pub use linalg::Matrix;
pub use packed::{AxisNodes, CoeffBreakdown, PackedDecoupled};
pub use store::Packable;
pub use tt::{ExactTt, TtDecoupled};
pub use ttcross::{
    ExactCrossInfo, ExactCrossOptions, cross_poly_model, cross_values_on_grid, exact_tt_cross,
    split_rank,
};

// ── rational domain (ℚ) ──
pub use rational::Rational;

// ── finite-field domain (𝔽_p) ──
pub use field::Fp;
