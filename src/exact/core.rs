//! The field abstraction shared by both exact domains.
//!
//! [`Field`] is the scalar interface the SVD-free Loewner / rational-KST
//! decoupling runs over: exact arithmetic (`+ − × ÷`) plus equality. Both the
//! rational domain ([`super::rational`], ℚ) and the finite-field domain
//! ([`super::field`], 𝔽_p) implement it, so the generic construction
//! ([`super::decouple`]), linear algebra ([`super::linalg`]), and compression
//! ([`super::tt`]) are written **once** against this trait.
//!
//! See `docs/EXACT_FIELDS.md` §1 for the per-step requirement table. Plain ℤ is
//! deliberately *not* a `Field` — node-gap division (`γ_j = 1/∏(λ_j−λ_i)`) leaves
//! ℤ — so integer data is embedded ℤ↪ℚ (exact) or reduced ℤ↠𝔽_p before running.

use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An exact field: the scalar type the generic Loewner/decoupling code runs over.
///
/// Implementors must give *exact* arithmetic (no rounding) — that is the whole
/// point of this branch. [`Field::inv`] returns `None` only for the additive
/// identity (the single non-invertible element of a field).
pub trait Field:
    Clone
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The additive identity `0`.
    fn zero() -> Self;
    /// The multiplicative identity `1`.
    fn one() -> Self;
    /// Build the field element for a (possibly negative) integer.
    fn from_int(n: i64) -> Self;
    /// `true` iff this is the additive identity.
    fn is_zero(&self) -> bool;
    /// Multiplicative inverse, or `None` for `0`.
    fn inv(&self) -> Option<Self>;
}
