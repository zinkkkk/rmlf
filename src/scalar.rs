//! Floating-point scalars supported by the numerical Loewner core.
//!
//! Algorithms are monomorphized for real and complex double precision. Rank
//! thresholds, residual magnitudes and singular-value spectra remain real.

use faer::traits::ComplexField;
use num_complex::Complex64;
use num_traits::NumAssign;
use std::iter::{Product, Sum};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex64 {}
}

/// A real or complex double-precision scalar. Implemented for `f64` and
/// [`Complex64`]; exact fields use the separate `exact` module.
pub trait Scalar:
    sealed::Sealed
    + ComplexField<Real = f64>
    + NumAssign
    + Copy
    + Send
    + Sync
    + Sum<Self>
    + Product<Self>
{
    /// Embed a real number in this scalar type.
    fn from_real(value: f64) -> Self;
    /// Euclidean magnitude (absolute value for a real scalar).
    fn magnitude(self) -> f64;
    /// Whether every component is finite.
    fn finite(self) -> bool;
    /// Real component, used to extract the real diagonal of an SVD.
    fn real(self) -> f64;
    /// Diagnostic for a coincident node, preserving real error compatibility.
    #[doc(hidden)]
    fn duplicate_node(axis: usize, value: Self) -> crate::LoewnerError;
}

impl Scalar for f64 {
    #[inline]
    fn from_real(value: f64) -> Self {
        value
    }
    #[inline]
    fn magnitude(self) -> f64 {
        self.abs()
    }
    #[inline]
    fn finite(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn real(self) -> f64 {
        self
    }
    fn duplicate_node(axis: usize, value: Self) -> crate::LoewnerError {
        crate::LoewnerError::NonDisjoint { axis, value }
    }
}

impl Scalar for Complex64 {
    #[inline]
    fn from_real(value: f64) -> Self {
        Self::new(value, 0.0)
    }
    #[inline]
    fn magnitude(self) -> f64 {
        self.norm()
    }
    #[inline]
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline]
    fn real(self) -> f64 {
        self.re
    }
    fn duplicate_node(axis: usize, value: Self) -> crate::LoewnerError {
        crate::LoewnerError::ComplexNonDisjoint { axis, value }
    }
}
