//! Compile-time selection of the existing real TT accelerator or complex dense
//! evaluation. The complex core does not require TT compression support.

use crate::{Barycentric, Complex64, CompressedBarycentric, Scalar};

/// Optional acceleration of Algorithm 2's candidate scan.
#[doc(hidden)]
pub trait ScanAccelerator<S>: Sync {
    fn eval(&self, x: &[S]) -> S;
    fn eval_full_grid(&self, axes: &[Vec<S>]) -> Vec<S>;
    fn eval_cost(&self) -> usize;
}

impl ScanAccelerator<f64> for CompressedBarycentric {
    fn eval(&self, x: &[f64]) -> f64 {
        self.eval(x)
    }
    fn eval_full_grid(&self, axes: &[Vec<f64>]) -> Vec<f64> {
        self.eval_full_grid(axes)
    }
    fn eval_cost(&self) -> usize {
        self.eval_cost()
    }
}

/// No complex acceleration object can be constructed in this milestone.
#[doc(hidden)]
pub enum DenseOnly {}

impl ScanAccelerator<Complex64> for DenseOnly {
    fn eval(&self, _: &[Complex64]) -> Complex64 {
        match *self {}
    }
    fn eval_full_grid(&self, _: &[Vec<Complex64>]) -> Vec<Complex64> {
        match *self {}
    }
    fn eval_cost(&self) -> usize {
        match *self {}
    }
}

/// Scalar supported by the greedy fitter. Real models retain their TT scan
/// acceleration; complex models use dense evaluation regardless of `compress`.
pub trait FitScalar: Scalar {
    #[doc(hidden)]
    type Accelerator: ScanAccelerator<Self>;
    #[doc(hidden)]
    fn compress(model: &Barycentric<Self>, tol: f64) -> Option<Self::Accelerator>;
}

impl FitScalar for f64 {
    type Accelerator = CompressedBarycentric;
    #[inline]
    fn compress(model: &Barycentric<Self>, tol: f64) -> Option<Self::Accelerator> {
        Some(CompressedBarycentric::from_model(model, tol))
    }
}

impl FitScalar for Complex64 {
    type Accelerator = DenseOnly;
    #[inline]
    fn compress(_: &Barycentric<Self>, _: f64) -> Option<Self::Accelerator> {
        None
    }
}
