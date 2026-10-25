//! How the framework obtains sample values: from a runtime function (closure) or
//! from a precomputed data tensor.
//!
//! The convention is fixed: **`n` inputs, single output**. A target is anything
//! that maps an `n`-vector `[x_1, ..., x_n]` to a scalar. Any
//! `Fn(&[S]) -> S` qualifies via a blanket impl, so callers pass a closure
//! directly — no string variable names, no per-arity machinery.

/// A scalar-valued function of `n` real or complex variables (`f64` by default).
///
/// Implemented automatically for every `Fn(&[S]) -> S`, so the typical
/// caller writes `|x| x[0] * x[1] + x[2]` and is done.
pub trait TargetFn<S = f64> {
    /// Evaluate at the point `x` (length `n`).
    fn eval(&self, x: &[S]) -> S;
}

impl<S, F> TargetFn<S> for F
where
    F: Fn(&[S]) -> S,
{
    #[inline]
    fn eval(&self, x: &[S]) -> S {
        (self)(x)
    }
}
