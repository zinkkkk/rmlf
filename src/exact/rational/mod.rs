//! The **ℚ domain** — exact reconstruction of integer/rational polynomials and
//! rational functions, with a one-call front door.
//!
//! - [`Rational`] — the field (ℚ, BigInt-backed).
//! - [`fit_poly`] / [`fit_rational`] — **the user entry points**: pass a closure
//!   and per-axis degrees, get back a [`RationalModel`] you can `eval`. They build
//!   the disjoint unisolvent integer grid for you (the part users should not have
//!   to hand-write).
//! - [`catalog`] — a browsable example set + a `fit`-style API, doubling as the
//!   regression/sweep corpus.

pub mod catalog;
mod scalars;

pub use scalars::Rational;

use super::ExactDecoupled;
use super::decouple::{ExactGrid, decouple_from_closure};
use super::packed::PackedDecoupled;
use crate::error::Result;

/// A fitted exact rational model over ℚ — the user-facing result of [`fit_poly`] /
/// [`fit_rational`]. Wraps the [`ExactDecoupled`] with ergonomic evaluation and
/// one-call compression.
#[derive(Clone, Debug)]
pub struct RationalModel {
    inner: ExactDecoupled<Rational>,
}

impl RationalModel {
    /// The underlying decoupled model (coefficient tensors, nodes).
    pub fn inner(&self) -> &ExactDecoupled<Rational> {
        &self.inner
    }
    /// Consume into the underlying decoupled model.
    pub fn into_inner(self) -> ExactDecoupled<Rational> {
        self.inner
    }
    /// Number of variables.
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Evaluate at exact rational coordinates. `None` on a genuine `0/0`.
    pub fn eval(&self, x: &[Rational]) -> Option<Rational> {
        self.inner.eval(x)
    }

    /// Evaluate at **integer** coordinates (the common case), returning the exact
    /// rational value. `None` on a genuine `0/0`.
    pub fn eval_q(&self, x: &[i64]) -> Option<Rational> {
        let pt: Vec<Rational> = x.iter().map(|&v| Rational::integer(v)).collect();
        self.inner.eval(&pt)
    }

    /// Evaluate at integer coordinates when the result is known to be an **integer**
    /// (an integer-valued polynomial on the integer grid). `None` if `0/0` or the
    /// exact value is not an integer.
    pub fn eval_int(&self, x: &[i64]) -> Option<i64> {
        let v = self.eval_q(x)?;
        if v.is_integer() {
            v.numer().try_into().ok()
        } else {
            None
        }
    }

    /// The canonical, eval-invariant, all-integer form (joint primitive rescale) —
    /// the smallest-bit representative. Returns a new model; the original is intact.
    pub fn canonical(&self) -> RationalModel {
        let mut inner = self.inner.clone();
        inner.canonicalize();
        RationalModel { inner }
    }

    /// The exact bit footprint of the stored coefficients.
    pub fn bit_size(&self) -> u64 {
        self.inner.bit_size()
    }

    /// The compact packed + pooled storage form (lossless).
    pub fn packed(&self) -> PackedDecoupled<Rational> {
        PackedDecoupled::pack(&self.inner)
    }
}

/// **Fit a multivariate polynomial** exactly from a closure and its per-axis
/// degrees. The minimal unisolvent integer grid (`degree_l + 1` column nodes +
/// disjoint row nodes per axis) is built automatically. Exact for any
/// integer/rational polynomial of the stated degrees.
///
/// ```
/// use rmlf::exact::rational::fit_poly;
/// let m = fit_poly(|x| &x[0] * &x[1] + &x[1] * &x[1], &[1, 2]).unwrap();
/// assert_eq!(m.eval_int(&[3, 4]).unwrap(), 3 * 4 + 4 * 4);
/// ```
pub fn fit_poly<F>(f: F, degrees: &[usize]) -> Result<RationalModel>
where
    F: Fn(&[Rational]) -> Rational,
{
    let grid = integer_grid(degrees);
    let inner = decouple_from_closure(&grid, f)?;
    Ok(RationalModel { inner })
}

/// **Fit a multivariate rational function** `n/d` exactly from a closure. The grid
/// is sized per axis for `max(num_degree_l, den_degree_l)` (enough to pin both),
/// built disjoint and unisolvent automatically. Exact for any tensor-product
/// rational at the stated degrees on a unisolvent grid.
///
/// ```
/// use rmlf::exact::rational::{fit_rational, Rational};
/// // (t² + s − 2)/(t² + 2s + 1), s = x0, t = x1.
/// let m = fit_rational(|x| {
///     let (s, t) = (x[0].clone(), x[1].clone());
///     (t.clone()*t.clone() + s.clone() - Rational::integer(2))
///         / (t.clone()*t.clone() + Rational::integer(2)*s + Rational::integer(1))
/// }, &[1, 2], &[1, 2]).unwrap();
/// assert_eq!(m.eval_q(&[7, 9]), Some(Rational::new(43, 48)));
/// ```
pub fn fit_rational<F>(f: F, num_degree: &[usize], den_degree: &[usize]) -> Result<RationalModel>
where
    F: Fn(&[Rational]) -> Rational,
{
    assert_eq!(
        num_degree.len(),
        den_degree.len(),
        "numerator and denominator degree vectors must have the same length"
    );
    let degrees: Vec<usize> = num_degree
        .iter()
        .zip(den_degree)
        .map(|(&a, &b)| a.max(b))
        .collect();
    fit_poly(f, &degrees)
}

/// Build the minimal disjoint unisolvent **integer** grid for per-axis `degrees`:
/// axis `l` gets column nodes and disjoint row nodes laid out in a block of
/// `2(degree_l + 1)` consecutive integers, **offset across axes** so coupled
/// targets never hit an on-grid diagonal (e.g. `x0 = x1`). This is the grid the
/// fitters use; exposed for callers who want to inspect or reuse it.
pub fn integer_grid(degrees: &[usize]) -> ExactGrid<Rational> {
    let mut lambda = Vec::with_capacity(degrees.len());
    let mut mu = Vec::with_capacity(degrees.len());
    let mut base = 0i64;
    for &d in degrees {
        let k = d as i64 + 1;
        lambda.push((base..base + k).map(Rational::integer).collect());
        mu.push((base + k..base + 2 * k).map(Rational::integer).collect());
        base += 2 * k;
    }
    ExactGrid { lambda, mu }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_poly_polya_szego_from_closure() {
        let m = fit_poly(
            |x| {
                x[0].clone() * x[1].clone()
                    + x[0].clone() * x[2].clone()
                    + x[1].clone() * x[2].clone()
            },
            &[1, 1, 1],
        )
        .unwrap();
        assert_eq!(m.eval_int(&[5, 7, 11]).unwrap(), 5 * 7 + 5 * 11 + 7 * 11);
        assert_eq!(m.eval_int(&[-3, 4, 9]).unwrap(), -3 * 4 + -3 * 9 + 4 * 9);
    }

    #[test]
    fn fit_rational_appendix_c_from_closure() {
        let m = fit_rational(
            |x| {
                let (s, t) = (x[0].clone(), x[1].clone());
                (t.clone() * t.clone() + s.clone() - Rational::integer(2))
                    / (t.clone() * t.clone() + Rational::integer(2) * s + Rational::integer(1))
            },
            &[1, 2],
            &[1, 2],
        )
        .unwrap();
        // (81 + 7 - 2)/(81 + 14 + 1) = 86/96 = 43/48.
        assert_eq!(m.eval_q(&[7, 9]), Some(Rational::new(43, 48)));
    }

    #[test]
    fn canonical_is_integral_and_eval_invariant() {
        let m = fit_rational(
            |x| Rational::one() / (x[0].clone() + Rational::integer(1)),
            &[0],
            &[1],
        )
        .unwrap();
        let c = m.canonical();
        assert!(c.bit_size() <= m.bit_size());
        for t in [3i64, -5, 20] {
            assert_eq!(c.eval_q(&[t]), m.eval_q(&[t]));
        }
    }
}
