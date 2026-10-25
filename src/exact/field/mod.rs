//! The **finite-field domain** — 𝔽_p (prime fields), with a one-call front door.
//!
//! - [`Fp`] — prime field 𝔽_p arithmetic.
//! - [`fit_fp`] — **the user entry point**: pass a closure and per-axis degrees,
//!   get a [`FieldModel`] you can `eval`. The disjoint unisolvent grid is built
//!   for you; the model stores compactly (the prime is hoisted **once**, see
//!   [`super::packed`]).
//! - [`catalog`] — the prime catalog ([`catalog::all`]).

pub mod catalog;
pub mod fp;

pub use fp::Fp;

use super::ExactDecoupled;
use super::core::Field;
use super::decouple::{ExactGrid, decouple_from_closure};
use super::packed::PackedDecoupled;
use super::store::Packable;
use crate::error::{LoewnerError, Result};

/// A fitted exact model over a finite field `F` (𝔽_p) — the user-facing result of
/// [`fit_fp`]. Stores compactly via [`Self::packed`] (the prime/context hoisted
/// once, no per-coefficient repetition). Generic over `F: Field` so other field
/// instances can reuse it.
#[derive(Clone, Debug)]
pub struct FieldModel<F: Field> {
    inner: ExactDecoupled<F>,
}

impl<F: Field> FieldModel<F> {
    /// The underlying decoupled model.
    pub fn inner(&self) -> &ExactDecoupled<F> {
        &self.inner
    }
    /// Consume into the underlying decoupled model.
    pub fn into_inner(self) -> ExactDecoupled<F> {
        self.inner
    }
    /// Number of variables.
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }
    /// Evaluate at field coordinates. `None` on a genuine `0/0` / pole.
    pub fn eval(&self, x: &[F]) -> Option<F> {
        self.inner.eval(x)
    }
}

impl FieldModel<Fp> {
    /// Evaluate at **integer** coordinates over 𝔽_p, returning the canonical
    /// representative in `[0, p)`. `None` on a genuine pole.
    pub fn eval_int(&self, x: &[i64]) -> Option<u64> {
        let p = self.prime();
        let pt: Vec<Fp> = x.iter().map(|&v| Fp::new(v as i128, p)).collect();
        self.inner.eval(&pt).map(|v| v.value())
    }

    /// The model's prime modulus.
    pub fn prime(&self) -> u64 {
        Fp::context_of(
            &self
                .inner
                .lambda
                .iter()
                .flatten()
                .chain(&self.inner.c_den)
                .cloned()
                .collect::<Vec<_>>(),
        )
    }
}

impl<F: Packable> FieldModel<F> {
    /// The compact packed + pooled storage form (lossless). For 𝔽_p this hoists the
    /// prime once and stores bare values — the actually-compressed representation.
    pub fn packed(&self) -> PackedDecoupled<F> {
        PackedDecoupled::pack(&self.inner)
    }
}

/// **Fit a finite-field map** `𝔽_p^n → 𝔽_p` exactly from a closure and its per-axis
/// degrees. Builds the minimal disjoint integer grid mod `p` automatically. The
/// disjoint column∪row requirement caps the recoverable per-axis degree at
/// `⌊p/2⌋ − 1` (each axis needs `2(degree_l+1) ≤ p` distinct nodes); a higher
/// degree returns `Err` (the sharp structural ceiling, `docs/EXACT_FIELDS.md` §7).
///
/// ```
/// use rmlf::exact::field::fit_fp;
/// let m = fit_fp(|x| x[0] * x[1] + x[0], 97, &[1, 1]).unwrap();
/// assert_eq!(m.eval_int(&[5, 7]).unwrap(), (5 * 7 + 5) % 97);
/// ```
pub fn fit_fp<F>(f: F, p: u64, degrees: &[usize]) -> Result<FieldModel<Fp>>
where
    F: Fn(&[Fp]) -> Fp,
{
    let grid = fp_grid(p, degrees)?;
    let inner = decouple_from_closure(&grid, f)?;
    Ok(FieldModel { inner })
}

/// Build the minimal disjoint 𝔽_p grid for per-axis `degrees`: axis `l` gets column
/// nodes `0..k` and row nodes `k..2k` (`k = degree_l + 1`) reduced mod `p`, requiring
/// `2k ≤ p` so the nodes stay distinct. Offset across axes (so coupled maps avoid the
/// on-grid diagonal). Errors if any axis exceeds the `⌊p/2⌋−1` ceiling.
pub fn fp_grid(p: u64, degrees: &[usize]) -> Result<ExactGrid<Fp>> {
    let mut lambda = Vec::with_capacity(degrees.len());
    let mut mu = Vec::with_capacity(degrees.len());
    let mut base = 0i128;
    for &d in degrees {
        let k = d as i128 + 1;
        if 2 * k > p as i128 {
            return Err(LoewnerError::Degenerate(format!(
                "degree {d} needs {} distinct nodes > p={p} (disjoint-grid ceiling ⌊p/2⌋−1)",
                2 * k
            )));
        }
        lambda.push((base..base + k).map(|v| Fp::new(v, p)).collect());
        mu.push((base + k..base + 2 * k).map(|v| Fp::new(v, p)).collect());
        base += 2 * k;
    }
    Ok(ExactGrid { lambda, mu })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_fp_from_closure() {
        let m = fit_fp(|x| x[0] * x[1] + x[0] + x[1], 97, &[1, 1]).unwrap();
        for &(a, b) in &[(5i64, 7), (40, 60), (90, 3)] {
            assert_eq!(
                m.eval_int(&[a, b]).unwrap(),
                ((a * b + a + b) % 97) as u64,
                "at {:?}",
                (a, b)
            );
        }
        assert_eq!(m.prime(), 97);
        // Packed storage hoists the prime once.
        assert_eq!(m.packed().ctx, 97);
    }

    #[test]
    fn fit_fp_rejects_above_ceiling() {
        // Degree 4 needs 10 nodes > 7 → Err over GF(7).
        assert!(fit_fp(|x| x[0], 7, &[4]).is_err());
    }
}
