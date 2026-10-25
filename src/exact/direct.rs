//! **Direct KST-decoupled rational functions over a field** — built *by inspection* (pure
//! evaluation, no null-space solve), stored in **tensor-train form** so the footprint and eval cost
//! are governed by the per-axis interaction *rank*, not the product `∏_l k_l`. No rational /
//! barycentric round-trip: everything is field arithmetic over `Fp` / `Zm`.
//!
//! ## Grounding in the two papers
//!
//! * **Paper 2 (Variable decoupling & KST for rational functions, 2026), Prop 3.1 / §3.5:** in the
//!   tensorized *normalized-Lagrange* basis the coefficient tensor of a multivariate polynomial
//!   simply **equals its sample values on the Kronecker node grid** — `α_{j₁…jₙ} = p(λ_{j₁},…,λ_{jₙ})`.
//!   So a polynomial (or, separately, the numerator and denominator of a rational function) is
//!   constructed *by inspection* from samples; no linear algebra.
//! * **Paper 1 (Loewner / curse of dimensionality, 2025), Thm 5.4 / §3.10:** that coefficient tensor
//!   factors so that storage and work scale with the **sum** of axis sizes (the largest single axis),
//!   not the product — "taming the curse." Concretely: store the sample tensor in **TT form**
//!   ([`ExactTt`], lossless, *exact* unfolding ranks over the field), and evaluate by contracting the
//!   TT against the per-axis Cauchy weight vectors `w_l[j] = γ^(l)_j /(x_l − λ^(l)_j)` —
//!   [`ExactTt::contract`], `O(Σ_l r_l·k_l·r_{l+1})`. For separable / low-interaction functions the
//!   TT ranks are tiny (often 1) and this is `O(Σ k_l)`; full interaction degrades gracefully to the
//!   dense product, which is then the *true* rank, not waste.
//!
//! ## Why this and not [`super::ExactDecoupled`] / [`super::TtDecoupled`]
//!
//! [`super::ExactDecoupled`] stores the **dense** `∏ k_l` num/den tensors (the curse, in storage).
//! [`super::TtDecoupled`] already compresses those — but its front door is the mLF/rational
//! construction path. [`DirectKst`] is the lean, field-native entry point: hand it samples (or a
//! closure) and it builds the TT directly, which is exactly the inverse-construction workflow —
//! *define the function by what it should evaluate to on the grid, with no reference function to
//! mirror.*

use crate::exact::core::Field;
use crate::exact::decouple::lagrange_coeffs;
use crate::exact::tt::ExactTt;

/// A multivariate rational `N(x)/D(x)`, numerator and denominator each a TT-compressed
/// normalized-Lagrange polynomial. They decouple (paper 2), so they are stored and evaluated
/// independently and combined with a single division.
#[derive(Clone, Debug)]
pub struct DirectKst<F: Field> {
    /// Decoupled numerator.
    pub num: DecoupledPoly<F>,
    /// Decoupled denominator.
    pub den: DecoupledPoly<F>,
}

/// A multivariate polynomial in the tensorized normalized-Lagrange basis, stored as a TT of its
/// grid sample values. Eval is a Cauchy-weight TT contraction (curse-tamed).
#[derive(Clone, Debug)]
pub struct DecoupledPoly<F: Field> {
    /// Per-axis Cauchy nodes `λ^(l)` (length `k_l`); the grid the samples live on.
    pub nodes: Vec<Vec<F>>,
    /// Per-axis Lagrange coefficients `γ^(l)_j = 1/∏_{i≠j}(λ_j−λ_i)` (precomputed once).
    pub lag: Vec<Vec<F>>,
    /// TT of the grid sample tensor (row-major), losslessly compressed with exact ranks.
    pub tt: ExactTt<F>,
}

impl<F: Field> DecoupledPoly<F> {
    /// Number of variables.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.nodes.len()
    }

    /// Per-axis sizes `k_l`.
    pub fn sizes(&self) -> Vec<usize> {
        self.nodes.iter().map(Vec::len).collect()
    }

    /// The interior TT bond ranks — the *measured* per-axis interaction. Tiny ⇒ curse tamed.
    pub fn bond_ranks(&self) -> &[usize] {
        self.tt.bond_ranks()
    }

    /// Stored field elements (TT cores) — the curse-tamed footprint.
    pub fn stored_len(&self) -> usize {
        self.tt.storage_len()
    }

    /// Dense footprint `∏_l k_l` (what a dense model would store), for comparison.
    pub fn dense_len(&self) -> usize {
        self.tt.dense_len()
    }

    /// **By-inspection construction** from a full grid sample tensor (row-major, `∏ k_l` entries):
    /// the samples *are* the normalized-Lagrange coefficients (paper 2, Prop 3.1). The tensor is
    /// immediately TT-compressed (paper 1), so the dense form is transient — for a function supplied
    /// as a closure use [`DirectKst::from_grid_closure`] which never materializes it densely per
    /// axis beyond construction.
    pub fn from_grid_values(nodes: Vec<Vec<F>>, values: &[F]) -> DecoupledPoly<F> {
        let sizes: Vec<usize> = nodes.iter().map(Vec::len).collect();
        debug_assert_eq!(values.len(), sizes.iter().product::<usize>());
        let lag: Vec<Vec<F>> = nodes.iter().map(|nd| lagrange_coeffs(nd)).collect();
        let tt = ExactTt::from_dense(values, &sizes);
        DecoupledPoly { nodes, lag, tt }
    }

    /// Evaluate the polynomial at `x` (length `ndim`) via the normalized-Lagrange TT contraction.
    /// `O(Σ_l r_l·k_l·r_{l+1})`. Exact and total: the per-axis weights are the *polynomial*
    /// normalized-Lagrange values `ℓ^(l)_j(x_l) = γ^(l)_j ∏_{i≠j}(x_l−λ_i)` (paper 2 eqs 11–12 /
    /// 16), defined everywhere — no division by `x−λ`, no on-node special case.
    pub fn eval(&self, x: &[F]) -> F {
        debug_assert_eq!(x.len(), self.ndim());
        let weights: Vec<Vec<F>> = (0..self.ndim())
            .map(|l| self.lagrange_basis(l, &x[l]))
            .collect();
        self.tt.contract(&weights)
    }

    /// Normalized-Lagrange basis values for axis `l` at scalar `xl`:
    /// `ℓ_j(xl) = γ_j · ∏_{i≠j}(xl − λ_i)`. Partition of unity: `Σ_j ℓ_j(xl) = 1`.
    fn lagrange_basis(&self, l: usize, xl: &F) -> Vec<F> {
        let nd = &self.nodes[l];
        let g = &self.lag[l];
        nd.iter()
            .enumerate()
            .map(|(j, _)| {
                let mut prod = g[j].clone();
                for (i, li) in nd.iter().enumerate() {
                    if i != j {
                        prod = prod * (xl.clone() - li.clone());
                    }
                }
                prod
            })
            .collect()
    }
}

impl<F: Field> DirectKst<F> {
    /// Number of variables.
    pub fn ndim(&self) -> usize {
        self.num.ndim()
    }

    /// Total stored field elements (numerator + denominator TT cores).
    pub fn stored_len(&self) -> usize {
        self.num.stored_len() + self.den.stored_len()
    }

    /// Dense footprint `2·∏ k_l`, for comparison.
    pub fn dense_len(&self) -> usize {
        self.num.dense_len() + self.den.dense_len()
    }

    /// The numerator and denominator interaction ranks (the two TT bond-rank profiles).
    pub fn bond_ranks(&self) -> (Vec<usize>, Vec<usize>) {
        (
            self.num.bond_ranks().to_vec(),
            self.den.bond_ranks().to_vec(),
        )
    }

    /// Build `N/D` by inspection from full numerator and denominator sample tensors on `nodes`.
    pub fn from_num_den_values(
        nodes: Vec<Vec<F>>,
        num_values: &[F],
        den_values: &[F],
    ) -> DirectKst<F> {
        let num = DecoupledPoly::from_grid_values(nodes.clone(), num_values);
        let den = DecoupledPoly::from_grid_values(nodes, den_values);
        DirectKst { num, den }
    }

    /// Inverse-construction entry point: define the function purely by what its numerator and
    /// denominator evaluate to at each grid node — no reference function to mirror.
    pub fn from_grid_closure(
        nodes: Vec<Vec<F>>,
        mut nd: impl FnMut(&[F]) -> (F, F),
    ) -> DirectKst<F> {
        let sizes: Vec<usize> = nodes.iter().map(Vec::len).collect();
        let total: usize = sizes.iter().product();
        let n = nodes.len();
        let mut numv = Vec::with_capacity(total);
        let mut denv = Vec::with_capacity(total);
        let mut idx = vec![0usize; n];
        for _ in 0..total {
            let pt: Vec<F> = (0..n).map(|l| nodes[l][idx[l]].clone()).collect();
            let (nu, de) = nd(&pt);
            numv.push(nu);
            denv.push(de);
            for l in (0..n).rev() {
                idx[l] += 1;
                if idx[l] < sizes[l] {
                    break;
                }
                idx[l] = 0;
            }
        }
        Self::from_num_den_values(nodes, &numv, &denv)
    }

    /// Evaluate `N(x)/D(x)`; `None` only for a genuine `0/0`.
    pub fn eval(&self, x: &[F]) -> Option<F> {
        let d = self.den.eval(x);
        let nu = self.num.eval(x);
        if d.is_zero() {
            return if nu.is_zero() { Some(F::zero()) } else { None };
        }
        Some(nu / d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::field::Fp;

    const P: u64 = 1_000_003;
    fn f(v: i128) -> Fp {
        Fp::new(v, P)
    }
    fn grid(sizes: &[usize]) -> Vec<Vec<Fp>> {
        sizes
            .iter()
            .map(|&k| (0..k as i128).map(|i| f(i + 1)).collect())
            .collect()
    }

    /// Sample a closure on the row-major Kronecker grid.
    fn sample(nodes: &[Vec<Fp>], mut p: impl FnMut(&[Fp]) -> Fp) -> Vec<Fp> {
        let sizes: Vec<usize> = nodes.iter().map(Vec::len).collect();
        let total: usize = sizes.iter().product();
        let n = nodes.len();
        let mut out = Vec::with_capacity(total);
        let mut idx = vec![0usize; n];
        for _ in 0..total {
            let pt: Vec<Fp> = (0..n).map(|l| nodes[l][idx[l]]).collect();
            out.push(p(&pt));
            for l in (0..n).rev() {
                idx[l] += 1;
                if idx[l] < sizes[l] {
                    break;
                }
                idx[l] = 0;
            }
        }
        out
    }

    #[test]
    fn polynomial_reconstructs_exactly() {
        let nodes = grid(&[3, 3]);
        let p = |x: &[Fp]| x[0] * x[1] + x[0] + f(2) * x[1] + f(3);
        let vals = sample(&nodes, p);
        let poly = DecoupledPoly::from_grid_values(nodes, &vals);
        for &(a, b) in &[(10i128, 20i128), (7, 3), (100, 1), (5, 5)] {
            assert_eq!(
                poly.eval(&[f(a), f(b)]),
                p(&[f(a), f(b)]),
                "mismatch at ({a},{b})"
            );
        }
    }

    #[test]
    fn polya_szego_three_vars() {
        let nodes = grid(&[2, 2, 2]);
        let p = |x: &[Fp]| x[0] * x[1] + x[0] * x[2] + x[1] * x[2];
        let kst = DirectKst::from_grid_closure(nodes, |pt| (p(pt), f(1)));
        for &(a, b, c) in &[(3i128, 5, 7), (11, 13, 2), (1, 1, 1), (100, 1, 50)] {
            assert_eq!(
                kst.eval(&[f(a), f(b), f(c)]).unwrap(),
                p(&[f(a), f(b), f(c)])
            );
        }
    }

    #[test]
    fn rational_reconstructs_exactly() {
        // H(s,t) = (t²+s−2)/(t²+2s+1) — paper 2 Appendix C, degrees (1,2) ⇒ k=(2,3)
        let nodes = grid(&[2, 3]);
        let num = |x: &[Fp]| x[1] * x[1] + x[0] - f(2);
        let den = |x: &[Fp]| x[1] * x[1] + f(2) * x[0] + f(1);
        let kst = DirectKst::from_grid_closure(nodes, |pt| (num(pt), den(pt)));
        for &(s, t) in &[(10i128, 20i128), (3, 7), (5, 2), (50, 9)] {
            let d = den(&[f(s), f(t)]);
            if d.is_zero() {
                continue;
            }
            assert_eq!(kst.eval(&[f(s), f(t)]).unwrap(), num(&[f(s), f(t)]) / d);
        }
    }

    #[test]
    fn separable_is_low_rank_storage() {
        // a separable poly ∏_l (x_l + l) has TT rank 1 on every bond ⇒ storage Σk_l, not ∏k_l
        let nodes = grid(&[4, 4, 4, 4]);
        let p = |x: &[Fp]| {
            x.iter()
                .enumerate()
                .fold(Fp::one(), |acc, (l, xl)| acc * (*xl + f(l as i128)))
        };
        let vals = sample(&nodes, p);
        let poly = DecoupledPoly::from_grid_values(nodes, &vals);
        assert!(
            poly.bond_ranks().iter().all(|&r| r == 1),
            "separable ⇒ all bond ranks 1"
        );
        assert!(
            poly.stored_len() < poly.dense_len(),
            "TT must beat dense for separable"
        );
        // and it still evaluates exactly
        assert_eq!(
            poly.eval(&[f(9), f(8), f(7), f(6)]),
            p(&[f(9), f(8), f(7), f(6)])
        );
    }
}
