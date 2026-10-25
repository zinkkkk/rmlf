//! [`PackedDecoupled`] — the **compact stored form** of an exact decoupled model.
//!
//! This is the answer to "the compression did nothing for finite fields": the
//! working model [`ExactDecoupled<F>`] holds `Vec<F>`, and for 𝔽_p each `F = Fp` is
//! 16 bytes carrying a *copy of the prime*. `PackedDecoupled` stores the prime
//! **once** (in the [`Packable::Context`]) and the coefficients as bare payloads
//! (`u64` for 𝔽_p, `BigRational` for ℚ, `u8` for GF(2^8)) — see [`super::store`].
//! It also drops exact zeros and **pools repeated values** through a dictionary,
//! so the two structural redundancies (per-element prime + repeated coefficients)
//! are removed together.
//!
//! It round-trips losslessly to [`ExactDecoupled`] ([`PackedDecoupled::unpack`]) and reports
//! the **honest byte footprint** ([`PackedDecoupled::stored_bytes`]) — the metric that earlier
//! abstract "nonzero count" hid.

use super::decouple::ExactDecoupled;
use super::store::{Packable, PayloadBytes, payload_bytes};

/// One stored term: flat row-major index + dictionary code. Both `u32` —
/// coefficient tensors here are far smaller than 2³² entries, so an index is 4
/// bytes, not a `usize`.
type Term = (u32, u32);

/// Bytes a stored term occupies (`u32` index + `u32` code).
const TERM_BYTES: usize = 8;

/// One axis's column nodes, **compacted** when they form an arithmetic run.
///
/// The auto-built grids place nodes at `0, 1, 2, …` (a unit-step integer run), so
/// storing the explicit vector is pure waste. [`AxisNodes::Run`] keeps only
/// `(start, step, count)`; an irregular axis falls back to [`AxisNodes::Explicit`].
/// (This is the "incrementing λ is pointless" redundancy made compact.)
#[derive(Clone, Debug, PartialEq)]
pub enum AxisNodes<P> {
    /// `count` nodes at `start, start+step, …` — three numbers, not `count` payloads.
    Run {
        start: i128,
        step: i128,
        count: usize,
    },
    /// Irregular nodes, stored explicitly.
    Explicit(Vec<P>),
}

impl<P> AxisNodes<P> {
    /// Number of nodes on this axis.
    pub fn len(&self) -> usize {
        match self {
            AxisNodes::Run { count, .. } => *count,
            AxisNodes::Explicit(v) => v.len(),
        }
    }
    /// Whether the axis has no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A compact, packed, dictionary-pooled exact decoupled model.
///
/// Storage = one [`Packable::Context`] (e.g. the prime, once) + the per-axis node
/// descriptors (an arithmetic [`AxisNodes::Run`] is three numbers) + a value
/// dictionary of distinct nonzero **payloads** + `(index, code)` term lists for
/// numerator and denominator. Lossless: [`PackedDecoupled::unpack`] reproduces the working
/// [`ExactDecoupled`] bit-for-bit.
#[derive(Clone, Debug)]
pub struct PackedDecoupled<F: Packable> {
    /// Per-axis column nodes, compacted to a run where possible.
    pub lambda: Vec<AxisNodes<F::Payload>>,
    /// The shared field context (the prime for 𝔽_p; unit otherwise) — stored once.
    pub ctx: F::Context,
    /// Per-axis node counts (so indices decode without expanding λ).
    pub dims: Vec<usize>,
    /// Distinct nonzero coefficient payloads, each stored once.
    pub dict: Vec<F::Payload>,
    /// Nonzero numerator terms `(flat index, dict code)`.
    pub num_terms: Vec<Term>,
    /// Nonzero denominator terms `(flat index, dict code)`.
    pub den_terms: Vec<Term>,
}

/// Detect an arithmetic run in a packed node vector and compact it; else keep it
/// explicit. Run detection uses [`Packable::payload_as_int`] (all 𝔽_p/GF(2^8)
/// nodes and integer ℚ nodes are integer-valued).
fn compact_axis<F: Packable>(packed: &[F::Payload]) -> AxisNodes<F::Payload> {
    if packed.len() >= 2 {
        let ints: Option<Vec<i128>> = packed.iter().map(F::payload_as_int).collect();
        if let Some(ints) = ints {
            let step = ints[1] - ints[0];
            if ints.windows(2).all(|w| w[1] - w[0] == step) {
                return AxisNodes::Run {
                    start: ints[0],
                    step,
                    count: ints.len(),
                };
            }
        }
    }
    AxisNodes::Explicit(packed.to_vec())
}

/// Expand an axis descriptor back to explicit working nodes under `ctx`.
fn expand_axis<F: Packable>(axis: &AxisNodes<F::Payload>, ctx: &F::Context) -> Vec<F> {
    match axis {
        AxisNodes::Run { start, step, count } => (0..*count)
            .map(|i| F::unpack(&F::int_payload(start + step * i as i128, ctx), ctx))
            .collect(),
        AxisNodes::Explicit(v) => v.iter().map(|p| F::unpack(p, ctx)).collect(),
    }
}

impl<F: Packable> PackedDecoupled<F> {
    /// Pack a working model: hoist the field context, drop zeros, pool values.
    pub fn pack(model: &ExactDecoupled<F>) -> Self {
        // Context from all scalars in the model (nodes + coefficients).
        let all: Vec<F> = model
            .lambda
            .iter()
            .flatten()
            .chain(&model.c_num)
            .chain(&model.c_den)
            .cloned()
            .collect();
        let ctx = F::context_of(&all);

        let lambda: Vec<AxisNodes<F::Payload>> = model
            .lambda
            .iter()
            .map(|axis| {
                let packed: Vec<F::Payload> = axis.iter().map(|v| v.pack(&ctx)).collect();
                compact_axis::<F>(&packed)
            })
            .collect();
        let dims = model.sizes();

        let mut dict: Vec<F::Payload> = Vec::new();
        let mut intern = |v: &F| -> u32 {
            let pl = v.pack(&ctx);
            match dict.iter().position(|d| *d == pl) {
                Some(i) => i as u32,
                None => {
                    dict.push(pl);
                    (dict.len() - 1) as u32
                }
            }
        };
        let gather = |tensor: &[F], intern: &mut dyn FnMut(&F) -> u32| -> Vec<Term> {
            tensor
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.is_zero())
                .map(|(j, c)| (j as u32, intern(c)))
                .collect()
        };
        let num_terms = gather(&model.c_num, &mut intern);
        let den_terms = gather(&model.c_den, &mut intern);

        Self {
            lambda,
            ctx,
            dims,
            dict,
            num_terms,
            den_terms,
        }
    }

    /// Unpack to the working [`ExactDecoupled`] — bit-exact inverse of [`Self::pack`].
    pub fn unpack(&self) -> ExactDecoupled<F> {
        let lambda: Vec<Vec<F>> = self
            .lambda
            .iter()
            .map(|axis| expand_axis::<F>(axis, &self.ctx))
            .collect();
        let total: usize = self.dims.iter().product();
        let mut c_num = vec![F::zero(); total];
        let mut c_den = vec![F::zero(); total];
        for &(j, code) in &self.num_terms {
            c_num[j as usize] = F::unpack(&self.dict[code as usize], &self.ctx);
        }
        for &(j, code) in &self.den_terms {
            c_den[j as usize] = F::unpack(&self.dict[code as usize], &self.ctx);
        }
        ExactDecoupled {
            lambda,
            c_num,
            c_den,
        }
    }

    /// Number of variables.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    /// Stored nonzero terms (numerator + denominator).
    pub fn nnz(&self) -> usize {
        self.num_terms.len() + self.den_terms.len()
    }
    /// Distinct values stored after pooling.
    pub fn dict_len(&self) -> usize {
        self.dict.len()
    }

    /// Evaluate `N(x)/D(x)` with the exact [`ExactDecoupled::eval`] semantics. For
    /// convenience this unpacks once and delegates (the working types are needed
    /// for the field arithmetic anyway); use [`PackedDecoupled::unpack`] + reuse for many
    /// queries.
    pub fn eval(&self, x: &[F]) -> Option<F> {
        self.unpack().eval(x)
    }
}

impl<F: Packable> PackedDecoupled<F>
where
    F::Payload: PayloadBytes,
{
    /// The **honest** stored byte footprint: the context, the packed nodes, the
    /// dictionary payloads, and the term lists (`usize + u32` per term). This is
    /// the number the earlier "nonzero count" hid — for 𝔽_p it now counts one
    /// prime, not one per coefficient.
    pub fn stored_bytes(&self) -> usize {
        self.node_bytes() + self.coeff_bytes() + std::mem::size_of::<F::Context>()
    }

    /// Bytes for the compacted node descriptors (a [`AxisNodes::Run`] is three
    /// `i64`s regardless of length; an explicit axis is its payloads).
    pub fn node_bytes(&self) -> usize {
        self.lambda
            .iter()
            .map(|a| match a {
                AxisNodes::Run { .. } => 3 * std::mem::size_of::<i64>(),
                AxisNodes::Explicit(v) => payload_bytes::<F>(v),
            })
            .sum()
    }

    /// Bytes for the coefficient data: the pooled value dictionary plus the
    /// `(u32 index, u32 code)` term lists. This is where the prime-hoist and
    /// value-pooling wins land — compare to [`Self::dense_coeff_bytes`].
    pub fn coeff_bytes(&self) -> usize {
        payload_bytes::<F>(&self.dict) + self.nnz() * TERM_BYTES
    }

    /// The **dense** working footprint of the coefficients for comparison: all
    /// `2·∏k` entries as full working elements `F` (16 bytes each for 𝔽_p — the
    /// per-element prime that the packed form removes).
    pub fn dense_coeff_bytes(&self) -> usize {
        2 * self.dims.iter().product::<usize>() * std::mem::size_of::<F>()
    }

    /// The total **dense** working footprint (nodes + coefficients), for the
    /// headline comparison against [`PackedDecoupled::stored_bytes`].
    pub fn dense_bytes(&self) -> usize {
        let elem = std::mem::size_of::<F>();
        let nodes: usize = self.dims.iter().sum();
        nodes * elem + self.dense_coeff_bytes()
    }

    /// A per-lever breakdown of the coefficient compression, in bytes — so the
    /// individual contributions (prime-hoist, sparse zero-drop, value pooling) are
    /// each visible, not just the final number. Each stage is the *running* packed
    /// footprint after adding that lever.
    pub fn coeff_breakdown(&self) -> CoeffBreakdown {
        let total_coeffs = 2 * self.dims.iter().product::<usize>();
        // Sum of all nonzero payload bytes (real sizes, fair for variable-width ℚ).
        let nz_payload_bytes = payload_bytes::<F>(&self.dict_all_nonzero());
        CoeffBreakdown {
            // Dense: every coefficient as a full working element `F`.
            dense: total_coeffs * std::mem::size_of::<F>(),
            // + hoist the shared context out: every coefficient as a bare payload
            // (no zero-drop, no pooling). For 𝔽_p this is the prime-hoist halving;
            // for ℚ payload == element so this equals `dense`.
            hoisted: total_coeffs * self.dense_payload_size(),
            // + sparse: only nonzero coefficients, each its own payload + index.
            sparse: nz_payload_bytes + self.nnz() * TERM_BYTES,
            // + pooled (the final packed form): distinct values once + term codes.
            pooled: self.coeff_bytes(),
        }
    }

    /// The byte width of one *dense* coefficient after hoisting the shared context.
    /// For a fixed-width payload smaller than the working element (𝔽_p `u64` vs 16-B
    /// `Fp`; GF(2^8) `u8`) this is the hoist saving; for ℚ the payload *is* the
    /// `BigRational` element, so there is no hoist and it equals `size_of::<F>()`.
    fn dense_payload_size(&self) -> usize {
        // `size_of::<Payload>()` is the in-memory width: 8 for `u64`, 1 for `u8`,
        // and `size_of::<BigRational>()` (== `size_of::<Rational>()`) for ℚ. The
        // hoist only saves when the payload type is strictly narrower than `F`.
        std::mem::size_of::<F::Payload>().min(std::mem::size_of::<F>())
    }

    /// All nonzero coefficient payloads (numerator + denominator), expanded from the
    /// pooled dictionary — for sizing the sparse (un-pooled) footprint.
    fn dict_all_nonzero(&self) -> Vec<F::Payload> {
        self.num_terms
            .iter()
            .chain(&self.den_terms)
            .map(|&(_, code)| self.dict[code as usize].clone())
            .collect()
    }
}

/// Per-lever byte breakdown of coefficient compression — each field is the running
/// packed footprint after the named lever, so the contribution of each is the gap
/// to the previous. See [`PackedDecoupled::coeff_breakdown`].
#[derive(Clone, Copy, Debug)]
pub struct CoeffBreakdown {
    /// Dense working footprint (`2·∏k` full elements).
    pub dense: usize,
    /// After hoisting the shared context (𝔽_p prime) out: bare payloads, still
    /// dense (no zero-drop, no pooling).
    pub hoisted: usize,
    /// After also dropping exact zeros (the *sparse* footprint): nonzero payloads
    /// + index per term, no pooling.
    pub sparse: usize,
    /// After also pooling repeated values (the final packed form): distinct
    /// payloads once + `(index, code)` term codes.
    pub pooled: usize,
}

#[cfg(test)]
mod tests {
    use super::super::decouple::{ExactGrid, decouple_from_closure};
    use super::super::field::Fp;
    use super::super::rational::Rational;
    use super::*;

    #[test]
    fn fp_packed_model_hoists_prime_and_roundtrips() {
        // field product a·b over GF(257): the dump that motivated this refactor.
        let p = 257;
        let e = |v: i128| Fp::new(v, p);
        let grid = ExactGrid {
            lambda: vec![vec![e(0), e(1), e(2), e(3)]],
            mu: vec![vec![e(4), e(5), e(6), e(7)]],
        };
        // Single-axis "product"-ish degree-3 poly so it's a real 𝔽_p model.
        let f = |x: &[Fp]| x[0] * x[0] * x[0];
        let model = decouple_from_closure(&grid, f).unwrap();
        let packed = PackedDecoupled::pack(&model);
        // The prime appears exactly once, in the context.
        assert_eq!(packed.ctx, p);
        // Lossless round-trip.
        let back = packed.unpack();
        assert_eq!(back.c_num, model.c_num);
        assert_eq!(back.c_den, model.c_den);
        assert_eq!(back.lambda, model.lambda);
        // Honest footprint: packed beats dense (bare u64 + one prime vs 16B each).
        assert!(
            packed.stored_bytes() < packed.dense_bytes(),
            "packed {} should beat dense {}",
            packed.stored_bytes(),
            packed.dense_bytes()
        );
        // Eval parity.
        for t in [10i128, 99, 200] {
            let pt = [Fp::new(t, p)];
            assert_eq!(packed.eval(&pt), model.eval(&pt));
        }
    }

    #[test]
    fn lambda_run_is_compacted_not_stored_per_node() {
        // The auto-built grid uses nodes 0,1,2,… on each axis — an arithmetic run.
        let int = Rational::integer;
        let grid = ExactGrid {
            lambda: vec![vec![int(0), int(1), int(2)], vec![int(0), int(1), int(2)]],
            mu: vec![vec![int(3), int(4), int(5)], vec![int(3), int(4), int(5)]],
        };
        let f = |x: &[Rational]| x[0].clone() * x[1].clone();
        let model = decouple_from_closure(&grid, f).unwrap();
        let packed = PackedDecoupled::pack(&model);
        // Both axes collapse to a Run (start 0, step 1, count 3) — not 3 nodes each.
        for axis in &packed.lambda {
            assert!(
                matches!(
                    axis,
                    AxisNodes::Run {
                        start: 0,
                        step: 1,
                        count: 3
                    }
                ),
                "node run should compact, got {axis:?}"
            );
        }
        // Lossless: unpack restores the explicit nodes and the model.
        let back = packed.unpack();
        assert_eq!(back.lambda, model.lambda);
        assert_eq!(back.c_num, model.c_num);
    }

    #[test]
    fn rational_packed_pools_symmetric_values() {
        // Pólya–Szegő — symmetry pools coefficients.
        let int = Rational::integer;
        let grid = ExactGrid {
            lambda: vec![vec![int(0), int(1)]; 3],
            mu: vec![vec![int(2), int(3)]; 3],
        };
        let f = |x: &[Rational]| {
            x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone()
        };
        let model = decouple_from_closure(&grid, f).unwrap();
        let packed = PackedDecoupled::pack(&model);
        assert!(packed.dict_len() < packed.nnz(), "symmetry pools values");
        let back = packed.unpack();
        assert_eq!(back.c_num, model.c_num);
    }
}
