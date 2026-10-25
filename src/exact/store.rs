//! Packed storage — the layer that makes a stored exact model *actually* compact.
//!
//! ## The problem this solves
//!
//! During construction the scalars are the ergonomic working types ([`Fp`],
//! [`Rational`]). [`Fp`] in particular carries its prime *in every element*
//! (`Fp { val: u64, p: u64 }` — 16 bytes), so a stored coefficient tensor drags
//! one copy of the prime per entry, and the modulus-free constant `p == 0` can
//! even leak into stored data. Counting "nonzero coefficients" hid this: the
//! abstract count was small, the byte footprint was not.
//!
//! ## The fix
//!
//! A [`Packable`] field declares a compact **payload** type and a per-model
//! **context** that holds whatever is shared across all elements (the prime for
//! 𝔽_p; nothing for ℚ). The model stores `Vec<Payload>` + one `Context`:
//!
//! | field      | working element | packed payload | context (stored once) |
//! |------------|-----------------|----------------|------------------------|
//! | [`Fp`]     | `{val,p}` 16 B  | `u64` 8 B      | the prime `p`          |
//! | [`Rational`] | `BigRational` | `BigRational`  | `()`                   |
//!
//! So 𝔽_p storage **halves** and never repeats the prime; ℚ is already minimal and
//! packs to itself. [`Packable::pack`]/[`Packable::unpack`] round-trip exactly, so the packed models
//! ([`super::tt::TtDecoupled`], [`super::PackedDecoupled`]) evaluate identically
//! to the dense working model.

use super::core::Field;
use super::field::Fp;
use super::rational::Rational;

/// A field whose elements can be stored in a compact payload plus a per-model
/// context (the data shared by every element — the prime, for 𝔽_p).
pub trait Packable: Field {
    /// The compact stored form of one element.
    type Payload: Clone + std::fmt::Debug + PartialEq;
    /// Per-model shared data reconstituting the working element from a payload.
    type Context: Clone + std::fmt::Debug + PartialEq;

    /// The shared context of a built model — derived from its elements. All
    /// elements of one model share a context (the same prime, for 𝔽_p).
    fn context_of(elems: &[Self]) -> Self::Context;
    /// Pack one element into its payload under `ctx`.
    fn pack(&self, ctx: &Self::Context) -> Self::Payload;
    /// Unpack a payload back to the working element under `ctx`.
    fn unpack(payload: &Self::Payload, ctx: &Self::Context) -> Self;

    /// The integer this payload represents, if it is integer-valued — used to
    /// detect and compact an arithmetic **run** of nodes (`0,1,2,…`) to a
    /// `(start, step, count)` descriptor instead of storing every node. `None`
    /// for a non-integer payload (a true rational). 𝔽_p values are always integers
    /// in `[0,p)`; GF(2^8) bytes are integers; a `Rational` is integer iff its
    /// denominator is 1.
    fn payload_as_int(payload: &Self::Payload) -> Option<i128>;
    /// The payload for an integer (the inverse of [`Self::payload_as_int`] on
    /// integer values) — to expand a compacted run back to explicit payloads.
    fn int_payload(n: i128, ctx: &Self::Context) -> Self::Payload;
}

// ── ℚ : already minimal, packs to itself (context is unit) ──
impl Packable for Rational {
    type Payload = Rational;
    type Context = ();
    fn context_of(_: &[Self]) {}
    fn pack(&self, _: &()) -> Rational {
        self.clone()
    }
    fn unpack(payload: &Rational, _: &()) -> Self {
        payload.clone()
    }
    fn payload_as_int(payload: &Rational) -> Option<i128> {
        if payload.is_integer() {
            payload.numer().try_into().ok()
        } else {
            None
        }
    }
    fn int_payload(n: i128, _: &()) -> Rational {
        Rational::integer(n as i64)
    }
}

// ── 𝔽_p : hoist the prime to the context, store bare u64 values ──
impl Packable for Fp {
    type Payload = u64;
    type Context = u64; // the prime

    /// The model's prime: the modulus of any element that carries one (a built
    /// model always has at least one genuine 𝔽_p element). Falls back to `0` only
    /// for an all-modulus-free model, which the framework never stores.
    fn context_of(elems: &[Self]) -> u64 {
        elems.iter().map(Fp::modulus).find(|&p| p != 0).unwrap_or(0)
    }
    fn pack(&self, &p: &u64) -> u64 {
        // Reduce into [0, p); a modulus-free constant (p_elem == 0) only ever holds
        // 0 or 1, which are already canonical.
        if p == 0 {
            self.value()
        } else {
            self.value() % p
        }
    }
    fn unpack(&payload: &u64, &p: &u64) -> Self {
        Fp::new(payload as i128, p)
    }
    fn payload_as_int(&payload: &u64) -> Option<i128> {
        Some(payload as i128)
    }
    fn int_payload(n: i128, &p: &u64) -> u64 {
        if p == 0 {
            n as u64
        } else {
            (n.rem_euclid(p as i128)) as u64
        }
    }
}

/// Pack a slice of working elements under a shared context.
pub fn pack_all<F: Packable>(elems: &[F], ctx: &F::Context) -> Vec<F::Payload> {
    elems.iter().map(|e| e.pack(ctx)).collect()
}

/// Unpack a slice of payloads under a shared context.
pub fn unpack_all<F: Packable>(payloads: &[F::Payload], ctx: &F::Context) -> Vec<F> {
    payloads.iter().map(|p| F::unpack(p, ctx)).collect()
}

/// Bytes used by a packed tensor `Vec<Payload>` (the honest stored footprint).
/// For 𝔽_p this is `8·len` plus one prime; for ℚ it is the sum of the bigint
/// magnitudes; for GF(256) it is `len` bytes.
pub fn payload_bytes<F: Packable>(payloads: &[F::Payload]) -> usize
where
    F::Payload: PayloadBytes,
{
    payloads.iter().map(PayloadBytes::bytes).sum()
}

/// The byte size of one packed payload (for honest footprint accounting — the
/// metric the earlier "abstract nonzero count" hid).
pub trait PayloadBytes {
    /// Bytes this payload occupies when stored.
    fn bytes(&self) -> usize;
}
impl PayloadBytes for u64 {
    fn bytes(&self) -> usize {
        8
    }
}
impl PayloadBytes for Rational {
    fn bytes(&self) -> usize {
        // BigInt magnitude in bytes (ceil bits/8), numerator + denominator, min 1 each.
        let b = |n: &num_bigint::BigInt| (n.bits() as usize).div_ceil(8).max(1);
        b(self.numer()) + b(self.denom())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_packs_to_bare_u64_with_one_prime() {
        let p = 257;
        let elems: Vec<Fp> = [0i128, 3, 233, 27].iter().map(|&v| Fp::new(v, p)).collect();
        let ctx = Fp::context_of(&elems);
        assert_eq!(ctx, p, "prime hoisted to context");
        let packed = pack_all(&elems, &ctx);
        assert_eq!(packed, vec![0u64, 3, 233, 27], "bare values, no prime");
        // Round-trips exactly, and the modulus is restored.
        let back = unpack_all::<Fp>(&packed, &ctx);
        assert_eq!(back, elems);
        assert!(
            back.iter().all(|e| e.modulus() == p),
            "prime restored on unpack"
        );
        // Footprint: 8 bytes/element (not 16), the prime counted once at the model.
        assert_eq!(payload_bytes::<Fp>(&packed), 4 * 8);
    }

    #[test]
    fn fp_pack_normalizes_modulus_free_constant_leak() {
        // The `p == 0` modulus-free `1` that used to leak into stored data
        // (`Fp { val: 1, p: 0 }`) packs to a clean value under the model's prime.
        let p = 257;
        let leaked = Fp::one(); // p == 0
        let ctx = 257u64;
        assert_eq!(leaked.pack(&ctx), 1);
        assert_eq!(Fp::unpack(&1, &ctx), Fp::new(1, p));
    }

    #[test]
    fn rational_packs_to_itself() {
        let r = Rational::new(3, 7);
        assert_eq!(Rational::unpack(&r.pack(&()), &()), r);
    }
}
