//! A catalog of finite fields 𝔽_p to exercise the exact decoupling across a wide
//! range of moduli — small, large, and number-theoretically notable.
//!
//! [`super::Fp`] implements the **prime** fields 𝔽_p (`ℤ/pℤ`). Genuine
//! prime-*power* Galois fields `GF(p^k)` (GF(4), GF(8), GF(256)=AES, …) are *not*
//! prime fields — their arithmetic is polynomials mod an irreducible — and are a
//! documented future extension (see `docs/EXACT_FIELDS.md`); they would slot in as
//! another [`crate::exact::Field`] impl with the decoupling code unchanged. This
//! catalog therefore enumerates **primes**, chosen to span the regimes that
//! actually stress the method:
//!
//! - **Tiny** (GF(2), GF(3), GF(5), GF(7)) — smaller than most catalog grids, so a
//!   case only runs over them if `p ≥ ExactExample::min_prime()`; used to show the
//!   *grid-too-large-for-the-field* boundary.
//! - **Small/medium** (13, 17, 31, 97, 251, 257) — comfortably fit the catalog;
//!   257 is the Fermat prime `F_3 = 2^8 + 1`.
//! - **Number-theory notables** — Mersenne `2^31 − 1` (a classic word-sized prime),
//!   the NIST/Solinas-style `2^61 − 1` (Mersenne), and the large generic
//!   `1_000_003`. These mirror primes used in hashing, NTT/FFT, and crypto.

/// A finite field 𝔽_p with a short human label describing its character.
#[derive(Debug, Clone, Copy)]
pub struct FieldSpec {
    /// The prime modulus.
    pub p: u64,
    /// A short descriptive name.
    pub name: &'static str,
    /// `true` for the tiny fields included mainly to probe the
    /// grid-larger-than-field boundary.
    pub tiny: bool,
}

/// The full finite-field catalog, ascending by modulus.
pub fn all() -> &'static [FieldSpec] {
    &[
        FieldSpec {
            p: 2,
            name: "GF(2) binary",
            tiny: true,
        },
        FieldSpec {
            p: 3,
            name: "GF(3) ternary",
            tiny: true,
        },
        FieldSpec {
            p: 5,
            name: "GF(5)",
            tiny: true,
        },
        FieldSpec {
            p: 7,
            name: "GF(7)",
            tiny: true,
        },
        FieldSpec {
            p: 13,
            name: "GF(13)",
            tiny: false,
        },
        FieldSpec {
            p: 17,
            name: "GF(17)",
            tiny: false,
        },
        FieldSpec {
            p: 31,
            name: "GF(31) Mersenne 2^5-1",
            tiny: false,
        },
        FieldSpec {
            p: 97,
            name: "GF(97)",
            tiny: false,
        },
        FieldSpec {
            p: 251,
            name: "GF(251) largest 8-bit prime",
            tiny: false,
        },
        FieldSpec {
            p: 257,
            name: "GF(257) Fermat 2^8+1",
            tiny: false,
        },
        FieldSpec {
            p: 65_537,
            name: "GF(65537) Fermat 2^16+1",
            tiny: false,
        },
        FieldSpec {
            p: 2_147_483_647,
            name: "GF(2^31-1) Mersenne",
            tiny: false,
        },
        FieldSpec {
            p: 2_305_843_009_213_693_951,
            name: "GF(2^61-1) Mersenne",
            tiny: false,
        },
    ]
}

/// Fields large enough for a case needing modulus `≥ min_prime`.
pub fn fitting(min_prime: u64) -> impl Iterator<Item = &'static FieldSpec> {
    all().iter().filter(move |f| f.p >= min_prime)
}
