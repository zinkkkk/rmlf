//! 𝔽_p — the prime fields, integers mod a runtime prime.
//!
//! [`Fp`] carries its modulus `p` *in the element* so a heterogeneous expression
//! knows its prime; build elements via [`Fp::new`]. This is the convenient form
//! for **arithmetic during construction** — note that the *stored* models never
//! keep this 16-byte-per-element form: they pack to bare `u64` with the prime
//! hoisted once onto the model (see [`crate::exact::store::Packable::Context`] and the
//! packed storage in [`crate::exact::store`]). That packing is what makes a
//! finite-field model actually compact (no repeated prime, no `p==0` constant
//! leaking into stored data).
//!
//! `p` must be prime for [`Field::inv`] to be total on nonzero elements (the
//! inverse uses Fermat's little theorem `a^(p−2)`); a composite modulus makes `Fp`
//! a ring, not a field (see `docs/EXACT_FIELDS.md` §1 "unlucky prime").

use crate::exact::core::Field;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An element of 𝔽_p.
///
/// The generic [`Field`] methods `zero`/`one`/`from_int` produce **modulus-free
/// constants** (`p == 0`) — there is no modulus context at those call sites.
/// Arithmetic resolves the working modulus from whichever operand carries one
/// (see `Fp::resolve`), so `Fp::zero() + Fp::new(3, 7)` is `3 mod 7`. Combining
/// two genuinely *different* nonzero moduli is a bug and is debug-asserted.
#[derive(Clone, Copy, Debug)]
pub struct Fp {
    val: u64, // in [0, p) when p > 0; a small constant when p == 0 (modulus-free)
    p: u64,
}

impl Fp {
    /// The class of `v mod p` in 𝔽_p. `v` may be any (signed) integer.
    pub fn new(v: i128, p: u64) -> Self {
        let m = p as i128;
        let r = ((v % m) + m) % m;
        Self { val: r as u64, p }
    }

    /// The canonical representative in `[0, p)`.
    pub fn value(&self) -> u64 {
        self.val
    }
    /// The modulus (`0` for a modulus-free constant).
    pub fn modulus(&self) -> u64 {
        self.p
    }

    /// Resolve the common modulus of two operands, lifting a modulus-free constant
    /// (`p == 0`) to the other's modulus. Reduces both `val`s mod that prime.
    fn resolve(self, other: Self) -> (u64, u64, u64) {
        let p = match (self.p, other.p) {
            (0, q) => q,
            (q, 0) => q,
            (a, b) => {
                debug_assert_eq!(a, b, "Fp arithmetic across different moduli");
                a
            }
        };
        if p == 0 {
            // Both modulus-free: stay in the constant world (only 0/1 ever arise).
            (self.val, other.val, 0)
        } else {
            (self.val % p, other.val % p, p)
        }
    }

    /// `base^exp mod p` by square-and-multiply (for the Fermat inverse).
    fn pow(base: u64, mut exp: u64, p: u64) -> u64 {
        let mut acc: u128 = 1;
        let mut b = base as u128;
        let pm = p as u128;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = (acc * b) % pm;
            }
            b = (b * b) % pm;
            exp >>= 1;
        }
        acc as u64
    }
}

impl PartialEq for Fp {
    fn eq(&self, other: &Self) -> bool {
        let (a, b, _) = self.resolve(*other);
        a == b
    }
}

impl Add for Fp {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (a, b, p) = self.resolve(rhs);
        if p == 0 {
            return Fp { val: a + b, p: 0 };
        }
        Fp {
            val: ((a as u128 + b as u128) % p as u128) as u64,
            p,
        }
    }
}
impl Sub for Fp {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let (a, b, p) = self.resolve(rhs);
        let pm = p as u128;
        Fp {
            val: ((a as u128 + pm - b as u128) % pm) as u64,
            p,
        }
    }
}
impl Mul for Fp {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (a, b, p) = self.resolve(rhs);
        if p == 0 {
            return Fp { val: a * b, p: 0 };
        }
        Fp {
            val: ((a as u128 * b as u128) % p as u128) as u64,
            p,
        }
    }
}
#[allow(clippy::suspicious_arithmetic_impl)] // division multiplies by the modular inverse
impl Div for Fp {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let inv = rhs.inv().expect("Fp division by zero");
        self * inv
    }
}
impl Neg for Fp {
    type Output = Self;
    fn neg(self) -> Self {
        if self.p == 0 {
            return Fp {
                val: self.val,
                p: 0,
            }; // only 0 is ever negated modulus-free
        }
        Fp {
            val: (self.p - self.val % self.p) % self.p,
            p: self.p,
        }
    }
}

impl Field for Fp {
    /// A modulus-free `0` (`p == 0`); arithmetic with a real element adopts its
    /// modulus. Callers wanting a 𝔽_p zero with a concrete prime use `Fp::new(0, p)`.
    fn zero() -> Self {
        Fp { val: 0, p: 0 }
    }
    /// A modulus-free `1` (`p == 0`).
    fn one() -> Self {
        Fp { val: 1, p: 0 }
    }
    /// A modulus-free small constant. Prefer [`Fp::new`] in 𝔽_p code; this exists to
    /// satisfy the trait and is only used for tiny non-negative literals (0, 1, 2)
    /// that the framework introduces, which resolve correctly on first contact with
    /// a real-modulus operand.
    fn from_int(n: i64) -> Self {
        debug_assert!(
            n >= 0,
            "Fp::from_int needs a concrete modulus for negatives"
        );
        Fp {
            val: n.max(0) as u64,
            p: 0,
        }
    }
    fn is_zero(&self) -> bool {
        self.val == 0
    }
    fn inv(&self) -> Option<Self> {
        if self.val == 0 {
            None
        } else if self.p == 0 {
            // Modulus-free: only `1` is ever inverted here, and 1⁻¹ = 1.
            debug_assert_eq!(self.val, 1, "modulus-free inverse only defined for 1");
            Some(*self)
        } else {
            // Fermat: a^(p−2) ≡ a⁻¹ (mod p), valid because p is prime.
            Some(Fp {
                val: Self::pow(self.val % self.p, self.p - 2, self.p),
                p: self.p,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_arithmetic_mod_7() {
        let p = 7;
        let a = Fp::new(3, p);
        let b = Fp::new(5, p);
        assert_eq!((a + b).value(), 1); // 8 mod 7
        assert_eq!((a - b).value(), 5); // -2 mod 7
        assert_eq!((a * b).value(), 1); // 15 mod 7
        assert_eq!(a.inv().unwrap(), b); // 3*5 = 1 ⇒ 5 = 3⁻¹
        assert_eq!((a / a).value(), 1);
        assert!(Fp::new(0, p).inv().is_none());
    }

    #[test]
    fn fp_negation_and_normalization() {
        let p = 11;
        assert_eq!(Fp::new(-1, p).value(), 10);
        assert_eq!((-Fp::new(4, p)).value(), 7);
        assert_eq!(Fp::new(25, p).value(), 3);
    }
}
