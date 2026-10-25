//! ℚ — the exact rationals, BigInt-backed.
//!
//! [`Rational`] is the field for integer/rational polynomial and rational-function
//! reconstruction with **zero rounding error**.  It stores a normalized pair of
//! [`num_bigint::BigInt`] values directly so the exact path uses the current
//! `num-bigint` API without depending on a rational crate tied to an older,
//! type-incompatible `BigInt` release.

use crate::exact::core::Field;
use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Zero};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An exact rational (ℚ).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rational {
    numer: BigInt,
    denom: BigInt,
}

impl Rational {
    /// Construct `num/den` in lowest terms. Panics on a zero denominator (a
    /// programming error, never produced by the framework).
    pub fn new(num: impl Into<BigInt>, den: impl Into<BigInt>) -> Self {
        let mut numer = num.into();
        let mut denom = den.into();
        assert!(!denom.is_zero(), "Rational denominator is zero");

        let gcd = numer.gcd(&denom);
        numer /= &gcd;
        denom /= gcd;
        if denom.sign() == Sign::Minus {
            numer = -numer;
            denom = -denom;
        }
        Self { numer, denom }
    }

    /// An integer as a rational `n/1`.
    pub fn integer(n: impl Into<BigInt>) -> Self {
        Self {
            numer: n.into(),
            denom: BigInt::one(),
        }
    }

    /// The rational `0` (inherent convenience; same as [`Field::zero`]).
    pub fn zero() -> Self {
        <Self as Field>::zero()
    }
    /// The rational `1` (inherent convenience; same as [`Field::one`]).
    pub fn one() -> Self {
        <Self as Field>::one()
    }

    /// The numerator (in lowest terms, sign on the numerator).
    pub fn numer(&self) -> &BigInt {
        &self.numer
    }
    /// The denominator (in lowest terms, always positive).
    pub fn denom(&self) -> &BigInt {
        &self.denom
    }

    /// `true` iff this rational is an integer (`den == 1`).
    pub fn is_integer(&self) -> bool {
        self.denom.is_one()
    }

    /// Lossy `f64` view (for diagnostics / comparison against the float core).
    pub fn to_f64(&self) -> f64 {
        let n: f64 = self.numer.to_string().parse().unwrap_or(f64::NAN);
        let d: f64 = self.denom.to_string().parse().unwrap_or(f64::NAN);
        n / d
    }
}

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.numer * &rhs.denom + rhs.numer * &self.denom,
            self.denom * rhs.denom,
        )
    }
}
impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.numer * &rhs.denom - rhs.numer * &self.denom,
            self.denom * rhs.denom,
        )
    }
}
impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.numer * rhs.numer, self.denom * rhs.denom)
    }
}
impl Div for Rational {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        assert!(!rhs.numer.is_zero(), "Rational division by zero");
        Self::new(self.numer * rhs.denom, self.denom * rhs.numer)
    }
}
impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            numer: -self.numer,
            denom: self.denom,
        }
    }
}

// Reference operators — so user closures can write `&x[0] * &x[1]` without
// cloning at every use (a big readability win for multivariate targets).
macro_rules! ref_binop {
    ($trait:ident, $method:ident) => {
        impl $trait<&Rational> for &Rational {
            type Output = Rational;
            fn $method(self, rhs: &Rational) -> Rational {
                $trait::$method(self.clone(), rhs.clone())
            }
        }
    };
}
ref_binop!(Add, add);
ref_binop!(Sub, sub);
ref_binop!(Mul, mul);
ref_binop!(Div, div);

impl Field for Rational {
    fn zero() -> Self {
        Self::integer(0)
    }
    fn one() -> Self {
        Self::integer(1)
    }
    fn from_int(n: i64) -> Self {
        Self::integer(n)
    }
    fn is_zero(&self) -> bool {
        self.numer.is_zero()
    }
    fn inv(&self) -> Option<Self> {
        if self.numer.is_zero() {
            None
        } else {
            Some(Self::new(self.denom.clone(), self.numer.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_arithmetic_is_exact() {
        let a = Rational::new(1, 3);
        let b = Rational::new(1, 6);
        assert_eq!(a.clone() + b.clone(), Rational::new(1, 2));
        assert_eq!(a.clone() - b.clone(), Rational::new(1, 6));
        assert_eq!(a.clone() * b.clone(), Rational::new(1, 18));
        assert_eq!(a.clone() / b.clone(), Rational::integer(2));
        assert_eq!(a.inv().unwrap(), Rational::integer(3));
        assert!(Rational::zero().inv().is_none());
    }

    #[test]
    fn rational_reduces_and_signs() {
        assert_eq!(Rational::new(2, 4), Rational::new(1, 2));
        assert_eq!(Rational::new(-1, -2), Rational::new(1, 2));
        assert_eq!(Rational::new(1, -2), Rational::new(-1, 2));
        assert!(Rational::integer(7).is_integer());
        assert!(!Rational::new(1, 2).is_integer());
    }
}
