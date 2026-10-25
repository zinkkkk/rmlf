//! A browsable catalog of ℚ examples — integer/rational polynomials and rational
//! functions — doubling as the regression corpus and the sweep input.
//!
//! Each [`Case`] carries a generic target over [`Rational`] plus its per-axis
//! degree, so [`Case::fit`] reconstructs it in one call (building the grid via the
//! [`super::fit_poly`] front door). Cases 0–9 are well-posed (recover exactly);
//! 10–12 are **limit cases** that probe the boundaries (under-resolved degree,
//! pole risk, non-coprime diagonal).

use super::{RationalModel, fit_poly};
use crate::error::Result;
use crate::exact::Rational;

/// Structural tag for a catalog case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    /// Multivariate polynomial (`d ≡ 1`).
    Polynomial,
    /// Multivariate rational `n/d`.
    Rational,
    /// Separable (`∏_l g_l(x_l)`) — TT-rank 1.
    Separable,
    /// Genuinely coupled (TT-rank > 1).
    Coupled,
}

/// One ℚ catalog entry.
#[derive(Clone)]
pub struct Case {
    /// Catalog id.
    pub id: u32,
    /// Number of variables.
    pub ndim: usize,
    /// Human-readable name.
    pub name: &'static str,
    /// Per-axis degree the grid is sized for. For a well-posed case this is the
    /// target's true degree; for a limit case it is deliberately too low.
    pub degree: &'static [usize],
    /// Structural tags.
    pub tags: &'static [Tag],
    /// `true` if the target recovers exactly on its grid.
    pub exact: bool,
    /// The generic target over ℚ.
    pub f: fn(&[Rational]) -> Rational,
}

impl Case {
    /// Fit this case via the [`super::fit_poly`] front door.
    pub fn fit(&self) -> Result<RationalModel> {
        fit_poly(self.f, self.degree)
    }
    /// Total grid samples taken (`∏ 2(degree_l + 1)`).
    pub fn samples(&self) -> usize {
        self.degree.iter().map(|d| 2 * (d + 1)).product()
    }
}

// ── target closures ──

fn polya(x: &[Rational]) -> Rational {
    x[0].clone() * x[1].clone() + x[0].clone() * x[2].clone() + x[1].clone() * x[2].clone()
}
fn sep3(x: &[Rational]) -> Rational {
    (x[0].clone() + Rational::integer(1))
        * (x[1].clone() + Rational::integer(2))
        * (x[2].clone() + Rational::integer(3))
}
fn coupled2(x: &[Rational]) -> Rational {
    let (a, b) = (x[0].clone(), x[1].clone());
    a.clone() * a.clone() * b.clone() + a.clone() * b.clone() * b.clone() + Rational::one()
}
fn coupled4(x: &[Rational]) -> Rational {
    x[0].clone() * x[1].clone() + x[2].clone() * x[3].clone() + x[0].clone() * x[3].clone()
}
fn sep5(x: &[Rational]) -> Rational {
    let mut acc = Rational::one();
    for (l, xl) in x.iter().enumerate() {
        acc = acc * (xl.clone() + Rational::integer(l as i64));
    }
    acc
}
fn highdeg(x: &[Rational]) -> Rational {
    let a = x[0].clone();
    let a2 = a.clone() * a.clone();
    let a4 = a2.clone() * a2.clone();
    a4 * x[1].clone() + a
}
fn seprat(x: &[Rational]) -> Rational {
    Rational::one()
        / ((x[0].clone() + Rational::integer(1)) * (x[1].clone() + Rational::integer(2)))
}
fn apxc(x: &[Rational]) -> Rational {
    let (s, t) = (x[0].clone(), x[1].clone());
    let num = t.clone() * t.clone() + s.clone() - Rational::integer(2);
    let den = t.clone() * t.clone() + Rational::integer(2) * s + Rational::one();
    num / den
}
fn rat3(x: &[Rational]) -> Rational {
    let num = x[0].clone() + x[1].clone() * x[2].clone();
    let den = Rational::one() + x[0].clone() * x[0].clone() + x[1].clone() * x[1].clone();
    num / den
}
fn fftab(x: &[Rational]) -> Rational {
    x[0].clone() * x[0].clone()
        + x[0].clone() * x[1].clone()
        + x[1].clone() * x[1].clone()
        + Rational::integer(7)
}
fn underdeg(x: &[Rational]) -> Rational {
    let (a, b) = (x[0].clone(), x[1].clone());
    a.clone() * a.clone() * a.clone() * b.clone() * b.clone() * b.clone()
        + a.clone() * b.clone()
        + Rational::one()
}
fn poled(x: &[Rational]) -> Rational {
    Rational::one() / (x[0].clone() + x[1].clone() + Rational::one())
}
fn noncoprime(x: &[Rational]) -> Rational {
    let (a, b) = (x[0].clone(), x[1].clone());
    (a.clone() * a.clone() - b.clone() * b.clone()) / (a.clone() - b.clone())
}

/// The full ℚ catalog.
pub fn all() -> &'static [Case] {
    use Tag::*;
    &[
        Case {
            id: 0,
            ndim: 3,
            name: "Polya-Szego xy+xz+yz",
            degree: &[1, 1, 1],
            tags: &[Polynomial, Coupled],
            exact: true,
            f: polya,
        },
        Case {
            id: 1,
            ndim: 3,
            name: "separable (x+1)(y+2)(z+3)",
            degree: &[1, 1, 1],
            tags: &[Polynomial, Separable],
            exact: true,
            f: sep3,
        },
        Case {
            id: 2,
            ndim: 2,
            name: "coupled x^2 y + x y^2 + 1",
            degree: &[2, 2],
            tags: &[Polynomial, Coupled],
            exact: true,
            f: coupled2,
        },
        Case {
            id: 3,
            ndim: 4,
            name: "4-var coupled x0x1+x2x3+x0x3",
            degree: &[1, 1, 1, 1],
            tags: &[Polynomial, Coupled],
            exact: true,
            f: coupled4,
        },
        Case {
            id: 4,
            ndim: 5,
            name: "separable 5-var prod(x_l+l)",
            degree: &[1, 1, 1, 1, 1],
            tags: &[Polynomial, Separable],
            exact: true,
            f: sep5,
        },
        Case {
            id: 5,
            ndim: 2,
            name: "high-degree x0^4 x1 + x0",
            degree: &[4, 1],
            tags: &[Polynomial, Coupled],
            exact: true,
            f: highdeg,
        },
        Case {
            id: 6,
            ndim: 2,
            name: "separable rational 1/((x0+1)(x1+2))",
            degree: &[1, 1],
            tags: &[Rational, Separable],
            exact: true,
            f: seprat,
        },
        Case {
            id: 7,
            ndim: 2,
            name: "Appendix C (t^2+s-2)/(t^2+2s+1)",
            degree: &[1, 2],
            tags: &[Rational, Coupled],
            exact: true,
            f: apxc,
        },
        Case {
            id: 8,
            ndim: 3,
            name: "coupled rational (x0+x1x2)/(1+x0^2+x1^2)",
            degree: &[2, 2, 1],
            tags: &[Rational, Coupled],
            exact: true,
            f: rat3,
        },
        Case {
            id: 9,
            ndim: 2,
            name: "quadratic table x0^2+x0x1+x1^2+7",
            degree: &[2, 2],
            tags: &[Polynomial, Coupled],
            exact: true,
            f: fftab,
        },
        // ── limit cases ──
        Case {
            id: 10,
            ndim: 2,
            name: "LIMIT under-resolved x0^3 x1^3 + x0 x1 + 1 (grid deg 1)",
            degree: &[1, 1],
            tags: &[Polynomial, Coupled],
            exact: false,
            f: underdeg,
        },
        Case {
            id: 11,
            ndim: 2,
            name: "LIMIT 1/(x0+x1+1)",
            degree: &[1, 1],
            tags: &[Rational, Coupled],
            exact: true,
            f: poled,
        },
        Case {
            id: 12,
            ndim: 2,
            name: "LIMIT (x0^2-x1^2)/(x0-x1) = x0+x1 (non-coprime)",
            degree: &[1, 1],
            tags: &[Rational, Coupled],
            exact: true,
            f: noncoprime,
        },
    ]
}

/// The well-posed cases (expected to recover exactly).
pub fn well_posed() -> impl Iterator<Item = &'static Case> {
    all().iter().filter(|c| c.exact)
}

/// Look up a case by id.
pub fn get(id: u32) -> Option<&'static Case> {
    all().iter().find(|c| c.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off-grid integer point that avoids nodes and the x0=x1 diagonal.
    fn point(ndim: usize, seed: i64) -> Vec<i64> {
        (0..ndim).map(|l| seed + 2 * l as i64 + 1).collect()
    }

    #[test]
    fn well_posed_cases_recover_exactly() {
        for c in well_posed() {
            let m = c.fit().unwrap_or_else(|e| panic!("{}: {e:?}", c.name));
            for seed in [13i64, -7, 100] {
                let pt = point(c.ndim, seed);
                let qpt: Vec<Rational> = pt.iter().map(|&v| Rational::integer(v)).collect();
                assert_eq!(
                    m.eval_q(&pt),
                    Some((c.f)(&qpt)),
                    "case {} ({})",
                    c.id,
                    c.name
                );
            }
        }
    }

    #[test]
    fn under_resolved_limit_fails() {
        let c = get(10).unwrap();
        let m = c.fit().unwrap();
        let disagrees = [13i64, -7, 100, 5].iter().any(|&seed| {
            let pt = point(c.ndim, seed);
            let qpt: Vec<Rational> = pt.iter().map(|&v| Rational::integer(v)).collect();
            m.eval_q(&pt) != Some((c.f)(&qpt))
        });
        assert!(disagrees, "under-resolved case unexpectedly recovered");
    }
}
