//! Benchmark / example function catalog, ported from `+mlf/examples.m`.
//!
//! Each [`Example`] bundles the target function as a plain `fn(&[f64]) -> f64`
//! (exactly what [`crate::alg1()`] / [`crate::alg2()`] consume), its per-axis
//! domain, a recommended interpolation-point count, and citation metadata. The
//! `id` matches the MATLAB `CAS` selector in `demo1.m`, so [`get`] is the direct
//! analogue of `mlf.examples(CAS)` — pick a case, build its points, fit, compare.
//!
//! This is a curated subset spanning the tag space (polynomial / rational /
//! irrational) and `n = 2..10`, including the named classics (ReLU, the Austin
//! et al. 2021 set, the high-dimensional rationals that showcase the recursion
//! taming the curse of dimensionality). Add cases here as needed; the `id`s are
//! kept aligned with the reference for cross-checking.

use core::f64;

use crate::points::Points;

pub mod complex {
    //! Complex-input/output examples from `mLF/+mlf/examples.m`.
    //!
    //! Cases 11–20 and 30 are the ENTIRE commented `name_domain = 'Complex'`
    //! block. Their IDs, formulas, per-axis orders and `lam` ranges are preserved.
    //! Case 59 is the active mixed imaginary/real-domain example. The legacy block
    //! specifies ranges but no particular complex contour; use explicit segments or
    //! complex query points when exercising complex arithmetic.
    //!
    //! These high-dimensional functions are callable without materializing a grid.
    //! A full Cartesian fit of the 30-variable case is generally too large; 1-D
    //! sections provide practical regression tests of every variable's dependence.

    use super::Tag;
    use crate::{Complex64 as C, Points, Result};

    /// A MATLAB complex catalog entry. IDs are local to this catalog.
    #[derive(Debug, Clone)]
    pub struct ComplexExample {
        pub id: u32,
        pub ndim: usize,
        pub name: &'static str,
        pub tags: &'static [Tag],
        /// Declared per-axis orders in the commented MATLAB block.
        pub degrees: Option<&'static [usize]>,
        /// Original `lam` range, used on every axis unless `bounds` is supplied.
        pub range: (f64, f64),
        /// Explicit complex segment bounds (the active case 59).
        pub bounds: &'static [(C, C)],
        /// Optional MATLAB `Nip`; the legacy block instead specifies degrees.
        pub recommended_nip: Option<usize>,
        pub f: fn(&[C]) -> C,
    }

    impl ComplexExample {
        pub fn axis_bounds(&self) -> Vec<(C, C)> {
            if self.bounds.is_empty() {
                vec![(C::new(self.range.0, 0.0), C::new(self.range.1, 0.0)); self.ndim]
            } else {
                self.bounds.to_vec()
            }
        }

        /// Build point ARRAYS only, using MATLAB's reference placement. Sampling
        /// their Cartesian product is a separate operation and can be very large.
        /// Without an override, legacy cases use degree+1 nodes per side per axis.
        pub fn points(&self, nip: Option<usize>) -> Result<Points<C>> {
            let counts = if let Some(n) = nip.or(self.recommended_nip) {
                vec![n; self.ndim]
            } else {
                self.degrees.unwrap().iter().map(|d| d + 1).collect()
            };
            Points::reference_segments(&self.axis_bounds(), &counts)
        }
    }

    pub fn get(id: u32) -> Option<ComplexExample> {
        all().iter().find(|e| e.id == id).cloned()
    }
    pub fn all() -> &'static [ComplexExample] {
        CATALOG
    }

    const RATIONAL: &[Tag] = &[Tag::Rational];
    const LEGACY: &[usize] = &[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30];

    /// IDs of the complete commented block, excluding active catalog additions.
    pub fn legacy_ids() -> &'static [usize] {
        LEGACY
    }

    const CATALOG: &[ComplexExample] = &[
        ComplexExample {
            id: 11,
            ndim: 11,
            name: "Legacy complex rational, n=11",
            tags: RATIONAL,
            degrees: Some(&[6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            range: (1.0, 5.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                s[0] / (s[0].powu(6)
                    + s[1] * s[2]
                    + s[3]
                    + s[4]
                    + s[5]
                    + s[6] * s[7]
                    + s[8]
                    + s[9]
                    + s[10])
            },
        },
        ComplexExample {
            id: 12,
            ndim: 12,
            name: "Legacy complex rational, n=12",
            tags: RATIONAL,
            degrees: Some(&[6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            range: (1.0, 5.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                1.0 / (s[0].powu(6)
                    + s[0].powu(2) * s[1] * s[2]
                    + s[3]
                    + s[4]
                    + s[5]
                    + s[6] * s[7]
                    + s[8]
                    + s[9]
                    + s[10]
                    + s[11])
            },
        },
        ComplexExample {
            id: 13,
            ndim: 13,
            name: "Legacy complex rational, n=13",
            tags: RATIONAL,
            degrees: Some(&[7, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2]),
            range: (0.1, 10.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0] + 4.0 * s[7] + s[11])
                    / (s[0].powu(7)
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4].powu(3)
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12].powu(2))
            },
        },
        ComplexExample {
            id: 14,
            ndim: 14,
            name: "Legacy complex rational, n=14",
            tags: RATIONAL,
            degrees: Some(&[7, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3]),
            range: (0.1, 10.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0] + 4.0 * s[7] + s[11] + 3.0 * s[13].powu(3))
                    / (s[0].powu(7)
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4].powu(3)
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12].powu(2)
                        + 2.0)
            },
        },
        ComplexExample {
            id: 15,
            ndim: 15,
            name: "Legacy complex rational, n=15",
            tags: RATIONAL,
            degrees: Some(&[8, 2, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 5]),
            range: (0.1, 10.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0] + 4.0 * s[7] + s[11] + 3.0 * s[13].powu(3))
                    / (s[0].powu(8)
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4].powu(3)
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12].powu(2)
                        + 2.0
                        + s[14].powu(5))
            },
        },
        ComplexExample {
            id: 16,
            ndim: 16,
            name: "Legacy complex rational, n=16",
            tags: RATIONAL,
            degrees: Some(&[8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]),
            range: (0.1, 2.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0].powu(8) + 4.0 * s[7] + s[11] + s[12] * s[13] + s[14])
                    / (s[0]
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4]
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12]
                        + s[15].powu(3) * std::f64::consts::PI)
            },
        },
        ComplexExample {
            id: 17,
            ndim: 17,
            name: "Legacy complex rational, n=17",
            tags: RATIONAL,
            degrees: Some(&[9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2]),
            range: (0.1, 2.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0].powu(9) + 4.0 * s[7] + s[11] + s[12] * s[13] + s[14])
                    / (s[0]
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4]
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12]
                        + s[15].powu(3) * std::f64::consts::PI
                        + s[16].powu(2))
            },
        },
        ComplexExample {
            id: 18,
            ndim: 18,
            name: "Legacy complex rational, n=18",
            tags: RATIONAL,
            degrees: Some(&[9, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2]),
            range: (0.1, 2.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0].powu(9) + 4.0 * s[7] + s[11] + s[12] * s[13] + s[14] + s[17].powu(2))
                    / (s[0]
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4]
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12]
                        + s[15].powu(3) * std::f64::consts::PI
                        + s[16].powu(2))
            },
        },
        ComplexExample {
            id: 19,
            ndim: 19,
            name: "Legacy complex rational, n=19",
            tags: RATIONAL,
            degrees: Some(&[10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 1]),
            range: (0.1, 2.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0].powu(10) + 4.0 * s[7] + s[11] + s[12] * s[13] + s[14] + s[17].powu(2))
                    / (s[0]
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4]
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12]
                        + s[15].powu(3) * std::f64::consts::PI
                        + s[16].powu(2)
                        + s[18])
            },
        },
        ComplexExample {
            id: 20,
            ndim: 20,
            name: "Legacy complex rational, n=20",
            tags: RATIONAL,
            degrees: Some(&[10, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1]),
            range: (0.1, 2.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0].powu(3) + 4.0 * s[7] + s[11] + s[12] * s[13] + s[14])
                    / (s[0].powu(10)
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4]
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12]
                        + s[15].powu(3) * std::f64::consts::PI
                        + s[16]
                        + s[17] * s[18]
                        - s[19])
            },
        },
        ComplexExample {
            id: 30,
            ndim: 30,
            name: "Legacy complex rational, n=30",
            tags: RATIONAL,
            degrees: Some(&[
                3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1,
                1, 3,
            ]),
            range: (0.1, 5.0),
            bounds: &[],
            recommended_nip: None,
            f: |s| {
                (3.0 * s[0].powu(3) + 4.0 * s[7] + s[11] + s[12] * s[13] + s[14] + s[20].powu(4)
                    - s[29] * s[1])
                    / (s[0]
                        + s[1].powu(2) * s[2]
                        + s[3]
                        + s[4]
                        + s[5]
                        + s[6] * s[7]
                        + s[8] * s[9] * s[10]
                        + s[12]
                        + s[15].powu(3) * std::f64::consts::PI
                        + s[16]
                        + s[17] * s[18]
                        - s[19]
                        + s[21] * s[22]
                        - s[23] / 2.0
                        + s[24] * s[25]
                        - std::f64::consts::SQRT_2 * s[26]
                        - s[27]
                        + s[28]
                        + s[29].powu(3))
            },
        },
        ComplexExample {
            id: 59,
            ndim: 2,
            name: "Marx sqrt/sinh on a mixed complex/real domain",
            tags: &[Tag::Irrational, Tag::CInf],
            degrees: None,
            range: (0.0, 0.0),
            bounds: &[
                (C::new(0.0, 0.001), C::new(0.0, 100.0)),
                (C::new(0.001, 0.0), C::new(20.0, 0.0)),
            ],
            recommended_nip: Some(30),
            f: |s| (s[1] * s[1] + 1.0) / (s[1] + s[0].sqrt() * s[0].sqrt().sinh()),
        },
    ];
}

/// A function tag, mirroring the MATLAB `tag` cell array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    Polynomial,
    Rational,
    Irrational,
    /// Continuous but not smooth (`$C^0$`).
    C0,
    /// Smooth (`$C^\infty$`).
    CInf,
}

/// One catalog entry: a target function plus its metadata and domain.
#[derive(Debug, Clone)]
pub struct Example {
    /// Matches the MATLAB `CAS` selector.
    pub id: u32,
    /// Number of input variables `n`.
    pub ndim: usize,
    /// Human-readable name.
    pub name: &'static str,
    /// Paper / source reference.
    pub reference: &'static str,
    /// Classification tags.
    pub tags: &'static [Tag],
    /// Per-axis domain `[lo, hi]` (length `ndim`), or empty for the default
    /// `[-1, 1]` on every axis. Use [`Self::axis_bounds`] to get the filled set.
    pub bounds: &'static [(f64, f64)],
    /// Recommended number of interpolation points per axis (MATLAB `Nip`).
    pub recommended_nip: usize,
    /// The target function: `n` inputs, single output.
    pub f: fn(&[f64]) -> f64,
}

impl Example {
    /// Per-axis domain, substituting the conventional `[-1, 1]` for any axis the
    /// catalog left unspecified (the MATLAB `xbnd = [-1 1]` default).
    pub fn axis_bounds(&self) -> Vec<(f64, f64)> {
        if self.bounds.is_empty() {
            vec![(-1.0, 1.0); self.ndim]
        } else {
            self.bounds.to_vec()
        }
    }

    /// Build the column/row support points using the **reference convention**
    /// from `examples.m`: along each axis, `p_c = linspace(lo, hi, nip)` and
    /// `p_r = p_c + dx` with `dx = |p_c[end] - p_c[end-1]| / 2` (a half-step
    /// shift, guaranteeing disjointness). Pass `nip = None` to use
    /// [`Self::recommended_nip`].
    ///
    /// This mirrors `mlf.examples` exactly, so the sampled tensor matches the
    /// MATLAB demo's `tab` and golden comparisons line up. For a quick
    /// well-conditioned alternative see [`Points::alternating`].
    pub fn points(&self, nip: Option<usize>) -> Points {
        let m = nip.unwrap_or(self.recommended_nip).max(2);
        let bounds = self.axis_bounds();
        let mut column = Vec::with_capacity(self.ndim);
        let mut row = Vec::with_capacity(self.ndim);
        for &(lo, hi) in &bounds {
            let pc: Vec<f64> = (0..m)
                .map(|i| lo + (hi - lo) * i as f64 / (m - 1) as f64)
                .collect();
            let dx = (pc[m - 1] - pc[m - 2]).abs() / 2.0;
            let pr: Vec<f64> = pc.iter().map(|&v| v + dx).collect();
            column.push(pc);
            row.push(pr);
        }
        Points::new(column, row).expect("reference points are disjoint by construction")
    }

    /// Build the support points with **Chebyshev** placement on the catalog
    /// bounds ([`Points::chebyshev`]) instead of the reference `linspace`
    /// convention. Better conditioned for rational/AAA fitting and keeps every
    /// point inside the domain. Pass `nip = None` to use [`Self::recommended_nip`].
    pub fn points_chebyshev(&self, nip: Option<usize>) -> Points {
        let m = nip.unwrap_or(self.recommended_nip).max(2);
        let nips = vec![m; self.ndim];
        Points::chebyshev(&self.axis_bounds(), &nips)
            .expect("chebyshev points are disjoint by construction")
    }
}

/// Look up an example by its MATLAB `CAS` id. Returns `None` for ids not in the
/// curated subset.
pub fn get(id: u32) -> Option<Example> {
    all().iter().find(|e| e.id == id).cloned()
}

/// All curated catalog entries.
pub fn all() -> &'static [Example] {
    CATALOG
}

// Helpers keeping the closures readable.
#[inline]
fn relu(x: f64) -> f64 {
    0.5 * (x + x.abs())
}

const CATALOG: &[Example] = &[
    Example {
        id: 0,
        ndim: 2,
        name: "x1 * x2^2",
        reference: "Personal communication",
        tags: &[Tag::Polynomial],
        bounds: &[],
        recommended_nip: 20,
        f: |x| x[0] * x[1] * x[1],
    },
    Example {
        id: 1,
        ndim: 2,
        name: "ReLU(x1) + x2/10",
        reference: "Personal communication",
        tags: &[Tag::Irrational, Tag::C0],
        bounds: &[(-1.0, 1.0), (-1.0, -1e-10)],
        recommended_nip: 20,
        f: |x| relu(x[0]) + 0.1 * x[1],
    },
    Example {
        id: 2,
        ndim: 2,
        name: "exp(sin(x1) + x2^2)",
        reference: "Liu et al. 2024",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| (x[0].sin() + x[1] * x[1]).exp(),
    },
    Example {
        id: 3,
        ndim: 2,
        name: "x1 * x2",
        reference: "Liu et al. 2024",
        tags: &[Tag::Polynomial, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| x[0] * x[1],
    },
    Example {
        id: 4,
        ndim: 3,
        name: "1/3 sum(sin(pi x_i / 2)^2)",
        reference: "Liu et al. 2024",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            let p = std::f64::consts::PI / 2.0;
            ((x[0] * p).sin().powi(2) + (x[1] * p).sin().powi(2) + (x[2] * p).sin().powi(2)) / 3.0
        },
    },
    Example {
        id: 5,
        ndim: 4,
        name: "exp(1/2 (sin(pi(x1^2+x2^2)) + sin(pi(x3^2+x4^2))))",
        reference: "Liu et al. 2024",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            let p = std::f64::consts::PI;
            (0.5 * ((p * (x[0] * x[0] + x[1] * x[1])).sin()
                + (p * (x[2] * x[2] + x[3] * x[3])).sin()))
            .exp()
        },
    },
    Example {
        id: 6,
        ndim: 2,
        name: "exp(x1 x2) / ((x1^2-1.44)(x2^2-1.44))",
        reference: "Austin et al. 2021 (A.5.1)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| (x[0] * x[1]).exp() / ((x[0] * x[0] - 1.44) * (x[1] * x[1] - 1.44)),
    },
    Example {
        id: 7,
        ndim: 2,
        name: "log2(2.25 - x1^2 - x2^2)",
        reference: "Austin et al. 2021 (A.5.2)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| (2.25 - x[0] * x[0] - x[1] * x[1]).log2(),
    },
    Example {
        id: 8,
        ndim: 2,
        name: "tanh(4(x1 - x2))",
        reference: "Austin et al. 2021 (A.5.3)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 37,
        f: |x| (4.0 * (x[0] - x[1])).tanh(),
    },
    Example {
        id: 9,
        ndim: 2,
        name: "exp(-(x1^2+x2^2)/1000)",
        reference: "Austin et al. 2021 (A.5.4)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 20,
        f: |x| (-(x[0] * x[0] + x[1] * x[1]) / 1000.0).exp(),
    },
    Example {
        id: 10,
        ndim: 2,
        name: "|x1 - x2|^3",
        reference: "Austin et al. 2021 (A.5.5)",
        tags: &[Tag::Irrational, Tag::C0],
        bounds: &[],
        recommended_nip: 41,
        f: |x| (x[0] - x[1]).abs().powi(3),
    },
    Example {
        id: 11,
        ndim: 2,
        name: "(x1 + x2^3) / (x1 x2^2 + 2)",
        reference: "Austin et al. 2021 (A.5.6)",
        tags: &[Tag::Rational],
        bounds: &[(1e-10, 1.0), (1e-10, 1.0)],
        recommended_nip: 20,
        f: |x| (x[0] + x[1] * x[1] * x[1]) / (x[0] * x[1] * x[1] + 2.0),
    },
    Example {
        id: 12,
        ndim: 2,
        name: "(x1^2+x2^2+x1-x2-1)/((x1-1.1)(x2-1.1))",
        reference: "Austin et al. 2021 (A.5.7)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| (x[0] * x[0] + x[1] * x[1] + x[0] - x[1] - 1.0) / ((x[0] - 1.1) * (x[1] - 1.1)),
    },
    Example {
        id: 13,
        ndim: 2,
        name: "(x1^4+x2^4+x1^2 x2^2+x1 x2)/((x1-1.1)(x2-1.1))",
        reference: "Austin et al. 2021 (A.5.8)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            (x[0].powi(4) + x[1].powi(4) + x[0].powi(2) * x[1].powi(2) + x[0] * x[1])
                / ((x[0] - 1.1) * (x[1] - 1.1))
        },
    },
    Example {
        id: 14,
        ndim: 4,
        name: "(x1^2+x2^2+x1-x2+1)/((x3-1.5)(x4-1.5))",
        reference: "Austin et al. 2021 (A.5.9)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 10,
        f: |x| (x[0] * x[0] + x[1] * x[1] + x[0] - x[1] + 1.0) / ((x[2] - 1.5) * (x[3] - 1.5)),
    },
    Example {
        id: 15,
        ndim: 2,
        name: "(x1^2+x2^2+x1-x2-1)/(x1^3+x2^3+4)",
        reference: "Austin et al. 2021 (A.5.10)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            (x[0] * x[0] + x[1] * x[1] + x[0] - x[1] - 1.0) / (x[0].powi(3) + x[1].powi(3) + 4.0)
        },
    },
    Example {
        id: 16,
        ndim: 2,
        name: "(x1^3+x2^3)/(x1^2+x2^2+3)",
        reference: "Austin et al. 2021 (A.5.11)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| (x[0].powi(3) + x[1].powi(3)) / (x[0] * x[0] + x[1] * x[1] + 3.0),
    },
    Example {
        id: 17,
        ndim: 2,
        name: "(x1^4+x2^4+x1^2 x2^2+x1 x2)/(x1^2 x2^2 - 2 x1^2 - 2 x2^2 + 4)",
        reference: "Austin et al. 2021 (A.5.12)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            let x1sq = x[0] * x[0];
            let x2sq = x[1] * x[1];
            (x1sq * x1sq + x2sq * x2sq + x1sq * x2sq + x[0] * x[1])
                / (x1sq * x2sq - 2.0 * x1sq - 2.0 * x2sq + 4.0)
        },
    },
    Example {
        id: 18,
        ndim: 2,
        name: "(x1^3+x2^3)/(x1^2 x2^2 - 2 x1^2 - 2 x2^2 + 4)",
        reference: "Austin et al. 2021 (A.5.13)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            let x1sq = x[0] * x[0];
            let x2sq = x[1] * x[1];
            (x[0].powi(3) + x[1].powi(3)) / (x1sq * x2sq - 2.0 * x1sq - 2.0 * x2sq + 4.0)
        },
    },
    Example {
        id: 19,
        ndim: 2,
        name: "(x1^4+x2^4+x1^2 x2^2+x1 x2)/(x1^3+x2^3+4)",
        reference: "Austin et al. 2021 (A.5.14)",
        tags: &[Tag::Rational],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            let x1sq = x[0] * x[0];
            let x2sq = x[1] * x[1];
            (x1sq * x1sq + x2sq * x2sq + x1sq * x2sq + x[0] * x[1])
                / (x[0].powi(3) + x[1].powi(3) + 4.0)
        },
    },
    Example {
        id: 20,
        ndim: 3,
        name: "Breit Wigner function",
        reference: "Austin et al. 2021 (A.5.15)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(80.0, 100.0), (5.0, 10.0), (90.0, 93.0)],
        recommended_nip: 10,
        f: |x| {
            let e = x[0];
            let g = x[1];
            let m = x[2];
            let m2 = m * m;
            let g2 = g * g;
            let mg_sqrt = (m2 * (m2 + g2)).sqrt();
            let num = 2.0 * std::f64::consts::SQRT_2 * m * g * mg_sqrt;
            let den =
                (std::f64::consts::PI * (m2 + mg_sqrt).sqrt()) * ((e * e - m2).powi(2) + m2 * g2);
            num / den
        },
    },
    Example {
        id: 21,
        ndim: 4,
        name: "sum_i atan(x_i) / (x1^2 x2^2 - x1^2 - x2^2 + 1)",
        reference: "Austin et al. 2021 (A.5.16)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(-0.95, 0.95); 4],
        recommended_nip: 10,
        f: |x| {
            (x[0].atan() + x[1].atan() + x[2].atan() + x[3].atan())
                / (x[0] * x[0] * x[1] * x[1] - x[0] * x[0] - x[1] * x[1] + 1.0)
        },
    },
    Example {
        id: 22,
        ndim: 4,
        name: "exp(x1 x2 x3 x4) / (x1^2+x2^2-x3 x4+3)",
        reference: "Austin et al. 2021 (A.5.17)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[],
        recommended_nip: 10,
        f: |x| (x[0] * x[1] * x[2] * x[3]).exp() / (x[0] * x[0] + x[1] * x[1] - x[2] * x[3] + 3.0),
    },
    Example {
        id: 23,
        ndim: 4,
        name: "10 prod_i sinc(x_i)",
        reference: "Austin et al. 2021 (A.5.18)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(1e-6, 4.0 * f64::consts::PI); 4],
        recommended_nip: 11,
        f: |x| {
            10.0 * (x[0].sin() * x[1].sin() * x[2].sin() * x[3].sin()) / (x[0] * x[1] * x[2] * x[3])
        },
    },
    Example {
        id: 24,
        ndim: 2,
        name: "10 sinc(x1) sinc(x2)",
        reference: "Austin et al. 2021 (A.5.19)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(1e-6, 4.0 * f64::consts::PI); 2],
        recommended_nip: 21,
        f: |x| 10.0 * (x[0].sin() * x[1].sin()) / (x[0] * x[1]),
    },
    Example {
        id: 25,
        ndim: 2,
        name: "x1^2 + x2^2 + x1 x2 - x2 + 1",
        reference: "Austin et al. 2021 (A.5.20)",
        tags: &[Tag::Polynomial],
        bounds: &[],
        recommended_nip: 20,
        f: |x| x[0] * x[0] + x[1] * x[1] + x[0] * x[1] - x[1] + 1.0,
    },
    Example {
        id: 26,
        ndim: 3,
        name: "(x1+x2+x3)/(6+cos(x1)+cos(x2)+cos(x3))",
        reference: "Balicki et al. 2025",
        tags: &[Tag::Irrational],
        bounds: &[(-10.0, 10.0); 3],
        recommended_nip: 30,
        f: |x| (x[0] + x[1] + x[2]) / (6.0 + x[0].cos() + x[1].cos() + x[2].cos()),
    },
    Example {
        id: 27,
        ndim: 5,
        name: "sum(x) / (10+sum(cos(x)))",
        reference: "Balicki et al. 2025",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(-4.0, 4.0); 5],
        recommended_nip: 13,
        f: |x| {
            (x[0] + x[1] + x[2] + x[3] + x[4])
                / (10.0 + x[0].cos() + x[1].cos() + x[2].cos() + x[3].cos() + x[4].cos())
        },
    },
    Example {
        id: 28,
        ndim: 2,
        name: "Toy function",
        reference: "J/al. 2024 (Toy function)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(1e-10, 10.0); 2],
        recommended_nip: 31,
        f: |x| {
            let x1 = x[0];
            let x2 = x[1];
            (x1 / (x1 + 1.0)).powi(4)
                * (1.0 + (-x2 * x2).exp())
                * (1.0 + x2 * x2.cos() * (-x1 * x2 / (x1 + 1.0)).exp())
        },
    },
    Example {
        id: 29,
        ndim: 2,
        name: "min(10|x1|,1)sign(x1) + x1 x2^3 / 10",
        reference: "Personal communication",
        tags: &[Tag::Irrational, Tag::C0],
        bounds: &[],
        recommended_nip: 20,
        f: |x| {
            let s1 = if x[0] == 0.0 { 0.0 } else { x[0].signum() };
            (10.0 * x[0].abs()).min(1.0) * s1 + (x[0] * x[1].powi(3)) / 10.0
        },
    },
    Example {
        id: 30,
        ndim: 8,
        name: "Borehole function",
        reference: "Surjanovic",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[
            (0.05, 0.15),
            (100.0, 50000.0),
            (63070.0, 115600.0),
            (990.0, 1110.0),
            (63.1, 116.0),
            (700.0, 820.0),
            (1120.0, 1680.0),
            (9855.0, 12045.0),
        ],
        recommended_nip: 4,
        f: |x| {
            let rw = x[0];
            let r = x[1];
            let tu = x[2];
            let hu = x[3];
            let tl = x[4];
            let hl = x[5];
            let l = x[6];
            let kw = x[7];

            let num = 2.0 * std::f64::consts::PI * tu * (hu - hl);
            let frac2a = 2.0 * l * tu / ((r / rw).ln() * rw * rw * kw);
            let frac2b = tu / tl;
            let den = (r / rw).ln() * (1.0 + frac2a + frac2b);
            num / den
        },
    },
    Example {
        id: 31,
        ndim: 6,
        name: "x1^2 x2^3 + x3 x4 - x5^2 + x6",
        reference: "Personal communication",
        tags: &[Tag::Polynomial],
        bounds: &[(-2.0, 2.0); 6],
        recommended_nip: 8,
        f: |x| x[0].powi(2) * x[1].powi(3) + x[2] * x[3] - x[4].powi(2) + x[5],
    },
    Example {
        id: 32,
        ndim: 2,
        name: "atan(x1) + x2^3",
        reference: "Personal communication",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(-2.0, 2.0); 2],
        recommended_nip: 20,
        f: |x| x[0].atan() + x[1].powi(3),
    },
    Example {
        id: 33,
        ndim: 2,
        name: "(x1+x2)/(cos(x1)^2+cos(x2)+3)",
        reference: "Personal communication",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(-10.0, 10.0); 2],
        recommended_nip: 30,
        f: |x| (x[0] + x[1]) / (x[0].cos().powi(2) + x[1].cos() + 3.0),
    },
    Example {
        id: 36,
        ndim: 3,
        name: "x2/(3+x2 x1/3-x3^2)",
        reference: "Personal communication",
        tags: &[Tag::Rational],
        bounds: &[(0.1, 1.0); 3],
        recommended_nip: 10,
        f: |x| x[1] / (3.0 + (x[1] * x[0]) / 3.0 - x[2] * x[2]),
    },
    Example {
        id: 37,
        ndim: 4,
        name: "x1 x4^3 + sin(2 x2) x3",
        reference: "Personal communication",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(1e-3, 1.0); 4],
        recommended_nip: 10,
        f: |x| x[0] * x[3].powi(3) + (2.0 * x[1]).sin() * x[2],
    },
    Example {
        id: 38,
        ndim: 3,
        name: "(x1^9 x2^7 + x1^3 + 5 x3^2) / (5 x1^4 + 4 x1^2 + x3 x2^3 + 1)",
        reference: "A.C. Antoulas (presentation)",
        tags: &[Tag::Rational],
        bounds: &[(-1.1, 1.1); 3],
        recommended_nip: 30,
        f: |x| {
            let num = x[0].powi(9) * x[1].powi(7) + x[0].powi(3) + 5.0 * x[2].powi(2);
            let den = 5.0 * x[0].powi(4) + 4.0 * x[0].powi(2) + x[2] * x[1].powi(3) + 1.0;
            num / den
        },
    },
    Example {
        id: 39,
        ndim: 3,
        name: "(x3 + x1^4) / (x1^3 + x2^2 + 1)",
        reference: "examples.m case 39",
        tags: &[Tag::Rational],
        bounds: &[(0.1, 10.0); 3],
        recommended_nip: 20,
        f: |x| (x[2] + x[0].powi(4)) / (x[0].powi(3) + x[1] * x[1] + 1.0),
    },
    Example {
        id: 40,
        ndim: 4,
        name: "x3 x1 / (x1^2+x2+x3^2+1) + x4^3",
        reference: "Personal communication",
        tags: &[Tag::Rational],
        bounds: &[(1.0, 4.0); 4],
        recommended_nip: 20,
        f: |x| x[2] * x[0] / (x[0] * x[0] + x[1] + x[2] * x[2] + 1.0) + x[3].powi(3),
    },
    Example {
        id: 41,
        ndim: 5,
        name: "(x5^3 x3 x1+x3^2)/(x1^3+x2 x3+x4)",
        reference: "Personal communication",
        tags: &[Tag::Rational],
        bounds: &[(0.1, 1.0); 5],
        recommended_nip: 5,
        f: |x| {
            let num = x[4].powi(3) * x[2] * x[0] + x[2].powi(2);
            let den = x[0].powi(3) + x[1] * x[2] + x[3];
            num / den
        },
    },
    Example {
        id: 42,
        ndim: 6,
        name: "(x1 + x3 - sqrt(2) x6^2) / (x1^4 + x2 x3 + x4^3 + x5^2 + x6)",
        reference: "examples.m case 42",
        tags: &[Tag::Rational],
        bounds: &[(0.1, 1.0); 6],
        recommended_nip: 5,
        f: |x| {
            (x[0] + x[2] - std::f64::consts::SQRT_2 * x[5] * x[5])
                / (x[0].powi(4) + x[1] * x[2] + x[3].powi(3) + x[4] * x[4] + x[5])
        },
    },
    Example {
        id: 43,
        ndim: 7,
        name: "(x3 x2^3+1)/(x1^4+x2^2 x3+x4^2+x5+x6^3+x7)",
        reference: "Personal communication",
        tags: &[Tag::Rational],
        bounds: &[(1.0, 10.0); 7],
        recommended_nip: 5,
        f: |x| {
            (x[2] * x[1].powi(3) + 1.0)
                / (x[0].powi(4) + x[1] * x[1] * x[2] + x[3] * x[3] + x[4] + x[5].powi(3) + x[6])
        },
    },
    Example {
        id: 44,
        ndim: 8,
        name: "1 / (x1^4 + x2^2 x3 + x4^2 + x5 + x6 + x7 + x8)",
        reference: "Personal communication (n=8)",
        tags: &[Tag::Rational],
        bounds: &[(0.1, 20.0); 8],
        recommended_nip: 5,
        f: |x| 1.0 / (x[0].powi(4) + x[1] * x[1] * x[2] + x[3] * x[3] + x[4] + x[5] + x[6] + x[7]),
    },
    Example {
        id: 45,
        ndim: 9,
        name: "1 / (x1^2+x2^2 x3+x4^2+x5+x6+x7+x8+x9)",
        reference: "Personal communication",
        tags: &[Tag::Rational],
        bounds: &[(1.0, 5.0); 9],
        recommended_nip: 3,
        f: |x| {
            1.0 / (x[0] * x[0]
                + x[1] * x[1] * x[2]
                + x[3] * x[3]
                + x[4]
                + x[5]
                + x[6]
                + x[7]
                + x[8])
        },
    },
    Example {
        id: 46,
        ndim: 10,
        name: "1 / (x1 + x1^2 x2 x3 + x4 + x5 + x6 + x7 x8 + x9^2 + x10)",
        reference: "Personal communication (n=10)",
        tags: &[Tag::Rational],
        bounds: &[(1.0, 5.0); 10],
        recommended_nip: 3,
        f: |x| {
            1.0 / (x[0]
                + x[0] * x[0] * x[1] * x[2]
                + x[3]
                + x[4]
                + x[5]
                + x[6] * x[7]
                + x[8] * x[8]
                + x[9])
        },
    },
    Example {
        id: 47,
        ndim: 5,
        name: "Polynomial Ex 3.1",
        reference: "GHK 2025 (Ex 3.1)",
        tags: &[Tag::Polynomial],
        bounds: &[(-2.0, 2.0); 5],
        recommended_nip: 6,
        f: |x| {
            12.0 * x[2] + 24.0 * x[0] * x[2] - 6.0 * x[1] * x[2]
                + 4.0 * x[2] * x[3]
                + 2.0 * x[1] * x[3]
                - 18.0 * x[2] * x[4]
                - 12.0 * x[0] * x[1] * x[2]
                + 8.0 * x[0] * x[2] * x[3]
                - 2.0 * x[0] * x[1] * x[3]
                - 36.0 * x[0] * x[2] * x[4]
                + 4.0 * x[1] * x[2] * x[3]
                + 9.0 * x[1] * x[2] * x[4]
                - 2.0 * x[1] * x[3] * x[4]
                - 6.0 * x[2] * x[3] * x[4]
                - 10.0 * x[0] * x[1] * x[2] * x[3]
                + 18.0 * x[0] * x[1] * x[2] * x[4]
                + 2.0 * x[0] * x[1] * x[3] * x[4]
                - 12.0 * x[0] * x[2] * x[3] * x[4]
                + 12.0 * x[0] * x[1] * x[2] * x[3] * x[4]
        },
    },
    Example {
        id: 48,
        ndim: 3,
        name: "x1 x2 + x1 x3 + x2 x3",
        reference: "Polya-Szego (getting started)",
        tags: &[Tag::Polynomial],
        bounds: &[(-0.5, 1.0); 3],
        recommended_nip: 6,
        f: |x| x[0] * x[1] + x[0] * x[2] + x[1] * x[2],
    },
    Example {
        id: 51,
        ndim: 2,
        name: "NLT Challenge #4",
        reference: "Trefethen (challenge document)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(-0.5, 0.5), (-0.5, 0.5)],
        recommended_nip: 1000,
        f: |x| {
            (50.0 * x[0]).sin().exp()
                + (60.0 * x[1].exp()).sin()
                + (70.0 * x[0].sin()).sin()
                + (80.0 * x[1]).sin().sin()
                - (10.0 * (x[0] + x[1])).sin()
                + 0.25 * (x[0] * x[0] + x[1] * x[1])
        },
    },
    Example {
        id: 52,
        ndim: 2,
        name: "x2 + sin(x1)^2",
        reference: "P. Vuillemin (discussion)",
        tags: &[Tag::Irrational, Tag::CInf],
        bounds: &[(-3.0, 3.0), (-4.0, 4.0)],
        recommended_nip: 20,
        f: |x| x[1] + x[0].sin().powi(2),
    },
    Example {
        id: 53,
        ndim: 2,
        name: "(2-x1)(1-2x2)+x1(5+x2^2)",
        reference: "L. Balada Gaggioli (discussion)",
        tags: &[Tag::Polynomial],
        bounds: &[(-1.0, 1.0); 2],
        recommended_nip: 10,
        f: |x| (2.0 - x[0]) * (1.0 - 2.0 * x[1]) + x[0] * (5.0 + x[1] * x[1]),
    },
    Example {
        id: 54,
        ndim: 2,
        name: "x1^4 x2^2 + x1^2 x2^4 - 3 x1^2 x2^2 + 1",
        reference: "V. Magron (Motzkin problem)",
        tags: &[Tag::Polynomial],
        bounds: &[(-1.1, 1.1); 2],
        recommended_nip: 20,
        f: |x| {
            x[0].powi(4) * x[1].powi(2) + x[0].powi(2) * x[1].powi(4)
                - 3.0 * x[0].powi(2) * x[1].powi(2)
                + 1.0
        },
    },
    Example {
        id: 55,
        ndim: 2,
        name: "(x1^2+2x2^2)/(x1^2+x2^2)",
        reference: "D. Henrion (discussion)",
        tags: &[Tag::Rational],
        bounds: &[(-1.0, 1.0); 2],
        recommended_nip: 20,
        f: |x| (x[0] * x[0] + 2.0 * x[1] * x[1]) / (x[0] * x[0] + x[1] * x[1]),
    },
    Example {
        id: 56,
        ndim: 2,
        name: "x1^2/(x1^2+x2^2)",
        reference: "D. Henrion (discussion)",
        tags: &[Tag::Rational],
        bounds: &[(-1.0, 1.0); 2],
        recommended_nip: 20,
        f: |x| x[0] * x[0] / (x[0] * x[0] + x[1] * x[1]),
    },
    Example {
        id: 57,
        ndim: 2,
        name: "x1 x2^2",
        reference: "A.C. Antoulas (discussion)",
        tags: &[Tag::Rational],
        bounds: &[(-1.1, 1.1); 2],
        recommended_nip: 10,
        f: |x| x[0] * x[1] * x[1],
    },
    Example {
        id: 58,
        ndim: 4,
        name: "(x1 x2+x3-x4+10)/(x1+2x2+x3+3x4+20)",
        reference: "A.C. Antoulas (\"brouillon\")",
        tags: &[Tag::Rational],
        bounds: &[(-2.0, 2.0); 4],
        recommended_nip: 2,
        f: |x| (x[0] * x[1] + x[2] - x[3] + 10.0) / (x[0] + 2.0 * x[1] + x[2] + 3.0 * x[3] + 20.0),
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Alg1Options, alg1};
    use approx::assert_relative_eq;

    #[test]
    fn catalog_bounds_match_ndim() {
        // Empty `bounds` means the default [-1, 1]^n; otherwise lengths agree.
        for e in all() {
            assert!(
                e.bounds.is_empty() || e.bounds.len() == e.ndim,
                "case {} bounds/ndim mismatch",
                e.id
            );
        }
    }

    #[test]
    fn default_bounds_filled_for_points() {
        // A case with no explicit bounds gets the reference default xbnd = [-1 1]
        // (examples.m header). Case 0 has empty bounds; case 48 (Pólya–Szegő)
        // carries its own xbnd = [-1/2, 1] per examples.m, so it is NOT a witness
        // for the fallback.
        let e = get(0).unwrap();
        assert!(e.bounds.is_empty(), "case 0 should have empty bounds");
        let pts = e.points(Some(8));
        assert_eq!(pts.ndim(), 2);
        assert_eq!(e.axis_bounds(), vec![(-1.0, 1.0); 2]);
    }

    #[test]
    fn alg1_recovers_separable_rational_case() {
        // A genuinely order-1 separable rational drives the recursion to ~machine
        // precision, exercising the full catalog -> points -> alg1 path. (Higher
        // catalog cases need more points; see the Octave golden tests for those.)
        let e = Example {
            id: 9001,
            ndim: 3,
            name: "test separable rational",
            reference: "test",
            tags: &[Tag::Rational],
            bounds: &[],
            recommended_nip: 8,
            f: |x| 1.0 / ((x[0] + 3.0) * (x[1] + 2.0) * (x[2] + 4.0)),
        };
        let pts = e.points(None);
        let (m1, info) = alg1(&e.f, &pts, &Alg1Options::default()).unwrap();
        assert_eq!(info.order, vec![1, 1, 1]);
        let x = [0.2, -0.3, 0.5];
        assert_relative_eq!(m1.eval(&x), (e.f)(&x), epsilon = 1e-9);
    }
}
