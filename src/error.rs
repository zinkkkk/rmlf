//! Error types for the multivariate Loewner framework.

use thiserror::Error;

/// Errors that can arise while building or evaluating a Loewner approximation.
#[derive(Debug, Error)]
pub enum LoewnerError {
    /// Column and row interpolation point sets share a value along some axis.
    ///
    /// The Loewner construction requires the two sets to be disjoint per axis
    /// (otherwise a Loewner entry divides by zero).
    #[error(
        "interpolation points along axis {axis} are not disjoint (value {value} appears in both column and row sets)"
    )]
    NonDisjoint { axis: usize, value: f64 },

    /// Coincident column/row nodes on a complex axis.
    #[error(
        "interpolation points along axis {axis} are not disjoint (value {value} appears in both column and row sets)"
    )]
    ComplexNonDisjoint {
        axis: usize,
        value: num_complex::Complex64,
    },

    /// A dimension mismatch between the supplied data and the point sets.
    #[error("dimension mismatch: {what} (expected {expected}, got {got})")]
    DimMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// Fewer than the required number of points were supplied along an axis.
    #[error("not enough points along axis {axis}: need at least {need}, got {got}")]
    TooFewPoints {
        axis: usize,
        need: usize,
        got: usize,
    },

    /// The SVD used for a null-space computation failed to converge.
    #[error("SVD failed during null-space computation")]
    SvdFailed,

    /// A degenerate Loewner block produced no usable null vector.
    #[error("empty or rank-deficient Loewner block: {0}")]
    Degenerate(String),

    /// The target produced a non-finite sample (NaN or ±Inf) and the active
    /// handling policy is `Raw` (the default), which fabricates nothing.
    #[error(
        "target produced a non-finite sample ({count} of {total}); set DecoupleOptions.non_finite to NonFinite::Excise or NonFinite::Pole to handle it"
    )]
    NonFinite { count: usize, total: usize },
}

/// Convenience result alias.
pub type Result<T> = core::result::Result<T, LoewnerError>;
