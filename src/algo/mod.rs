//! High-level algorithms that orchestrate the full pipeline.
//!
//! * [`alg1()`] — the direct, one-shot construction (`mlf.alg1`).
//! * [`alg2()`] — the greedy/adaptive (iterative) construction (`mlf.alg2`).
//! * [`alg3()`] — AAA-style resampling refinement (grows the grid each round).

pub mod alg1;
pub mod alg2;
pub mod alg3;
mod coefficient;
pub mod scan;

pub use alg1::{Alg1Info, Alg1Options, alg1, alg1_from_tensor};
pub use alg2::{Alg2Info, Alg2Options, Candidates, alg2, alg2_from_tensor};
pub use alg3::{Alg3Info, Alg3Options, alg3};
pub use coefficient::CoefficientMethod;
