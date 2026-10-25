//! Self-test: does the high-D rank-stability detector fire on a genuinely separable driver?
//! g(n) = -2^n reshaped to a matrix is rank-1 (geometric ⇒ outer product), stable across sizes.
use rmlf::exact::{ExactTt, Fp};
const P: u64 = 2_147_483_647;

fn reshape_rank(fp: &[i128], w: usize, h: usize) -> usize {
    let data: Vec<Fp> = fp[..w * h].iter().map(|&x| Fp::new(x, P)).collect();
    *ExactTt::from_dense(&data, &[w, h])
        .bond_ranks()
        .first()
        .unwrap_or(&1)
}
fn nd_max(fp: &[i128], dims: &[usize]) -> usize {
    let need: usize = dims.iter().product();
    let data: Vec<Fp> = fp[..need].iter().map(|&x| Fp::new(x, P)).collect();
    ExactTt::from_dense(&data, dims)
        .bond_ranks()
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
}
fn main() {
    // geometric driver -2^n mod P, 32 terms
    let mut pw = 1i128;
    let g: Vec<i128> = (1..=32)
        .map(|_| {
            pw = (pw * 2) % P as i128;
            (-pw).rem_euclid(P as i128)
        })
        .collect();
    println!("geometric g=-2^n:");
    println!("  4x4 (first16) rank = {}", reshape_rank(&g[..16], 4, 4));
    println!("  4x8 (first32) rank = {}", reshape_rank(&g[..32], 4, 8));
    println!(
        "  2x2x4(first16) maxrank = {}",
        nd_max(&g[..16], &[2, 2, 4])
    );
    println!(
        "  2x2x8(first32) maxrank = {}",
        nd_max(&g[..32], &[2, 2, 8])
    );

    // linear driver -n (matrix [i*h+j] = i*h + j ⇒ rank 2)
    let l: Vec<i128> = (1..=32)
        .map(|n| (-(n as i128)).rem_euclid(P as i128))
        .collect();
    println!("linear g=-n:");
    println!(
        "  4x4 rank = {}, 4x8 rank = {}",
        reshape_rank(&l[..16], 4, 4),
        reshape_rank(&l[..32], 4, 8)
    );
}
