//! Preimage sampling on integer lattices with a known basis.
//!
//! Given a basis `B` (rows are lattice generators), this module
//! produces integer coefficients `z` such that the lattice point
//! `Σ z_i · b_i ∈ Λ` is close to a target vector `t ∈ R^n`.
//!
//! - [`PreimageContext::nearest_plane`] is Babai's deterministic
//!   rounding; the output lattice point lies within the fundamental
//!   parallelepiped of the Gram-Schmidt basis around `t`.
//! - [`PreimageContext::gaussian_preimage`] is Klein's randomized
//!   algorithm; the output lattice point is distributed statistically
//!   close to the discrete Gaussian `D_{Λ, σ, t}` whenever `σ`
//!   exceeds the maximum Gram-Schmidt norm by a `ω(√log n)` factor.
//!
//! All arithmetic inside the orthogonalization and the nearest-plane
//! backward pass uses `f64`.  This is correct for small bases but
//! suffers precision loss on long bases; Nguyen-Regev (2006) broke
//! `NTRUSign` by exploiting exactly this effect.  This crate targets
//! correctness, not production side-channel resistance.

use crate::error::Error;
use crate::lattice::gaussian::discrete_gaussian;
use crate::lattice::signed::{ZMatrix, ZVec};

/// Minimum squared Gram-Schmidt norm accepted at construction.  A row
/// below this threshold implies the basis is singular or numerically
/// rank-deficient, and [`PreimageContext::new`] rejects it.
const GS_MIN_NORM_SQ: f64 = 1e-10;

/// Precomputed state for preimage sampling on a fixed integer lattice.
///
/// Caches the Gram-Schmidt orthogonalization of the basis rows along
/// with `1/||b*_i||²` and `1/||b*_i||` so that nearest-plane and
/// Klein sampling traverse the backward pass using only multiplies,
/// without recomputing the per-row `sqrt` or division on each query.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lattice::{PreimageContext, ZMatrix};
///
/// let basis = ZMatrix::new(2, 2, vec![1, 0, 0, 1]).ok();
/// let ctx = basis.and_then(|b| PreimageContext::new(b).ok());
/// assert!(ctx.is_some());
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct PreimageContext {
    basis: ZMatrix,
    gs: Vec<Vec<f64>>,
    gs_inv_norm: Vec<f64>,
    gs_inv_norm_sq: Vec<f64>,
}

impl PreimageContext {
    /// Build a preimage context by Gram-Schmidt orthogonalizing the
    /// rows of `basis` over f64.
    ///
    /// # Errors
    ///
    /// - [`Error::RandomGenerationFailed`] if any Gram-Schmidt row has
    ///   squared norm below `1e-10`, indicating a singular or
    ///   numerically rank-deficient basis.
    pub fn new(basis: ZMatrix) -> Result<Self, Error> {
        let rows_f: Vec<Vec<f64>> = (0..basis.rows())
            .map(|i| {
                basis
                    .row(i)
                    .unwrap_or(&[])
                    .iter()
                    .map(|&x| {
                        // i64 -> f64 loses precision beyond 2^53; bases used
                        // here are small-entry integer matrices.
                        #[allow(clippy::cast_precision_loss)]
                        let v = x as f64;
                        v
                    })
                    .collect()
            })
            .collect();
        let (gs, gs_norms_sq) = gram_schmidt(&rows_f);
        gs_norms_sq
            .iter()
            .enumerate()
            .find_map(|(i, &ns)| (ns < GS_MIN_NORM_SQ).then_some((i, ns)))
            .map_or_else(
                || {
                    let gs_inv_norm_sq: Vec<f64> =
                        gs_norms_sq.iter().map(|&n| 1.0 / n).collect();
                    let gs_inv_norm: Vec<f64> =
                        gs_inv_norm_sq.iter().map(|&inv| inv.sqrt()).collect();
                    Ok(Self {
                        basis,
                        gs,
                        gs_inv_norm,
                        gs_inv_norm_sq,
                    })
                },
                |(i, ns)| {
                    Err(Error::RandomGenerationFailed(format!(
                        "basis is singular: Gram-Schmidt row {i} has squared norm {ns}",
                    )))
                },
            )
    }

    /// Borrow of the underlying basis.
    pub fn basis(&self) -> &ZMatrix {
        &self.basis
    }

    /// Ambient dimension (number of columns of the basis).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.basis.cols()
    }

    /// Number of lattice-generating rows.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.basis.rows()
    }

    /// Babai's nearest-plane algorithm.  Returns integer coefficients
    /// `z = (z_0, ..., z_{m-1})` such that `Σ z_i · b_i` is close to
    /// `target`.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `target.len() != self.dim()`.
    pub fn nearest_plane(&self, target: &[f64]) -> Result<ZVec, Error> {
        if target.len() == self.dim() {
            Ok(babai_nearest_plane(
                target,
                &self.basis,
                &self.gs,
                &self.gs_inv_norm_sq,
            ))
        } else {
            Err(Error::DimensionMismatch {
                expected: self.dim(),
                actual: target.len(),
            })
        }
    }

    /// Klein's algorithm for Gaussian preimage sampling.  Returns
    /// integer coefficients `z` such that the lattice point
    /// `Σ z_i · b_i` is statistically close to the discrete Gaussian
    /// `D_{Λ, σ, target}`.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `target.len() != self.dim()`.
    /// - [`Error::RandomGenerationFailed`] propagated from the
    ///   underlying discrete Gaussian sampler.
    pub fn gaussian_preimage<F>(
        &self,
        target: &[f64],
        sigma: f64,
        rng: &F,
    ) -> Result<ZVec, Error>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error>,
    {
        if target.len() == self.dim() {
            klein_sample(
                target,
                sigma,
                &self.basis,
                &self.gs,
                &self.gs_inv_norm,
                &self.gs_inv_norm_sq,
                rng,
            )
        } else {
            Err(Error::DimensionMismatch {
                expected: self.dim(),
                actual: target.len(),
            })
        }
    }
}

/// Gram-Schmidt over f64.  For each row `b_i`, returns its orthogonal
/// component `b*_i` against all previous rows and the squared norm
/// `||b*_i||²`.
fn gram_schmidt(rows: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
    rows.iter().fold(
        (Vec::<Vec<f64>>::new(), Vec::<f64>::new()),
        |(gs, norms), bi| {
            let orth = orthogonalize(bi, &gs, &norms);
            let ns: f64 = orth.iter().map(|&x| x * x).sum();
            let next_gs: Vec<Vec<f64>> = gs.into_iter().chain(std::iter::once(orth)).collect();
            let next_norms: Vec<f64> = norms.into_iter().chain(std::iter::once(ns)).collect();
            (next_gs, next_norms)
        },
    )
}

/// Subtract the projection of `bi` onto each previous Gram-Schmidt row.
fn orthogonalize(bi: &[f64], prev_gs: &[Vec<f64>], prev_norms: &[f64]) -> Vec<f64> {
    prev_gs.iter().zip(prev_norms.iter()).fold(
        bi.to_vec(),
        |acc, (gs_j, &nj)| {
            let ip: f64 = acc.iter().zip(gs_j.iter()).map(|(a, b)| a * b).sum();
            let mu = ip / nj;
            acc.iter().zip(gs_j.iter()).map(|(a, b)| a - mu * b).collect()
        },
    )
}

/// Babai's nearest-plane: iterate `i` from `m-1` down to `0`, choosing
/// `z_i = round(<t_i, b*_i> · (1/||b*_i||²))` and updating
/// `t_{i-1} = t_i - z_i b_i`.  The inverse squared norms come
/// precomputed from [`PreimageContext`] so the backward pass is
/// division-free.
fn babai_nearest_plane(
    target: &[f64],
    basis: &ZMatrix,
    gs: &[Vec<f64>],
    gs_inv_norm_sq: &[f64],
) -> ZVec {
    let (_final_t, zs_rev) = (0..basis.rows()).rev().fold(
        (target.to_vec(), Vec::<i64>::new()),
        |(t, zs), i| {
            let gs_row = gs.get(i).map_or(&[][..], Vec::as_slice);
            let inv_norm_sq = gs_inv_norm_sq.get(i).copied().unwrap_or(0.0);
            let ip: f64 = t.iter().zip(gs_row.iter()).map(|(a, b)| a * b).sum();
            let mu = ip * inv_norm_sq;
            // f64 -> i64 saturating cast; mu is bounded by the lattice
            // coefficient scale (far below i64::MAX in well-posed bases).
            #[allow(clippy::cast_possible_truncation)]
            let z = mu.round() as i64;
            let new_t = subtract_scaled_row(&t, basis, i, z);
            let new_zs: Vec<i64> = zs.into_iter().chain(std::iter::once(z)).collect();
            (new_t, new_zs)
        },
    );
    let zs: Vec<i64> = zs_rev.into_iter().rev().collect();
    ZVec::new(zs)
}

/// Klein's algorithm: same backward pass as Babai, but `z_i` is drawn
/// from a discrete Gaussian on `Z` centered at the projection
/// coefficient with width `σ · (1/||b*_i||)`.  The precomputed inverse
/// norms turn the per-step `sigma / ||b*_i||` and `ip / ||b*_i||²` into
/// bare multiplies.
fn klein_sample<F>(
    target: &[f64],
    sigma: f64,
    basis: &ZMatrix,
    gs: &[Vec<f64>],
    gs_inv_norm: &[f64],
    gs_inv_norm_sq: &[f64],
    rng: &F,
) -> Result<ZVec, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    (0..basis.rows())
        .rev()
        .try_fold(
            (target.to_vec(), Vec::<i64>::new()),
            |(t, zs), i| {
                let gs_row = gs.get(i).map_or(&[][..], Vec::as_slice);
                let inv_norm_sq = gs_inv_norm_sq.get(i).copied().unwrap_or(0.0);
                let inv_norm = gs_inv_norm.get(i).copied().unwrap_or(0.0);
                let ip: f64 = t.iter().zip(gs_row.iter()).map(|(a, b)| a * b).sum();
                let mu = ip * inv_norm_sq;
                let sigma_i = sigma * inv_norm;
                discrete_gaussian(sigma_i, mu, rng).map(|z| {
                    let new_t = subtract_scaled_row(&t, basis, i, z);
                    let new_zs: Vec<i64> =
                        zs.into_iter().chain(std::iter::once(z)).collect();
                    (new_t, new_zs)
                })
            },
        )
        .map(|(_t, zs_rev)| {
            let zs: Vec<i64> = zs_rev.into_iter().rev().collect();
            ZVec::new(zs)
        })
}

/// Compute `t - z · basis[i]` element-wise in f64.
fn subtract_scaled_row(t: &[f64], basis: &ZMatrix, i: usize, z: i64) -> Vec<f64> {
    let row = basis.row(i).unwrap_or(&[]);
    // i64 -> f64 loses precision beyond 2^53; lattice coefficients stay
    // well inside that bound for bases sized for signature use.
    #[allow(clippy::cast_precision_loss)]
    let zf = z as f64;
    t.iter()
        .zip(row.iter())
        .map(|(&a, &b)| {
            #[allow(clippy::cast_precision_loss)]
            let bf = b as f64;
            a - zf * bf
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counter_rng() -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(0));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n)
                .map(|j| ((i.wrapping_mul(0x9E37_79B9) + j) & 0xFF) as u8)
                .collect())
        }
    }

    fn identity_basis(n: usize) -> ZMatrix {
        let data: Vec<i64> = (0..n * n)
            .map(|k| i64::from(u8::from(k / n == k % n)))
            .collect();
        ZMatrix::new(n, n, data)
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0))
    }

    #[test]
    fn construct_succeeds_for_identity() {
        assert!(PreimageContext::new(identity_basis(3)).is_ok());
    }

    #[test]
    fn construct_fails_for_singular_basis() {
        // Rows [1, 2] and [2, 4] are linearly dependent.
        let b = ZMatrix::new(2, 2, vec![1, 2, 2, 4])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        assert!(PreimageContext::new(b).is_err());
    }

    #[test]
    fn construct_fails_for_duplicate_rows() {
        let b = ZMatrix::new(2, 3, vec![1, 2, 3, 1, 2, 3])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        assert!(PreimageContext::new(b).is_err());
    }

    #[test]
    fn gram_schmidt_of_identity_is_identity() {
        let ctx = PreimageContext::new(identity_basis(2)).ok();
        let gs = ctx.as_ref().map(|c| c.gs.clone());
        let inv_norm_sq = ctx.as_ref().map(|c| c.gs_inv_norm_sq.clone());
        let inv_norm = ctx.as_ref().map(|c| c.gs_inv_norm.clone());
        assert_eq!(gs, Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]));
        assert_eq!(inv_norm_sq, Some(vec![1.0, 1.0]));
        assert_eq!(inv_norm, Some(vec![1.0, 1.0]));
    }

    #[test]
    fn gram_schmidt_orthogonalizes_leaning_basis() {
        // b_0 = [2, 1], b_1 = [1, 2].  μ_{1,0} = 4/5 = 0.8.
        // b*_1 = [1 - 1.6, 2 - 0.8] = [-0.6, 1.2], ||·||² = 0.36 + 1.44 = 1.8.
        let b = ZMatrix::new(2, 2, vec![2, 1, 1, 2])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        let ctx = PreimageContext::new(b).ok();
        let gs = ctx.as_ref().map(|c| c.gs.clone());
        let expected = [vec![2.0, 1.0], vec![-0.6, 1.2]];
        let within = gs.is_some_and(|g| {
            g.len() == 2
                && g[0].iter().zip(expected[0].iter()).all(|(a, b)| (a - b).abs() < 1e-9)
                && g[1].iter().zip(expected[1].iter()).all(|(a, b)| (a - b).abs() < 1e-9)
        });
        assert!(within);
    }

    #[test]
    fn nearest_plane_identity_just_rounds() {
        let ctx = PreimageContext::new(identity_basis(2)).ok();
        let z = ctx.and_then(|c| c.nearest_plane(&[3.4, -2.6]).ok());
        assert_eq!(z, Some(ZVec::new(vec![3, -3])));
    }

    #[test]
    fn nearest_plane_origin_is_zero() {
        let ctx = PreimageContext::new(identity_basis(3)).ok();
        let z = ctx.and_then(|c| c.nearest_plane(&[0.0, 0.0, 0.0]).ok());
        assert_eq!(z, Some(ZVec::new(vec![0, 0, 0])));
    }

    #[test]
    fn nearest_plane_lattice_point_recovered_exactly() {
        // Basis b_0 = [2, 1], b_1 = [1, 2].  Target (5, 4) = 2·b_0 + 1·b_1.
        let b = ZMatrix::new(2, 2, vec![2, 1, 1, 2])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        let ctx = PreimageContext::new(b).ok();
        let z = ctx.and_then(|c| c.nearest_plane(&[5.0, 4.0]).ok());
        assert_eq!(z, Some(ZVec::new(vec![2, 1])));
    }

    #[test]
    fn nearest_plane_diagonal_basis_rounds_componentwise() {
        // Basis diag(3, 5).  Target (7.4, -12.7):
        // i=1: <t, b*_1> / 25 = -63.5/25 = -2.54 → z_1 = -3.
        //   t' = (7.4, -12.7) - (-3)·(0,5) = (7.4, 2.3).
        // i=0: <t', b*_0> / 9 = 22.2/9 ≈ 2.467 → z_0 = 2.
        let b = ZMatrix::new(2, 2, vec![3, 0, 0, 5])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        let ctx = PreimageContext::new(b).ok();
        let z = ctx.and_then(|c| c.nearest_plane(&[7.4, -12.7]).ok());
        assert_eq!(z, Some(ZVec::new(vec![2, -3])));
    }

    #[test]
    fn nearest_plane_dim_mismatch_errors() {
        let ctx = PreimageContext::new(identity_basis(2)).ok();
        let err = ctx.as_ref().map(|c| c.nearest_plane(&[1.0]).is_err());
        assert_eq!(err, Some(true));
    }

    #[test]
    fn klein_returns_correct_length() {
        let ctx = PreimageContext::new(identity_basis(3)).ok();
        let rng = counter_rng();
        let z = ctx.and_then(|c| c.gaussian_preimage(&[0.0, 0.0, 0.0], 3.0, &rng).ok());
        assert_eq!(z.map(|v| v.len()), Some(3));
    }

    #[test]
    fn klein_deterministic_with_identical_rng() {
        let a = PreimageContext::new(identity_basis(2))
            .ok()
            .and_then(|c| c.gaussian_preimage(&[0.0, 0.0], 3.0, &counter_rng()).ok());
        let b = PreimageContext::new(identity_basis(2))
            .ok()
            .and_then(|c| c.gaussian_preimage(&[0.0, 0.0], 3.0, &counter_rng()).ok());
        assert_eq!(a, b);
    }

    #[test]
    fn klein_dim_mismatch_errors() {
        let ctx = PreimageContext::new(identity_basis(2)).ok();
        let rng = counter_rng();
        let err = ctx.as_ref().map(|c| {
            c.gaussian_preimage(&[1.0, 2.0, 3.0], 3.0, &rng).is_err()
        });
        assert_eq!(err, Some(true));
    }

    #[test]
    fn klein_output_is_bounded_near_target() {
        // σ=2 on identity basis puts most coefficients in [-12, 12].
        let ctx = PreimageContext::new(identity_basis(2)).ok();
        let rng = counter_rng();
        let z = ctx.and_then(|c| c.gaussian_preimage(&[0.0, 0.0], 2.0, &rng).ok());
        let ok = z.is_some_and(|v| v.entries().iter().all(|&e| e.abs() < 25));
        assert!(ok);
    }

    #[test]
    fn rank_and_dim_reflect_basis_shape() {
        let b = ZMatrix::new(3, 4, vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        let ctx = PreimageContext::new(b).ok();
        assert_eq!(ctx.as_ref().map(PreimageContext::rank), Some(3));
        assert_eq!(ctx.as_ref().map(PreimageContext::dim), Some(4));
    }
}
