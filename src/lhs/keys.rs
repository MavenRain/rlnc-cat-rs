//! Key generation and key types for the LHS scheme.
//!
//! The public key carries the public matrix `A = [A_0 | G - A_0·R]`
//! and the parameter set.  The secret key carries the trapdoor `R` —
//! the short matrix with ternary entries that lets the signer produce
//! short preimages under `A` via the gadget identity
//! `A · [R·w ; w] = G·w`.
//!
//! Hash targets `h_i` are *not* part of the keypair.  They are
//! per-generation data derived from caller-supplied metadata by the
//! authenticator layer (see [`crate::lhs::LatticeHomomorphicAuthenticator`]).
//! That keeps the keypair reusable across many generations and binds
//! each generation's commitment to the file it authenticates.

use crate::error::Error;
use crate::lattice::{PreimageContext, ZMatrix, Zq, ZqMatrix, ZqVec};
use crate::lhs::gadget::{gadget_entry, gadget_kernel_basis};
use crate::lhs::params::LhsParams;

/// A public key for the LHS scheme.
#[derive(Clone, Debug)]
#[must_use]
pub struct PublicKey<const Q: u32> {
    a: ZqMatrix<Q>,
    params: LhsParams<Q>,
}

impl<const Q: u32> PublicKey<Q> {
    /// Build a public key from the public matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `a`'s shape is not
    ///   `params.n() × params.m()`.
    pub fn new(a: ZqMatrix<Q>, params: LhsParams<Q>) -> Result<Self, Error> {
        let (n, m) = (params.n(), params.m());
        if a.rows() == n && a.cols() == m {
            Ok(Self { a, params })
        } else {
            Err(Error::DimensionMismatch {
                expected: n * m,
                actual: a.rows() * a.cols(),
            })
        }
    }

    /// The public matrix `A`.
    pub fn a(&self) -> &ZqMatrix<Q> {
        &self.a
    }

    /// The parameter set.
    pub fn params(&self) -> &LhsParams<Q> {
        &self.params
    }
}

/// The secret key: the trapdoor matrix `R`, the parameter set, and a
/// cached preimage context over the single-block gadget kernel
/// `Λ^⊥_q(g)` used by [`crate::lhs::sign::sign`] to draw a Gaussian
/// offset on top of the deterministic bit-decomposition preimage.
#[derive(Clone, Debug)]
#[must_use]
pub struct SecretKey<const Q: u32> {
    r: ZMatrix,
    params: LhsParams<Q>,
    gadget_kernel_ctx: PreimageContext,
}

impl<const Q: u32> SecretKey<Q> {
    /// Build a secret key from the trapdoor matrix.
    ///
    /// Also materialises the `k × k` gadget-kernel basis and its
    /// Gram-Schmidt orthogonalization (cached inside a
    /// [`PreimageContext`]), so repeated [`sign`][crate::lhs::sign::sign]
    /// calls can amortise the f64 linear-algebra work across the
    /// generation's pieces.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `r`'s shape is not
    ///   `params.m0() × params.gadget_width()`.
    /// - [`Error::RandomGenerationFailed`] if the gadget-kernel basis
    ///   derived from `Q` is numerically singular (`Q` is a power of
    ///   two, for instance); see [`PreimageContext::new`].
    pub fn new(r: ZMatrix, params: LhsParams<Q>) -> Result<Self, Error> {
        (r.rows() == params.m0() && r.cols() == params.gadget_width())
            .then_some(())
            .ok_or(Error::DimensionMismatch {
                expected: params.m0() * params.gadget_width(),
                actual: r.rows() * r.cols(),
            })
            .and_then(|()| gadget_kernel_basis::<Q>())
            .and_then(PreimageContext::new)
            .map(|gadget_kernel_ctx| Self {
                r,
                params,
                gadget_kernel_ctx,
            })
    }

    /// Borrow of the trapdoor matrix.
    pub fn r(&self) -> &ZMatrix {
        &self.r
    }

    /// The parameter set.
    pub fn params(&self) -> &LhsParams<Q> {
        &self.params
    }

    /// Borrow of the cached gadget-kernel preimage context.
    pub fn gadget_kernel_ctx(&self) -> &PreimageContext {
        &self.gadget_kernel_ctx
    }
}

/// Generate a fresh keypair.
///
/// Samples:
/// - a uniform `A_0 ∈ Zq^{n × m0}`,
/// - a ternary `R ∈ Z^{m0 × (n·k_gadget)}` with entries in `{-1, 0, 1}`,
///
/// and assembles `A = [A_0 | G - A_0·R]` so that
/// `A · [R·w ; w] = G·w` for every `w`.
///
/// # Errors
///
/// - [`Error::RandomGenerationFailed`] from any of the underlying
///   samplers (uniform `Zq`, ternary bytes).
/// - [`Error::DimensionMismatch`] if assembly of intermediate matrices
///   fails (unreachable for well-formed parameters).
pub fn keygen<const Q: u32, F>(
    params: LhsParams<Q>,
    rng: &F,
) -> Result<(PublicKey<Q>, SecretKey<Q>), Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    let (n, m0) = (params.n(), params.m0());
    let g_width = params.gadget_width();

    ZqMatrix::<Q>::sample(n, m0, rng).and_then(|a_0| {
        sample_ternary(m0 * g_width, rng).and_then(|r_data| {
            ZMatrix::new(m0, g_width, r_data).and_then(|r| {
                build_a1::<Q>(&a_0, &r, n, g_width).and_then(|a_1| {
                    assemble_full_a(&a_0, &a_1, n, m0, g_width).and_then(|a| {
                        PublicKey::new(a, params.clone())
                            .and_then(|pk| SecretKey::new(r, params).map(|sk| (pk, sk)))
                    })
                })
            })
        })
    })
}

/// Sample `count` integers uniformly from `{-1, 0, 1}` by reading
/// bytes and splitting the `[0, 255)` range into thirds.
fn sample_ternary<F>(count: usize, rng: &F) -> Result<Vec<i64>, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    sample_ternary_step(count, rng, Vec::new(), 32)
}

fn sample_ternary_step<F>(
    need: usize,
    rng: &F,
    acc: Vec<i64>,
    budget: u32,
) -> Result<Vec<i64>, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    if acc.len() >= need {
        Ok(acc.into_iter().take(need).collect())
    } else if budget == 0 {
        Err(Error::RandomGenerationFailed(
            "ternary sampling exhausted".to_string(),
        ))
    } else {
        let deficit = need - acc.len();
        rng(deficit * 2).and_then(|bytes| {
            let new_samples: Vec<i64> = bytes
                .iter()
                .filter(|&&b| b != 255)
                .map(|&b| i64::from(b / 85) - 1)
                .collect();
            let next_acc: Vec<i64> =
                acc.into_iter().chain(new_samples).collect();
            sample_ternary_step(need, rng, next_acc, budget - 1)
        })
    }
}

/// Compute `A_1 = G - A_0 · R (mod Q)` where `G` is the gadget matrix
/// and the product `A_0 · R` is computed column by column in `Zq`.
fn build_a1<const Q: u32>(
    a_0: &ZqMatrix<Q>,
    r: &ZMatrix,
    n: usize,
    g_width: usize,
) -> Result<ZqMatrix<Q>, Error> {
    let r_zq: ZqMatrix<Q> = r.reduce_mod();
    let cols: Vec<ZqVec<Q>> = (0..g_width)
        .map(|j| {
            let col: Vec<Zq<Q>> = (0..r_zq.rows())
                .map(|k| r_zq.entry(k, j).unwrap_or_else(Zq::zero))
                .collect();
            a_0.mul_vec(&ZqVec::new(col))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let data: Vec<Zq<Q>> = (0..n)
        .flat_map(|i| (0..g_width).map(move |j| (i, j)))
        .map(|(i, j)| {
            let g_ij = gadget_entry::<Q>(i, j);
            let a0r_ij = cols[j].entries()[i];
            g_ij - a0r_ij
        })
        .collect();
    ZqMatrix::new(n, g_width, data)
}

/// Horizontally concatenate `A_0` (n × m0) and `A_1` (n × `g_width`)
/// into `A` (n × m).
fn assemble_full_a<const Q: u32>(
    a_0: &ZqMatrix<Q>,
    a_1: &ZqMatrix<Q>,
    n: usize,
    m0: usize,
    g_width: usize,
) -> Result<ZqMatrix<Q>, Error> {
    let m = m0 + g_width;
    let data: Vec<Zq<Q>> = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| {
            if j < m0 {
                a_0.entry(i, j).unwrap_or_else(Zq::zero)
            } else {
                a_1.entry(i, j - m0).unwrap_or_else(Zq::zero)
            }
        })
        .collect();
    ZqMatrix::new(n, m, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::ZVec;
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

    #[test]
    fn keygen_dimensions_match_params() {
        let params = LhsParams::<97>::new(2, 2, 3, 10_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params.clone(), &counter_rng()).ok();
        assert_eq!(keys.as_ref().map(|(pk, _)| pk.a().rows()), Some(2));
        assert_eq!(
            keys.as_ref().map(|(pk, _)| pk.a().cols()),
            Some(params.m()),
        );
        assert_eq!(
            keys.as_ref().map(|(_, sk)| sk.r().rows()),
            Some(params.m0()),
        );
        assert_eq!(
            keys.as_ref().map(|(_, sk)| sk.r().cols()),
            Some(params.gadget_width()),
        );
    }

    #[test]
    fn keygen_secret_has_gadget_kernel_context() {
        // k × k basis cached in the secret key lets `sign` skip the
        // Gram-Schmidt cost on every invocation.
        let params = LhsParams::<97>::new(2, 2, 3, 10_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng()).ok();
        let k = LhsParams::<97>::k_gadget();
        let shape = keys.map(|(_, sk)| {
            let ctx = sk.gadget_kernel_ctx();
            (ctx.rank(), ctx.dim())
        });
        assert_eq!(shape, Some((k, k)));
    }

    #[test]
    fn keygen_trapdoor_identity_holds() {
        // A · [R·w ; w] = G · w  for every w.  Verify with a random
        // bit vector `w` and confirm the left side reduces to G·w
        // in Zq.
        let params = LhsParams::<97>::new(2, 2, 1, 10_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params.clone(), &counter_rng()).ok();
        let ok = keys.is_some_and(|(pk, sk)| {
            let g_width = params.gadget_width();
            let w: Vec<i64> = (0..g_width).map(|j| i64::from(u8::from(j % 2 == 0))).collect();
            let w_zvec = ZVec::new(w.clone());
            let r_w = sk.r().mul_vec(&w_zvec).ok();
            r_w.is_some_and(|r_w| {
                let sigma_entries: Vec<i64> =
                    r_w.entries().iter().copied().chain(w.iter().copied()).collect();
                let sigma_zq: ZqVec<97> = ZVec::new(sigma_entries).reduce_mod();
                let a_sigma = pk.a().mul_vec(&sigma_zq).ok();
                let g_w = crate::lhs::gadget::gadget_apply::<97>(&w, params.n());
                a_sigma == Some(g_w)
            })
        });
        assert!(ok);
    }

    #[test]
    fn sample_ternary_entries_are_in_range() {
        let t = sample_ternary(200, &counter_rng()).ok();
        let ok = t.is_some_and(|v| {
            v.len() == 200 && v.iter().all(|&x| (-1..=1).contains(&x))
        });
        assert!(ok);
    }

    #[test]
    fn public_key_construction_rejects_mismatched_shape() {
        let params = LhsParams::<97>::new(2, 2, 3, 10_000, 3.0)
            .ok()
            .unwrap_or_else(unreachable_params);
        let a = ZqMatrix::<97>::zeros(1, params.m());
        assert!(PublicKey::new(a, params).is_err());
    }

    fn unreachable_params<const Q: u32>() -> LhsParams<Q> {
        LhsParams::<Q>::new(1, 1, 1, 1, 1.0)
            .ok()
            .unwrap_or_else(unreachable_params)
    }
}
