//! Vectors over `Z/QZ`.
//!
//! [`ZqVec`] is the vector-space counterpart to [`Zq`]: an owned,
//! immutable sequence of `Z/QZ` entries with linear-combination and
//! squared-L2-norm operations.  The norm uses the *signed*
//! representative of each entry, since "short" in a lattice always
//! means "close to zero" in the balanced interval `(-Q/2, Q/2]`.

use crate::error::Error;
use crate::lattice::zq::Zq;

/// An immutable vector over `Z/QZ`.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lattice::{Zq, ZqVec};
///
/// let v = ZqVec::<97>::new(vec![Zq::new(1), Zq::new(2), Zq::new(3)]);
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.squared_l2_norm(), 1 + 4 + 9);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct ZqVec<const Q: u32> {
    entries: Vec<Zq<Q>>,
}

impl<const Q: u32> ZqVec<Q> {
    /// Create a vector from an owned entry list.
    pub fn new(entries: Vec<Zq<Q>>) -> Self {
        Self { entries }
    }

    /// The zero vector of the given length.
    pub fn zeros(len: usize) -> Self {
        Self {
            entries: vec![Zq::zero(); len],
        }
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vector has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A borrowed slice of the entries.
    pub fn entries(&self) -> &[Zq<Q>] {
        &self.entries
    }

    /// The entry at `index`, if in bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Zq<Q>> {
        self.entries.get(index).copied()
    }

    /// The linear combination `Σ_i scalars[i] * vecs[i]`.
    ///
    /// All input vectors must share a common length; the output has
    /// that same length.  An empty input yields a length-zero vector.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `scalars.len() != vecs.len()`
    ///   or if any vector in `vecs` has a different length from the
    ///   first.
    pub fn linear_combine(scalars: &[Zq<Q>], vecs: &[Self]) -> Result<Self, Error> {
        if scalars.len() == vecs.len() {
            let dim = vecs.first().map_or(0, Self::len);
            vecs.iter().find(|v| v.len() != dim).map_or_else(
                || {
                    let init: Vec<Zq<Q>> = vec![Zq::<Q>::zero(); dim];
                    let entries = scalars.iter().zip(vecs.iter()).fold(
                        init,
                        |acc, (&s, v)| {
                            acc.into_iter()
                                .zip(v.entries.iter())
                                .map(|(a, &b)| a + s * b)
                                .collect()
                        },
                    );
                    Ok(Self { entries })
                },
                |bad| {
                    Err(Error::DimensionMismatch {
                        expected: dim,
                        actual: bad.len(),
                    })
                },
            )
        } else {
            Err(Error::DimensionMismatch {
                expected: scalars.len(),
                actual: vecs.len(),
            })
        }
    }

    /// Pointwise addition.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if lengths differ.
    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        if self.len() == other.len() {
            let entries = self
                .entries
                .iter()
                .zip(other.entries.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            Ok(Self { entries })
        } else {
            Err(Error::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            })
        }
    }

    /// Multiply every entry by `scalar`.
    pub fn scale(&self, scalar: Zq<Q>) -> Self {
        Self {
            entries: self.entries.iter().map(|&a| scalar * a).collect(),
        }
    }

    /// The sum of squared *signed* representatives.
    ///
    /// Each entry is interpreted in the balanced interval `(-Q/2, Q/2]`
    /// before squaring, so a value `Q - 1` contributes `1`, not
    /// `(Q-1)^2`.  The return type is `u128` so large vectors cannot
    /// overflow (for `Q < 2^32` each square is under `2^62`).
    #[must_use]
    pub fn squared_l2_norm(&self) -> u128 {
        self.entries
            .iter()
            .map(|z| {
                let a = u128::from(z.signed_repr().unsigned_abs());
                a * a
            })
            .sum()
    }

    /// Sample a uniform vector of the given length.
    ///
    /// Each entry is drawn independently from the uniform distribution
    /// on `Z/QZ` via rejection sampling in [`Zq::sample`].
    ///
    /// # Errors
    ///
    /// - [`Error::RandomGenerationFailed`] from the underlying
    ///   rejection sampler.
    pub fn sample<F>(len: usize, rng: &F) -> Result<Self, Error>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error>,
    {
        (0..len)
            .map(|_| Zq::sample(rng))
            .collect::<Result<Vec<_>, _>>()
            .map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u32 = 97;

    #[test]
    fn zeros_has_zero_norm() {
        assert_eq!(ZqVec::<Q>::zeros(5).squared_l2_norm(), 0);
    }

    #[test]
    fn basis_vector_norm_is_one() {
        let e1 = ZqVec::<Q>::new(vec![Zq::zero(), Zq::one(), Zq::zero()]);
        assert_eq!(e1.squared_l2_norm(), 1);
    }

    #[test]
    fn signed_repr_used_for_norm() {
        // Value Q-1 is signed_repr -1, so norm contribution is 1.
        let v = ZqVec::<Q>::new(vec![Zq::new(Q - 1)]);
        assert_eq!(v.squared_l2_norm(), 1);
    }

    #[test]
    fn linear_combine_empty_is_empty() {
        let out = ZqVec::<Q>::linear_combine(&[], &[]).ok();
        assert_eq!(out, Some(ZqVec::zeros(0)));
    }

    #[test]
    fn linear_combine_single_is_scale() {
        let v = ZqVec::<Q>::new(vec![Zq::new(3), Zq::new(5)]);
        let out = ZqVec::<Q>::linear_combine(&[Zq::new(7)], std::slice::from_ref(&v)).ok();
        let expected = ZqVec::<Q>::new(vec![Zq::new(21), Zq::new(35)]);
        assert_eq!(out, Some(expected));
    }

    #[test]
    fn linear_combine_two_combines() {
        let v1 = ZqVec::<Q>::new(vec![Zq::new(1), Zq::new(2)]);
        let v2 = ZqVec::<Q>::new(vec![Zq::new(3), Zq::new(4)]);
        let out = ZqVec::<Q>::linear_combine(&[Zq::new(2), Zq::new(3)], &[v1, v2]).ok();
        let expected = ZqVec::<Q>::new(vec![Zq::new(11), Zq::new(16)]);
        assert_eq!(out, Some(expected));
    }

    #[test]
    fn linear_combine_length_mismatch_errors() {
        let v1 = ZqVec::<Q>::new(vec![Zq::new(1), Zq::new(2)]);
        let v2 = ZqVec::<Q>::new(vec![Zq::new(1)]);
        let out = ZqVec::<Q>::linear_combine(&[Zq::new(1), Zq::new(1)], &[v1, v2]);
        assert!(out.is_err());
    }

    #[test]
    fn linear_combine_scalar_count_mismatch_errors() {
        let v = ZqVec::<Q>::new(vec![Zq::new(1), Zq::new(2)]);
        let out = ZqVec::<Q>::linear_combine(&[Zq::new(1), Zq::new(2)], &[v]);
        assert!(out.is_err());
    }

    #[test]
    fn add_is_pointwise() {
        let v1 = ZqVec::<Q>::new(vec![Zq::new(1), Zq::new(2)]);
        let v2 = ZqVec::<Q>::new(vec![Zq::new(3), Zq::new(4)]);
        let out = v1.add(&v2).ok();
        assert_eq!(
            out,
            Some(ZqVec::new(vec![Zq::new(4), Zq::new(6)])),
        );
    }

    #[test]
    fn scale_multiplies_each_entry() {
        let v = ZqVec::<Q>::new(vec![Zq::new(1), Zq::new(2), Zq::new(3)]);
        let out = v.scale(Zq::new(4));
        assert_eq!(
            out,
            ZqVec::new(vec![Zq::new(4), Zq::new(8), Zq::new(12)]),
        );
    }

    #[test]
    fn sample_produces_correct_length() {
        let rng = |n: usize| -> Result<Vec<u8>, Error> { Ok(vec![0x42u8; n]) };
        let v = ZqVec::<Q>::sample(7, &rng).ok();
        assert_eq!(v.map(|x| x.len()), Some(7));
    }
}
