//! Matrices over `Z/QZ` in row-major storage.
//!
//! [`ZqMatrix`] is the matrix counterpart to [`ZqVec`].  Only the
//! operations needed by the BFKW09 homomorphic-signature scheme live
//! here: construction, entry access, matrix-vector multiplication,
//! and uniform sampling.  Matrix-matrix multiplication and
//! factorizations are deliberately out of scope at this stage.

use crate::error::Error;
use crate::lattice::vector::ZqVec;
use crate::lattice::zq::Zq;

/// A matrix over `Z/QZ`, stored row-major.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lattice::{Zq, ZqMatrix, ZqVec};
///
/// let a = ZqMatrix::<97>::new(
///     2, 2,
///     vec![Zq::new(1), Zq::new(0), Zq::new(0), Zq::new(1)],
/// ).ok();
/// let v = ZqVec::<97>::new(vec![Zq::new(5), Zq::new(7)]);
/// let out = a.and_then(|m| m.mul_vec(&v).ok());
/// assert_eq!(out, Some(v));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct ZqMatrix<const Q: u32> {
    rows: usize,
    cols: usize,
    data: Vec<Zq<Q>>,
}

impl<const Q: u32> ZqMatrix<Q> {
    /// Create a matrix from row-major data.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<Zq<Q>>) -> Result<Self, Error> {
        if data.len() == rows * cols {
            Ok(Self { rows, cols, data })
        } else {
            Err(Error::DimensionMismatch {
                expected: rows * cols,
                actual: data.len(),
            })
        }
    }

    /// The zero matrix of the given dimensions.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Zq::zero(); rows * cols],
        }
    }

    /// The row count.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The column count.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The entry at `(row, col)`, if in bounds.
    #[must_use]
    #[inline]
    pub fn entry(&self, row: usize, col: usize) -> Option<Zq<Q>> {
        (col < self.cols)
            .then(|| self.data.get(row * self.cols + col).copied())
            .flatten()
    }

    /// Multiply this matrix by a column vector.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `v.len() != self.cols`.
    pub fn mul_vec(&self, v: &ZqVec<Q>) -> Result<ZqVec<Q>, Error> {
        if v.len() == self.cols {
            let v_entries = v.entries();
            let entries: Vec<Zq<Q>> = self
                .data
                .chunks(self.cols)
                .map(|row_slice| {
                    row_slice
                        .iter()
                        .zip(v_entries.iter())
                        .fold(Zq::<Q>::zero(), |acc, (&a, &b)| acc + a * b)
                })
                .collect();
            Ok(ZqVec::new(entries))
        } else {
            Err(Error::DimensionMismatch {
                expected: self.cols,
                actual: v.len(),
            })
        }
    }

    /// Sample a uniform `rows x cols` matrix.
    ///
    /// # Errors
    ///
    /// - [`Error::RandomGenerationFailed`] from the underlying
    ///   per-entry rejection sampler.
    pub fn sample<F>(rows: usize, cols: usize, rng: &F) -> Result<Self, Error>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error>,
    {
        (0..rows * cols)
            .map(|_| Zq::sample(rng))
            .collect::<Result<Vec<_>, _>>()
            .map(|data| Self { rows, cols, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u32 = 97;

    fn identity(n: usize) -> ZqMatrix<Q> {
        let data: Vec<Zq<Q>> = (0..n * n)
            .map(|k| {
                if k / n == k % n {
                    Zq::one()
                } else {
                    Zq::zero()
                }
            })
            .collect();
        ZqMatrix::new(n, n, data).ok().unwrap_or_else(|| ZqMatrix::zeros(0, 0))
    }

    #[test]
    fn new_rejects_wrong_data_length() {
        let bad = ZqMatrix::<Q>::new(2, 3, vec![Zq::zero(); 5]);
        assert!(bad.is_err());
    }

    #[test]
    fn zero_matrix_kills_any_vector() {
        let a = ZqMatrix::<Q>::zeros(3, 4);
        let v = ZqVec::<Q>::new((0..4).map(|i| Zq::new(i + 1)).collect());
        let result = a.mul_vec(&v).ok();
        assert_eq!(result, Some(ZqVec::<Q>::zeros(3)));
    }

    #[test]
    fn identity_matrix_fixes_vector() {
        let a = identity(3);
        let v = ZqVec::<Q>::new(vec![Zq::new(5), Zq::new(7), Zq::new(11)]);
        let result = a.mul_vec(&v).ok();
        assert_eq!(result, Some(v));
    }

    #[test]
    fn mul_vec_dim_mismatch_errors() {
        let a = ZqMatrix::<Q>::zeros(3, 4);
        let v = ZqVec::<Q>::zeros(3);
        assert!(a.mul_vec(&v).is_err());
    }

    #[test]
    fn entry_lookup_is_row_major() {
        let data = vec![
            Zq::new(1),
            Zq::new(2),
            Zq::new(3),
            Zq::new(4),
            Zq::new(5),
            Zq::new(6),
        ];
        let a = ZqMatrix::<Q>::new(2, 3, data)
            .ok()
            .unwrap_or_else(|| ZqMatrix::zeros(0, 0));
        assert_eq!(a.entry(0, 0), Some(Zq::new(1)));
        assert_eq!(a.entry(0, 2), Some(Zq::new(3)));
        assert_eq!(a.entry(1, 2), Some(Zq::new(6)));
        assert_eq!(a.entry(2, 0), None);
        assert_eq!(a.entry(0, 3), None);
    }

    #[test]
    fn mul_vec_computes_dot_products() {
        // A = [[1, 2], [3, 4]], v = [5, 6].
        // A v = [1*5 + 2*6, 3*5 + 4*6] = [17, 39].
        let data = vec![Zq::new(1), Zq::new(2), Zq::new(3), Zq::new(4)];
        let a = ZqMatrix::<Q>::new(2, 2, data)
            .ok()
            .unwrap_or_else(|| ZqMatrix::zeros(0, 0));
        let v = ZqVec::<Q>::new(vec![Zq::new(5), Zq::new(6)]);
        let result = a.mul_vec(&v).ok();
        assert_eq!(
            result,
            Some(ZqVec::new(vec![Zq::new(17), Zq::new(39)])),
        );
    }

    #[test]
    fn sample_produces_correct_dims() {
        let rng = |n: usize| -> Result<Vec<u8>, Error> { Ok(vec![0x42u8; n]) };
        let m = ZqMatrix::<Q>::sample(3, 5, &rng).ok();
        assert_eq!(m.as_ref().map(ZqMatrix::rows), Some(3));
        assert_eq!(m.as_ref().map(ZqMatrix::cols), Some(5));
    }
}
