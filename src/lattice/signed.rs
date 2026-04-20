//! Integer vectors and matrices over `Z`.
//!
//! [`ZVec`] and [`ZMatrix`] carry unbounded integer entries (stored as
//! `i64`) without any modular reduction.  They are the natural domain
//! of signature vectors and short lattice bases: shortness is measured
//! in the `Z` interpretation via [`ZVec::squared_l2_norm`], and the
//! projection into `Z/qZ` happens only at verification time via
//! [`ZVec::reduce_mod`] and [`ZMatrix::reduce_mod`].

use crate::error::Error;
use crate::lattice::matrix::ZqMatrix;
use crate::lattice::vector::ZqVec;
use crate::lattice::zq::Zq;

/// An immutable vector of `i64` entries.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lattice::ZVec;
///
/// let v = ZVec::new(vec![1, -2, 3]);
/// assert_eq!(v.squared_l2_norm(), 1 + 4 + 9);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct ZVec {
    entries: Vec<i64>,
}

impl ZVec {
    /// Wrap an owned entry list.
    pub fn new(entries: Vec<i64>) -> Self {
        Self { entries }
    }

    /// The zero vector of the given length.
    pub fn zeros(len: usize) -> Self {
        Self {
            entries: vec![0i64; len],
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
    #[must_use]
    pub fn entries(&self) -> &[i64] {
        &self.entries
    }

    /// The entry at `index`, if in bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<i64> {
        self.entries.get(index).copied()
    }

    /// Pointwise addition.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if lengths differ.
    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        if self.len() == other.len() {
            Ok(Self {
                entries: self
                    .entries
                    .iter()
                    .zip(other.entries.iter())
                    .map(|(&a, &b)| a + b)
                    .collect(),
            })
        } else {
            Err(Error::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            })
        }
    }

    /// Pointwise subtraction `self - other`.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if lengths differ.
    pub fn sub(&self, other: &Self) -> Result<Self, Error> {
        if self.len() == other.len() {
            Ok(Self {
                entries: self
                    .entries
                    .iter()
                    .zip(other.entries.iter())
                    .map(|(&a, &b)| a - b)
                    .collect(),
            })
        } else {
            Err(Error::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            })
        }
    }

    /// Multiply every entry by the integer scalar `k`.
    pub fn scale(&self, k: i64) -> Self {
        Self {
            entries: self.entries.iter().map(|&a| k * a).collect(),
        }
    }

    /// The linear combination `Σ_i scalars[i] * vecs[i]` over `Z`.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `scalars.len() != vecs.len()`
    ///   or if the vectors in `vecs` do not all share a common length.
    pub fn linear_combine(scalars: &[i64], vecs: &[Self]) -> Result<Self, Error> {
        if scalars.len() == vecs.len() {
            let dim = vecs.first().map_or(0, Self::len);
            vecs.iter().find(|v| v.len() != dim).map_or_else(
                || {
                    let init: Vec<i64> = vec![0i64; dim];
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

    /// Sum of squared entries as a `u128`.  Safe against overflow for
    /// any `i64` entries.
    #[must_use]
    pub fn squared_l2_norm(&self) -> u128 {
        self.entries
            .iter()
            .map(|&x| {
                let a = u128::from(x.unsigned_abs());
                a * a
            })
            .sum()
    }

    /// Reduce each entry modulo `Q`, yielding a [`ZqVec<Q>`].
    ///
    /// Negative entries are lifted into `[0, Q)` via Euclidean
    /// remainder so the output matches the canonical representative
    /// of `Zq`.
    pub fn reduce_mod<const Q: u32>(&self) -> ZqVec<Q> {
        let entries = self
            .entries
            .iter()
            .map(|&x| {
                let r = x.rem_euclid(i64::from(Q));
                Zq::<Q>::new(u32::try_from(r).unwrap_or(0))
            })
            .collect();
        ZqVec::new(entries)
    }
}

/// A matrix of `i64` entries in row-major storage.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lattice::{ZMatrix, ZVec};
///
/// let m = ZMatrix::new(2, 2, vec![1, 2, 3, 4]).ok();
/// let v = ZVec::new(vec![5, 6]);
/// let out = m.and_then(|a| a.mul_vec(&v).ok());
/// assert_eq!(out.map(|z| z.entries().to_vec()), Some(vec![17, 39]));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct ZMatrix {
    rows: usize,
    cols: usize,
    data: Vec<i64>,
}

impl ZMatrix {
    /// Create from row-major data.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<i64>) -> Result<Self, Error> {
        if data.len() == rows * cols {
            Ok(Self { rows, cols, data })
        } else {
            Err(Error::DimensionMismatch {
                expected: rows * cols,
                actual: data.len(),
            })
        }
    }

    /// The zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0i64; rows * cols],
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

    /// The entry at `(row, col)` if in bounds.
    #[must_use]
    pub fn entry(&self, row: usize, col: usize) -> Option<i64> {
        self.data.get(row * self.cols + col).copied().filter(|_| col < self.cols && row < self.rows)
    }

    /// A borrowed slice covering row `row`, if in bounds.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[i64]> {
        (row < self.rows)
            .then(|| self.data.get(row * self.cols..(row + 1) * self.cols))
            .flatten()
    }

    /// Multiply by an integer column vector.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `v.len() != self.cols`.
    pub fn mul_vec(&self, v: &ZVec) -> Result<ZVec, Error> {
        if v.len() == self.cols {
            let v_entries = v.entries();
            let entries: Vec<i64> = self
                .data
                .chunks(self.cols)
                .map(|row_slice| {
                    row_slice
                        .iter()
                        .zip(v_entries.iter())
                        .fold(0i64, |acc, (&a, &b)| acc + a * b)
                })
                .collect();
            Ok(ZVec::new(entries))
        } else {
            Err(Error::DimensionMismatch {
                expected: self.cols,
                actual: v.len(),
            })
        }
    }

    /// Reduce every entry modulo `Q`, yielding a [`ZqMatrix<Q>`].
    pub fn reduce_mod<const Q: u32>(&self) -> ZqMatrix<Q> {
        let data: Vec<Zq<Q>> = self
            .data
            .iter()
            .map(|&x| {
                let r = x.rem_euclid(i64::from(Q));
                Zq::<Q>::new(u32::try_from(r).unwrap_or(0))
            })
            .collect();
        ZqMatrix::new(self.rows, self.cols, data)
            .ok()
            .unwrap_or_else(|| ZqMatrix::zeros(self.rows, self.cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_l2_norm_counts_signed_squares() {
        let v = ZVec::new(vec![3, -4, 0]);
        assert_eq!(v.squared_l2_norm(), 9 + 16);
    }

    #[test]
    fn add_is_pointwise() {
        let a = ZVec::new(vec![1, 2, 3]);
        let b = ZVec::new(vec![10, 20, 30]);
        assert_eq!(
            a.add(&b).ok(),
            Some(ZVec::new(vec![11, 22, 33])),
        );
    }

    #[test]
    fn sub_is_pointwise() {
        let a = ZVec::new(vec![10, 20, 30]);
        let b = ZVec::new(vec![1, 2, 3]);
        assert_eq!(
            a.sub(&b).ok(),
            Some(ZVec::new(vec![9, 18, 27])),
        );
    }

    #[test]
    fn scale_multiplies_each_entry() {
        let v = ZVec::new(vec![1, -2, 3]);
        assert_eq!(v.scale(-5), ZVec::new(vec![-5, 10, -15]));
    }

    #[test]
    fn linear_combine_computes_sum_of_scaled_vecs() {
        let v1 = ZVec::new(vec![1, 2]);
        let v2 = ZVec::new(vec![10, 20]);
        let out = ZVec::linear_combine(&[3, -1], &[v1, v2]).ok();
        assert_eq!(out, Some(ZVec::new(vec![3 - 10, 6 - 20])));
    }

    #[test]
    fn linear_combine_mismatch_errors() {
        let v1 = ZVec::new(vec![1, 2]);
        let v2 = ZVec::new(vec![10]);
        assert!(ZVec::linear_combine(&[1, 1], &[v1, v2]).is_err());
    }

    #[test]
    fn reduce_mod_handles_negatives() {
        let v = ZVec::new(vec![-1, -2, 50]);
        let reduced: ZqVec<7> = v.reduce_mod();
        assert_eq!(reduced.entries()[0], Zq::new(6));
        assert_eq!(reduced.entries()[1], Zq::new(5));
        assert_eq!(reduced.entries()[2], Zq::new(1));
    }

    #[test]
    fn zmatrix_mul_vec_matches_hand_calculation() {
        let m = ZMatrix::new(2, 2, vec![1, 2, 3, 4])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        let v = ZVec::new(vec![5, 6]);
        let out = m.mul_vec(&v).ok();
        assert_eq!(out, Some(ZVec::new(vec![17, 39])));
    }

    #[test]
    fn zmatrix_row_borrow_covers_full_row() {
        let m = ZMatrix::new(2, 3, vec![1, 2, 3, 4, 5, 6])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        assert_eq!(m.row(1), Some(&[4i64, 5, 6][..]));
        assert_eq!(m.row(2), None);
    }

    #[test]
    fn zmatrix_reduce_mod_dims_preserved() {
        let m = ZMatrix::new(2, 3, vec![-1, 0, 7, 8, -2, 100])
            .ok()
            .unwrap_or_else(|| ZMatrix::zeros(0, 0));
        let reduced: ZqMatrix<5> = m.reduce_mod();
        assert_eq!(reduced.rows(), 2);
        assert_eq!(reduced.cols(), 3);
        assert_eq!(reduced.entry(0, 0), Some(Zq::new(4))); // -1 mod 5 = 4
        assert_eq!(reduced.entry(1, 2), Some(Zq::new(0))); // 100 mod 5 = 0
    }
}
