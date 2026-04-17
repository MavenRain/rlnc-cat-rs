//! Immutable matrices over GF(2^8).
//!
//! [`GfMatrix`] stores rows as `Arc<GfVec>` internally so that every row
//! operation (swap, scale, add) only reallocates the row(s) it touches;
//! the rest of the matrix is reconstructed by refcount bumps.  For
//! Gaussian elimination this turns the per-op cost from O(rows × cols)
//! byte copies into O(rows) atomic increments plus one fresh row.

use std::sync::Arc;

use crate::error::Error;
use crate::field::Gf256;
use crate::vector::GfVec;

/// An immutable matrix of GF(2^8) elements in row-major order.
///
/// Used for the augmented decoder matrix in Gaussian elimination.
/// Each row operation (swap, scale, add) returns a fresh matrix whose
/// untouched rows share storage with `self` via [`Arc`], so the
/// per-op byte-copy cost is bounded by the single row that changes.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::vector::{GfVec, GfMatrix};
///
/// let row0 = GfVec::from_bytes(&[1, 0]);
/// let row1 = GfVec::from_bytes(&[0, 1]);
/// let m = GfMatrix::new(vec![row0, row1]);
/// assert!(m.is_ok());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct GfMatrix {
    rows: Vec<Arc<GfVec>>,
    col_count: usize,
}

impl GfMatrix {
    /// Create a matrix from a vector of rows.
    ///
    /// All rows must have the same length.  An empty row list produces
    /// a 0x0 matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if rows have inconsistent lengths.
    pub fn new(rows: Vec<GfVec>) -> Result<Self, Error> {
        let col_count = rows.first().map_or(0, GfVec::len);
        let bad_row = rows.iter().find(|r| r.len() != col_count);
        if let Some(r) = bad_row {
            Err(Error::DimensionMismatch {
                expected: col_count,
                actual: r.len(),
            })
        } else {
            Ok(Self {
                rows: rows.into_iter().map(Arc::new).collect(),
                col_count,
            })
        }
    }

    /// Create a matrix with zero rows and the given column count.
    pub fn empty(col_count: usize) -> Self {
        Self {
            rows: Vec::new(),
            col_count,
        }
    }

    /// The number of rows.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The number of columns.
    #[must_use]
    pub fn col_count(&self) -> usize {
        self.col_count
    }

    /// A borrowed reference to the given row.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&GfVec> {
        self.rows.get(index).map(Arc::as_ref)
    }

    /// The element at (row, col).
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<Gf256> {
        self.rows.get(row).and_then(|r| r.get(col))
    }

    /// An iterator over the rows.
    pub fn rows(&self) -> impl Iterator<Item = &GfVec> {
        self.rows.iter().map(Arc::as_ref)
    }

    /// Append a row to the matrix, returning a new matrix.
    ///
    /// Untouched rows are shared via `Arc`; only the new row is freshly
    /// allocated.
    ///
    /// # Errors
    ///
    /// Returns an error if the new row's length differs from `col_count`.
    pub fn append_row(&self, row: GfVec) -> Result<Self, Error> {
        if row.len() == self.col_count {
            let new_rows: Vec<Arc<GfVec>> = self
                .rows
                .iter()
                .map(Arc::clone)
                .chain(core::iter::once(Arc::new(row)))
                .collect();
            Ok(Self {
                rows: new_rows,
                col_count: self.col_count,
            })
        } else {
            Err(Error::DimensionMismatch {
                expected: self.col_count,
                actual: row.len(),
            })
        }
    }

    /// Swap two rows, returning a new matrix.
    ///
    /// Returns `None` if either index is out of bounds.  Swaps are
    /// O(rows) refcount bumps; no row bytes are copied.
    #[must_use]
    pub fn with_rows_swapped(&self, i: usize, j: usize) -> Option<Self> {
        if i >= self.rows.len() || j >= self.rows.len() {
            None
        } else {
            let new_rows: Vec<Arc<GfVec>> = self
                .rows
                .iter()
                .enumerate()
                .map(|(idx, row)| {
                    if idx == i {
                        Arc::clone(&self.rows[j])
                    } else if idx == j {
                        Arc::clone(&self.rows[i])
                    } else {
                        Arc::clone(row)
                    }
                })
                .collect();
            Some(Self {
                rows: new_rows,
                col_count: self.col_count,
            })
        }
    }

    /// Scale a row by a scalar, returning a new matrix.
    ///
    /// Only the scaled row is freshly allocated; all others share
    /// storage via `Arc`.  Returns `None` if the index is out of bounds.
    #[must_use]
    pub fn with_row_scaled(&self, index: usize, scalar: Gf256) -> Option<Self> {
        if index >= self.rows.len() {
            None
        } else {
            let scaled = Arc::new(self.rows[index].scale(scalar));
            let new_rows: Vec<Arc<GfVec>> = self
                .rows
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    if i == index {
                        Arc::clone(&scaled)
                    } else {
                        Arc::clone(row)
                    }
                })
                .collect();
            Some(Self {
                rows: new_rows,
                col_count: self.col_count,
            })
        }
    }

    /// Add `scalar * row[source]` to `row[target]`, returning a new matrix.
    ///
    /// Computes: `new_row[target] = row[target] + scalar * row[source]`
    /// as a single fused multiply-accumulate.  Only the modified row is
    /// freshly allocated; all other rows share storage with `self` via
    /// `Arc`.
    ///
    /// Returns `None` if either index is out of bounds.
    #[must_use]
    pub fn with_row_added(
        &self,
        target: usize,
        source: usize,
        scalar: Gf256,
    ) -> Option<Self> {
        if target >= self.rows.len() || source >= self.rows.len() {
            None
        } else {
            // mac can only fail on dimension mismatch, which cannot
            // happen here since all rows have the same length.
            self.rows[target]
                .mac(&self.rows[source], scalar)
                .ok()
                .map(|new_target| {
                    let new_target_arc = Arc::new(new_target);
                    let new_rows: Vec<Arc<GfVec>> = self
                        .rows
                        .iter()
                        .enumerate()
                        .map(|(i, row)| {
                            if i == target {
                                Arc::clone(&new_target_arc)
                            } else {
                                Arc::clone(row)
                            }
                        })
                        .collect();
                    Self {
                        rows: new_rows,
                        col_count: self.col_count,
                    }
                })
        }
    }

    /// Split the matrix at the given column, returning (left, right).
    ///
    /// The left matrix has columns `0..col` and the right has `col..col_count`.
    ///
    /// # Errors
    ///
    /// Returns an error if `col > col_count`.
    pub fn split_at_col(&self, col: usize) -> Result<(Self, Self), Error> {
        if col > self.col_count {
            Err(Error::DimensionMismatch {
                expected: self.col_count,
                actual: col,
            })
        } else {
            let (left_rows, right_rows): (Vec<Arc<GfVec>>, Vec<Arc<GfVec>>) = self
                .rows
                .iter()
                .map(|row| {
                    let bytes = row.to_bytes();
                    let (l, r) = bytes.split_at(col);
                    (
                        Arc::new(GfVec::from_bytes(l)),
                        Arc::new(GfVec::from_bytes(r)),
                    )
                })
                .unzip();
            Ok((
                Self {
                    rows: left_rows,
                    col_count: col,
                },
                Self {
                    rows: right_rows,
                    col_count: self.col_count - col,
                },
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_2x2() -> GfMatrix {
        GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 0]),
            GfVec::from_bytes(&[0, 1]),
        ])
        .unwrap_or(GfMatrix::empty(0))
    }

    #[test]
    fn new_validates_row_lengths() {
        let bad = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 2]),
            GfVec::from_bytes(&[3, 4, 5]),
        ]);
        assert!(bad.is_err());
    }

    #[test]
    fn append_row_validates_length() {
        let m = identity_2x2();
        assert!(m.append_row(GfVec::from_bytes(&[1, 2, 3])).is_err());
        assert!(m.append_row(GfVec::from_bytes(&[1, 2])).is_ok());
    }

    #[test]
    fn swap_rows_works() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 0]),
            GfVec::from_bytes(&[0, 2]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let swapped = m.with_rows_swapped(0, 1);
        assert!(swapped.is_some());
        let s = swapped.unwrap_or(GfMatrix::empty(0));
        assert_eq!(s.get(0, 0), Some(Gf256::zero()));
        assert_eq!(s.get(0, 1), Some(Gf256::new(2)));
        assert_eq!(s.get(1, 0), Some(Gf256::one()));
    }

    #[test]
    fn split_at_col_works() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 2, 3, 4]),
            GfVec::from_bytes(&[5, 6, 7, 8]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let split = m.split_at_col(2);
        assert!(split.is_ok());
        let (left, right) = split.unwrap_or((GfMatrix::empty(0), GfMatrix::empty(0)));
        assert_eq!(left.col_count(), 2);
        assert_eq!(right.col_count(), 2);
        assert_eq!(left.get(0, 0), Some(Gf256::new(1)));
        assert_eq!(right.get(0, 0), Some(Gf256::new(3)));
    }

    #[test]
    fn row_op_shares_untouched_rows() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 2, 3]),
            GfVec::from_bytes(&[4, 5, 6]),
            GfVec::from_bytes(&[7, 8, 9]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let modified = m.with_row_added(0, 1, Gf256::one()).unwrap_or(m.clone());
        // Row 2 is unchanged; the Arc handle inside `modified` must point
        // at the same allocation as the corresponding Arc in `m`.
        let m_row2 = &m.rows[2];
        let modified_row2 = &modified.rows[2];
        assert!(Arc::ptr_eq(m_row2, modified_row2));
    }
}
