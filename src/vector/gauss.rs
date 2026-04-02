//! Gaussian elimination over GF(2^8).
//!
//! Pure functions that reduce a [`GfMatrix`] to reduced row echelon
//! form (RREF) via forward and backward elimination.  Each step
//! returns a new matrix; nothing is mutated.

use crate::error::Error;
use crate::field::Gf256;
use crate::vector::GfMatrix;

/// Find the first nonzero entry in the given column at or below `start_row`.
fn find_pivot(matrix: &GfMatrix, col: usize, start_row: usize) -> Option<usize> {
    (start_row..matrix.row_count()).find(|&r| {
        matrix.get(r, col).is_some_and(|e| !e.is_zero())
    })
}

/// Perform forward elimination on a single column.
///
/// Given a pivot at (`pivot_row`, `pivot_col`), eliminate all entries
/// below the pivot by subtracting the appropriate multiple of the
/// pivot row from each lower row.
fn eliminate_below(
    matrix: &GfMatrix,
    pivot_row: usize,
    pivot_col: usize,
) -> Result<GfMatrix, Error> {
    let pivot_val = matrix
        .get(pivot_row, pivot_col)
        .ok_or(Error::DimensionMismatch {
            expected: pivot_col,
            actual: matrix.col_count(),
        })?;

    // Scale pivot row so the pivot entry becomes 1
    let pivot_inv = pivot_val.inv()?;
    let scaled = matrix
        .with_row_scaled(pivot_row, pivot_inv)
        .ok_or(Error::DimensionMismatch {
            expected: pivot_row,
            actual: matrix.row_count(),
        })?;

    // Eliminate entries below the pivot
    ((pivot_row + 1)..scaled.row_count()).try_fold(scaled, |acc, r| {
        let entry = acc.get(r, pivot_col).unwrap_or(Gf256::zero());
        if entry.is_zero() {
            Ok(acc)
        } else {
            // In GF(2^8), subtraction = addition, so adding
            // entry * pivot_row zeros out position (r, pivot_col)
            acc.with_row_added(r, pivot_row, entry)
                .ok_or(Error::DimensionMismatch {
                    expected: r,
                    actual: acc.row_count(),
                })
        }
    })
}

/// Perform backward elimination on a single column.
///
/// Given a pivot at (`pivot_row`, `pivot_col`) with value 1, eliminate
/// all entries above the pivot.
fn eliminate_above(
    matrix: &GfMatrix,
    pivot_row: usize,
    pivot_col: usize,
) -> Result<GfMatrix, Error> {
    (0..pivot_row).try_fold(matrix.clone(), |acc, r| {
        let entry = acc.get(r, pivot_col).unwrap_or(Gf256::zero());
        if entry.is_zero() {
            Ok(acc)
        } else {
            acc.with_row_added(r, pivot_row, entry)
                .ok_or(Error::DimensionMismatch {
                    expected: r,
                    actual: acc.row_count(),
                })
        }
    })
}

/// Forward elimination: produce row echelon form.
///
/// Returns the matrix in row echelon form together with the pivot
/// column positions (one per pivot row, in order).
fn forward_eliminate(matrix: &GfMatrix) -> Result<(GfMatrix, Vec<usize>), Error> {
    let cols = matrix.col_count();
    let rows = matrix.row_count();

    // Fold over columns, tracking (matrix, pivots, current_pivot_row)
    let (result, pivots, _) = (0..cols).try_fold(
        (matrix.clone(), Vec::<usize>::new(), 0usize),
        |(mat, pivots, current_row), col| {
            if current_row >= rows {
                Ok((mat, pivots, current_row))
            } else {
                match find_pivot(&mat, col, current_row) {
                    None => Ok((mat, pivots, current_row)),
                    Some(pivot_row) => {
                        // Swap pivot row into position
                        let swapped = if pivot_row == current_row {
                            mat
                        } else {
                            mat.with_rows_swapped(current_row, pivot_row)
                                .ok_or(Error::DimensionMismatch {
                                    expected: pivot_row,
                                    actual: mat.row_count(),
                                })?
                        };

                        let eliminated = eliminate_below(&swapped, current_row, col)?;

                        let new_pivots: Vec<usize> = pivots
                            .into_iter()
                            .chain(core::iter::once(col))
                            .collect();

                        Ok((eliminated, new_pivots, current_row + 1))
                    }
                }
            }
        },
    )?;

    Ok((result, pivots))
}

/// Reduce a matrix to reduced row echelon form (RREF).
///
/// Returns the fully reduced matrix and the pivot column positions.
///
/// This is the composition of forward elimination (producing row
/// echelon form) followed by backward elimination (zeroing entries
/// above each pivot).
///
/// # Errors
///
/// Returns an error if a division by zero occurs during pivot
/// normalization (this should not happen for well-formed inputs
/// since we only normalize nonzero pivot entries).
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::vector::{GfVec, GfMatrix};
/// use rlnc_cat_rs::vector::gauss::reduced_row_echelon_form;
/// use rlnc_cat_rs::field::Gf256;
///
/// // 2x2 identity matrix is already in RREF
/// let m = GfMatrix::new(vec![
///     GfVec::from_bytes(&[1, 0]),
///     GfVec::from_bytes(&[0, 1]),
/// ]);
/// let (rref, pivots) = m.and_then(|m| reduced_row_echelon_form(&m))
///     .unwrap_or_else(|_| (GfMatrix::empty(0), vec![]));
/// assert_eq!(pivots, vec![0, 1]);
/// ```
pub fn reduced_row_echelon_form(matrix: &GfMatrix) -> Result<(GfMatrix, Vec<usize>), Error> {
    let (ref_matrix, pivots) = forward_eliminate(matrix)?;

    // Backward elimination: for each pivot (in reverse), eliminate above
    let result = pivots
        .iter()
        .enumerate()
        .rev()
        .try_fold(ref_matrix, |acc, (pivot_row, &pivot_col)| {
            eliminate_above(&acc, pivot_row, pivot_col)
        })?;

    Ok((result, pivots))
}

/// Count the rank of a matrix (number of linearly independent rows).
///
/// # Errors
///
/// Returns an error if Gaussian elimination fails.
pub fn rank(matrix: &GfMatrix) -> Result<usize, Error> {
    reduced_row_echelon_form(matrix).map(|(_, pivots)| pivots.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::GfVec;

    #[test]
    fn identity_is_already_rref() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 0]),
            GfVec::from_bytes(&[0, 1]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let (rref, pivots) =
            reduced_row_echelon_form(&m).unwrap_or((GfMatrix::empty(0), vec![]));
        assert_eq!(rref, m);
        assert_eq!(pivots, vec![0, 1]);
    }

    #[test]
    fn rref_swaps_rows_when_needed() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[0, 1]),
            GfVec::from_bytes(&[1, 0]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let (rref, pivots) =
            reduced_row_echelon_form(&m).unwrap_or((GfMatrix::empty(0), vec![]));
        assert_eq!(pivots, vec![0, 1]);
        assert_eq!(rref.get(0, 0), Some(Gf256::one()));
        assert_eq!(rref.get(0, 1), Some(Gf256::zero()));
        assert_eq!(rref.get(1, 0), Some(Gf256::zero()));
        assert_eq!(rref.get(1, 1), Some(Gf256::one()));
    }

    #[test]
    fn rref_eliminates_correctly() {
        // [2, 3]   ->   [1, 0]
        // [1, 1]        [0, 1]
        // (in GF(2^8))
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[2, 3]),
            GfVec::from_bytes(&[1, 1]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let (rref, pivots) =
            reduced_row_echelon_form(&m).unwrap_or((GfMatrix::empty(0), vec![]));
        assert_eq!(pivots.len(), 2);
        assert_eq!(rref.get(0, 0), Some(Gf256::one()));
        assert_eq!(rref.get(1, 1), Some(Gf256::one()));
        assert_eq!(rref.get(0, 1), Some(Gf256::zero()));
        assert_eq!(rref.get(1, 0), Some(Gf256::zero()));
    }

    #[test]
    fn rank_of_identity() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 0, 0]),
            GfVec::from_bytes(&[0, 1, 0]),
            GfVec::from_bytes(&[0, 0, 1]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        assert_eq!(rank(&m).unwrap_or(0), 3);
    }

    #[test]
    fn rank_of_dependent_rows() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 2]),
            GfVec::from_bytes(&[2, 4]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        assert_eq!(rank(&m).unwrap_or(0), 1);
    }

    #[test]
    fn rref_is_idempotent() {
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[5, 3, 7]),
            GfVec::from_bytes(&[2, 8, 1]),
            GfVec::from_bytes(&[9, 4, 6]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let (rref1, pivots1) =
            reduced_row_echelon_form(&m).unwrap_or((GfMatrix::empty(0), vec![]));
        let (rref2, pivots2) =
            reduced_row_echelon_form(&rref1).unwrap_or((GfMatrix::empty(0), vec![]));
        assert_eq!(rref1, rref2);
        assert_eq!(pivots1, pivots2);
    }

    #[test]
    fn rref_augmented_matrix() {
        // Solve x + 2y = 5, 3x + 4y = 6 over GF(2^8)
        let m = GfMatrix::new(vec![
            GfVec::from_bytes(&[1, 2, 5]),
            GfVec::from_bytes(&[3, 4, 6]),
        ])
        .unwrap_or(GfMatrix::empty(0));
        let (rref, pivots) =
            reduced_row_echelon_form(&m).unwrap_or((GfMatrix::empty(0), vec![]));
        assert_eq!(pivots, vec![0, 1]);
        assert_eq!(rref.get(0, 0), Some(Gf256::one()));
        assert_eq!(rref.get(0, 1), Some(Gf256::zero()));
        assert_eq!(rref.get(1, 0), Some(Gf256::zero()));
        assert_eq!(rref.get(1, 1), Some(Gf256::one()));
        let x = rref.get(0, 2).unwrap_or(Gf256::zero());
        let y = rref.get(1, 2).unwrap_or(Gf256::zero());
        assert_eq!(x + Gf256::new(2) * y, Gf256::new(5));
        assert_eq!(Gf256::new(3) * x + Gf256::new(4) * y, Gf256::new(6));
    }
}
