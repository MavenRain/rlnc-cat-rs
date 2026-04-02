//! Decoding: reconstruct original data from coded pieces.
//!
//! [`DecoderState`] absorbs coded pieces one at a time, performing
//! partial forward elimination as each arrives.  When enough linearly
//! independent pieces have been received, [`DecoderState::decode`]
//! completes the backward elimination and extracts the original data.
//!
//! [`decode_stream`] integrates with comp-cat-rs's [`Stream`] by
//! folding a stream of coded pieces through the decoder.

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;
use crate::vector::gauss::reduced_row_echelon_form;
use crate::vector::{GfMatrix, GfVec};

/// The state of a decoder accumulating coded pieces.
///
/// Each call to [`absorb`](DecoderState::absorb) adds one coded piece
/// as a new row in an augmented matrix `[coding_vectors | data]` and
/// performs partial forward elimination to maintain row echelon form.
///
/// When [`is_complete`](DecoderState::is_complete) returns true,
/// calling [`decode`](DecoderState::decode) finishes the backward
/// elimination and reconstructs the original data.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::coding::piece::{CodingVector, CodedPiece, OriginalData};
/// use rlnc_cat_rs::coding::encode::encode_with_vector;
/// use rlnc_cat_rs::coding::decode::DecoderState;
/// use rlnc_cat_rs::vector::GfVec;
///
/// let data = vec![10, 20, 30, 40];
/// let orig = OriginalData::from_bytes(&data, 2).ok();
/// let decoder = orig.as_ref().map(|o| {
///     // Encode with identity vectors to get the pieces directly
///     let p0 = encode_with_vector(o, CodingVector::from_bytes(&[1, 0])).ok();
///     let p1 = encode_with_vector(o, CodingVector::from_bytes(&[0, 1])).ok();
///     let state = DecoderState::new(o.piece_count(), o.piece_byte_len());
///     let state = p0.and_then(|p| state.absorb(&p).ok());
///     let state = state.and_then(|s| p1.and_then(|p| s.absorb(&p).ok()));
///     state.filter(|s| s.is_complete()).and_then(|s| s.decode().ok())
/// });
/// let decoded = decoder.flatten();
/// assert_eq!(decoded, Some(data));
/// ```
#[must_use]
pub struct DecoderState {
    matrix: GfMatrix,
    piece_count: usize,
    piece_byte_len: usize,
    useful_count: usize,
}

impl DecoderState {
    /// Create an initial empty decoder state.
    ///
    /// `piece_count` is the number of original pieces (k), and
    /// `piece_byte_len` is the byte length of each piece.
    pub fn new(piece_count: usize, piece_byte_len: usize) -> Self {
        Self {
            matrix: GfMatrix::empty(piece_count + piece_byte_len),
            piece_count,
            piece_byte_len,
            useful_count: 0,
        }
    }

    /// The number of original pieces expected.
    #[must_use]
    pub fn piece_count(&self) -> usize {
        self.piece_count
    }

    /// The byte length of each piece.
    #[must_use]
    pub fn piece_byte_len(&self) -> usize {
        self.piece_byte_len
    }

    /// The number of linearly independent pieces absorbed so far.
    #[must_use]
    pub fn useful_count(&self) -> usize {
        self.useful_count
    }

    /// The number of pieces still needed to decode.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.piece_count.saturating_sub(self.useful_count)
    }

    /// Whether enough linearly independent pieces have been received.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.useful_count >= self.piece_count
    }

    /// Absorb one coded piece, returning the updated state.
    ///
    /// The piece is appended as a new row in the augmented matrix.
    /// Forward elimination is performed via RREF.  If the piece is
    /// linearly dependent on previously received pieces, the rank
    /// does not increase, but the operation still succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if the piece dimensions are incompatible.
    pub fn absorb(self, piece: &CodedPiece) -> Result<Self, Error> {
        if piece.coding_vector().len() != self.piece_count {
            return Err(Error::CodingVectorLengthMismatch {
                expected: self.piece_count,
                actual: piece.coding_vector().len(),
            });
        }
        if piece.data().len() != self.piece_byte_len {
            return Err(Error::PieceSizeMismatch {
                expected: self.piece_byte_len,
                actual: piece.data().len(),
            });
        }
        // Build the augmented row: [coding_vector | data]
        let row_bytes: Vec<u8> = piece
            .coding_vector()
            .as_vec()
            .to_bytes()
            .into_iter()
            .chain(piece.data().to_bytes())
            .collect();
        let row = GfVec::from_bytes(&row_bytes);
        let new_matrix = self.matrix.append_row(row)?;

        // Run RREF to determine the new rank
        let (rref_matrix, pivots) = reduced_row_echelon_form(&new_matrix)?;

        // Count pivots in the coding vector columns (0..piece_count)
        let useful = pivots
            .iter()
            .filter(|&&col| col < self.piece_count)
            .count();

        Ok(Self {
            matrix: rref_matrix,
            piece_count: self.piece_count,
            piece_byte_len: self.piece_byte_len,
            useful_count: useful,
        })
    }

    /// Finish decoding and extract the original data.
    ///
    /// The matrix should already be in RREF from incremental
    /// absorption.  This method splits off the data columns and
    /// reconstructs the original bytes, stripping the boundary
    /// marker padding.
    ///
    /// # Errors
    ///
    /// Returns `Error::InsufficientPieces` if not enough linearly
    /// independent pieces have been received, or `Error::InvalidPadding`
    /// if the boundary marker cannot be found.
    pub fn decode(self) -> Result<Vec<u8>, Error> {
        if !self.is_complete() {
            return Err(Error::InsufficientPieces {
                needed: self.piece_count,
                received: self.useful_count,
            });
        }
        // The matrix is already in RREF from absorb.
        // Split off the data columns.
        let (_, data_matrix) = self.matrix.split_at_col(self.piece_count)?;

        // Extract the first piece_count rows (the pivot rows)
        let decoded_pieces: Vec<GfVec> = (0..self.piece_count)
            .filter_map(|i| data_matrix.row(i).cloned())
            .collect();

        OriginalData::reconstruct(&decoded_pieces)
    }
}

/// Decode original data from a stream of coded pieces.
///
/// Folds the stream through [`DecoderState::absorb`], stopping
/// early when the decoder has received enough linearly independent
/// pieces, then finalizes with [`DecoderState::decode`].
///
/// # Errors
///
/// The returned `Io` fails with `Error::InsufficientPieces` if
/// the stream ends before enough useful pieces arrive.
#[must_use]
pub fn decode_stream(
    piece_count: usize,
    piece_byte_len: usize,
    pieces: Stream<Error, CodedPiece>,
) -> Io<Error, Vec<u8>> {
    let init = DecoderState::new(piece_count, piece_byte_len);
    let pc = piece_count;
    let pbl = piece_byte_len;
    pieces
        .fold(
            init,
            Arc::new(move |state, piece| {
                if state.is_complete() {
                    state
                } else {
                    state.absorb(&piece).unwrap_or_else(|_| {
                        DecoderState::new(pc, pbl)
                    })
                }
            }),
        )
        .flat_map(|state| Io::suspend(move || state.decode()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::encode::encode_with_vector;
    use crate::coding::piece::{CodingVector, OriginalData};

    fn make_test_data() -> (OriginalData, Vec<u8>) {
        let data = vec![10, 20, 30, 40, 50, 60];
        let orig = OriginalData::from_bytes(&data, 3)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        (orig, data)
    }

    #[test]
    fn decode_with_identity_vectors() {
        let (orig, data) = make_test_data();
        let k = orig.piece_count();

        // Encode with standard basis vectors
        let pieces: Vec<CodedPiece> = (0..k)
            .map(|i| {
                let cv_bytes: Vec<u8> = (0..k)
                    .map(|j| if j == i { 1u8 } else { 0u8 })
                    .collect();
                encode_with_vector(&orig, CodingVector::from_bytes(&cv_bytes))
                    .unwrap_or_else(|_| CodedPiece::new(
                        CodingVector::from_bytes(&[]),
                        GfVec::zeros(0),
                    ))
            })
            .collect();

        // Feed to decoder
        let state = pieces.iter().try_fold(
            DecoderState::new(orig.piece_count(), orig.piece_byte_len()),
            |s, p| s.absorb(p),
        );
        assert!(state.is_ok());
        let state = state.unwrap_or_else(|_| DecoderState::new(0, 0));
        assert!(state.is_complete());
        let decoded = state.decode();
        assert!(decoded.is_ok());
        assert_eq!(decoded.unwrap_or_default(), data);
    }

    #[test]
    fn decode_with_random_vectors() {
        let (orig, data) = make_test_data();
        let k = orig.piece_count();

        // Encode with non-trivial coding vectors
        let cvs: Vec<Vec<u8>> = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];
        let pieces: Vec<CodedPiece> = cvs[..k]
            .iter()
            .map(|cv| {
                encode_with_vector(&orig, CodingVector::from_bytes(cv))
                    .unwrap_or_else(|_| CodedPiece::new(
                        CodingVector::from_bytes(&[]),
                        GfVec::zeros(0),
                    ))
            })
            .collect();

        let state = pieces.iter().try_fold(
            DecoderState::new(orig.piece_count(), orig.piece_byte_len()),
            |s, p| s.absorb(p),
        );
        assert!(state.is_ok());
        let state = state.unwrap_or_else(|_| DecoderState::new(0, 0));
        assert!(state.is_complete());
        let decoded = state.decode();
        assert!(decoded.is_ok());
        assert_eq!(decoded.unwrap_or_default(), data);
    }

    #[test]
    fn decode_incomplete_fails() {
        let (orig, _data) = make_test_data();
        let state = DecoderState::new(orig.piece_count(), orig.piece_byte_len());
        assert!(!state.is_complete());
        assert!(state.decode().is_err());
    }

    #[test]
    fn decode_stream_roundtrip() {
        let (orig, data) = make_test_data();
        let k = orig.piece_count();

        let cvs: Vec<Vec<u8>> = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ];
        let pieces: Vec<CodedPiece> = cvs[..k]
            .iter()
            .map(|cv| {
                encode_with_vector(&orig, CodingVector::from_bytes(cv))
                    .unwrap_or_else(|_| CodedPiece::new(
                        CodingVector::from_bytes(&[]),
                        GfVec::zeros(0),
                    ))
            })
            .collect();

        let stream = Stream::from_vec(pieces);
        let result = decode_stream(
            orig.piece_count(),
            orig.piece_byte_len(),
            stream,
        )
        .run();
        assert!(result.is_ok());
        assert_eq!(result.unwrap_or_default(), data);
    }

    #[test]
    fn linearly_dependent_piece_does_not_increase_rank() {
        let (orig, _data) = make_test_data();

        let p1 = encode_with_vector(&orig, CodingVector::from_bytes(&[1, 0, 0]));
        // p2 is 2 * p1 in GF(2^8), so linearly dependent
        let p2 = encode_with_vector(&orig, CodingVector::from_bytes(&[2, 0, 0]));

        let state = DecoderState::new(orig.piece_count(), orig.piece_byte_len());
        let state = p1
            .as_ref()
            .map_err(|_| Error::EmptyData)
            .and_then(|p| state.absorb(p))
            .unwrap_or_else(|_| DecoderState::new(0, 0));
        assert_eq!(state.useful_count(), 1);

        let state = p2
            .as_ref()
            .map_err(|_| Error::EmptyData)
            .and_then(|p| state.absorb(p))
            .unwrap_or_else(|_| DecoderState::new(0, 0));
        // Rank should still be 1
        assert_eq!(state.useful_count(), 1);
    }
}
