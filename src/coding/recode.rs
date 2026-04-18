//! Recoding: generate new coded pieces from existing coded pieces.
//!
//! A [`Recoder`] stores received coded pieces and produces new
//! random linear combinations of them, without decoding back to
//! the original data.  This enables multi-hop distribution in
//! networks where intermediate nodes do not need the original data.

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::coding::piece::{CodedPiece, CodingVector};
use crate::error::Error;
use crate::field::Gf256;
use crate::vector::GfVec;

/// A recoder that produces new coded pieces from existing ones.
///
/// Stores a collection of received coded pieces and creates new
/// random linear combinations of them.  The new piece's coding
/// vector is the corresponding linear combination of the stored
/// coding vectors, preserving the algebraic relationship to the
/// original data.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::coding::piece::{CodingVector, CodedPiece, OriginalData};
/// use rlnc_cat_rs::coding::encode::encode_with_vector;
/// use rlnc_cat_rs::coding::recode::Recoder;
/// use rlnc_cat_rs::vector::GfVec;
///
/// let data = vec![10, 20, 30, 40];
/// let orig = OriginalData::from_bytes(&data, 2).ok();
/// let pieces = orig.as_ref().map(|o| vec![
///     encode_with_vector(o, CodingVector::from_bytes(&[1, 0])).ok(),
///     encode_with_vector(o, CodingVector::from_bytes(&[0, 1])).ok(),
/// ]);
/// let recoder = pieces.and_then(|ps| {
///     ps.into_iter()
///       .try_fold(Recoder::new(2, 3), |r, p| {
///           p.ok_or(rlnc_cat_rs::error::Error::EmptyData)
///             .and_then(|p| r.add_piece(&p))
///       })
///       .ok()
/// });
/// let recoded = recoder.and_then(|r| {
///     r.recode_one(|n| Ok(vec![3u8; n])).run().ok()
/// });
/// assert!(recoded.is_some());
/// ```
#[derive(Clone)]
#[must_use]
pub struct Recoder {
    pieces: Vec<CodedPiece>,
    piece_count: usize,
    piece_byte_len: usize,
}

impl Recoder {
    /// Create an empty recoder expecting pieces for `piece_count`
    /// original data pieces, each of `piece_byte_len` bytes.
    pub fn new(piece_count: usize, piece_byte_len: usize) -> Self {
        Self {
            pieces: Vec::new(),
            piece_count,
            piece_byte_len,
        }
    }

    /// Add a coded piece to the recoder's pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the piece dimensions are incompatible.
    pub fn add_piece(self, piece: &CodedPiece) -> Result<Self, Error> {
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
        let new_pieces: Vec<CodedPiece> = self
            .pieces
            .into_iter()
            .chain(core::iter::once(piece.clone()))
            .collect();
        Ok(Self {
            pieces: new_pieces,
            piece_count: self.piece_count,
            piece_byte_len: self.piece_byte_len,
        })
    }

    /// The number of stored coded pieces.
    #[must_use]
    pub fn stored_count(&self) -> usize {
        self.pieces.len()
    }

    /// The expected piece count (k).
    #[must_use]
    pub fn piece_count(&self) -> usize {
        self.piece_count
    }

    /// The expected piece byte length.
    #[must_use]
    pub fn piece_byte_len(&self) -> usize {
        self.piece_byte_len
    }

    /// Generate a single recoded piece.
    ///
    /// The `rng` closure receives the number of random bytes needed
    /// (equal to `stored_count()`) and must return that many random
    /// bytes.  These are the coefficients for the linear combination
    /// of stored pieces.
    ///
    /// # Errors
    ///
    /// The returned `Io` fails if no pieces are stored or if the
    /// random source fails.
    #[must_use]
    pub fn recode_one(
        &self,
        rng: impl FnOnce(usize) -> Result<Vec<u8>, Error> + Send + 'static,
    ) -> Io<Error, CodedPiece> {
        let pieces = self.pieces.clone();
        Io::suspend(move || recode_from(&pieces, rng))
    }

    /// Generate an unbounded stream of recoded pieces.
    ///
    /// The `rng_factory` closure is called once per recoded piece.
    /// The stream is infinite; use `.take(n)` to bound it.
    #[must_use]
    pub fn recode_stream<F>(
        &self,
        rng_factory: F,
    ) -> Stream<Error, CodedPiece>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static,
    {
        let pieces = self.pieces.clone();
        let rng_factory = Arc::new(rng_factory);
        Stream::unfold(
            (),
            Arc::new(move |()| {
                let ps = pieces.clone();
                let rng = Arc::clone(&rng_factory);
                Io::suspend(move || {
                    if ps.is_empty() {
                        Ok(None)
                    } else {
                        recode_from(&ps, |n| rng(n)).map(|piece| Some((piece, ())))
                    }
                })
            }),
        )
    }
}

/// Core recoding logic: produce one recoded piece from a set of
/// stored pieces using the provided random bytes.
fn recode_from(
    pieces: &[CodedPiece],
    rng: impl FnOnce(usize) -> Result<Vec<u8>, Error>,
) -> Result<CodedPiece, Error> {
    if pieces.is_empty() {
        return Err(Error::InsufficientPieces {
            needed: 1,
            received: 0,
        });
    }
    let random_bytes = rng(pieces.len())?;
    let coeffs: Vec<Gf256> = random_bytes.iter().copied().map(Gf256::new).collect();

    // Linear combination of coding vectors
    let cv_refs: Vec<&GfVec> = pieces
        .iter()
        .map(|p| p.coding_vector().as_vec())
        .collect();
    let new_cv = GfVec::linear_combine(&coeffs, &cv_refs)?;

    // Linear combination of data vectors
    let data_refs: Vec<&GfVec> = pieces.iter().map(CodedPiece::data).collect();
    let new_data = GfVec::linear_combine(&coeffs, &data_refs)?;

    Ok(CodedPiece::new(CodingVector::new(new_cv), new_data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::decode::DecoderState;
    use crate::coding::encode::encode_with_vector;
    use crate::coding::piece::OriginalData;

    #[test]
    fn recode_produces_valid_piece() {
        let data = vec![10, 20, 30, 40, 50, 60];
        let orig = OriginalData::from_bytes(&data, 3)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();

        // Encode with identity vectors
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

        // Add to recoder
        let recoder = pieces.iter().try_fold(
            Recoder::new(orig.piece_count(), orig.piece_byte_len()),
            |r, p| r.add_piece(p),
        );
        assert!(recoder.is_ok());
        let recoder = recoder.unwrap_or_else(|_| Recoder::new(0, 0));

        // Recode
        let recoded = recoder.recode_one(|n| Ok(vec![1u8; n])).run();
        assert!(recoded.is_ok());
        let recoded = recoded.unwrap_or_else(|_| CodedPiece::new(
            CodingVector::from_bytes(&[]),
            GfVec::zeros(0),
        ));

        // The recoded piece should have the right dimensions
        assert_eq!(recoded.coding_vector().len(), orig.piece_count());
        assert_eq!(recoded.data().len(), orig.piece_byte_len());
    }

    #[test]
    fn recode_then_decode_roundtrip() {
        let data = vec![10, 20, 30, 40];
        let orig = OriginalData::from_bytes(&data, 2)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();

        // Encode with identity vectors
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

        // Recode: produce k new pieces from the k originals
        let recoder = pieces.iter().try_fold(
            Recoder::new(orig.piece_count(), orig.piece_byte_len()),
            |r, p| r.add_piece(p),
        )
        .unwrap_or_else(|_| Recoder::new(0, 0));

        let recoded_pieces: Vec<CodedPiece> = (0..k)
            .map(|i| {
                let seed = (i as u8).wrapping_add(1);
                recoder
                    .recode_one(move |n| {
                        Ok((0..n).map(|j| seed.wrapping_add(j as u8)).collect())
                    })
                    .run()
                    .unwrap_or_else(|_| CodedPiece::new(
                        CodingVector::from_bytes(&[]),
                        GfVec::zeros(0),
                    ))
            })
            .collect();

        // Decode the recoded pieces
        let state = recoded_pieces.iter().try_fold(
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
    fn recode_empty_recoder_fails() {
        let recoder = Recoder::new(3, 10);
        let result = recoder.recode_one(|n| Ok(vec![1u8; n])).run();
        assert!(result.is_err());
    }
}
