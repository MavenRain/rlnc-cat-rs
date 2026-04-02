//! Encoding: generate coded pieces from original data.
//!
//! The pure core is [`encode_with_vector`], which computes a single
//! coded piece given explicit coefficients.  [`Encoder`] wraps this
//! with [`Io`] and [`Stream`] for random coefficient generation.

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::coding::piece::{CodedPiece, CodingVector, OriginalData};
use crate::error::Error;
use crate::field::Gf256;
use crate::vector::GfVec;

/// Compute a single coded piece from explicit coding coefficients.
///
/// The coded data is the linear combination
/// `sum(coding_vector[i] * original.piece(i))` over all pieces.
///
/// This is a pure function with no side effects.
///
/// # Errors
///
/// Returns an error if the coding vector length does not match
/// the number of original pieces.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
/// use rlnc_cat_rs::coding::encode::encode_with_vector;
///
/// let data = vec![10, 20, 30, 40];
/// let orig = OriginalData::from_bytes(&data, 2).ok();
/// let cv = orig.as_ref().map(|o| {
///     CodingVector::from_bytes(&vec![1; o.piece_count()])
/// });
/// let piece = orig.as_ref().zip(cv.as_ref()).and_then(|(o, c)| {
///     encode_with_vector(o, c.clone()).ok()
/// });
/// assert!(piece.is_some());
/// ```
#[must_use = "this returns the coded piece without side effects"]
pub fn encode_with_vector(
    original: &OriginalData,
    coding_vector: CodingVector,
) -> Result<CodedPiece, Error> {
    if coding_vector.len() == original.piece_count() {
        let coeffs: Vec<Gf256> = coding_vector.as_vec().elements().to_vec();
        let piece_refs: Vec<&GfVec> = original.pieces().iter().collect();
        GfVec::linear_combine(&coeffs, &piece_refs)
            .map(|data| CodedPiece::new(coding_vector, data))
    } else {
        Err(Error::CodingVectorLengthMismatch {
            expected: original.piece_count(),
            actual: coding_vector.len(),
        })
    }
}

/// An encoder that generates coded pieces from original data.
///
/// The encoder holds the original data and provides methods to
/// generate coded pieces using injected randomness.  The pure
/// encoding logic lives in [`encode_with_vector`]; the encoder
/// wraps it with effect types.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::coding::piece::OriginalData;
/// use rlnc_cat_rs::coding::encode::Encoder;
///
/// let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
/// let orig = OriginalData::from_bytes(&data, 4).ok();
/// let encoder = orig.map(Encoder::new);
/// assert!(encoder.is_some());
/// ```
#[must_use]
pub struct Encoder {
    original: OriginalData,
}

impl Encoder {
    /// Create an encoder from the original data.
    pub fn new(original: OriginalData) -> Self {
        Self { original }
    }

    /// The underlying original data.
    pub fn original(&self) -> &OriginalData {
        &self.original
    }

    /// Generate a single coded piece.
    ///
    /// The `rng` closure receives the number of random bytes needed
    /// (equal to `piece_count`) and must return that many random bytes.
    /// It is called inside an [`Io::suspend`], so the side effect
    /// is deferred until the returned `Io` is run.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlnc_cat_rs::coding::piece::OriginalData;
    /// use rlnc_cat_rs::coding::encode::Encoder;
    ///
    /// let orig = OriginalData::from_bytes(&[1, 2, 3, 4], 2).ok();
    /// let encoder = orig.map(Encoder::new);
    /// // Deterministic "random" source for testing
    /// let io = encoder.map(|enc| {
    ///     enc.encode_one(|n| Ok(vec![42u8; n]))
    /// });
    /// let piece = io.and_then(|io| io.run().ok());
    /// assert!(piece.is_some());
    /// ```
    #[must_use]
    pub fn encode_one(
        &self,
        rng: impl FnOnce(usize) -> Result<Vec<u8>, Error> + Send + 'static,
    ) -> Io<Error, CodedPiece> {
        let original = self.original.clone();
        Io::suspend(move || {
            let random_bytes = rng(original.piece_count())?;
            let cv = CodingVector::from_bytes(&random_bytes);
            encode_with_vector(&original, cv)
        })
    }

    /// Generate an unbounded stream of coded pieces.
    ///
    /// The `rng_factory` closure is called once per piece to produce
    /// the random bytes for that piece's coding vector.  The stream
    /// is infinite; use `.take(n)` to bound it.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlnc_cat_rs::coding::piece::OriginalData;
    /// use rlnc_cat_rs::coding::encode::Encoder;
    ///
    /// let orig = OriginalData::from_bytes(&[1, 2, 3, 4], 2).ok();
    /// let encoder = orig.map(Encoder::new);
    /// let stream = encoder.map(|enc| {
    ///     enc.encode_stream(|n| Ok(vec![42u8; n]))
    ///         .take(3)
    /// });
    /// let pieces = stream.map(|s| s.collect().run());
    /// let pieces = pieces.and_then(|r| r.ok());
    /// assert_eq!(pieces.map(|p| p.len()), Some(3));
    /// ```
    #[must_use]
    pub fn encode_stream<F>(
        &self,
        rng_factory: F,
    ) -> Stream<Error, CodedPiece>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static,
    {
        let original = self.original.clone();
        let piece_count = original.piece_count();
        let rng_factory = Arc::new(rng_factory);
        Stream::unfold(
            (),
            Arc::new(move |()| {
                let orig = original.clone();
                let rng = Arc::clone(&rng_factory);
                Io::suspend(move || {
                    let random_bytes = rng(piece_count)?;
                    let cv = CodingVector::from_bytes(&random_bytes);
                    encode_with_vector(&orig, cv).map(|piece| Some((piece, ())))
                })
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_rng(seed: u8) -> impl FnOnce(usize) -> Result<Vec<u8>, Error> {
        move |n| {
            Ok((0..n)
                .map(|i| seed.wrapping_add(i as u8))
                .collect())
        }
    }

    #[test]
    fn encode_with_identity_vector() {
        // Coding vector [1, 0] selects the first piece
        let data = vec![10, 20, 30, 40];
        let orig = OriginalData::from_bytes(&data, 2)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let cv = CodingVector::from_bytes(&[1, 0]);
        let piece = encode_with_vector(&orig, cv);
        assert!(piece.is_ok());
        let piece = piece.unwrap_or_else(|_| CodedPiece::new(
            CodingVector::from_bytes(&[]),
            GfVec::zeros(0),
        ));
        // The coded data should be exactly the first piece
        assert_eq!(piece.data(), orig.piece(0).unwrap_or(&GfVec::zeros(0)));
    }

    #[test]
    fn encode_with_wrong_vector_length_fails() {
        let data = vec![1, 2, 3, 4];
        let orig = OriginalData::from_bytes(&data, 2)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let cv = CodingVector::from_bytes(&[1, 2, 3]); // wrong length
        assert!(encode_with_vector(&orig, cv).is_err());
    }

    #[test]
    fn encoder_io_produces_piece() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let orig = OriginalData::from_bytes(&data, 3)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let encoder = Encoder::new(orig);
        let io = encoder.encode_one(deterministic_rng(1));
        let result = io.run();
        assert!(result.is_ok());
    }

    #[test]
    fn encoder_stream_produces_correct_count() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let orig = OriginalData::from_bytes(&data, 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let encoder = Encoder::new(orig);
        let stream = encoder.encode_stream(|n| {
            Ok((0..n).map(|i| (i as u8).wrapping_add(1)).collect())
        });
        let pieces = stream.take(5).collect().run();
        assert!(pieces.is_ok());
        assert_eq!(pieces.unwrap_or_default().len(), 5);
    }
}
