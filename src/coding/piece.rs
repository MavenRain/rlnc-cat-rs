//! Coded pieces and original data representation.
//!
//! A [`CodedPiece`] pairs a [`CodingVector`] (the GF(2^8) coefficients
//! describing how this piece was formed) with the coded data payload.
//! [`OriginalData`] splits raw bytes into fixed-size pieces with
//! padding for the encoder.

use crate::error::Error;
use crate::vector::GfVec;

/// Boundary marker appended to the original data before zero-padding.
///
/// The decoder scans backwards from the end to find this marker and
/// strip the padding.
const BOUNDARY_MARKER: u8 = 0x81;

/// A coding vector: the GF(2^8) coefficients that describe how
/// a coded piece was formed as a linear combination of original pieces.
///
/// For an encoding of k original pieces, the coding vector has k
/// elements.  Element i is the coefficient of original piece i.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct CodingVector {
    inner: GfVec,
}

impl CodingVector {
    /// Create a coding vector from a [`GfVec`].
    pub fn new(inner: GfVec) -> Self {
        Self { inner }
    }

    /// Create a coding vector from raw bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            inner: GfVec::from_bytes(data),
        }
    }

    /// The number of coefficients (equal to the piece count).
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether this coding vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// A borrowed reference to the underlying vector.
    pub fn as_vec(&self) -> &GfVec {
        &self.inner
    }

    /// Consume into the underlying vector.
    pub fn into_vec(self) -> GfVec {
        self.inner
    }
}

/// A coded piece: a coding vector paired with the coded data.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::coding::piece::{CodingVector, CodedPiece};
/// use rlnc_cat_rs::vector::GfVec;
///
/// let cv = CodingVector::from_bytes(&[1, 0, 0]);
/// let data = GfVec::from_bytes(&[10, 20, 30, 40]);
/// let piece = CodedPiece::new(cv, data);
/// assert_eq!(piece.coding_vector().len(), 3);
/// assert_eq!(piece.data().len(), 4);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct CodedPiece {
    coding_vector: CodingVector,
    data: GfVec,
}

impl CodedPiece {
    /// Create a coded piece from a coding vector and data.
    pub fn new(coding_vector: CodingVector, data: GfVec) -> Self {
        Self {
            coding_vector,
            data,
        }
    }

    /// The coding vector (coefficients).
    pub fn coding_vector(&self) -> &CodingVector {
        &self.coding_vector
    }

    /// The coded data payload.
    pub fn data(&self) -> &GfVec {
        &self.data
    }

    /// Serialize to bytes: coding vector bytes followed by data bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.coding_vector
            .as_vec()
            .to_bytes()
            .into_iter()
            .chain(self.data.to_bytes())
            .collect()
    }

    /// Deserialize from bytes, given the piece count (coding vector length).
    ///
    /// The first `piece_count` bytes are the coding vector; the rest
    /// is the data payload.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte slice is shorter than `piece_count`.
    pub fn from_bytes(bytes: &[u8], piece_count: usize) -> Result<Self, Error> {
        if bytes.len() < piece_count {
            Err(Error::PieceSizeMismatch {
                expected: piece_count,
                actual: bytes.len(),
            })
        } else {
            let (cv_bytes, data_bytes) = bytes.split_at(piece_count);
            Ok(Self {
                coding_vector: CodingVector::from_bytes(cv_bytes),
                data: GfVec::from_bytes(data_bytes),
            })
        }
    }
}

/// The original data split into k pieces, padded for encoding.
///
/// Padding adds a boundary marker (0x81) followed by zeros so that
/// the total data length is divisible by the piece count.  The
/// decoder uses the marker to find and strip the padding.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::coding::piece::OriginalData;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let orig = OriginalData::from_bytes(&data, 2);
/// assert!(orig.is_ok());
/// ```
#[derive(Clone, Debug)]
#[must_use]
pub struct OriginalData {
    pieces: Vec<GfVec>,
    piece_count: usize,
    piece_byte_len: usize,
    original_len: usize,
}

impl OriginalData {
    /// Split raw data into `piece_count` equal-sized pieces.
    ///
    /// Appends a 0x81 boundary marker and zero-pads so that the
    /// total length is divisible by `piece_count`.
    ///
    /// # Errors
    ///
    /// Returns `Error::EmptyData` if `data` is empty, or
    /// `Error::InvalidPieceCount` if `piece_count` is zero.
    pub fn from_bytes(data: &[u8], piece_count: usize) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::EmptyData);
        }
        if piece_count == 0 {
            return Err(Error::InvalidPieceCount);
        }
        let original_len = data.len();
        // Append boundary marker + enough zeros
        let with_marker: Vec<u8> = data
            .iter()
            .copied()
            .chain(core::iter::once(BOUNDARY_MARKER))
            .collect();
        let remainder = with_marker.len() % piece_count;
        let padding_needed = if remainder == 0 { 0 } else { piece_count - remainder };
        let padded: Vec<u8> = with_marker
            .into_iter()
            .chain(core::iter::repeat_n(0u8, padding_needed))
            .collect();
        let piece_byte_len = padded.len() / piece_count;
        let pieces: Vec<GfVec> = padded
            .chunks(piece_byte_len)
            .map(GfVec::from_bytes)
            .collect();
        Ok(Self {
            pieces,
            piece_count,
            piece_byte_len,
            original_len,
        })
    }

    /// The number of pieces.
    #[must_use]
    pub fn piece_count(&self) -> usize {
        self.piece_count
    }

    /// The byte length of each piece.
    #[must_use]
    pub fn piece_byte_len(&self) -> usize {
        self.piece_byte_len
    }

    /// The original (unpadded) data length.
    #[must_use]
    pub fn original_len(&self) -> usize {
        self.original_len
    }

    /// A borrowed reference to the piece at the given index.
    #[must_use]
    pub fn piece(&self, index: usize) -> Option<&GfVec> {
        self.pieces.get(index)
    }

    /// A borrowed slice of all pieces.
    pub fn pieces(&self) -> &[GfVec] {
        &self.pieces
    }

    /// Reconstruct the original data from decoded pieces.
    ///
    /// Concatenates the pieces, locates the boundary marker, and
    /// returns the bytes before it.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidPadding` if the boundary marker is
    /// not found or the padding is malformed.
    pub fn reconstruct(pieces: &[GfVec]) -> Result<Vec<u8>, Error> {
        let concatenated: Vec<u8> = pieces.iter().flat_map(GfVec::to_bytes).collect();
        // Scan backwards for the boundary marker
        concatenated
            .iter()
            .rposition(|&b| b == BOUNDARY_MARKER)
            .filter(|&pos| {
                // Everything after the marker must be zero
                concatenated.get(pos + 1..).unwrap_or(&[]).iter().all(|&b| b == 0)
            })
            .map(|pos| concatenated.get(..pos).unwrap_or(&[]).to_vec())
            .ok_or(Error::InvalidPadding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_data_empty_fails() {
        assert!(OriginalData::from_bytes(&[], 3).is_err());
    }

    #[test]
    fn original_data_zero_pieces_fails() {
        assert!(OriginalData::from_bytes(&[1, 2, 3], 0).is_err());
    }

    #[test]
    fn original_data_splits_correctly() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let orig = OriginalData::from_bytes(&data, 2);
        assert!(orig.is_ok());
        let orig = orig.unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        assert_eq!(orig.piece_count(), 2);
        assert!(orig.piece(0).is_some());
        assert!(orig.piece(1).is_some());
        assert_eq!(
            orig.piece(0).map(GfVec::len),
            orig.piece(1).map(GfVec::len)
        );
    }

    #[test]
    fn reconstruct_roundtrip() {
        let data = vec![10, 20, 30, 40, 50];
        let orig = OriginalData::from_bytes(&data, 3);
        assert!(orig.is_ok());
        let orig = orig.unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let reconstructed = OriginalData::reconstruct(orig.pieces());
        assert!(reconstructed.is_ok());
        assert_eq!(reconstructed.unwrap_or_default(), data);
    }

    #[test]
    fn reconstruct_single_piece() {
        let data = vec![42];
        let orig = OriginalData::from_bytes(&data, 1);
        assert!(orig.is_ok());
        let orig = orig.unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let reconstructed = OriginalData::reconstruct(orig.pieces());
        assert_eq!(reconstructed.unwrap_or_default(), data);
    }

    #[test]
    fn coded_piece_serialization_roundtrip() {
        let cv = CodingVector::from_bytes(&[1, 2, 3]);
        let data = GfVec::from_bytes(&[10, 20, 30, 40]);
        let piece = CodedPiece::new(cv, data);
        let bytes = piece.to_bytes();
        let restored = CodedPiece::from_bytes(&bytes, 3);
        assert!(restored.is_ok());
        assert_eq!(restored.unwrap_or_else(|_| piece.clone()), piece);
    }
}
