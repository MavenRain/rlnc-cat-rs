//! A no-op [`Authenticator`] that accepts every piece.
//!
//! Use only when pollution is not part of the threat model.

use super::Authenticator;
use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;

/// An authenticator that performs no verification.
///
/// Commitments and tags are both the unit type, so wire overhead is
/// zero when this impl is used.  Useful for tests and for trusted
/// networks where pollution is not a concern.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::auth::{Authenticator, NullAuthenticator};
/// use rlnc_cat_rs::coding::encode::encode_with_vector;
/// use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
///
/// let auth = NullAuthenticator;
/// let orig = OriginalData::from_bytes(&[1, 2, 3, 4], 2).ok();
/// let piece = orig.as_ref().and_then(|o| {
///     encode_with_vector(o, CodingVector::from_bytes(&[1, 0])).ok()
/// });
///
/// let commitment = orig.as_ref().map(|o| auth.commit(o));
/// let tag = commitment.as_ref().zip(piece.as_ref()).map(|(c, p)| auth.tag(c, p));
///
/// let verified = commitment
///     .as_ref()
///     .zip(piece.as_ref())
///     .zip(tag.as_ref())
///     .map(|((c, p), t)| auth.verify(c, p, t).is_ok());
/// assert_eq!(verified, Some(true));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[must_use]
pub struct NullAuthenticator;

impl Authenticator for NullAuthenticator {
    type Commitment = ();
    type Tag = ();

    fn commit(&self, _original: &OriginalData) {}

    fn tag(&self, _commitment: &(), _piece: &CodedPiece) {}

    fn verify(
        &self,
        _commitment: &(),
        _piece: &CodedPiece,
        _tag: &(),
    ) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::encode::encode_with_vector;
    use crate::coding::piece::CodingVector;

    fn sample_piece() -> (OriginalData, CodedPiece) {
        let orig = OriginalData::from_bytes(&[1, 2, 3, 4, 5, 6], 3)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let piece = encode_with_vector(&orig, CodingVector::from_bytes(&[1, 0, 0]))
            .unwrap_or_else(|_| {
                use crate::vector::GfVec;
                CodedPiece::new(CodingVector::from_bytes(&[]), GfVec::zeros(0))
            });
        (orig, piece)
    }

    #[test]
    fn null_verify_accepts_any_piece() {
        let auth = NullAuthenticator;
        let (orig, piece) = sample_piece();
        // commit/tag both return unit; feed the unit values straight into verify.
        auth.commit(&orig);
        auth.tag(&(), &piece);
        assert!(auth.verify(&(), &piece, &()).is_ok());
    }
}
