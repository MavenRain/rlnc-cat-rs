//! BLAKE3 keyed-hash [`Authenticator`].
//!
//! The source and verifier share a 32-byte key.  Each emitted piece
//! carries a 32-byte tag computed as `blake3::keyed_hash(key,
//! commitment || coding_vector || data)`.  Verification recomputes
//! the tag and compares it in constant time (via `blake3::Hash`'s
//! built-in constant-time equality).
//!
//! This authenticator is **not homomorphic**: a recoder cannot produce
//! a valid tag for a recoded piece without the shared key.  That makes
//! it suitable for:
//!
//! - Source-signed-only deployments (no recoding).
//! - Permissioned networks where every relay holds the key and can
//!   independently tag the pieces it emits.
//!
//! For public-network gossip with recoding, swap to a future
//! homomorphic-signature implementation.

use super::Authenticator;
use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;

/// A commitment to an [`OriginalData`] generation: a BLAKE3 hash of the
/// concatenated source piece bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct Commitment([u8; 32]);

impl From<[u8; 32]> for Commitment {
    fn from(bytes: [u8; 32]) -> Self {
        Commitment(bytes)
    }
}

impl From<Commitment> for [u8; 32] {
    fn from(c: Commitment) -> Self {
        c.0
    }
}

impl Commitment {
    /// Borrow the commitment bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// An authentication tag on a [`CodedPiece`]: a 32-byte BLAKE3
/// keyed-hash output.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct Tag([u8; 32]);

impl From<[u8; 32]> for Tag {
    fn from(bytes: [u8; 32]) -> Self {
        Tag(bytes)
    }
}

impl From<Tag> for [u8; 32] {
    fn from(t: Tag) -> Self {
        t.0
    }
}

impl Tag {
    /// Borrow the tag bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl PartialEq for Tag {
    /// Constant-time equality via [`blake3::Hash`]'s ct-eq.
    fn eq(&self, other: &Self) -> bool {
        blake3::Hash::from_bytes(self.0) == blake3::Hash::from_bytes(other.0)
    }
}

impl Eq for Tag {}

/// An [`Authenticator`] using BLAKE3 keyed-hash as a per-piece MAC.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::auth::{Authenticator, KeyedHashAuthenticator};
/// use rlnc_cat_rs::coding::encode::encode_with_vector;
/// use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
///
/// let key = [0u8; 32];
/// let auth = KeyedHashAuthenticator::new(key);
///
/// let orig = OriginalData::from_bytes(&[1, 2, 3, 4], 2).ok();
/// let piece = orig.as_ref().and_then(|o| {
///     encode_with_vector(o, CodingVector::from_bytes(&[1, 0])).ok()
/// });
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
#[derive(Clone, Debug)]
#[must_use]
pub struct KeyedHashAuthenticator {
    key: [u8; 32],
}

impl KeyedHashAuthenticator {
    /// Construct from a 32-byte shared key.
    pub fn new(key: [u8; 32]) -> Self {
        Self { key }
    }
}

impl Authenticator for KeyedHashAuthenticator {
    type Commitment = Commitment;
    type Tag = Tag;

    fn commit(&self, original: &OriginalData) -> Commitment {
        let bytes: Vec<u8> = original
            .pieces()
            .iter()
            .flat_map(crate::vector::GfVec::to_bytes)
            .collect();
        Commitment::from(*blake3::hash(&bytes).as_bytes())
    }

    fn tag(&self, commitment: &Commitment, piece: &CodedPiece) -> Tag {
        let payload: Vec<u8> = commitment
            .as_bytes()
            .iter()
            .copied()
            .chain(piece.to_bytes())
            .collect();
        Tag::from(*blake3::keyed_hash(&self.key, &payload).as_bytes())
    }

    fn verify(
        &self,
        commitment: &Commitment,
        piece: &CodedPiece,
        tag: &Tag,
    ) -> Result<(), Error> {
        let expected = self.tag(commitment, piece);
        if expected == *tag {
            Ok(())
        } else {
            Err(Error::AuthenticatorRejected)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::encode::encode_with_vector;
    use crate::coding::piece::CodingVector;
    use crate::vector::GfVec;

    fn sample() -> (KeyedHashAuthenticator, OriginalData, CodedPiece) {
        let auth = KeyedHashAuthenticator::new([7u8; 32]);
        let orig = OriginalData::from_bytes(&[1, 2, 3, 4, 5, 6, 7, 8], 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let piece = encode_with_vector(&orig, CodingVector::from_bytes(&[1, 0, 0, 0]))
            .unwrap_or_else(|_| {
                CodedPiece::new(CodingVector::from_bytes(&[]), GfVec::zeros(0))
            });
        (auth, orig, piece)
    }

    #[test]
    fn keyed_hash_roundtrip_accepts() {
        let (auth, orig, piece) = sample();
        let c = auth.commit(&orig);
        let t = auth.tag(&c, &piece);
        assert!(auth.verify(&c, &piece, &t).is_ok());
    }

    #[test]
    fn keyed_hash_rejects_wrong_tag() {
        let (auth, orig, piece) = sample();
        let c = auth.commit(&orig);
        let bad = Tag::from([0u8; 32]);
        assert!(auth.verify(&c, &piece, &bad).is_err());
    }

    #[test]
    fn keyed_hash_rejects_tampered_piece() {
        let (auth, orig, piece) = sample();
        let c = auth.commit(&orig);
        let t = auth.tag(&c, &piece);
        // Flip one byte in the data payload
        let tampered_data = GfVec::from_bytes(
            &piece
                .data()
                .to_bytes()
                .iter()
                .enumerate()
                .map(|(i, &b)| if i == 0 { b ^ 0x01 } else { b })
                .collect::<Vec<u8>>(),
        );
        let tampered = CodedPiece::new(piece.coding_vector().clone(), tampered_data);
        assert!(auth.verify(&c, &tampered, &t).is_err());
    }

    #[test]
    fn keyed_hash_rejects_wrong_key() {
        let (auth, orig, piece) = sample();
        let c = auth.commit(&orig);
        let t = auth.tag(&c, &piece);
        let other = KeyedHashAuthenticator::new([9u8; 32]);
        assert!(other.verify(&c, &piece, &t).is_err());
    }

    #[test]
    fn keyed_hash_rejects_wrong_commitment() {
        let (auth, _orig, piece) = sample();
        let other_orig = OriginalData::from_bytes(&[9, 9, 9, 9], 2)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let c_real = auth.commit(&other_orig);
        let c_real_tag = auth.tag(&c_real, &piece);
        let c_fake = Commitment::from([0u8; 32]);
        assert!(auth.verify(&c_fake, &piece, &c_real_tag).is_err());
    }
}
