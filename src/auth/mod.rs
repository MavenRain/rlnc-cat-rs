//! Piece authentication for pollution-resistant RLNC gossip.
//!
//! A recoding network is vulnerable to *pollution*: a byzantine relay
//! can inject coded pieces whose `(coding_vector, data)` pair does not
//! correspond to any linear combination of the source pieces.  Such
//! pieces enter downstream decoders and contaminate the reconstructed
//! data.
//!
//! The [`Authenticator`] trait abstracts the defense layer.  A source
//! commits to a generation of [`OriginalData`] and tags each coded
//! piece it emits.  Verifiers reject pieces whose tag does not match
//! the commitment.
//!
//! # Choosing an implementation
//!
//! - [`NullAuthenticator`]: no verification.  Use only when pollution
//!   is not part of the threat model (testing, trusted network).
//! - [`KeyedHashAuthenticator`]: BLAKE3 keyed-hash per piece.  Suitable
//!   for permissioned networks where source and verifier share a key.
//!   Not homomorphic: recoders cannot tag recoded pieces without the key.
//!
//! For public-network gossip with recoding, a linear homomorphic
//! signature scheme is required.  That is a separate work item; future
//! implementations will live alongside these two under the same trait.
//!
//! [`OriginalData`]: crate::coding::piece::OriginalData

mod keyed_hash;
mod null;

pub use keyed_hash::{Commitment as KeyedHashCommitment, KeyedHashAuthenticator, Tag as KeyedHashTag};
pub use null::NullAuthenticator;

use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;

/// Authenticates coded pieces against a per-generation commitment.
///
/// A source commits to an [`OriginalData`] and tags each [`CodedPiece`]
/// it emits; verifiers check `(commitment, piece, tag)` tuples.
pub trait Authenticator {
    /// A commitment to a generation of original data.
    type Commitment;

    /// An authentication tag attached to an emitted piece.
    type Tag;

    /// Commit to a fresh generation of original data.
    fn commit(&self, original: &OriginalData) -> Self::Commitment;

    /// Produce an authentication tag for a coded piece derived from
    /// the committed generation.
    fn tag(&self, commitment: &Self::Commitment, piece: &CodedPiece) -> Self::Tag;

    /// Verify that `piece` was produced from the generation committed
    /// to by `commitment` and carries a valid `tag`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AuthenticatorRejected`] when the tag does not
    /// match the expected value for `(commitment, piece)`.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        piece: &CodedPiece,
        tag: &Self::Tag,
    ) -> Result<(), Error>;
}
