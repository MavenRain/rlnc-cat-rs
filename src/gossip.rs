//! Gossip-layer integration.
//!
//! Ties the encode / decode paths to a pluggable [`Authenticator`] and
//! surfaces them as [`Stream`] / [`Io`] arrows that a transport can wire
//! directly into its inbound and outbound queues.
//!
//! # Roles
//!
//! - [`source`]: takes [`OriginalData`] and emits a [`Stream`] of
//!   authenticated [`WirePiece`]s to broadcast.
//! - [`receive`]: takes a [`Stream`] of authenticated [`WirePiece`]s
//!   and returns an [`Io`] that resolves to the reconstructed bytes
//!   when enough linearly-independent pieces have arrived.
//!
//! Recoding relays are a separate role (multi-peer fan-out, verify +
//! recode + re-tag) and will land in a later module alongside the
//! transport abstraction.
//!
//! # Example: full round-trip with `NullAuthenticator`
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//! use rlnc_cat_rs::auth::NullAuthenticator;
//! use rlnc_cat_rs::coding::piece::OriginalData;
//! use rlnc_cat_rs::gossip::{receive, source};
//!
//! let orig = OriginalData::from_bytes(&[1, 2, 3, 4, 5, 6], 3).ok();
//! let k = orig.as_ref().map(OriginalData::piece_count);
//! let b = orig.as_ref().map(OriginalData::piece_byte_len);
//!
//! // Each call returns the next standard basis vector, so the k emitted
//! // coding vectors are trivially linearly independent.  A real deployment
//! // would use a CSPRNG (or the source's choice of randomness) here.
//! let counter = Arc::new(AtomicUsize::new(0));
//! let rng = move |n: usize| -> Result<Vec<u8>, rlnc_cat_rs::error::Error> {
//!     let i = counter.fetch_add(1, Ordering::Relaxed);
//!     Ok((0..n).map(|j| u8::from(j == i)).collect())
//! };
//!
//! let decoded = orig.zip(k).zip(b).and_then(|((o, k), b)| {
//!     let auth = Arc::new(NullAuthenticator);
//!     let (commitment, stream) = source(Arc::clone(&auth), o, rng);
//!     receive(auth, commitment, k, b, stream.take(k)).run().ok()
//! });
//! assert_eq!(decoded, Some(vec![1, 2, 3, 4, 5, 6]));
//! ```

use std::sync::Arc;

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::auth::Authenticator;
use crate::coding::decode::DecoderState;
use crate::coding::encode::Encoder;
use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;

/// The on-wire package for a single authenticated coded piece.
///
/// Bundles the generation commitment, the coded piece itself, and the
/// authenticator's tag so a relay can verify without out-of-band state.
#[must_use]
pub struct WirePiece<A: Authenticator>
where
    A::Commitment: Clone,
    A::Tag: Clone,
{
    commitment: A::Commitment,
    piece: CodedPiece,
    tag: A::Tag,
}

impl<A: Authenticator> WirePiece<A>
where
    A::Commitment: Clone,
    A::Tag: Clone,
{
    /// Construct a wire piece from its three components.
    pub fn new(commitment: A::Commitment, piece: CodedPiece, tag: A::Tag) -> Self {
        Self { commitment, piece, tag }
    }

    /// The commitment to the source generation this piece was derived from.
    pub fn commitment(&self) -> &A::Commitment {
        &self.commitment
    }

    /// The coded piece payload.
    pub fn piece(&self) -> &CodedPiece {
        &self.piece
    }

    /// The authentication tag.
    pub fn tag(&self) -> &A::Tag {
        &self.tag
    }
}

impl<A: Authenticator> Clone for WirePiece<A>
where
    A::Commitment: Clone,
    A::Tag: Clone,
{
    fn clone(&self) -> Self {
        Self {
            commitment: self.commitment.clone(),
            piece: self.piece.clone(),
            tag: self.tag.clone(),
        }
    }
}

/// Source-side: commit to the generation, encode pieces, tag each one,
/// and emit them on a [`Stream`].
///
/// Returns `(commitment, stream)`.  The caller conveys the commitment
/// to the receiver out-of-band (or trusts the one embedded in the
/// first wire piece received) so the receiver can call [`receive`]
/// with the expected commitment.
///
/// The returned stream is unbounded: cap it with [`Stream::take`].
pub fn source<A, F>(
    auth: Arc<A>,
    orig: OriginalData,
    rng_factory: F,
) -> (A::Commitment, Stream<Error, WirePiece<A>>)
where
    A: Authenticator + Send + Sync + 'static,
    A::Commitment: Clone + Send + Sync + 'static,
    A::Tag: Clone + Send + Sync + 'static,
    F: Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static,
{
    let commitment = auth.commit(&orig);
    let c_arc: Arc<A::Commitment> = Arc::new(commitment.clone());
    let encoder = Encoder::new(orig);
    let stream = encoder
        .encode_stream(rng_factory)
        .map(Arc::new(move |piece: CodedPiece| {
            let tag = auth.tag(c_arc.as_ref(), &piece);
            WirePiece::new(c_arc.as_ref().clone(), piece, tag)
        }));
    (commitment, stream)
}

/// Terminal side: verify every incoming [`WirePiece`] against the
/// supplied `commitment` and absorb it into a [`DecoderState`].
/// Pieces that fail authentication are dropped; the stream continues
/// until enough linearly-independent pieces have accumulated to
/// reconstruct the data.
///
/// # Errors
///
/// The returned [`Io`] resolves to the decoded byte vector, or
/// propagates any error surfaced by the underlying decoder (including
/// `Error::InsufficientPieces` if the stream ends before enough
/// valid pieces arrive).
pub fn receive<A>(
    auth: Arc<A>,
    commitment: A::Commitment,
    piece_count: usize,
    piece_byte_len: usize,
    incoming: Stream<Error, WirePiece<A>>,
) -> Io<Error, Vec<u8>>
where
    A: Authenticator + Send + Sync + 'static,
    A::Commitment: Clone + Send + Sync + 'static,
    A::Tag: Clone + Send + Sync + 'static,
{
    let c_arc: Arc<A::Commitment> = Arc::new(commitment);
    let initial = DecoderState::new(piece_count, piece_byte_len);
    incoming
        .fold(
            initial,
            Arc::new(move |state: DecoderState, wp: WirePiece<A>| {
                if auth.verify(c_arc.as_ref(), wp.piece(), wp.tag()).is_ok() {
                    let backup = state.clone();
                    state.absorb(wp.piece()).unwrap_or(backup)
                } else {
                    state
                }
            }),
        )
        .flat_map(|s| Io::suspend(move || s.decode()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{KeyedHashAuthenticator, NullAuthenticator};
    use crate::field::Gf256;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// An RNG that returns a Vandermonde row per call: the i-th invocation
    /// yields `[1, base, base^2, ..., base^(n-1)]` where `base = (i % 255) + 1`.
    /// k < 255 rows are guaranteed linearly independent in GF(2^8).
    fn vandermonde_rng() -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicU32::new(0));
        #[allow(clippy::cast_possible_truncation)]
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            let base = Gf256::new(((i % 255 + 1) & 0xFF) as u8);
            Ok((0..n)
                .map(|j| (0..j).fold(Gf256::one(), |acc, _| acc * base).value())
                .collect())
        }
    }

    #[test]
    fn null_auth_roundtrip() {
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        let orig = OriginalData::from_bytes(&data, 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();
        let b = orig.piece_byte_len();

        let auth = Arc::new(NullAuthenticator);
        let (commitment, stream) = source(Arc::clone(&auth), orig, vandermonde_rng());
        let decoded = receive(auth, commitment, k, b, stream.take(k)).run();
        assert!(decoded.is_ok());
        assert_eq!(decoded.unwrap_or_default(), data);
    }

    #[test]
    fn keyed_hash_auth_roundtrip() {
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        let orig = OriginalData::from_bytes(&data, 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();
        let b = orig.piece_byte_len();

        let auth = Arc::new(KeyedHashAuthenticator::new([42u8; 32]));
        let (commitment, stream) = source(Arc::clone(&auth), orig, vandermonde_rng());
        let decoded = receive(auth, commitment, k, b, stream.take(k)).run();
        assert!(decoded.is_ok());
        assert_eq!(decoded.unwrap_or_default(), data);
    }

    #[test]
    fn keyed_hash_auth_rejects_wrong_key() {
        // Source uses one key, receiver uses another.  Every piece's tag
        // fails verification, so no piece is absorbed and decode errors
        // with InsufficientPieces.
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80];
        let orig = OriginalData::from_bytes(&data, 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();
        let b = orig.piece_byte_len();

        let source_auth = Arc::new(KeyedHashAuthenticator::new([1u8; 32]));
        let (commitment, stream) = source(source_auth, orig, vandermonde_rng());

        let receiver_auth = Arc::new(KeyedHashAuthenticator::new([2u8; 32]));
        let decoded = receive(receiver_auth, commitment, k, b, stream.take(k)).run();
        assert!(decoded.is_err());
    }
}
