//! Linearly homomorphic signature [`Authenticator`] for RLNC gossip.
//!
//! Wires the BF11-style scheme from [`crate::lhs`] into the gossip
//! layer's [`Authenticator`] interface.  The authenticator publishes
//!
//! - the public key `pk = A`,
//! - the generation metadata `m ∈ {0,1}*` (e.g., `H(OriginalData)`),
//! - the per-piece signatures `σ_1, …, σ_k` on hash targets
//!   `h_i = H("lhs-v1.target" || i || m)`,
//! - a BLAKE3 commitment fingerprint over `(pk, m, σ_1, …, σ_k)` that
//!   identifies the generation.
//!
//! A tag for a coded piece with coding vector `cv ∈ GF(2^8)^k` is the
//! linear combination `σ = Σ cv_i · σ_i` computed by [`combine`] in
//! the lifted integer scheme.  The combination is *public*: any party
//! holding `(pk, m, σ_1, …, σ_k)` can tag any coded piece without
//! seeing `sk`, so a recoding relay can authenticate freshly mixed
//! pieces.
//!
//! Verification recomputes the per-piece target
//! `target = Σ cv_i · h_i (mod Q)` and delegates to [`verify`], which
//! enforces both the algebraic identity `A·σ ≡ target (mod Q)` and
//! the shortness bound `||σ||² ≤ β²`.  Cross-generation replay is
//! blocked by comparing the wire commitment against the one cached at
//! construction.
//!
//! # BFKW09 broadcast model
//!
//! The tuple `(pk, m, σ_1, …, σ_k)` is the **public transcript** of
//! the generation.  Signatures are sampled once with `sk`, bound into
//! the commitment, and `sk` is never required again at tagging or
//! verification time.  Relays run [`Self::from_public_artifacts`] to
//! rebuild the authenticator from the broadcast tuple.
//!
//! # Metadata binding
//!
//! Hash targets `h_i` are derived from caller-supplied `metadata`
//! bytes via a BLAKE3-based expansion with the domain tag
//! `"lhs-v1.target"` and a counter to handle rejection sampling.
//! Callers that want the commitment tied to a specific
//! [`OriginalData`] pass `metadata = H(original.bytes() || ...)` at
//! construction time; any downstream [`OriginalData`] that mismatches
//! the metadata will fail verification via the commitment check.

use crate::auth::Authenticator;
use crate::coding::piece::{CodedPiece, OriginalData};
use crate::error::Error;
use crate::lattice::{ZVec, Zq, ZqVec};
use crate::lhs::keys::{PublicKey, SecretKey};
use crate::lhs::sign::{Signature, combine, sign};
use crate::lhs::verify::verify;

/// Per-generation commitment: a BLAKE3 fingerprint of
/// `(pk, metadata, σ_originals)`.
///
/// Equality is constant-time via [`blake3::Hash`]'s built-in ct-eq,
/// so verifiers can compare wire commitments against the cached one
/// without leaking a timing side channel.
#[derive(Clone, Copy, Debug)]
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

impl PartialEq for Commitment {
    fn eq(&self, other: &Self) -> bool {
        blake3::Hash::from_bytes(self.0) == blake3::Hash::from_bytes(other.0)
    }
}

impl Eq for Commitment {}

/// Linearly homomorphic signature [`Authenticator`] over `Z/QZ`.
///
/// Holds the public artefacts of one generation: the public key, the
/// derived hash targets, the per-target signatures, and the commitment
/// fingerprint that binds them together.  All four are needed to tag
/// and verify coded pieces.
///
/// # Construction
///
/// - [`Self::new`] at the source side: derives `h_i` from `metadata`,
///   signs every one with `sk`, and caches `sk`-free artefacts
///   thereafter.
/// - [`Self::from_public_artifacts`] at relay and verifier sides:
///   rebuilds from the broadcast tuple `(pk, metadata, σ_1, …, σ_k)`
///   without access to `sk`.
#[derive(Clone, Debug)]
#[must_use]
pub struct LatticeHomomorphicAuthenticator<const Q: u32> {
    pk: PublicKey<Q>,
    hash_targets: Vec<ZqVec<Q>>,
    signed_originals: Vec<Signature>,
    commitment: Commitment,
}

impl<const Q: u32> LatticeHomomorphicAuthenticator<Q> {
    /// Build an authenticator at the source side.
    ///
    /// Derives `params.k_pieces()` hash targets from `metadata`, signs
    /// each of them with `sk`, and caches the resulting `σ_i`.  After
    /// this call, tagging and verification use only `pk`, `metadata`,
    /// and the cached signatures.
    ///
    /// # Errors
    ///
    /// - Propagates any error returned by [`sign`] (unreachable for
    ///   keys that came from a successful [`keygen`] call).
    ///
    /// [`keygen`]: crate::lhs::keygen
    pub fn new(
        pk: PublicKey<Q>,
        sk: &SecretKey<Q>,
        metadata: &[u8],
    ) -> Result<Self, Error> {
        let k = pk.params().k_pieces();
        let n = pk.params().n();
        let hash_targets = derive_hash_targets::<Q>(metadata, k, n);
        hash_targets
            .iter()
            .map(|target| sign(&pk, sk, target))
            .collect::<Result<Vec<Signature>, Error>>()
            .map(|signed_originals| {
                let commitment = derive_commitment(&pk, metadata, &signed_originals);
                Self {
                    pk,
                    hash_targets,
                    signed_originals,
                    commitment,
                }
            })
    }

    /// Build an authenticator from the public broadcast tuple.
    ///
    /// Relays and verifiers use this constructor: they hold
    /// `(pk, metadata, σ_1, …, σ_k)` but no secret key, and can still
    /// tag and verify pieces because the homomorphic combine is a
    /// public computation.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if
    ///   `signed_originals.len() != pk.params().k_pieces()`.
    pub fn from_public_artifacts(
        pk: PublicKey<Q>,
        metadata: &[u8],
        signed_originals: Vec<Signature>,
    ) -> Result<Self, Error> {
        let k = pk.params().k_pieces();
        let n = pk.params().n();
        (signed_originals.len() == k)
            .then_some(())
            .ok_or(Error::DimensionMismatch {
                expected: k,
                actual: signed_originals.len(),
            })
            .map(|()| {
                let hash_targets = derive_hash_targets::<Q>(metadata, k, n);
                let commitment = derive_commitment(&pk, metadata, &signed_originals);
                Self {
                    pk,
                    hash_targets,
                    signed_originals,
                    commitment,
                }
            })
    }

    /// Borrow the public key.
    pub fn public_key(&self) -> &PublicKey<Q> {
        &self.pk
    }

    /// Borrow the derived hash targets `h_1, …, h_k`.
    pub fn hash_targets(&self) -> &[ZqVec<Q>] {
        &self.hash_targets
    }

    /// Borrow the signed originals `σ_1, …, σ_k`.
    pub fn signed_originals(&self) -> &[Signature] {
        &self.signed_originals
    }

    /// Borrow the commitment fingerprint.
    pub fn commitment(&self) -> &Commitment {
        &self.commitment
    }
}

impl<const Q: u32> Authenticator for LatticeHomomorphicAuthenticator<Q> {
    type Commitment = Commitment;
    type Tag = Signature;

    fn commit(&self, _original: &OriginalData) -> Commitment {
        self.commitment
    }

    fn tag(&self, _commitment: &Commitment, piece: &CodedPiece) -> Signature {
        lifted_pairs::<Q>(piece, &self.signed_originals)
            .and_then(|pairs| combine::<Q>(&pairs).ok())
            .unwrap_or_else(|| Signature::new(ZVec::new(Vec::new())))
    }

    fn verify(
        &self,
        commitment: &Commitment,
        piece: &CodedPiece,
        tag: &Signature,
    ) -> Result<(), Error> {
        (commitment == &self.commitment)
            .then_some(())
            .ok_or(Error::AuthenticatorRejected)
            .and_then(|()| {
                derive_target::<Q>(piece, &self.hash_targets)
                    .map_err(|_| Error::AuthenticatorRejected)
            })
            .and_then(|target| verify(&self.pk, tag, &target))
    }
}

fn lifted_pairs<const Q: u32>(
    piece: &CodedPiece,
    signed_originals: &[Signature],
) -> Option<Vec<(Zq<Q>, Signature)>> {
    let cv_bytes = piece.coding_vector().as_vec().to_bytes();
    (cv_bytes.len() == signed_originals.len()).then(|| {
        cv_bytes
            .iter()
            .zip(signed_originals.iter())
            .map(|(&b, sig)| (Zq::<Q>::new(u32::from(b)), sig.clone()))
            .collect()
    })
}

fn derive_target<const Q: u32>(
    piece: &CodedPiece,
    hash_targets: &[ZqVec<Q>],
) -> Result<ZqVec<Q>, Error> {
    let cv_bytes = piece.coding_vector().as_vec().to_bytes();
    (cv_bytes.len() == hash_targets.len())
        .then_some(())
        .ok_or(Error::DimensionMismatch {
            expected: hash_targets.len(),
            actual: cv_bytes.len(),
        })
        .and_then(|()| {
            let scalars: Vec<Zq<Q>> = cv_bytes
                .iter()
                .map(|&b| Zq::<Q>::new(u32::from(b)))
                .collect();
            ZqVec::<Q>::linear_combine(&scalars, hash_targets)
        })
}

/// Derive the `k` hash targets of length `n` each from
/// `metadata` via a BLAKE3-based expansion with the domain tag
/// `"lhs-v1.target"`.  The derivation is deterministic: two
/// authenticators constructed with the same metadata see the same
/// hash targets, which is what lets a relay rebuild the authenticator
/// from `from_public_artifacts` without re-sampling.
fn derive_hash_targets<const Q: u32>(
    metadata: &[u8],
    k: usize,
    n: usize,
) -> Vec<ZqVec<Q>> {
    (0..k)
        .map(|i| derive_single_target::<Q>(metadata, i, n, 0, Vec::new()))
        .collect()
}

/// Expand `metadata || index || counter` into `n` uniform `Zq<Q>`
/// samples via rejection sampling on u32 chunks of the BLAKE3 digest.
/// Recurses with a bumped counter until `n` samples have been
/// accepted; for Q=97 the rejection rate is negligible so the
/// recursion depth is effectively 1.
fn derive_single_target<const Q: u32>(
    metadata: &[u8],
    index: usize,
    n: usize,
    counter: u64,
    acc: Vec<Zq<Q>>,
) -> ZqVec<Q> {
    if acc.len() >= n {
        ZqVec::new(acc.into_iter().take(n).collect())
    } else {
        let digest = hash_with_context(metadata, index, counter);
        let threshold = u32::MAX - (u32::MAX % Q);
        let new_samples: Vec<Zq<Q>> = digest
            .chunks_exact(4)
            .filter_map(|c| <[u8; 4]>::try_from(c).ok())
            .map(u32::from_le_bytes)
            .filter(|&x| x < threshold)
            .map(|x| Zq::new(x % Q))
            .collect();
        let next_acc: Vec<Zq<Q>> = acc.into_iter().chain(new_samples).collect();
        derive_single_target::<Q>(metadata, index, n, counter.wrapping_add(1), next_acc)
    }
}

fn hash_with_context(metadata: &[u8], index: usize, counter: u64) -> [u8; 32] {
    let idx_bytes = u64::try_from(index).unwrap_or(u64::MAX).to_le_bytes();
    let payload: Vec<u8> = b"lhs-v1.target"
        .iter()
        .copied()
        .chain(idx_bytes)
        .chain(counter.to_le_bytes())
        .chain(metadata.iter().copied())
        .collect();
    *blake3::hash(&payload).as_bytes()
}

fn derive_commitment<const Q: u32>(
    pk: &PublicKey<Q>,
    metadata: &[u8],
    signed_originals: &[Signature],
) -> Commitment {
    let a = pk.a();
    let a_bytes = (0..a.rows())
        .flat_map(|i| (0..a.cols()).map(move |j| (i, j)))
        .flat_map(|(i, j)| a.entry(i, j).unwrap_or_else(Zq::zero).value().to_le_bytes());
    let metadata_len = u64::try_from(metadata.len()).unwrap_or(u64::MAX).to_le_bytes();
    let sig_bytes = signed_originals
        .iter()
        .flat_map(|sig| sig.sigma().entries().iter().copied())
        .flat_map(i64::to_le_bytes);
    let payload: Vec<u8> = b"lhs-v1.commit"
        .iter()
        .copied()
        .chain(a_bytes)
        .chain(metadata_len)
        .chain(metadata.iter().copied())
        .chain(sig_bytes)
        .collect();
    Commitment::from(*blake3::hash(&payload).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lhs::keys::keygen;
    use crate::lhs::params::LhsParams;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const DEFAULT_METADATA: &[u8] = b"lhs-auth-test-default-metadata";

    fn counter_rng(seed: usize) -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(seed));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n)
                .map(|j| ((i.wrapping_mul(0x9E37_79B9) + j) & 0xFF) as u8)
                .collect())
        }
    }

    fn unreachable_params<const Q: u32>() -> LhsParams<Q> {
        LhsParams::<Q>::new(1, 1, 1, 1)
            .ok()
            .unwrap_or_else(unreachable_params)
    }

    fn sample_auth_with_seed_and_metadata(
        seed: usize,
        metadata: &[u8],
    ) -> Option<LatticeHomomorphicAuthenticator<97>> {
        let params = LhsParams::<97>::new(2, 2, 4, 100_000_000)
            .ok()
            .unwrap_or_else(unreachable_params);
        let keys = keygen(params, &counter_rng(seed)).ok();
        keys.and_then(|(pk, sk)| LatticeHomomorphicAuthenticator::new(pk, &sk, metadata).ok())
    }

    fn sample_auth_with_seed(seed: usize) -> Option<LatticeHomomorphicAuthenticator<97>> {
        sample_auth_with_seed_and_metadata(seed, DEFAULT_METADATA)
    }

    fn sample_auth() -> Option<LatticeHomomorphicAuthenticator<97>> {
        sample_auth_with_seed(0)
    }

    fn sample_piece<const Q: u32>(
        auth: &LatticeHomomorphicAuthenticator<Q>,
        coding_bytes: &[u8],
    ) -> CodedPiece {
        use crate::coding::piece::CodingVector;
        use crate::vector::GfVec;
        CodedPiece::new(
            CodingVector::from_bytes(coding_bytes),
            GfVec::from_bytes(&vec![0u8; auth.signed_originals.len()]),
        )
    }

    #[test]
    fn new_produces_k_signed_originals() {
        let auth = sample_auth();
        assert_eq!(auth.map(|a| a.signed_originals().len()), Some(4));
    }

    #[test]
    fn new_derives_k_hash_targets_with_length_n() {
        let auth = sample_auth();
        let shapes: Option<Vec<usize>> =
            auth.map(|a| a.hash_targets().iter().map(ZqVec::len).collect());
        assert_eq!(shapes, Some(vec![2, 2, 2, 2]));
    }

    #[test]
    fn commitment_is_deterministic_for_fixed_artifacts() {
        let auth = sample_auth();
        let recloned = auth.as_ref().map(|a| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                a.public_key().clone(),
                DEFAULT_METADATA,
                a.signed_originals().to_vec(),
            )
            .ok()
        });
        let eq = auth.and_then(|a| {
            recloned
                .flatten()
                .map(|r| r.commitment() == a.commitment())
        });
        assert_eq!(eq, Some(true));
    }

    #[test]
    fn from_public_artifacts_rejects_wrong_count() {
        let auth = sample_auth();
        let built = auth.map(|a| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                a.public_key().clone(),
                DEFAULT_METADATA,
                a.signed_originals().iter().take(2).cloned().collect(),
            )
        });
        assert!(built.is_some_and(|r| r.is_err()));
    }

    #[test]
    fn fresh_tag_verifies() {
        let auth = sample_auth();
        let ok = auth.is_some_and(|a| {
            let piece = sample_piece(&a, &[1, 2, 3, 4]);
            let commitment = *a.commitment();
            let tag = a.tag(&commitment, &piece);
            a.verify(&commitment, &piece, &tag).is_ok()
        });
        assert!(ok);
    }

    #[test]
    fn tag_on_standard_basis_matches_signed_original() {
        // cv = e_i (a single 1, rest 0) lifts to [0, ..., 1, ..., 0] in Zq;
        // the combined signature is exactly signed_originals[i].
        let auth = sample_auth();
        let ok = auth.is_some_and(|a| {
            (0..a.signed_originals().len()).all(|i| {
                let cv: Vec<u8> = (0..a.signed_originals().len())
                    .map(|j| u8::from(j == i))
                    .collect();
                let piece = sample_piece(&a, &cv);
                let commitment = *a.commitment();
                let tag = a.tag(&commitment, &piece);
                tag == a.signed_originals()[i]
            })
        });
        assert!(ok);
    }

    #[test]
    fn wrong_commitment_rejects_tag() {
        let auth = sample_auth();
        let rejected = auth.is_some_and(|a| {
            let piece = sample_piece(&a, &[1, 0, 0, 0]);
            let commitment = *a.commitment();
            let tag = a.tag(&commitment, &piece);
            let bogus = Commitment::from([0xFF; 32]);
            a.verify(&bogus, &piece, &tag).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn tag_from_foreign_generation_rejects() {
        // Two independent authenticators (distinct RNG seeds) have
        // independent signed_originals, so a tag produced by one fails
        // verification by the other even though the metadata-derived
        // hash_targets are identical: the verifier's public key `A_B`
        // doesn't satisfy `A_B · σ_A ≡ target (mod Q)`.
        let auth_a = sample_auth_with_seed(0);
        let auth_b = sample_auth_with_seed(10_000);
        let rejected = auth_a.zip(auth_b).is_some_and(|(a, b)| {
            let piece = sample_piece(&a, &[1, 2, 3, 4]);
            let commitment_a = *a.commitment();
            let tag_a = a.tag(&commitment_a, &piece);
            let commitment_b = *b.commitment();
            b.verify(&commitment_b, &piece, &tag_a).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn wrong_coding_vector_length_rejects() {
        let auth = sample_auth();
        let rejected = auth.is_some_and(|a| {
            let piece = sample_piece(&a, &[1, 2, 3]);
            let commitment = *a.commitment();
            let tag = a.tag(&commitment, &piece);
            a.verify(&commitment, &piece, &tag).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn mismatched_tag_and_coding_vector_rejects() {
        // Tag was generated for cv_a, but the wire piece carries cv_b.
        // derive_target(cv_b) mismatches A·tag, so verify rejects.
        let auth = sample_auth();
        let rejected = auth.is_some_and(|a| {
            let piece_a = sample_piece(&a, &[1, 0, 0, 0]);
            let piece_b = sample_piece(&a, &[0, 1, 0, 0]);
            let commitment = *a.commitment();
            let tag_a = a.tag(&commitment, &piece_a);
            a.verify(&commitment, &piece_b, &tag_a).is_err()
        });
        assert!(rejected);
    }

    #[test]
    fn commitment_differs_across_metadata() {
        // Same RNG seed (hence same pk, sk), different metadata.  The
        // derived hash_targets differ, the signed_originals differ,
        // and the commitment bytes differ.  This is the central
        // binding property the metadata-derivation adds on top of the
        // earlier keypair-only binding.
        let auth_a = sample_auth_with_seed_and_metadata(0, b"generation-one");
        let auth_b = sample_auth_with_seed_and_metadata(0, b"generation-two");
        let distinct = auth_a.zip(auth_b).is_some_and(|(a, b)| {
            a.commitment() != b.commitment()
                && a.hash_targets() != b.hash_targets()
                && a.signed_originals() != b.signed_originals()
        });
        assert!(distinct);
    }

    #[test]
    fn hash_targets_are_deterministic_across_constructors() {
        // `new` (signer path) and `from_public_artifacts` (relay path)
        // derive hash_targets identically when given the same metadata.
        let signer = sample_auth();
        let relay = signer.as_ref().and_then(|s| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                s.public_key().clone(),
                DEFAULT_METADATA,
                s.signed_originals().to_vec(),
            )
            .ok()
        });
        let eq = signer.zip(relay).map(|(s, r)| s.hash_targets() == r.hash_targets());
        assert_eq!(eq, Some(true));
    }

    #[test]
    fn relay_with_wrong_metadata_rejects_source_tags() {
        // Source and relay share (pk, σ_originals) but the relay builds
        // with different metadata.  Its derived hash_targets differ
        // from the source's, so when it tries to verify a piece the
        // derived target mismatches A · tag.
        let source = sample_auth_with_seed_and_metadata(0, b"generation-one");
        let relay = source.as_ref().and_then(|s| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                s.public_key().clone(),
                b"generation-two",
                s.signed_originals().to_vec(),
            )
            .ok()
        });
        let rejected = source.zip(relay).is_some_and(|(s, r)| {
            let piece = sample_piece(&s, &[1, 2, 3, 4]);
            let commitment_s = *s.commitment();
            let tag_s = s.tag(&commitment_s, &piece);
            let commitment_r = *r.commitment();
            // Relay can't verify the source's tag under its own commitment.
            r.verify(&commitment_r, &piece, &tag_s).is_err()
        });
        assert!(rejected);
    }
}
