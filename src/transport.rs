//! In-memory transport for wiring gossip nodes together.
//!
//! Three types:
//!
//! - [`InMemorySender`]: clone-able handle for sending packets to any
//!   peer in the network.  Sends land in an `mpsc` buffer; receivers
//!   see them as soon as they pull.
//! - [`InMemoryTransport`]: one peer's view of the network.  Owns
//!   the peer's unique `Receiver` for inbound packets, plus a sender
//!   handle for outbound.
//! - [`InMemoryNetwork`]: factory that wires N peers into a full-mesh
//!   topology (every peer can send to every peer, every peer has its
//!   own inbox).
//!
//! This is the testing transport; a real-network transport (TCP, QUIC,
//! iroh, whatever) can be added later alongside this one.
//!
//! # Example: 3-node source → relay → receive pipeline
//!
//! See the `three_node_pipeline_null_auth` test in this module for a
//! full end-to-end demonstration.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::error::Error;
use crate::gossip::PeerIndex;

/// An mpsc channel pair carrying `(source_peer, packet)` tuples.
type Channel<P> = (Sender<(PeerIndex, P)>, Receiver<(PeerIndex, P)>);

/// A clone-able sender handle: can publish packets to any peer in the
/// network.  Backed by a shared `Arc<Vec<mpsc::Sender>>` indexed by
/// destination [`PeerIndex`].
#[must_use]
pub struct InMemorySender<P: Send + 'static> {
    inboxes: Arc<Vec<Sender<(PeerIndex, P)>>>,
    self_peer: PeerIndex,
}

impl<P: Send + 'static> Clone for InMemorySender<P> {
    fn clone(&self) -> Self {
        Self {
            inboxes: Arc::clone(&self.inboxes),
            self_peer: self.self_peer,
        }
    }
}

impl<P: Send + 'static> InMemorySender<P> {
    /// Send a packet to the peer at `to`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TransportFailed`] if `to` is out of range for
    /// the network or the destination's inbox has been dropped.
    pub fn send(&self, to: PeerIndex, packet: P) -> Io<Error, ()> {
        let inboxes = Arc::clone(&self.inboxes);
        let from = self.self_peer;
        Io::suspend(move || {
            inboxes
                .get(to.value())
                .ok_or_else(|| {
                    Error::TransportFailed(format!("unknown peer {}", to.value()))
                })
                .and_then(|tx| {
                    tx.send((from, packet))
                        .map_err(|e| Error::TransportFailed(format!("{e}")))
                })
        })
    }

    /// This sender's own peer index.
    pub fn self_peer(&self) -> PeerIndex {
        self.self_peer
    }
}

/// One peer's transport: owns the inbox receiver and provides a sender.
#[must_use]
pub struct InMemoryTransport<P: Send + 'static> {
    sender: InMemorySender<P>,
    inbox: Receiver<(PeerIndex, P)>,
    peer_count: usize,
}

impl<P: Send + 'static> InMemoryTransport<P> {
    /// A clone of this transport's sender handle.
    pub fn sender(&self) -> InMemorySender<P> {
        self.sender.clone()
    }

    /// This transport's own peer index.
    pub fn self_peer(&self) -> PeerIndex {
        self.sender.self_peer()
    }

    /// Number of peers in the network.
    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.peer_count
    }

    /// Consume this transport into a [`Stream`] of inbound
    /// `(source_peer, packet)` tuples.  The stream ends when every
    /// sender targeting this inbox has been dropped.
    #[must_use]
    pub fn into_inbound(self) -> Stream<Error, (PeerIndex, P)> {
        let inbox = Arc::new(Mutex::new(self.inbox));
        Stream::unfold(
            (),
            Arc::new(move |()| {
                let inbox = Arc::clone(&inbox);
                Io::suspend(move || {
                    inbox
                        .lock()
                        .map_err(|e| {
                            Error::TransportFailed(format!("mutex poisoned: {e}"))
                        })
                        .and_then(|guard| {
                            guard
                                .recv()
                                .ok()
                                .map_or(Ok(None), |item| Ok(Some((item, ()))))
                        })
                })
            }),
        )
    }
}

/// Factory that wires N peers into a full-mesh in-memory network.
#[must_use]
pub struct InMemoryNetwork<P: Send + 'static> {
    transports: Vec<InMemoryTransport<P>>,
}

impl<P: Send + 'static> InMemoryNetwork<P> {
    /// Build a fresh network with `peer_count` peers.  Every peer can
    /// send to every peer, including itself.
    pub fn new(peer_count: usize) -> Self {
        let channels: Vec<Channel<P>> = (0..peer_count).map(|_| channel()).collect();
        let all_senders: Arc<Vec<Sender<(PeerIndex, P)>>> =
            Arc::new(channels.iter().map(|(s, _)| s.clone()).collect());
        let transports: Vec<InMemoryTransport<P>> = channels
            .into_iter()
            .enumerate()
            .map(|(i, (_s, r))| InMemoryTransport {
                sender: InMemorySender {
                    inboxes: Arc::clone(&all_senders),
                    self_peer: PeerIndex::from(i),
                },
                inbox: r,
                peer_count,
            })
            .collect();
        Self { transports }
    }

    /// Consume the network into its `Vec` of per-peer transports.
    #[must_use]
    pub fn into_transports(self) -> Vec<InMemoryTransport<P>> {
        self.transports
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::NullAuthenticator;
    use crate::coding::piece::OriginalData;
    use crate::field::Gf256;
    use crate::gossip::{WirePiece, receive, relay_fanout, source};
    use crate::lhs::{LatticeHomomorphicAuthenticator, LhsParams, keygen};
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

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

    fn keygen_rng(seed: usize) -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(seed));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n)
                .map(|j| ((i.wrapping_mul(0x9E37_79B9) + j) & 0xFF) as u8)
                .collect())
        }
    }

    /// Metadata string every LHS test in this module shares; binds the
    /// commitment to a fixed per-test generation identifier.  A real
    /// deployment would pass `H(OriginalData)` or a generation nonce.
    const LHS_TEST_METADATA: &[u8] = b"transport-test-generation";

    fn build_lhs_auth(seed: usize) -> Option<LatticeHomomorphicAuthenticator<97>> {
        let params = LhsParams::<97>::new(2, 2, 4, 100_000_000, 3.0).ok()?;
        let (pk, sk) = keygen(params, &keygen_rng(seed)).ok()?;
        LatticeHomomorphicAuthenticator::new(
            pk,
            &sk,
            LHS_TEST_METADATA,
            &keygen_rng(seed.wrapping_add(1)),
        )
        .ok()
    }

    #[test]
    fn sender_delivers_to_named_peer() {
        let transports = InMemoryNetwork::<u32>::new(2).into_transports();
        let pair: [InMemoryTransport<u32>; 2] = transports.try_into().ok().unwrap();
        let [t0, t1] = pair;
        let t0_sender = t0.sender();
        let t1_inbound = t1.into_inbound();

        // Send three u32s from peer 0 to peer 1, then drop t0_sender so
        // peer 1's inbox sees EOF after draining the buffered items.
        t0_sender.send(PeerIndex::from(1), 10u32).run().unwrap_or(());
        t0_sender.send(PeerIndex::from(1), 20u32).run().unwrap_or(());
        t0_sender.send(PeerIndex::from(1), 30u32).run().unwrap_or(());
        drop(t0_sender);
        drop(t0);

        let received = t1_inbound.collect().run().unwrap_or_default();
        assert_eq!(
            received,
            vec![
                (PeerIndex::from(0), 10u32),
                (PeerIndex::from(0), 20u32),
                (PeerIndex::from(0), 30u32),
            ]
        );
    }

    #[test]
    fn three_node_pipeline_null_auth() {
        // Node 0 (source) -> Node 1 (relay_fanout peer_count=1) -> Node 2 (receive).
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let orig = OriginalData::from_bytes(&data, 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();
        let b = orig.piece_byte_len();
        let auth = Arc::new(NullAuthenticator);

        let transports = InMemoryNetwork::<WirePiece<NullAuthenticator>>::new(3).into_transports();
        let triple: [InMemoryTransport<WirePiece<NullAuthenticator>>; 3] =
            transports.try_into().ok().unwrap();
        let [t0, t1, t2] = triple;
        let t0_sender = t0.sender();
        let t1_sender = t1.sender();
        let t1_inbox = t1.into_inbound();
        let t2_inbox = t2.into_inbound();
        drop(t0);

        // Node 0: produce k coded pieces and send them to Node 1.
        let (commitment, src_stream) = source(Arc::clone(&auth), orig, vandermonde_rng());
        src_stream
            .take(k)
            .fold(
                (),
                Arc::new(move |(), wp: WirePiece<NullAuthenticator>| {
                    t0_sender
                        .send(PeerIndex::from(1), wp)
                        .run()
                        .unwrap_or(());
                }),
            )
            .run()
            .unwrap_or(());

        // Node 1: drain k inbound, relay_fanout to Node 2, push each output.
        let node_1_inbox = t1_inbox
            .take(k)
            .map(Arc::new(|(_, wp): (PeerIndex, WirePiece<NullAuthenticator>)| wp));
        let fanned = relay_fanout(
            Arc::clone(&auth),
            commitment,
            k,
            b,
            1,
            node_1_inbox,
            vandermonde_rng(),
        )
        .run()
        .unwrap_or_else(|_| Stream::from_vec(Vec::new()));
        fanned
            .fold(
                (),
                Arc::new(
                    move |(),
                          (_, wp): (PeerIndex, WirePiece<NullAuthenticator>)| {
                        t1_sender
                            .send(PeerIndex::from(2), wp)
                            .run()
                            .unwrap_or(());
                    },
                ),
            )
            .run()
            .unwrap_or(());

        // Node 2: drain k inbound pieces and decode.
        let node_2_inbox = t2_inbox
            .take(k)
            .map(Arc::new(|(_, wp): (PeerIndex, WirePiece<NullAuthenticator>)| wp));
        let decoded = receive(auth, commitment, k, b, node_2_inbox).run();
        assert!(decoded.is_ok());
        assert_eq!(decoded.unwrap_or_default(), data);
    }

    #[test]
    fn three_node_pipeline_lhs_auth() {
        // Node 0 (source) → Node 1 (relay_fanout peer_count=1) → Node 2
        // (receive), with every hop authenticating pieces through the
        // BF11 linearly homomorphic signature scheme.  The relay holds
        // only the public artefacts (pk, σ_originals) and still tags
        // recoded pieces, exercising the homomorphic property across
        // the full wire transport.
        let data = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let orig = OriginalData::from_bytes(&data, 4)
            .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());
        let k = orig.piece_count();
        let b = orig.piece_byte_len();

        let source_auth = build_lhs_auth(0).map(Arc::new);
        let relay_auth = source_auth.as_ref().and_then(|sa| {
            LatticeHomomorphicAuthenticator::from_public_artifacts(
                sa.public_key().clone(),
                LHS_TEST_METADATA,
                sa.signed_originals().to_vec(),
            )
            .ok()
            .map(Arc::new)
        });

        type Packet = WirePiece<LatticeHomomorphicAuthenticator<97>>;

        let decoded = source_auth.zip(relay_auth).and_then(|(sa, ra)| {
            let transports = InMemoryNetwork::<Packet>::new(3).into_transports();
            let triple: [InMemoryTransport<Packet>; 3] = transports.try_into().ok()?;
            let [t0, t1, t2] = triple;
            let t0_sender = t0.sender();
            let t1_sender = t1.sender();
            let t1_inbox = t1.into_inbound();
            let t2_inbox = t2.into_inbound();
            drop(t0);

            let (commitment, src_stream) = source(Arc::clone(&sa), orig, vandermonde_rng());
            src_stream
                .take(k)
                .fold(
                    (),
                    Arc::new(move |(), wp: Packet| {
                        t0_sender
                            .send(PeerIndex::from(1), wp)
                            .run()
                            .unwrap_or(());
                    }),
                )
                .run()
                .unwrap_or(());

            let node_1_inbox = t1_inbox
                .take(k)
                .map(Arc::new(|(_, wp): (PeerIndex, Packet)| wp));
            let fanned = relay_fanout(
                ra,
                commitment,
                k,
                b,
                1,
                node_1_inbox,
                vandermonde_rng(),
            )
            .run()
            .unwrap_or_else(|_| Stream::from_vec(Vec::new()));
            fanned
                .fold(
                    (),
                    Arc::new(
                        move |(), (_, wp): (PeerIndex, Packet)| {
                            t1_sender
                                .send(PeerIndex::from(2), wp)
                                .run()
                                .unwrap_or(());
                        },
                    ),
                )
                .run()
                .unwrap_or(());

            let node_2_inbox = t2_inbox
                .take(k)
                .map(Arc::new(|(_, wp): (PeerIndex, Packet)| wp));
            receive(sa, commitment, k, b, node_2_inbox).run().ok()
        });
        assert_eq!(decoded, Some(data));
    }
}
