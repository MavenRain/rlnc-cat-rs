//! Random Linear Network Coding over GF(2^8), built on
//! [comp-cat-rs](https://docs.rs/comp-cat-rs).
//!
//! This crate implements the three core operations of RLNC:
//!
//! - **Encoding**: split data into k pieces and generate coded pieces
//!   as random linear combinations over GF(2^8).
//! - **Decoding**: given k linearly independent coded pieces,
//!   reconstruct the original data via Gaussian elimination.
//! - **Recoding**: generate new coded pieces from existing coded
//!   pieces without decoding, enabling multi-hop distribution.
//!
//! The pure linear algebra core (finite field arithmetic, vector
//! operations, Gaussian elimination) lives in [`field`] and
//! [`vector`].  The effectful operations (random coefficient
//! generation, streaming) use comp-cat-rs's [`Io`] and [`Stream`]
//! effect types via the [`coding`] module.
//!
//! [`Io`]: comp_cat_rs::effect::io::Io
//! [`Stream`]: comp_cat_rs::effect::stream::Stream
//!
//! # Quick start
//!
//! ```
//! use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
//! use rlnc_cat_rs::coding::encode::{encode_with_vector, Encoder};
//! use rlnc_cat_rs::coding::decode::DecoderState;
//!
//! // 1. Prepare original data
//! let data = vec![72, 101, 108, 108, 111];
//! let orig = OriginalData::from_bytes(&data, 3).ok();
//!
//! // 2. Encode with the Io-based encoder
//! let encoder = orig.map(Encoder::new);
//! let piece = encoder.as_ref().map(|enc: &Encoder| {
//!     enc.encode_one(|n| Ok(vec![42u8; n])).run()
//! });
//! assert!(piece.and_then(|r| r.ok()).is_some());
//! ```

pub mod auth;
pub mod coding;
pub mod error;
pub mod field;
pub mod gossip;
pub mod transport;
pub mod vector;
