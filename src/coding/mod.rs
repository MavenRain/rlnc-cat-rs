//! Random Linear Network Coding operations.
//!
//! This module provides the three core RLNC operations:
//!
//! - [`encode`]: Generate coded pieces from original data.
//! - [`decode`]: Reconstruct original data from coded pieces.
//! - [`recode`]: Generate new coded pieces from existing ones
//!   without decoding.
//!
//! The [`piece`] module defines the data types shared across
//! all three operations.

pub mod decode;
pub mod encode;
pub mod piece;
pub mod recode;
