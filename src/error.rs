//! Single project-wide error type.
//!
//! Every fallible operation in this crate returns [`Error`].

use core::fmt;

/// The error type for all RLNC operations.
#[derive(Debug)]
pub enum Error {
    /// The input data slice was empty.
    EmptyData,

    /// The requested piece count was zero.
    InvalidPieceCount,

    /// A coded piece had a different byte length than expected.
    PieceSizeMismatch {
        /// The expected piece byte length.
        expected: usize,
        /// The actual piece byte length.
        actual: usize,
    },

    /// A coding vector had a different length than the piece count.
    CodingVectorLengthMismatch {
        /// The expected length (piece count).
        expected: usize,
        /// The actual length.
        actual: usize,
    },

    /// Not enough linearly independent pieces to decode.
    InsufficientPieces {
        /// The number of pieces needed.
        needed: usize,
        /// The number of useful pieces received.
        received: usize,
    },

    /// Division by zero in GF(2^8).
    DivisionByZero,

    /// The boundary marker was not found or padding was invalid.
    InvalidPadding,

    /// Two operands had incompatible dimensions.
    DimensionMismatch {
        /// The expected dimension.
        expected: usize,
        /// The actual dimension.
        actual: usize,
    },

    /// The random byte source returned an error.
    RandomGenerationFailed(String),

    /// An authenticator rejected a piece: its tag did not match the
    /// expected value for the given commitment.
    AuthenticatorRejected,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyData => write!(f, "input data is empty"),
            Self::InvalidPieceCount => write!(f, "piece count must be at least 1"),
            Self::PieceSizeMismatch { expected, actual } => {
                write!(f, "piece size mismatch: expected {expected}, got {actual}")
            }
            Self::CodingVectorLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "coding vector length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InsufficientPieces { needed, received } => {
                write!(
                    f,
                    "insufficient pieces: need {needed}, received {received}"
                )
            }
            Self::DivisionByZero => write!(f, "division by zero in GF(2^8)"),
            Self::InvalidPadding => {
                write!(f, "boundary marker not found or padding is invalid")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::RandomGenerationFailed(msg) => {
                write!(f, "random generation failed: {msg}")
            }
            Self::AuthenticatorRejected => {
                write!(f, "authenticator rejected the piece")
            }
        }
    }
}

impl std::error::Error for Error {}
