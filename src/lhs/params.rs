//! Scheme parameters for the BF11-style linearly homomorphic signature.
//!
//! [`LhsParams`] packages the dimensional constants, the Gaussian width
//! `σ_g` used by the Klein preimage sampler, and the verification norm
//! bound that must be known by every KeyGen/Sign/Verify caller.  The
//! modulus `Q` is a const generic and the gadget width `k_gadget`
//! derives from it at compile time via `(Q - 1).ilog2() + 1`.

use crate::error::Error;

/// Parameters for the BF11-style linearly homomorphic signature
/// scheme over `Z/QZ`.
///
/// The full matrix width `m` is `m0 + n * k_gadget` where
/// `k_gadget = ⌈log₂ Q⌉` is the number of bits needed to represent
/// an element of `Z/QZ`.  `sigma_g` is the Gaussian width used by the
/// Klein preimage sampler in [`crate::lhs::sign`] to draw a short
/// preimage of each per-piece target.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lhs::LhsParams;
///
/// let p = LhsParams::<97>::new(2, 2, 3, 10_000_000, 3.0).ok();
/// assert_eq!(p.as_ref().map(LhsParams::n), Some(2));
/// assert_eq!(p.as_ref().map(LhsParams::m), Some(2 + 2 * 7));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct LhsParams<const Q: u32> {
    n: usize,
    m0: usize,
    k_pieces: usize,
    sig_norm_bound_sq: u128,
    sigma_g: f64,
}

impl<const Q: u32> LhsParams<Q> {
    /// Build a parameter set.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any of `n`, `m0`, `k_pieces`
    ///   is zero, if `Q < 2`, or if `sigma_g` is not a finite positive
    ///   number.
    pub fn new(
        n: usize,
        m0: usize,
        k_pieces: usize,
        sig_norm_bound_sq: u128,
        sigma_g: f64,
    ) -> Result<Self, Error> {
        if Q >= 2 && n > 0 && m0 > 0 && k_pieces > 0 && sigma_g.is_finite() && sigma_g > 0.0 {
            Ok(Self {
                n,
                m0,
                k_pieces,
                sig_norm_bound_sq,
                sigma_g,
            })
        } else {
            Err(Error::DimensionMismatch {
                expected: 1,
                actual: 0,
            })
        }
    }

    /// Number of rows of the public matrix `A` (security dimension).
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Width of the primary block `A_0` in `A = [A_0 | G - A_0·R]`.
    #[must_use]
    pub fn m0(&self) -> usize {
        self.m0
    }

    /// Number of piece indices signed at setup time.
    #[must_use]
    pub fn k_pieces(&self) -> usize {
        self.k_pieces
    }

    /// Squared verification norm bound `β²`.  A signature `σ` is
    /// accepted only when `||σ||² ≤ β²`.
    #[must_use]
    pub fn sig_norm_bound_sq(&self) -> u128 {
        self.sig_norm_bound_sq
    }

    /// Gaussian width `σ_g` for the Klein preimage sampler.  Drives
    /// the statistical closeness of signatures to a canonical discrete
    /// Gaussian (BFKW09).  Must exceed the maximum Gram-Schmidt norm of
    /// the gadget-kernel basis times the smoothing factor for provable
    /// distribution bounds; in practice `σ_g ≥ 3` is comfortable for
    /// `Q ≤ 2^{16}`.
    #[must_use]
    pub fn sigma_g(&self) -> f64 {
        self.sigma_g
    }

    /// Gadget width per ambient coordinate: the number of bits needed
    /// to represent an element of `Z/QZ`.  Equal to `⌈log₂ Q⌉`.
    #[must_use]
    pub const fn k_gadget() -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let bits = (Q - 1).ilog2() as usize + 1;
        bits
    }

    /// Total gadget-block width `n · k_gadget`.
    #[must_use]
    pub fn gadget_width(&self) -> usize {
        self.n * Self::k_gadget()
    }

    /// Signature dimension `m = m0 + n · k_gadget`.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m0 + self.gadget_width()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_gadget_is_bit_length_of_q_minus_one() {
        assert_eq!(LhsParams::<97>::k_gadget(), 7);
        assert_eq!(LhsParams::<128>::k_gadget(), 7);
        assert_eq!(LhsParams::<129>::k_gadget(), 8);
        assert_eq!(LhsParams::<2>::k_gadget(), 1);
        assert_eq!(LhsParams::<3>::k_gadget(), 2);
    }

    #[test]
    fn m_combines_primary_and_gadget_blocks() {
        let p = LhsParams::<97>::new(2, 4, 3, 10_000, 3.0).ok();
        assert_eq!(p.as_ref().map(LhsParams::m), Some(4 + 2 * 7));
        assert_eq!(p.as_ref().map(LhsParams::gadget_width), Some(14));
    }

    #[test]
    fn rejects_zero_dimensions() {
        assert!(LhsParams::<97>::new(0, 1, 1, 1, 3.0).is_err());
        assert!(LhsParams::<97>::new(1, 0, 1, 1, 3.0).is_err());
        assert!(LhsParams::<97>::new(1, 1, 0, 1, 3.0).is_err());
    }

    #[test]
    fn rejects_trivial_modulus() {
        assert!(LhsParams::<1>::new(1, 1, 1, 1, 3.0).is_err());
    }

    #[test]
    fn rejects_nonpositive_sigma_g() {
        assert!(LhsParams::<97>::new(1, 1, 1, 1, 0.0).is_err());
        assert!(LhsParams::<97>::new(1, 1, 1, 1, -1.0).is_err());
        assert!(LhsParams::<97>::new(1, 1, 1, 1, f64::NAN).is_err());
        assert!(LhsParams::<97>::new(1, 1, 1, 1, f64::INFINITY).is_err());
    }

    #[test]
    fn accessors_return_stored_values() {
        let p = LhsParams::<97>::new(5, 7, 11, 9_999, 4.5).ok();
        assert_eq!(p.as_ref().map(LhsParams::n), Some(5));
        assert_eq!(p.as_ref().map(LhsParams::m0), Some(7));
        assert_eq!(p.as_ref().map(LhsParams::k_pieces), Some(11));
        assert_eq!(p.as_ref().map(LhsParams::sig_norm_bound_sq), Some(9_999));
        assert_eq!(p.as_ref().map(LhsParams::sigma_g), Some(4.5));
    }
}
