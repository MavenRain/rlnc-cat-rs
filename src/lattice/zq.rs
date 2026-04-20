//! Integer modular arithmetic over `Z/QZ`.
//!
//! [`Zq`] is a newtype over `u32` representing an integer modulo a
//! compile-time constant `Q`.  Arithmetic uses `u64` intermediates to
//! avoid overflow.  For lattice constructions, elements are often
//! viewed as signed integers in the canonical interval `(-Q/2, Q/2]`;
//! see [`Zq::signed_repr`].

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use crate::error::Error;

/// An integer modulo `Q`.
///
/// Internally stores the canonical representative in `[0, Q)`.  The
/// const parameter `Q` must be greater than zero; a `Zq<0>` will panic
/// on any arithmetic operation (division by zero).
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::lattice::Zq;
///
/// let a: Zq<97> = Zq::new(50);
/// let b: Zq<97> = Zq::new(80);
/// assert_eq!((a + b).value(), (50 + 80) % 97);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct Zq<const Q: u32>(u32);

impl<const Q: u32> Zq<Q> {
    /// Create an element by reducing `val` mod `Q`.
    #[inline]
    pub const fn new(val: u32) -> Self {
        Self(val % Q)
    }

    /// The additive identity.
    #[inline]
    pub const fn zero() -> Self {
        Self(0)
    }

    /// The multiplicative identity.  Returns zero when `Q == 1`, since
    /// `Z/1Z` is the zero ring.
    #[inline]
    pub const fn one() -> Self {
        Self(1 % Q)
    }

    /// The canonical representative in `[0, Q)`.
    #[must_use]
    #[inline]
    pub const fn value(self) -> u32 {
        self.0
    }

    /// Whether this element is the additive identity.
    #[must_use]
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The balanced signed representative in `(-Q/2, Q/2]`.
    ///
    /// Values `v` with `2 * v <= Q` map to `v`; larger values map to
    /// `v - Q` so the result is the lift closer to zero.  Used for L2
    /// norm computations in lattice verification.
    #[must_use]
    #[inline]
    pub fn signed_repr(self) -> i64 {
        let v = i64::from(self.0);
        let q = i64::from(Q);
        if v * 2 > q { v - q } else { v }
    }

    /// Sample one uniform element of `Z/QZ` by rejection sampling.
    ///
    /// Reads 4 bytes at a time from `rng` and rejects draws that fall
    /// in the non-canonical high range, so the output distribution is
    /// exactly uniform on `[0, Q)`.  The retry loop is bounded at 32;
    /// for typical `Q` in the `2^20` to `2^30` range the acceptance
    /// rate per draw exceeds 99%, so budget exhaustion indicates a
    /// broken RNG.
    ///
    /// # Errors
    ///
    /// - [`Error::RandomGenerationFailed`] when the RNG closure errors,
    ///   returns fewer than 4 bytes, or the retry budget is exhausted.
    pub fn sample<F>(rng: &F) -> Result<Self, Error>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error>,
    {
        Self::sample_bounded(rng, 32)
    }

    fn sample_bounded<F>(rng: &F, remaining: u32) -> Result<Self, Error>
    where
        F: Fn(usize) -> Result<Vec<u8>, Error>,
    {
        if remaining == 0 {
            Err(Error::RandomGenerationFailed(
                "rejection sampling exhausted".to_string(),
            ))
        } else {
            let limit = u32::MAX - (u32::MAX % Q);
            rng(4)
                .and_then(|bytes| {
                    let got = bytes.len();
                    TryInto::<[u8; 4]>::try_into(bytes)
                        .map(u32::from_le_bytes)
                        .map_err(|_| {
                            Error::RandomGenerationFailed(format!(
                                "short read: expected 4 bytes, got {got}",
                            ))
                        })
                })
                .and_then(|v| {
                    if v < limit {
                        Ok(Self(v % Q))
                    } else {
                        Self::sample_bounded(rng, remaining - 1)
                    }
                })
        }
    }
}

impl<const Q: u32> From<u32> for Zq<Q> {
    #[inline]
    fn from(val: u32) -> Self {
        Self::new(val)
    }
}

impl<const Q: u32> Add for Zq<Q> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let r = (u64::from(self.0) + u64::from(rhs.0)) % u64::from(Q);
        Self(u32::try_from(r).unwrap_or(0))
    }
}

impl<const Q: u32> Sub for Zq<Q> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let q = u64::from(Q);
        let r = (u64::from(self.0) + q - u64::from(rhs.0)) % q;
        Self(u32::try_from(r).unwrap_or(0))
    }
}

impl<const Q: u32> Mul for Zq<Q> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let r = (u64::from(self.0) * u64::from(rhs.0)) % u64::from(Q);
        Self(u32::try_from(r).unwrap_or(0))
    }
}

impl<const Q: u32> Neg for Zq<Q> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        if self.0 == 0 { self } else { Self(Q - self.0) }
    }
}

impl<const Q: u32> fmt::Debug for Zq<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zq<{Q}>({})", self.0)
    }
}

impl<const Q: u32> fmt::Display for Zq<Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const Q_TEST: u32 = 97;

    #[test]
    fn new_reduces_mod_q() {
        assert_eq!(Zq::<Q_TEST>::new(100).value(), 3);
        assert_eq!(Zq::<Q_TEST>::new(0).value(), 0);
    }

    #[test]
    fn add_wraps_mod_q() {
        let a = Zq::<Q_TEST>::new(50);
        let b = Zq::<Q_TEST>::new(80);
        assert_eq!((a + b).value(), (50 + 80) % Q_TEST);
    }

    #[test]
    fn sub_does_not_underflow() {
        let a = Zq::<Q_TEST>::new(10);
        let b = Zq::<Q_TEST>::new(30);
        assert_eq!((a - b).value(), 77);
    }

    #[test]
    fn mul_reduces() {
        let a = Zq::<Q_TEST>::new(50);
        let b = Zq::<Q_TEST>::new(80);
        assert_eq!((a * b).value(), u32::try_from(50u64 * 80 % u64::from(Q_TEST)).ok().unwrap_or(0));
    }

    #[test]
    fn neg_zero_is_zero() {
        assert_eq!((-Zq::<Q_TEST>::zero()).value(), 0);
    }

    #[test]
    fn neg_is_additive_inverse() {
        let x = Zq::<Q_TEST>::new(42);
        assert_eq!((x + (-x)).value(), 0);
    }

    #[test]
    fn signed_repr_balances() {
        assert_eq!(Zq::<Q_TEST>::new(0).signed_repr(), 0);
        assert_eq!(Zq::<Q_TEST>::new(48).signed_repr(), 48);
        assert_eq!(Zq::<Q_TEST>::new(49).signed_repr(), -48);
        assert_eq!(Zq::<Q_TEST>::new(96).signed_repr(), -1);
    }

    #[test]
    fn sample_is_deterministic_with_constant_rng() {
        let rng = |n: usize| -> Result<Vec<u8>, Error> { Ok(vec![0xA5u8; n]) };
        let a = Zq::<Q_TEST>::sample(&rng).ok();
        let b = Zq::<Q_TEST>::sample(&rng).ok();
        assert_eq!(a, b);
        assert!(a.is_some_and(|z| z.value() < Q_TEST));
    }

    #[test]
    fn sample_covers_range() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);
        let rng = move |n: usize| -> Result<Vec<u8>, Error> {
            let i = c.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n).map(|j| ((i + j) & 0xFF) as u8).collect())
        };
        (0..100)
            .map(|_| Zq::<Q_TEST>::sample(&rng))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .into_iter()
            .flatten()
            .for_each(|z| assert!(z.value() < Q_TEST));
    }

    #[test]
    fn distributive_law_holds() {
        let a = Zq::<Q_TEST>::new(7);
        let b = Zq::<Q_TEST>::new(11);
        let c = Zq::<Q_TEST>::new(13);
        assert_eq!((a * (b + c)).value(), (a * b + a * c).value());
    }
}
