//! The Galois field element GF(2^8).
//!
//! [`Gf256`] is a newtype over `u8` representing an element of
//! GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1.
//!
//! Addition and subtraction are both XOR (characteristic 2).
//! Multiplication uses log/exp table lookup.  Division is provided
//! via [`Gf256::checked_div`] since division by zero cannot be
//! expressed through `core::ops::Div`.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

use super::tables::{EXP_TABLE, LOG_TABLE};
use crate::error::Error;

/// An element of the Galois field GF(2^8).
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::field::Gf256;
///
/// let a = Gf256::new(0x53);
/// let b = Gf256::new(0xCA);
/// let sum = a + b;
/// assert_eq!(sum.value(), 0x53 ^ 0xCA);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct Gf256(u8);

impl Gf256 {
    /// Create a field element from a raw byte.
    pub const fn new(val: u8) -> Self {
        Self(val)
    }

    /// The additive identity (zero).
    pub const fn zero() -> Self {
        Self(0)
    }

    /// The multiplicative identity (one).
    pub const fn one() -> Self {
        Self(1)
    }

    /// The raw byte value of this element.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Whether this element is the additive identity.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Multiplicative inverse.
    ///
    /// # Errors
    ///
    /// Returns `Err(Error::DivisionByZero)` when called on zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlnc_cat_rs::field::Gf256;
    ///
    /// let a = Gf256::new(0x53);
    /// let a_inv = a.inv().ok();
    /// let product = a_inv.map(|inv| a * inv);
    /// assert_eq!(product, Some(Gf256::one()));
    /// ```
    pub fn inv(self) -> Result<Self, Error> {
        if self.is_zero() {
            Err(Error::DivisionByZero)
        } else {
            let log_a = usize::from(LOG_TABLE.get(usize::from(self.0)).copied().unwrap_or(0));
            Ok(Self(EXP_TABLE.get(255 - log_a).copied().unwrap_or(0)))
        }
    }

    /// Checked division in GF(2^8).
    ///
    /// # Errors
    ///
    /// Returns `Err(Error::DivisionByZero)` when dividing by zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlnc_cat_rs::field::Gf256;
    ///
    /// let a = Gf256::new(0x53);
    /// let b = Gf256::new(0xCA);
    /// let quotient = a.checked_div(b);
    /// let roundtrip = quotient.map(|q| q * b);
    /// assert_eq!(roundtrip.ok(), Some(a));
    /// ```
    pub fn checked_div(self, rhs: Self) -> Result<Self, Error> {
        rhs.inv().map(|rhs_inv| self * rhs_inv)
    }
}

impl Add for Gf256 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for Gf256 {
    type Output = Self;

    /// Subtraction in GF(2^8) is identical to addition (characteristic 2).
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl Mul for Gf256 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else {
            let log_a = usize::from(LOG_TABLE.get(usize::from(self.0)).copied().unwrap_or(0));
            let log_b = usize::from(LOG_TABLE.get(usize::from(rhs.0)).copied().unwrap_or(0));
            Self(EXP_TABLE.get(log_a + log_b).copied().unwrap_or(0))
        }
    }
}

impl Neg for Gf256 {
    type Output = Self;

    /// Negation in GF(2^8) is identity (characteristic 2: -x = x).
    fn neg(self) -> Self {
        self
    }
}

impl fmt::Debug for Gf256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gf256(0x{:02X})", self.0)
    }
}

impl fmt::Display for Gf256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:02X}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_identity() {
        (0u16..256).for_each(|x| {
            let a = Gf256::new(x as u8);
            assert_eq!(a + Gf256::zero(), a);
        });
    }

    #[test]
    fn multiplicative_identity() {
        (0u16..256).for_each(|x| {
            let a = Gf256::new(x as u8);
            assert_eq!(a * Gf256::one(), a);
        });
    }

    #[test]
    fn additive_inverse_is_self() {
        // In characteristic 2, a + a = 0
        (0u16..256).for_each(|x| {
            let a = Gf256::new(x as u8);
            assert_eq!(a + a, Gf256::zero());
        });
    }

    #[test]
    fn multiplicative_inverse_roundtrip() {
        (1u16..256).for_each(|x| {
            let a = Gf256::new(x as u8);
            let a_inv = a.inv();
            assert!(a_inv.is_ok());
            assert_eq!(a * a_inv.unwrap_or(Gf256::zero()), Gf256::one());
        });
    }

    #[test]
    fn zero_has_no_inverse() {
        assert!(Gf256::zero().inv().is_err());
    }

    #[test]
    fn checked_div_by_zero_fails() {
        let a = Gf256::new(42);
        assert!(a.checked_div(Gf256::zero()).is_err());
    }

    #[test]
    fn checked_div_roundtrip() {
        (1u16..256).for_each(|x| {
            (1u16..256).for_each(|y| {
                let a = Gf256::new(x as u8);
                let b = Gf256::new(y as u8);
                let q = a.checked_div(b);
                assert!(q.is_ok());
                assert_eq!(q.unwrap_or(Gf256::zero()) * b, a);
            });
        });
    }

    #[test]
    fn sub_equals_add() {
        (0u16..256).for_each(|x| {
            (0u16..256).for_each(|y| {
                let a = Gf256::new(x as u8);
                let b = Gf256::new(y as u8);
                assert_eq!(a - b, a + b);
            });
        });
    }

    #[test]
    fn negation_is_identity() {
        (0u16..256).for_each(|x| {
            let a = Gf256::new(x as u8);
            assert_eq!(-a, a);
        });
    }
}
