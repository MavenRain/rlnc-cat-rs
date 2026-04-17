//! Immutable vectors over GF(2^8).
//!
//! [`GfVec`] is the fundamental building block for coding vectors
//! and data pieces.  All operations return new vectors; nothing
//! is mutated.

use crate::error::Error;
use crate::field::Gf256;

/// An immutable vector of GF(2^8) elements.
///
/// Used for coding vectors (the coefficients that describe how a
/// coded piece was formed) and for data pieces (the payload bytes
/// interpreted as field elements).
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::vector::GfVec;
/// use rlnc_cat_rs::field::Gf256;
///
/// let v = GfVec::from_bytes(&[1, 2, 3]);
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.elements()[0], Gf256::new(1));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct GfVec {
    elements: Vec<Gf256>,
}

impl GfVec {
    /// Create a vector from a slice of field elements.
    pub fn new(elements: Vec<Gf256>) -> Self {
        Self { elements }
    }

    /// Create a zero vector of the given length.
    pub fn zeros(len: usize) -> Self {
        Self {
            elements: vec![Gf256::zero(); len],
        }
    }

    /// Create a vector from raw bytes, interpreting each byte as
    /// a GF(2^8) element.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            elements: data.iter().copied().map(Gf256::new).collect(),
        }
    }

    /// Convert back to raw bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.elements.iter().map(|e| e.value()).collect()
    }

    /// The number of elements in this vector.
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Whether this vector has zero length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// A borrowed slice of the elements.
    pub fn elements(&self) -> &[Gf256] {
        &self.elements
    }

    /// The element at the given index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Gf256> {
        self.elements.get(index).copied()
    }

    /// Element-wise addition (XOR) of two vectors.
    ///
    /// Returns `Err(Error::DimensionMismatch)` if lengths differ.
    ///
    /// # Errors
    ///
    /// Returns an error when `self` and `other` have different lengths.
    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        if self.len() == other.len() {
            Ok(Self {
                elements: self
                    .elements
                    .iter()
                    .zip(other.elements.iter())
                    .map(|(&a, &b)| a + b)
                    .collect(),
            })
        } else {
            Err(Error::DimensionMismatch {
                expected: self.len(),
                actual: other.len(),
            })
        }
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, scalar: Gf256) -> Self {
        Self {
            elements: self.elements.iter().map(|&e| e * scalar).collect(),
        }
    }

    /// Fused multiply-accumulate: `self + scalar * other`.
    ///
    /// Equivalent to `self.add(&other.scale(scalar))?` but performs a
    /// single pass with one allocation.  This is the inner kernel of
    /// every Gaussian elimination row reduction and every RLNC
    /// coded-piece generation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] when `self` and `other` have
    /// different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use rlnc_cat_rs::field::Gf256;
    /// use rlnc_cat_rs::vector::GfVec;
    ///
    /// let acc = GfVec::from_bytes(&[0x10, 0x20, 0x30]);
    /// let v = GfVec::from_bytes(&[0x01, 0x02, 0x03]);
    /// let scalar = Gf256::new(0x05);
    ///
    /// let result = acc.mac(&v, scalar).ok();
    /// assert_eq!(result.map(|r| r.len()), Some(3));
    /// ```
    pub fn mac(&self, other: &Self, scalar: Gf256) -> Result<Self, Error> {
        crate::field::mac(&self.elements, &other.elements, scalar).map(Self::new)
    }

    /// Compute the linear combination `sum(coeffs[i] * vecs[i])`.
    ///
    /// All vectors must have the same length.  The number of coefficients
    /// must equal the number of vectors.  Each term folds through the
    /// fused [`mac`](Self::mac) primitive, so the traversal does one
    /// allocation per input vector instead of two.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions are inconsistent.
    pub fn linear_combine(coeffs: &[Gf256], vecs: &[&Self]) -> Result<Self, Error> {
        if coeffs.len() == vecs.len() {
            let dim = vecs.first().map_or(0, |v| v.len());
            let mismatch = vecs.iter().find(|v| v.len() != dim);
            if let Some(bad) = mismatch {
                Err(Error::DimensionMismatch {
                    expected: dim,
                    actual: bad.len(),
                })
            } else {
                coeffs
                    .iter()
                    .zip(vecs.iter())
                    .try_fold(Self::zeros(dim), |acc, (&c, v)| acc.mac(v, c))
            }
        } else {
            Err(Error::DimensionMismatch {
                expected: coeffs.len(),
                actual: vecs.len(),
            })
        }
    }

    /// Create a new vector with the element at `index` replaced.
    ///
    /// Returns `None` if `index` is out of bounds.
    #[must_use]
    pub fn with_element_at(&self, index: usize, value: Gf256) -> Option<Self> {
        if index >= self.len() {
            None
        } else {
            Some(Self {
                elements: self
                    .elements
                    .iter()
                    .enumerate()
                    .map(|(i, &e)| if i == index { value } else { e })
                    .collect(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bytes_roundtrip() {
        let data = vec![0, 1, 2, 255, 128];
        let v = GfVec::from_bytes(&data);
        assert_eq!(v.to_bytes(), data);
    }

    #[test]
    fn add_is_xor() {
        let a = GfVec::from_bytes(&[0x53, 0xCA]);
        let b = GfVec::from_bytes(&[0x11, 0x22]);
        let sum = a.add(&b);
        assert!(sum.is_ok());
        assert_eq!(sum.unwrap_or(GfVec::zeros(0)).to_bytes(), vec![0x42, 0xE8]);
    }

    #[test]
    fn add_dimension_mismatch() {
        let a = GfVec::from_bytes(&[1, 2]);
        let b = GfVec::from_bytes(&[1, 2, 3]);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn scale_by_zero_is_zero() {
        let v = GfVec::from_bytes(&[0x53, 0xCA, 0xFF]);
        let scaled = v.scale(Gf256::zero());
        assert_eq!(scaled, GfVec::zeros(3));
    }

    #[test]
    fn scale_by_one_is_identity() {
        let v = GfVec::from_bytes(&[0x53, 0xCA, 0xFF]);
        let scaled = v.scale(Gf256::one());
        assert_eq!(scaled, v);
    }

    #[test]
    fn linear_combine_single_vector() {
        let v = GfVec::from_bytes(&[1, 2, 3]);
        let scalar = Gf256::new(5);
        let result = GfVec::linear_combine(&[scalar], &[&v]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_or(GfVec::zeros(0)), v.scale(scalar));
    }

    #[test]
    fn linear_combine_empty() {
        let result = GfVec::linear_combine(&[], &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap_or(GfVec::from_bytes(&[1])), GfVec::zeros(0));
    }

    #[test]
    fn mac_matches_scale_then_add() {
        let acc = GfVec::from_bytes(&[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]);
        let v = GfVec::from_bytes(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        let scalar = Gf256::new(0x1B);

        let via_mac = acc.mac(&v, scalar).ok();
        let via_scale_add = acc.add(&v.scale(scalar)).ok();

        assert_eq!(via_mac, via_scale_add);
    }

    #[test]
    fn mac_rejects_dimension_mismatch() {
        let a = GfVec::from_bytes(&[1, 2, 3]);
        let b = GfVec::from_bytes(&[4, 5]);
        assert!(a.mac(&b, Gf256::one()).is_err());
    }

    #[test]
    fn mac_by_zero_equals_acc() {
        let acc = GfVec::from_bytes(&[0x11, 0x22, 0x33, 0x44]);
        let v = GfVec::from_bytes(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let result = acc.mac(&v, Gf256::zero()).ok();
        assert_eq!(result, Some(acc));
    }
}
