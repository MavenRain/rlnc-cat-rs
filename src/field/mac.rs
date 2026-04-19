//! Fused multiply-accumulate over GF(2^8).
//!
//! [`mac`] computes `result[i] = acc[i] + scalar * v[i]` element-wise,
//! where `+` is XOR and `*` is GF(2^8) multiplication.  This is the
//! inner kernel of every RLNC operation: source encoding, recoding,
//! and the row-reduction step of Gaussian elimination.
//!
//! # Implementation
//!
//! Pure safe Rust using iterator combinators.  For a fixed `scalar`,
//! we compute two 16-byte *half-nibble* tables once per call:
//!
//! - `low[i] = scalar * i` for `i` in `0..16`.
//! - `high[i] = scalar * (i << 4)` for `i` in `0..16`.
//!
//! Every byte `b` decomposes as `b = (b & 0x0F) | ((b >> 4) << 4)`,
//! and GF(2^8) multiplication distributes over XOR, so
//! `scalar * b = low[b & 0x0F] XOR high[b >> 4]`.
//!
//! The body is the iterator pipeline
//! `acc.iter().zip(v.iter()).map(mix_byte).collect()`, with all
//! table reads routed through panic-free `Option::unwrap_or`
//! fallthroughs and all nibble-to-index conversions through
//! [`usize::from`] / [`u8::try_from`].  Today LLVM does **not**
//! auto-vectorize this indirect table-lookup pattern into `TBL`
//! (aarch64) or `pshufb` (`x86_64`), so the compute loop runs at
//! scalar throughput.  A true SIMD lowering would require explicit
//! intrinsics or the nightly `core::simd` API; neither is used here,
//! in keeping with the crate's no-`unsafe`, stable-only policy.

use crate::error::Error;
use crate::field::Gf256;

/// Compute `acc + scalar * v` element-wise over GF(2^8), returning a fresh vector.
///
/// # Errors
///
/// Returns [`Error::DimensionMismatch`] when `acc` and `v` have different lengths.
///
/// # Examples
///
/// ```
/// use rlnc_cat_rs::field::{Gf256, mac};
///
/// let acc = [Gf256::new(0x10), Gf256::new(0x20)];
/// let v = [Gf256::new(0x01), Gf256::new(0x02)];
/// let scalar = Gf256::new(0x03);
///
/// assert!(mac(&acc, &v, scalar).is_ok());
///
/// let v_short = [Gf256::new(0x01)];
/// assert!(mac(&acc, &v_short, scalar).is_err());
/// ```
pub fn mac(acc: &[Gf256], v: &[Gf256], scalar: Gf256) -> Result<Vec<Gf256>, Error> {
    if acc.len() == v.len() {
        let (low, high) = half_nibble_tables(scalar);
        Ok(acc
            .iter()
            .zip(v.iter())
            .map(|(&a, &b)| mix_byte(a, b, &low, &high))
            .collect())
    } else {
        Err(Error::DimensionMismatch {
            expected: acc.len(),
            actual: v.len(),
        })
    }
}

/// Combine one accumulator byte with one data byte via the half-nibble
/// table lookup and XOR.  Uses `get(...).copied().unwrap_or(0)` for the
/// table reads: the nibble mask guarantees the index lies in `0..16`,
/// and `0` is the identity for XOR so an out-of-range read would leave
/// the accumulator byte unchanged rather than panic.
#[inline]
fn mix_byte(a: Gf256, b: Gf256, low: &[u8; 16], high: &[u8; 16]) -> Gf256 {
    let bv = b.value();
    let lo_idx = usize::from(bv & 0x0F);
    let hi_idx = usize::from(bv >> 4);
    let lo = low.get(lo_idx).copied().unwrap_or(0);
    let hi = high.get(hi_idx).copied().unwrap_or(0);
    Gf256::new(a.value() ^ lo ^ hi)
}

/// Compute the 16-byte half-nibble multiplication tables for `scalar`.
fn half_nibble_tables(scalar: Gf256) -> ([u8; 16], [u8; 16]) {
    let low: [u8; 16] = core::array::from_fn(|i| {
        let nibble = u8::try_from(i).unwrap_or(0);
        (scalar * Gf256::new(nibble)).value()
    });
    let high: [u8; 16] = core::array::from_fn(|i| {
        let nibble = u8::try_from(i).unwrap_or(0);
        (scalar * Gf256::new(nibble << 4)).value()
    });
    (low, high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mac_matches_elementwise_definition(
            bytes in proptest::collection::vec(any::<u8>(), 0..=512),
            scalar_byte in any::<u8>(),
        ) {
            let half = bytes.len() / 2;
            let acc: Vec<Gf256> = bytes[..half].iter().copied().map(Gf256::new).collect();
            let v: Vec<Gf256> = bytes[half..half * 2].iter().copied().map(Gf256::new).collect();
            let scalar = Gf256::new(scalar_byte);

            let expected: Vec<Gf256> = acc
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| a + scalar * b)
                .collect();

            prop_assert_eq!(mac(&acc, &v, scalar).ok(), Some(expected));
        }
    }

    proptest! {
        #[test]
        fn half_nibble_tables_recover_product(
            scalar_byte in any::<u8>(),
            v_byte in any::<u8>(),
        ) {
            let scalar = Gf256::new(scalar_byte);
            let (low, high) = half_nibble_tables(scalar);
            let expected = (scalar * Gf256::new(v_byte)).value();
            let got = low[(v_byte & 0x0F) as usize] ^ high[(v_byte >> 4) as usize];
            prop_assert_eq!(expected, got);
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let acc = [Gf256::new(1), Gf256::new(2)];
        let v = [Gf256::new(1)];
        assert!(mac(&acc, &v, Gf256::one()).is_err());
    }

    #[test]
    fn empty_inputs_return_empty_output() {
        let acc: [Gf256; 0] = [];
        let v: [Gf256; 0] = [];
        assert_eq!(mac(&acc, &v, Gf256::new(0x42)).ok(), Some(Vec::new()));
    }

    #[test]
    fn scaling_by_zero_equals_acc() {
        let acc: Vec<Gf256> = (0u8..64).map(Gf256::new).collect();
        let v: Vec<Gf256> = (0u8..64).map(|i| Gf256::new(i.wrapping_mul(5))).collect();
        assert_eq!(mac(&acc, &v, Gf256::zero()).ok(), Some(acc));
    }

    #[test]
    fn scaling_by_one_equals_xor() {
        let acc: Vec<Gf256> = (0u8..64).map(Gf256::new).collect();
        let v: Vec<Gf256> = (0u8..64).map(|i| Gf256::new(i.wrapping_mul(7))).collect();
        let expected: Vec<Gf256> = acc.iter().zip(v.iter()).map(|(&a, &b)| a + b).collect();
        assert_eq!(mac(&acc, &v, Gf256::one()).ok(), Some(expected));
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn handles_sizes_through_boundary() {
        // Exercise every length in 0..=33 so the loop behavior over
        // partial-chunk remainders is thoroughly tested.
        (0usize..=33).for_each(|len| {
            let acc: Vec<Gf256> = (0..len).map(|i| Gf256::new((i & 0xFF) as u8)).collect();
            let v: Vec<Gf256> = (0..len)
                .map(|i| Gf256::new(((i * 13) & 0xFF) as u8))
                .collect();
            let s = Gf256::new(0xA7);
            let got = mac(&acc, &v, s).ok();
            let expected: Vec<Gf256> = acc.iter().zip(v.iter()).map(|(&a, &b)| a + s * b).collect();
            assert_eq!(got, Some(expected), "len={len}");
        });
    }
}
