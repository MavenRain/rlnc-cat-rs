//! Gadget matrix and its bit-decomposition preimage.
//!
//! The gadget matrix `G ∈ (Z/QZ)^{n × n·k}` is the block-diagonal
//! matrix `I_n ⊗ [1, 2, 4, …, 2^{k-1}]`, where `k = ⌈log₂ Q⌉`.  It
//! admits a trivial short preimage: the bits of any target
//! `u ∈ (Z/QZ)^n`, concatenated into `{0,1}^{n·k}`, satisfy `G·w ≡ u`.
//!
//! This module exposes the two primitives [`gadget_preimage`]
//! (bits of `u`) and [`gadget_apply`] (reassembly `G·w`), plus the
//! entry lookup [`gadget_entry`] used to materialise `G` inside the
//! KeyGen step `A_1 = G - A_0 · R`.

use crate::lattice::{Zq, ZqVec};
use crate::lhs::params::LhsParams;

/// The bit-decomposition preimage `w ∈ {0,1}^{n·k}` of `u ∈ Zq^n`
/// under the gadget matrix.  Bits are ordered low-order first within
/// each coordinate block, and blocks are concatenated in the same
/// order as the entries of `u`.
#[must_use]
pub fn gadget_preimage<const Q: u32>(u: &ZqVec<Q>) -> Vec<i64> {
    let k = LhsParams::<Q>::k_gadget();
    u.entries()
        .iter()
        .flat_map(|z| {
            let v = z.value();
            (0..k).map(move |j| {
                #[allow(clippy::cast_possible_truncation)]
                let shift = j as u32;
                i64::from((v >> shift) & 1)
            })
        })
        .collect()
}

/// Apply the gadget matrix: compute `G · w (mod Q)` for `w ∈ Z^{n·k}`
/// and return the resulting `Zq^n`.
///
/// Used by the trapdoor-identity sanity checks in the tests for
/// [`crate::lhs::keys`] and this module; not part of the public API.
///
/// The slice `w` must have length `n * k_gadget`; shorter slices
/// truncate and longer slices are read only through index
/// `n · k_gadget - 1`.
#[cfg(test)]
pub(crate) fn gadget_apply<const Q: u32>(w: &[i64], n: usize) -> ZqVec<Q> {
    let k = LhsParams::<Q>::k_gadget();
    let entries: Vec<Zq<Q>> = (0..n)
        .map(|i| {
            (0..k).fold(Zq::<Q>::zero(), |acc, j| {
                let idx = i * k + j;
                let w_ij = w.get(idx).copied().unwrap_or(0);
                #[allow(clippy::cast_possible_truncation)]
                let shift = j as u32;
                let coef_u64 = 1u64 << shift;
                let coef_q = reduce_u64::<Q>(coef_u64);
                let w_q = reduce_i64::<Q>(w_ij);
                acc + coef_q * w_q
            })
        })
        .collect();
    ZqVec::new(entries)
}

/// The `(i, j)` entry of the gadget matrix `G = I_n ⊗ [1, 2, …, 2^{k-1}]`.
/// Returns `0` when the index falls outside the `i`-th block.
pub fn gadget_entry<const Q: u32>(i: usize, j: usize) -> Zq<Q> {
    let k = LhsParams::<Q>::k_gadget();
    if j / k == i {
        let bit = j % k;
        #[allow(clippy::cast_possible_truncation)]
        let shift = bit as u32;
        reduce_u64::<Q>(1u64 << shift)
    } else {
        Zq::<Q>::zero()
    }
}

fn reduce_u64<const Q: u32>(v: u64) -> Zq<Q> {
    #[allow(clippy::cast_possible_truncation)]
    let r = (v % u64::from(Q)) as u32;
    Zq::<Q>::new(r)
}

#[cfg(test)]
fn reduce_i64<const Q: u32>(v: i64) -> Zq<Q> {
    let q = i64::from(Q);
    let r = v.rem_euclid(q);
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    let r_u = r as u32;
    Zq::<Q>::new(r_u)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preimage_then_apply_is_identity_on_small_targets() {
        // All targets in Zq^2 for Q = 7 are tested.
        (0u32..7).for_each(|a| {
            (0u32..7).for_each(|b| {
                let u = ZqVec::<7>::new(vec![Zq::new(a), Zq::new(b)]);
                let w = gadget_preimage(&u);
                let back = gadget_apply::<7>(&w, 2);
                assert_eq!(back, u);
            });
        });
    }

    #[test]
    fn preimage_has_bit_entries_only() {
        let u = ZqVec::<97>::new(vec![Zq::new(42), Zq::new(96)]);
        let w = gadget_preimage(&u);
        w.iter().for_each(|&b| assert!(b == 0 || b == 1));
    }

    #[test]
    fn preimage_length_is_n_times_k_gadget() {
        let u = ZqVec::<97>::new(vec![Zq::new(0), Zq::new(0), Zq::new(0)]);
        let w = gadget_preimage(&u);
        assert_eq!(w.len(), 3 * LhsParams::<97>::k_gadget());
    }

    #[test]
    fn gadget_entry_is_power_of_two_inside_block_and_zero_outside() {
        let k = LhsParams::<97>::k_gadget();
        // Row 0, first block (cols 0..k): entries are 1, 2, 4, ..., 64.
        (0..k).for_each(|j| {
            let expected = u32::try_from(1u64 << j).ok().unwrap_or(0) % 97;
            assert_eq!(gadget_entry::<97>(0, j), Zq::new(expected));
        });
        // Row 0, second block (cols k..2k): all zero.
        (k..2 * k).for_each(|j| {
            assert_eq!(gadget_entry::<97>(0, j), Zq::zero());
        });
        // Row 1, second block: non-zero powers of 2.
        (k..2 * k).for_each(|j| {
            let bit = j - k;
            let expected = u32::try_from(1u64 << bit).ok().unwrap_or(0) % 97;
            assert_eq!(gadget_entry::<97>(1, j), Zq::new(expected));
        });
    }

    #[test]
    fn apply_on_zero_is_zero() {
        let out = gadget_apply::<97>(&[0i64; 14], 2);
        assert_eq!(out, ZqVec::zeros(2));
    }
}
