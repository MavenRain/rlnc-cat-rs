//! Discrete Gaussian sampler over `Z`.
//!
//! [`discrete_gaussian`] produces an `i64` drawn from `D_{Z, σ, c}`:
//! the distribution on integers with density proportional to
//! `exp(-(x - c)^2 / (2σ^2))`.  The implementation uses rejection
//! sampling inside a `±12σ` tail cutoff, which drops an `e^{-72}`
//! mass (well below any security parameter we will set).
//!
//! The sampler relies on `f64` arithmetic for the exponential; it is
//! therefore not constant-time and is **not** suitable for production
//! side-channel-resistant deployments.  It is algebraically correct
//! and sufficient for the correctness goals of this crate.

use crate::error::Error;

const TAIL_SIGMAS: f64 = 12.0;
const RETRY_BUDGET: u32 = 128;

/// Sample one integer from `D_{Z, σ, c}` by rejection sampling.
///
/// # Errors
///
/// - [`Error::RandomGenerationFailed`] when `σ` or `c` is not finite,
///   `σ ≤ 0`, the RNG returns fewer bytes than requested, or the retry
///   budget of `128` is exhausted (indicating a broken RNG or
///   pathological parameters).
pub fn discrete_gaussian<F>(sigma: f64, center: f64, rng: &F) -> Result<i64, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    if !sigma.is_finite() || sigma <= 0.0 {
        Err(Error::RandomGenerationFailed(format!(
            "invalid sigma: {sigma}",
        )))
    } else if !center.is_finite() {
        Err(Error::RandomGenerationFailed(format!(
            "invalid center: {center}",
        )))
    } else {
        let tail = TAIL_SIGMAS * sigma;
        let low_f = (center - tail).floor();
        let high_f = (center + tail).ceil();
        // f64 -> i64 saturating cast; bounded by sigma validation above.
        #[allow(clippy::cast_possible_truncation)]
        let low = low_f as i64;
        #[allow(clippy::cast_possible_truncation)]
        let high = high_f as i64;
        let range = u64::try_from((high - low + 1).max(1)).unwrap_or(1);
        let denom = 2.0 * sigma * sigma;
        sample_step(rng, low, range, center, denom, RETRY_BUDGET)
    }
}

fn sample_step<F>(
    rng: &F,
    low: i64,
    range: u64,
    center: f64,
    denom: f64,
    remaining: u32,
) -> Result<i64, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    if remaining == 0 {
        Err(Error::RandomGenerationFailed(
            "discrete Gaussian rejection exhausted".to_string(),
        ))
    } else {
        draw_candidate(rng, low, range, center, denom).and_then(|maybe_x| {
            maybe_x.map_or_else(
                || sample_step(rng, low, range, center, denom, remaining - 1),
                Ok,
            )
        })
    }
}

/// Produce one candidate and either accept (`Some(x)`) or reject (`None`).
fn draw_candidate<F>(
    rng: &F,
    low: i64,
    range: u64,
    center: f64,
    denom: f64,
) -> Result<Option<i64>, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    read_u64(rng).and_then(|proposal_u64| {
        read_u64(rng).map(|accept_u64| {
            let offset = i64::try_from(proposal_u64 % range).unwrap_or(0);
            let x = low + offset;
            // 53-bit uniform on [0, 1): `>> 11` drops the low bits so the
            // u64 fits exactly in f64's 53-bit mantissa.
            #[allow(clippy::cast_precision_loss)]
            let u = (accept_u64 >> 11) as f64 / ((1u64 << 53) as f64);
            // i64 -> f64 is lossy for |x| >= 2^53 but the sampler bound
            // is 12 * sigma, far below that threshold in practice.
            #[allow(clippy::cast_precision_loss)]
            let delta = x as f64 - center;
            let accept_prob = (-delta * delta / denom).exp();
            if u < accept_prob { Some(x) } else { None }
        })
    })
}

fn read_u64<F>(rng: &F) -> Result<u64, Error>
where
    F: Fn(usize) -> Result<Vec<u8>, Error>,
{
    rng(8).and_then(|bytes| {
        let got = bytes.len();
        TryInto::<[u8; 8]>::try_into(bytes)
            .map(u64::from_le_bytes)
            .map_err(|_| {
                Error::RandomGenerationFailed(format!(
                    "short read: expected 8 bytes, got {got}",
                ))
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counter_rng() -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
        let counter = Arc::new(AtomicUsize::new(0));
        move |n: usize| -> Result<Vec<u8>, Error> {
            let i = counter.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            Ok((0..n).map(|j| ((i.wrapping_mul(0x9E37_79B9) + j) & 0xFF) as u8).collect())
        }
    }

    #[test]
    fn rejects_nonpositive_sigma() {
        let rng = |n: usize| Ok(vec![0u8; n]);
        assert!(discrete_gaussian(0.0, 0.0, &rng).is_err());
        assert!(discrete_gaussian(-1.0, 0.0, &rng).is_err());
        assert!(discrete_gaussian(f64::NAN, 0.0, &rng).is_err());
        assert!(discrete_gaussian(f64::INFINITY, 0.0, &rng).is_err());
    }

    #[test]
    fn rejects_non_finite_center() {
        let rng = |n: usize| Ok(vec![0u8; n]);
        assert!(discrete_gaussian(1.0, f64::NAN, &rng).is_err());
        assert!(discrete_gaussian(1.0, f64::INFINITY, &rng).is_err());
    }

    #[test]
    fn output_lies_in_tail_cutoff() {
        let rng = counter_rng();
        let sigma = 3.0;
        let center = 7.0;
        let bound = TAIL_SIGMAS * sigma;
        #[allow(clippy::cast_possible_truncation)]
        let low = (center - bound).floor() as i64;
        #[allow(clippy::cast_possible_truncation)]
        let high = (center + bound).ceil() as i64;
        (0..50)
            .map(|_| discrete_gaussian(sigma, center, &rng))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .into_iter()
            .flatten()
            .for_each(|x| assert!(x >= low && x <= high));
    }

    #[test]
    fn empirical_mean_is_near_center() {
        let rng = counter_rng();
        let sigma = 4.0;
        let center = 2.0;
        let samples: Vec<i64> = (0..4000)
            .filter_map(|_| discrete_gaussian(sigma, center, &rng).ok())
            .collect();
        assert!(samples.len() > 3500);
        #[allow(clippy::cast_precision_loss)]
        let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
        // Loose tolerance: rejection sampler + 4k samples, mean in [center ± 0.5].
        assert!((mean - center).abs() < 0.5, "mean = {mean}, center = {center}");
    }

    #[test]
    fn empirical_variance_is_near_sigma_squared() {
        let rng = counter_rng();
        let sigma = 5.0;
        let samples: Vec<i64> = (0..4000)
            .filter_map(|_| discrete_gaussian(sigma, 0.0, &rng).ok())
            .collect();
        assert!(samples.len() > 3500);
        #[allow(clippy::cast_precision_loss)]
        let n = samples.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
        #[allow(clippy::cast_precision_loss)]
        let var = samples
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        // Should be near sigma² = 25.  Wide tolerance.
        assert!(var > 15.0 && var < 40.0, "variance = {var}");
    }

    #[test]
    fn determinism_with_identical_rng_streams() {
        let a = discrete_gaussian(3.0, 0.0, &counter_rng()).ok();
        let b = discrete_gaussian(3.0, 0.0, &counter_rng()).ok();
        assert_eq!(a, b);
    }
}
