//! Criterion benchmarks for the BFKW09 linearly homomorphic signature
//! scheme: `sign`, `combine` (swept across `k_pieces`), and `verify`.

#![allow(clippy::needless_for_each)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rlnc_cat_rs::error::Error;
use rlnc_cat_rs::lattice::{Zq, ZqVec};
use rlnc_cat_rs::lhs::{LhsParams, Signature, combine, keygen, sign, verify};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const K_PIECES: &[usize] = &[2, 4, 8, 16];
const N: usize = 4;
const M0: usize = 4;
const SIGMA_G: f64 = 3.0;
const NORM_BOUND_SQ: u128 = 100_000_000;
const Q: u32 = 97;

fn counter_rng() -> impl Fn(usize) -> Result<Vec<u8>, Error> + Send + Sync + 'static {
    let counter = Arc::new(AtomicU64::new(1));
    move |n: usize| {
        Ok((0..n.div_ceil(8))
            .flat_map(|_| counter.fetch_add(1, Ordering::Relaxed).to_le_bytes())
            .take(n)
            .collect())
    }
}

fn fresh_target(i: u32) -> ZqVec<Q> {
    ZqVec::new(
        (0..N)
            .map(|j| {
                let ju = u32::try_from(j).unwrap_or(0);
                Zq::new(i.wrapping_add(ju).wrapping_mul(7).wrapping_add(3))
            })
            .collect(),
    )
}

fn make_params(k: usize) -> Option<LhsParams<Q>> {
    LhsParams::<Q>::new(N, M0, k, NORM_BOUND_SQ, SIGMA_G).ok()
}

fn bench_sign(c: &mut Criterion) {
    let setup = make_params(2).and_then(|p| keygen(p, &counter_rng()).ok());
    setup.iter().for_each(|(pk, sk)| {
        let target = fresh_target(0);
        c.bench_function("lhs::sign", |b| {
            b.iter(|| std::hint::black_box(sign(pk, sk, &target, &counter_rng()).ok()));
        });
    });
}

fn bench_combine(c: &mut Criterion) {
    K_PIECES
        .iter()
        .filter_map(|&k| {
            make_params(k)
                .and_then(|p| keygen(p, &counter_rng()).ok())
                .and_then(|(pk, sk)| {
                    (0..k)
                        .map(|i| {
                            let u = u32::try_from(i).unwrap_or(0);
                            sign(&pk, &sk, &fresh_target(u), &counter_rng()).ok()
                        })
                        .collect::<Option<Vec<Signature>>>()
                })
                .map(|sigs| (k, sigs))
        })
        .for_each(|(k, sigs)| {
            let coeffs: Vec<(Zq<Q>, Signature)> = (0..k)
                .map(|i| {
                    let idx = u32::try_from(i).unwrap_or(0);
                    Zq::<Q>::new(idx.wrapping_mul(7).wrapping_add(1))
                })
                .zip(sigs.iter().cloned())
                .collect();
            c.benchmark_group("lhs::combine")
                .bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
                    b.iter(|| std::hint::black_box(combine::<Q>(&coeffs).ok()));
                });
        });
}

fn bench_verify(c: &mut Criterion) {
    let setup = make_params(2)
        .and_then(|p| keygen(p, &counter_rng()).ok())
        .and_then(|(pk, sk)| {
            let target = fresh_target(0);
            sign(&pk, &sk, &target, &counter_rng())
                .ok()
                .map(|sig| (pk, sig, target))
        });
    setup.iter().for_each(|(pk, sig, target)| {
        c.bench_function("lhs::verify", |b| {
            b.iter(|| std::hint::black_box(verify(pk, sig, target).is_ok()));
        });
    });
}

criterion_group!(benches, bench_sign, bench_combine, bench_verify);
criterion_main!(benches);
