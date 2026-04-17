//! Criterion benchmark for the GF(2^8) multiply-accumulate kernel.
//!
//! Measures the fused `mac` primitive at several payload sizes against
//! the legacy `scale` + `add` composition (two passes, two allocations).

// Criterion's BenchmarkGroup API requires &mut self method chains, so the
// bench functions keep a mut-bound group variable.  The needless_for_each
// lint is silenced at the module level because the project convention is to
// use combinators over explicit `for` loops.
#![allow(clippy::needless_for_each)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rlnc_cat_rs::field::{Gf256, mac};
use rlnc_cat_rs::vector::GfVec;

const SIZES: &[usize] = &[256, 1024, 4096, 8192, 32_768];

#[allow(clippy::cast_possible_truncation)]
fn fixture(size: usize) -> (Vec<Gf256>, Vec<Gf256>, Gf256) {
    let acc: Vec<Gf256> = (0..size).map(|i| Gf256::new((i & 0xFF) as u8)).collect();
    let v: Vec<Gf256> = (0..size)
        .map(|i| Gf256::new(((i * 7) & 0xFF) as u8))
        .collect();
    let scalar = Gf256::new(0xA7);
    (acc, v, scalar)
}

fn bench_field_mac(c: &mut Criterion) {
    let mut group = c.benchmark_group("field::mac");
    SIZES.iter().for_each(|&size| {
        let (acc, v, scalar) = fixture(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| std::hint::black_box(mac(&acc, &v, scalar).ok()));
        });
    });
    group.finish();
}

fn bench_gfvec_mac(c: &mut Criterion) {
    let mut group = c.benchmark_group("GfVec::mac");
    SIZES.iter().for_each(|&size| {
        let (acc, v, scalar) = fixture(size);
        let acc_vec = GfVec::new(acc);
        let v_vec = GfVec::new(v);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| std::hint::black_box(acc_vec.mac(&v_vec, scalar).ok()));
        });
    });
    group.finish();
}

fn bench_scale_then_add(c: &mut Criterion) {
    // Baseline: the pre-mac path (scale, then add) with two allocations.
    let mut group = c.benchmark_group("GfVec::scale+add (baseline)");
    SIZES.iter().for_each(|&size| {
        let (acc, v, scalar) = fixture(size);
        let acc_vec = GfVec::new(acc);
        let v_vec = GfVec::new(v);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let scaled = v_vec.scale(scalar);
                std::hint::black_box(acc_vec.add(&scaled).ok())
            });
        });
    });
    group.finish();
}

criterion_group!(benches, bench_field_mac, bench_gfvec_mac, bench_scale_then_add);
criterion_main!(benches);
