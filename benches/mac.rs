//! Criterion benchmark for the GF(2^8) multiply-accumulate kernel and
//! the `DecoderState::absorb` path.

// Combinators over explicit `for` loops.  Criterion's pedantic lint
// prefers a for loop, but the project convention is iterators.
#![allow(clippy::needless_for_each)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rlnc_cat_rs::coding::decode::DecoderState;
use rlnc_cat_rs::coding::encode::encode_with_vector;
use rlnc_cat_rs::coding::piece::{CodedPiece, CodingVector, OriginalData};
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
    SIZES.iter().for_each(|&size| {
        let (acc, v, scalar) = fixture(size);
        c.benchmark_group("field::mac")
            .throughput(Throughput::Bytes(size as u64))
            .bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
                b.iter(|| std::hint::black_box(mac(&acc, &v, scalar).ok()));
            });
    });
}

fn bench_gfvec_mac(c: &mut Criterion) {
    SIZES.iter().for_each(|&size| {
        let (acc, v, scalar) = fixture(size);
        let acc_vec = GfVec::new(acc);
        let v_vec = GfVec::new(v);
        c.benchmark_group("GfVec::mac")
            .throughput(Throughput::Bytes(size as u64))
            .bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
                b.iter(|| std::hint::black_box(acc_vec.mac(&v_vec, scalar).ok()));
            });
    });
}

fn bench_scale_then_add(c: &mut Criterion) {
    SIZES.iter().for_each(|&size| {
        let (acc, v, scalar) = fixture(size);
        let acc_vec = GfVec::new(acc);
        let v_vec = GfVec::new(v);
        c.benchmark_group("GfVec::scale+add (baseline)")
            .throughput(Throughput::Bytes(size as u64))
            .bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
                b.iter(|| {
                    let scaled = v_vec.scale(scalar);
                    std::hint::black_box(acc_vec.add(&scaled).ok())
                });
            });
    });
}

#[allow(clippy::cast_possible_truncation)]
fn vandermonde_cv(row: usize, k: usize) -> Vec<u8> {
    let base = Gf256::new((row as u8).wrapping_add(1));
    (0..k)
        .map(|j| {
            (0..j)
                .fold(Gf256::one(), |acc, _| acc * base)
                .value()
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn decoder_fixture(k: usize, piece_bytes: usize) -> (OriginalData, Vec<CodedPiece>) {
    // Unpadded data length chosen so piece_byte_len lands on the target.
    let data_len = k * piece_bytes - 1;
    let data: Vec<u8> = (0..data_len).map(|i| (i & 0xFF) as u8).collect();
    // Fallback mirrors the pattern used in existing tests: a minimal valid
    // OriginalData(&[0], 1).  Unreachable for our bench inputs (data non-empty
    // and k > 0 in every case below).
    let orig = OriginalData::from_bytes(&data, k)
        .unwrap_or_else(|_| OriginalData::from_bytes(&[0], 1).ok().unwrap());

    let pieces: Vec<CodedPiece> = (0..k)
        .map(|i| {
            let cv_bytes = vandermonde_cv(i, k);
            encode_with_vector(&orig, CodingVector::from_bytes(&cv_bytes)).unwrap_or_else(
                |_| CodedPiece::new(CodingVector::from_bytes(&[]), GfVec::zeros(0)),
            )
        })
        .collect();

    (orig, pieces)
}

fn bench_decoder_absorb(c: &mut Criterion) {
    let cases = &[(8usize, 64usize), (16, 256), (32, 1024), (64, 2048)];
    cases.iter().for_each(|&(k, piece_bytes)| {
        let (orig, pieces) = decoder_fixture(k, piece_bytes);
        let payload_bytes = k * piece_bytes;
        let label = format!("k={k} b={piece_bytes}");
        c.benchmark_group("DecoderState::absorb_all+decode")
            .throughput(Throughput::Bytes(payload_bytes as u64))
            .bench_with_input(BenchmarkId::from_parameter(&label), &label, |b, _| {
                b.iter(|| {
                    let state = DecoderState::new(orig.piece_count(), orig.piece_byte_len());
                    let result = pieces
                        .iter()
                        .try_fold(state, DecoderState::absorb)
                        .and_then(DecoderState::decode);
                    std::hint::black_box(result.ok())
                });
            });
    });
}

criterion_group!(
    benches,
    bench_field_mac,
    bench_gfvec_mac,
    bench_scale_then_add,
    bench_decoder_absorb,
);
criterion_main!(benches);
