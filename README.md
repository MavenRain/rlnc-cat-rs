# rlnc-cat-rs

Random Linear Network Coding over GF(2^8), built on [comp-cat-rs](https://github.com/MavenRain/comp-cat-rs).

Unlike Reed-Solomon codes, RLNC allows recovery from *any* k linearly independent coded pieces out of n total, making it ideal for lossy networks, multicast, peer-to-peer distribution, and distributed storage.  Intermediate nodes can recode without decoding.

## Installation

```toml
[dependencies]
rlnc-cat-rs = "0.1"
```

## Quick start

```rust
use std::sync::Arc;
use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
use rlnc_cat_rs::coding::encode::Encoder;
use rlnc_cat_rs::coding::decode::DecoderState;

// 1. Prepare original data and split into k pieces
let data = vec![72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100];
let orig = OriginalData::from_bytes(&data, 4)?;
let encoder = Encoder::new(orig.clone());

// 2. Generate coded pieces (here with a deterministic source for demo)
let mut seed: u8 = 1;
let pieces: Vec<_> = (0..4)
    .map(|_| {
        seed = seed.wrapping_add(7);
        let s = seed;
        encoder
            .encode_one(move |n| Ok((0..n).map(|i| s.wrapping_add(i as u8)).collect()))
            .run()
    })
    .collect::<Result<Vec<_>, _>>()?;

// 3. Decode from the k coded pieces
let state = pieces.iter().try_fold(
    DecoderState::new(orig.piece_count(), orig.piece_byte_len()),
    |s, p| s.absorb(p),
)?;
assert!(state.is_complete());
let recovered = state.decode()?;
assert_eq!(recovered, data);
# Ok::<(), rlnc_cat_rs::error::Error>(())
```

## Architecture

```
field/      GF(2^8) finite field arithmetic (Gf256, log/exp tables)
vector/     Immutable vectors (GfVec), matrices (GfMatrix), Gaussian elimination
coding/     The three RLNC operations: encode, decode, recode
error/      Single project-wide Error enum
```

The pure linear algebra core has zero side effects.  The effectful layer uses comp-cat-rs's `Io` and `Stream` to defer randomness and enable streaming.

## Core operations

### Encoding

Split data into k pieces, generate coded pieces as random linear combinations over GF(2^8).

```rust
use rlnc_cat_rs::coding::piece::OriginalData;
use rlnc_cat_rs::coding::encode::Encoder;

let orig = OriginalData::from_bytes(&[1, 2, 3, 4, 5, 6], 3)?;
let encoder = Encoder::new(orig);

// Single piece via Io (deferred until .run())
let piece = encoder.encode_one(|n| Ok(vec![42u8; n])).run()?;

// Unbounded stream, take 10
let pieces = encoder
    .encode_stream(|n| Ok(vec![42u8; n]))
    .take(10)
    .collect()
    .run()?;
# Ok::<(), rlnc_cat_rs::error::Error>(())
```

The pure core (`encode_with_vector`) is also available for deterministic or pre-chosen coding vectors.

### Decoding

Feed coded pieces to a `DecoderState`.  Each `absorb` performs incremental Gaussian elimination.  When `is_complete()`, call `decode()` to extract the original data.

```rust
use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
use rlnc_cat_rs::coding::encode::encode_with_vector;
use rlnc_cat_rs::coding::decode::DecoderState;

let data = vec![10, 20, 30, 40];
let orig = OriginalData::from_bytes(&data, 2)?;

// Encode with explicit coding vectors
let p0 = encode_with_vector(&orig, CodingVector::from_bytes(&[1, 2]))?;
let p1 = encode_with_vector(&orig, CodingVector::from_bytes(&[3, 4]))?;

// Decode
let state = [&p0, &p1].iter().try_fold(
    DecoderState::new(orig.piece_count(), orig.piece_byte_len()),
    |s, p| s.absorb(p),
)?;
let recovered = state.decode()?;
assert_eq!(recovered, data);
# Ok::<(), rlnc_cat_rs::error::Error>(())
```

Linearly dependent pieces are silently absorbed without increasing rank.

`decode_stream` folds a comp-cat-rs `Stream<Error, CodedPiece>` through the decoder automatically.

### Recoding

Generate new coded pieces from existing coded pieces without decoding.  This enables multi-hop distribution where intermediate nodes do not need the original data.

```rust
use rlnc_cat_rs::coding::piece::{CodingVector, OriginalData};
use rlnc_cat_rs::coding::encode::encode_with_vector;
use rlnc_cat_rs::coding::decode::DecoderState;
use rlnc_cat_rs::coding::recode::Recoder;

let data = vec![10, 20, 30, 40];
let orig = OriginalData::from_bytes(&data, 2)?;
let p0 = encode_with_vector(&orig, CodingVector::from_bytes(&[1, 0]))?;
let p1 = encode_with_vector(&orig, CodingVector::from_bytes(&[0, 1]))?;

// Intermediate node recodes without knowing the original data
let recoder = [&p0, &p1].iter().try_fold(
    Recoder::new(orig.piece_count(), orig.piece_byte_len()),
    |r, p| r.add_piece(p),
)?;

let r0 = recoder.recode_one(|n| Ok(vec![3u8; n])).run()?;
let r1 = recoder.recode_one(|n| Ok(vec![7u8; n])).run()?;

// Final node decodes the recoded pieces
let state = [&r0, &r1].iter().try_fold(
    DecoderState::new(orig.piece_count(), orig.piece_byte_len()),
    |s, p| s.absorb(p),
)?;
assert_eq!(state.decode()?, data);
# Ok::<(), rlnc_cat_rs::error::Error>(())
```

## Randomness

This crate has no `rand` dependency.  Randomness is injected as a closure:

- `encode_one` takes `impl FnOnce(usize) -> Result<Vec<u8>, Error>`
- `encode_stream` takes `impl Fn(usize) -> Result<Vec<u8>, Error>`

The argument is the number of random bytes needed.  This keeps the core pure and lets you plug in any source: system CSPRNG, deterministic seeds for testing, or hardware RNG.

## Effect integration

Encoding and recoding produce `Io<Error, CodedPiece>` or `Stream<Error, CodedPiece>`.  Nothing executes until `.run()` is called at the boundary.  This follows comp-cat-rs's delay-run catamorphism pattern.

```rust
use rlnc_cat_rs::coding::piece::OriginalData;
use rlnc_cat_rs::coding::encode::Encoder;

let orig = OriginalData::from_bytes(&[1, 2, 3, 4], 2)?;
let encoder = Encoder::new(orig);

// Build a lazy computation (no side effects yet)
let io = encoder.encode_one(|n| Ok(vec![42u8; n]));

// Nothing has happened. Now run it:
let piece = io.run()?;
# Ok::<(), rlnc_cat_rs::error::Error>(())
```

## Finite field

All arithmetic is over GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 and primitive element 3.

```rust
use rlnc_cat_rs::field::Gf256;

let a = Gf256::new(0x53);
let b = Gf256::new(0xCA);

// Addition and subtraction are both XOR (characteristic 2)
assert_eq!((a + b).value(), 0x53 ^ 0xCA);
assert_eq!(a - b, a + b);

// Multiplication via log/exp tables
let product = a * b;

// Division is fallible (no impl Div, only checked_div)
let quotient = a.checked_div(b);
assert_eq!(quotient.map(|q| q * b), Ok(a));
```

## Design choices

| Decision | Rationale |
|----------|-----------|
| Full immutability | Every matrix row op returns a new `GfMatrix`.  Correctness first; for typical k <= 256 the allocation cost is negligible. |
| No `core::ops::Div` for `Gf256` | The trait signature cannot express division-by-zero failure, and panic is forbidden. |
| No `dyn` in public API | Static dispatch everywhere; `dyn` appears only internally at the comp-cat-rs `Stream::unfold`/`fold` boundary. |
| Scalar GF(2^8) only | SIMD (NEON, AVX2, GFNI) deferred to v0.2 behind a feature flag. |

## License

MIT OR Apache-2.0
