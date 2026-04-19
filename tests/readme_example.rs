use rlnc_cat_rs::lhs::{LatticeHomomorphicAuthenticator, LhsParams, keygen};
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn readme_authenticated_recoding_example_compiles_and_runs() {
    let counter = AtomicU64::new(1);
    let rng = |n: usize| {
        let bytes: Vec<u8> = (0..n.div_ceil(8))
            .flat_map(|_| counter.fetch_add(1, Ordering::Relaxed).to_le_bytes())
            .take(n)
            .collect();
        Ok::<_, rlnc_cat_rs::error::Error>(bytes)
    };
    let result = LhsParams::<97>::new(4, 4, 3, 10_000_000, 3.0)
        .and_then(|params| keygen(params, &rng))
        .and_then(|(pk, sk)| {
            let metadata = b"generation-id";
            LatticeHomomorphicAuthenticator::new(pk, &sk, metadata, &rng)
        });
    assert!(result.is_ok(), "{result:?}");
}
