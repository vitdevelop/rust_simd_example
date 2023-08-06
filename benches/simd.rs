#![feature(test)]
#![feature(stdsimd)]

extern crate test;
use test::Bencher;

#[path = "../src/main.rs"]
mod simd;

#[bench]
fn bench_scalar_add(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::scalar_add(&a, &a));
    });
}

#[bench]
fn bench_simd_256_add(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::simd_256_add(&a, &a));
    });
}

#[bench]
fn bench_simd_512_add(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::simd_512_add(&a, &a));
    });
}