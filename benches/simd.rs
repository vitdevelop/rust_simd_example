#![feature(test)]
#![feature(stdsimd)]

extern crate test;
use test::Bencher;

#[path = "../src/main.rs"]
mod simd;

#[bench]
fn bench_scalar_add_f64(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::scalar_add_f64(&a, &a));
    });
}

#[bench]
fn bench_scalar_add_i64(b: &mut Bencher) {
    let a = vec![1i64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::scalar_add_i64(&a, &a));
    });
}

#[bench]
fn bench_simd_256_add_f64(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::simd_256_add_f64(&a, &a));
    });
}

#[bench]
fn bench_simd_256_add_i64(b: &mut Bencher) {
    let a = vec![1i64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::simd_256_add_i64(&a, &a));
    });
}

#[bench]
fn bench_simd_512_add_f64(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::simd_512_add_f64(&a, &a));
    });
}

#[bench]
fn bench_simd_512_add_i64(b: &mut Bencher) {
    let a = vec![1i64; 1_000_000];
    b.iter(|| {
        test::black_box(simd::simd_512_add_i64(&a, &a));
    });
}