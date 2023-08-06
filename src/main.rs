#![feature(stdsimd)]
#![feature(test)]

extern crate test;

use std::arch::x86_64::{_mm512_add_pd, _mm512_loadu_pd, _mm512_storeu_pd};
use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
use test::Bencher;

type Vector = Vec<f64>;

fn simd_256_add(a: &Vector, b: &Vector) -> Vector {
    let mut result = vec![0f64; a.len()];

    const FLOATS_IN_AVX2_REGISTER: usize = 4;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX2_REGISTER) * FLOATS_IN_AVX2_REGISTER;

    let mut i = 0usize;
    unsafe {
        while i < vectorization_samples {
            let a_register = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_register = _mm256_loadu_pd(b.as_ptr().add(i));

            let intermediate_sum = _mm256_add_pd(a_register, b_register);

            _mm256_storeu_pd(result.as_mut_ptr().add(i), intermediate_sum);

            i += FLOATS_IN_AVX2_REGISTER
        }
    }

    for i in i..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

fn simd_512_add(a: &Vector, b: &Vector) -> Vector {
    let mut result = vec![0f64; a.len()];

    const FLOATS_IN_AVX512_REGISTER: usize = 8;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX512_REGISTER) * FLOATS_IN_AVX512_REGISTER;

    let mut i = 0usize;
    unsafe {
        while i < vectorization_samples {
            let a_register = _mm512_loadu_pd(a.as_ptr().add(i));
            let b_register = _mm512_loadu_pd(b.as_ptr().add(i));

            let intermediate_sum = _mm512_add_pd(a_register, b_register);

            _mm512_storeu_pd(result.as_mut_ptr().add(i), intermediate_sum);

            i += FLOATS_IN_AVX512_REGISTER
        }
    }

    for i in i..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

fn scalar_add(a: &Vector, b: &Vector) -> Vector {
    let mut result = vec![0f64; a.len()];
    for i in 0..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

fn print_result(result: &Vector) {
    println!("{} elements", result.len());
    for x in result {
        print!("{} ", x);
    }
    println!();
}

fn main() {
    let a = vec![1f64; 17];
    let b = vec![1f64; 33];

    let result_256 = simd_256_add(&a, &a);
    let result_512 = simd_512_add(&b, &b);
    let result_scalar = scalar_add(&b, &b);

    print_result(&result_256);
    print_result(&result_512);
    print_result(&result_scalar);
}

#[bench]
fn bench_scalar_add(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(scalar_add(&a, &a));
    });
}

#[bench]
fn bench_simd_256_add(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd_256_add(&a, &a));
    });
}

#[bench]
fn bench_simd_512_add(b: &mut Bencher) {
    let a = vec![1f64; 1_000_000];
    b.iter(|| {
        test::black_box(simd_512_add(&a, &a));
    });
}
