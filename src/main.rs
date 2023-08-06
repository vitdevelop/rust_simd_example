#![feature(stdsimd)]

use std::arch::x86_64::{_mm512_add_pd, _mm512_loadu_pd, _mm512_storeu_pd};
use std::arch::x86_64::{_mm512_add_epi64, _mm512_loadu_epi64, _mm512_storeu_epi64};
use std::arch::x86_64::{_mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
use std::arch::x86_64::{_mm256_add_epi64, _mm256_loadu_epi64, _mm256_storeu_epi64};

type VectorF64 = Vec<f64>;
type VectorI64 = Vec<i64>;

pub fn simd_256_add_i64(a: &VectorI64, b: &VectorI64) -> VectorI64 {
    let mut result = vec![0i64; a.len()];

    const FLOATS_IN_AVX2_REGISTER: usize = 4;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX2_REGISTER) * FLOATS_IN_AVX2_REGISTER;

    let mut i = 0usize;
    unsafe {
        for _ in 0..result.len() / vectorization_samples {
            let a_register = _mm256_loadu_epi64(a.as_ptr().add(i));
            let b_register = _mm256_loadu_epi64(b.as_ptr().add(i));

            let intermediate_sum = _mm256_add_epi64(a_register, b_register);

            _mm256_storeu_epi64(result.as_mut_ptr().add(i), intermediate_sum);

            i += FLOATS_IN_AVX2_REGISTER
        }
    }

    for i in i..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

pub fn simd_256_add_f64(a: &VectorF64, b: &VectorF64) -> VectorF64 {
    let mut result = vec![0f64; a.len()];

    const INTEGERS_IN_AVX2_REGISTER: usize = 4;

    let vectorization_samples = (a.len() / INTEGERS_IN_AVX2_REGISTER) * INTEGERS_IN_AVX2_REGISTER;

    let mut i = 0usize;
    unsafe {
        for _ in 0..result.len() / vectorization_samples {
            let a_register = _mm256_loadu_pd(a.as_ptr().add(i));
            let b_register = _mm256_loadu_pd(b.as_ptr().add(i));

            let intermediate_sum = _mm256_add_pd(a_register, b_register);

            _mm256_storeu_pd(result.as_mut_ptr().add(i), intermediate_sum);

            i += INTEGERS_IN_AVX2_REGISTER
        }
    }

    for i in i..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

pub fn simd_512_add_f64(a: &VectorF64, b: &VectorF64) -> VectorF64 {
    let mut result = vec![0f64; a.len()];

    const FLOATS_IN_AVX512_REGISTER: usize = 8;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX512_REGISTER) * FLOATS_IN_AVX512_REGISTER;

    let mut i = 0usize;
    unsafe {
        for _ in 0..result.len() / vectorization_samples {
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

pub fn simd_512_add_i64(a: &VectorI64, b: &VectorI64) -> VectorI64 {
    let mut result = vec![0i64; a.len()];

    const FLOATS_IN_AVX512_REGISTER: usize = 8;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX512_REGISTER) * FLOATS_IN_AVX512_REGISTER;

    let mut i = 0usize;
    unsafe {
        for _ in 0..result.len() / vectorization_samples {
            let a_register = _mm512_loadu_epi64(a.as_ptr().add(i));
            let b_register = _mm512_loadu_epi64(b.as_ptr().add(i));

            let intermediate_sum = _mm512_add_epi64(a_register, b_register);

            _mm512_storeu_epi64(result.as_mut_ptr().add(i), intermediate_sum);

            i += FLOATS_IN_AVX512_REGISTER
        }
    }

    for i in i..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

pub fn scalar_add_i64(a: &VectorI64, b: &VectorI64) -> VectorI64 {
    let mut result = vec![0i64; a.len()];
    for i in 0..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

pub fn scalar_add_f64(a: &VectorF64, b: &VectorF64) -> VectorF64 {
    let mut result = vec![0f64; a.len()];
    for i in 0..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

fn print_result_vector(result: &VectorF64) {
    println!("{} elements", result.len());
    for x in result {
        print!("{} ", x);
    }
    println!();
}

fn print_result_vector64(result: &VectorI64) {
    println!("{} elements", result.len());
    for x in result {
        print!("{} ", x);
    }
    println!();
}

fn main() {
    let a = vec![1i64; 33];
    let b = vec![1f64; 33];

    let result_256 = simd_256_add_f64(&b, &b);
    let result_512 = simd_512_add_f64(&b, &b);
    let result_scalar = scalar_add_i64(&a, &a);

    print_result_vector(&result_256);
    print_result_vector(&result_512);
    print_result_vector64(&result_scalar);
}

