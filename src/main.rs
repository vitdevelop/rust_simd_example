#![feature(stdsimd)]

use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};

use core::arch::x86_64::{_mm512_add_ps, _mm512_loadu_ps, _mm512_storeu_ps};

type Vector = Vec<f32>;

fn simd_256_add(a: &Vector, b: &Vector) -> Vector {
    let mut result = vec![0f32; a.len()];

    const FLOATS_IN_AVX2_REGISTER: usize = 8;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX2_REGISTER) * FLOATS_IN_AVX2_REGISTER;

    let mut i = 0usize;
    unsafe {
        while i < vectorization_samples {
            let a_register = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_register = _mm256_loadu_ps(b.as_ptr().add(i));

            let intermediate_sum = _mm256_add_ps(a_register, b_register);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), intermediate_sum);

            i += FLOATS_IN_AVX2_REGISTER
        }
    }

    for i in i..result.len() {
        result[i] = a[i] + b[i];
    }

    return result;
}

fn simd_512_add(a: &Vector, b: &Vector) -> Vector {
    let mut result = vec![0f32; a.len()];

    const FLOATS_IN_AVX512_REGISTER: usize = 16;

    let vectorization_samples = (a.len() / FLOATS_IN_AVX512_REGISTER) * FLOATS_IN_AVX512_REGISTER;

    let mut i = 0usize;
    unsafe {
        while i < vectorization_samples {
            let a_register = _mm512_loadu_ps(a.as_ptr().add(i));
            let b_register = _mm512_loadu_ps(b.as_ptr().add(i));

            let intermediate_sum = _mm512_add_ps(a_register, b_register);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), intermediate_sum);

            i += FLOATS_IN_AVX512_REGISTER
        }
    }

    for i in i..result.len() {
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
    let a = vec![1f32; 17];
    let b = vec![1f32; 33];

    let result_256 = simd_256_add(&a, &a);
    let result_512 = simd_512_add(&b, &b);

    print_result(&result_256);
    print_result(&result_512);
}
