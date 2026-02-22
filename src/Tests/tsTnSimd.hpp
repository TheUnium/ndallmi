// Created by Unium on 22.02.26

#pragma once

#include "../Tensor/mtTnOps_.hpp"
#include "../Tensor/mtTnSimd.hpp"
#include "../Tensor/mtTnTnsr.hpp"
#include "tsTsTstf.hpp"

#include <cmath>
#include <vector>

using namespace MT;

// <<<s_start(simd_element)
// --- simd element wise ops
TEST(simd_add_basic) {
    auto tA = CTensor::Fill({3, 4}, 2.0f);
    auto tB = CTensor::Fill({3, 4}, 3.0f);
    auto tC = SM::Add(tA, tB);
    Check(tC.m_iNdim == 2, "ndim");
    Check(tC.m_lShape[0] == 3, "shape[0]");
    Check(tC.m_lShape[1] == 4, "shape[1]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 5.0f);
}

TEST(simd_add_varied) {
    auto tA = CTensor::Zeros({4});
    auto tB = CTensor::Zeros({4});
    for (int64_t i = 0; i < 4; i++) {
        tA.fFlat(i) = (float)i;
        tB.fFlat(i) = (float)(10 + i);
    }
    auto tC = SM::Add(tA, tB);
    CheckClose(tC.fFlat(0), 10.0f);
    CheckClose(tC.fFlat(1), 12.0f);
    CheckClose(tC.fFlat(2), 14.0f);
    CheckClose(tC.fFlat(3), 16.0f);
}

TEST(simd_add_large_avx_path) {
    // ensure we hit the avx2 path (>8 elements)
    auto tA = CTensor::Fill({32}, 1.5f);
    auto tB = CTensor::Fill({32}, 2.5f);
    auto tC = SM::Add(tA, tB);
    for (int64_t i = 0; i < 32; i++)
        CheckClose(tC.fFlat(i), 4.0f);
}

TEST(simd_add_non_multiple_of_8) {
    // tests scalar tail handling
    auto tA = CTensor::Fill({13}, 3.0f);
    auto tB = CTensor::Fill({13}, 7.0f);
    auto tC = SM::Add(tA, tB);
    for (int64_t i = 0; i < 13; i++)
        CheckClose(tC.fFlat(i), 10.0f);
}

TEST(simd_add_does_not_modify_inputs) {
    auto tA = CTensor::Fill({8}, 2.0f);
    auto tB = CTensor::Fill({8}, 5.0f);
    auto tC = SM::Add(tA, tB);
    CheckClose(tA.fFlat(0), 2.0f);
    CheckClose(tB.fFlat(0), 5.0f);
}

TEST(simd_sub_basic) {
    auto tA = CTensor::Fill({2, 3}, 10.0f);
    auto tB = CTensor::Fill({2, 3}, 3.0f);
    auto tC = SM::Sub(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 7.0f);
}

TEST(simd_sub_negative_result) {
    auto tA = CTensor::Fill({5}, 1.0f);
    auto tB = CTensor::Fill({5}, 5.0f);
    auto tC = SM::Sub(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), -4.0f);
}

TEST(simd_sub_large) {
    auto tA = CTensor::Fill({64}, 100.0f);
    auto tB = CTensor::Fill({64}, 37.0f);
    auto tC = SM::Sub(tA, tB);
    for (int64_t i = 0; i < 64; i++)
        CheckClose(tC.fFlat(i), 63.0f);
}

TEST(simd_mul_basic) {
    auto tA = CTensor::Fill({2, 2}, 3.0f);
    auto tB = CTensor::Fill({2, 2}, 4.0f);
    auto tC = SM::Mul(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 12.0f);
}

TEST(simd_mul_by_zero) {
    auto tA = CTensor::Fill({16}, 99.0f);
    auto tB = CTensor::Zeros({16});
    auto tC = SM::Mul(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 0.0f);
}

TEST(simd_mul_large) {
    auto tA = CTensor::Fill({33}, 2.0f);
    auto tB = CTensor::Fill({33}, 3.0f);
    auto tC = SM::Mul(tA, tB);
    for (int64_t i = 0; i < 33; i++)
        CheckClose(tC.fFlat(i), 6.0f);
}

TEST(simd_add_1d) {
    auto tA = CTensor::Zeros({5});
    auto tB = CTensor::Zeros({5});
    for (int64_t i = 0; i < 5; i++) {
        tA.fFlat(i) = (float)i;
        tB.fFlat(i) = (float)(i * 2);
    }
    auto tC = SM::Add(tA, tB);
    for (int64_t i = 0; i < 5; i++)
        CheckClose(tC.fFlat(i), (float)(i * 3));
}

TEST(simd_add_3d) {
    auto tA = CTensor::Fill({2, 3, 4}, 1.0f);
    auto tB = CTensor::Fill({2, 3, 4}, 2.0f);
    auto tC = SM::Add(tA, tB);
    Check(tC.m_iNdim == 3, "ndim");
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 3, "shape[1]");
    Check(tC.m_lShape[2] == 4, "shape[2]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 3.0f);
}
// >>>s_end(simd_element)

// <<<s_start(simd_scalar)
// --- simd scalar ops
TEST(simd_add_scalar_basic) {
    auto tA = CTensor::Fill({3, 3}, 5.0f);
    auto tB = SM::AddScalar(tA, 10.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 15.0f);
}

TEST(simd_add_scalar_negative) {
    auto tA = CTensor::Fill({4}, 3.0f);
    auto tB = SM::AddScalar(tA, -5.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), -2.0f);
}

TEST(simd_add_scalar_zero) {
    auto tA = CTensor::Fill({16}, 7.0f);
    auto tB = SM::AddScalar(tA, 0.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 7.0f);
}

TEST(simd_add_scalar_large) {
    auto tA = CTensor::Fill({65}, 1.0f);
    auto tB = SM::AddScalar(tA, 99.0f);
    for (int64_t i = 0; i < 65; i++)
        CheckClose(tB.fFlat(i), 100.0f);
}

TEST(simd_mul_scalar_basic) {
    auto tA = CTensor::Fill({2, 2}, 3.0f);
    auto tB = SM::MulScalar(tA, 4.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 12.0f);
}

TEST(simd_mul_scalar_zero) {
    auto tA = CTensor::Fill({16}, 99.0f);
    auto tB = SM::MulScalar(tA, 0.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 0.0f);
}

TEST(simd_mul_scalar_negative) {
    auto tA = CTensor::Fill({17}, 2.0f);
    auto tB = SM::MulScalar(tA, -3.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), -6.0f);
}

TEST(simd_scalar_does_not_modify_input) {
    auto tA = CTensor::Fill({8}, 5.0f);
    auto tB = SM::AddScalar(tA, 10.0f);
    CheckClose(tA.fFlat(0), 5.0f);
}
// >>>s_end(simd_scalar)

// <<<s_start(simd_matrix)
// --- simd matmul
TEST(simd_matmul_identity) {
    auto tI = CTensor::Zeros({2, 2});
    tI.fAt({0, 0}) = 1.0f;
    tI.fAt({1, 1}) = 1.0f;
    auto tA = CTensor::Zeros({2, 2});
    tA.fAt({0, 0}) = 1.0f;
    tA.fAt({0, 1}) = 2.0f;
    tA.fAt({1, 0}) = 3.0f;
    tA.fAt({1, 1}) = 4.0f;
    auto tC = SM::Matmul(tI, tA);
    CheckClose(tC.fAt({0, 0}), 1.0f);
    CheckClose(tC.fAt({0, 1}), 2.0f);
    CheckClose(tC.fAt({1, 0}), 3.0f);
    CheckClose(tC.fAt({1, 1}), 4.0f);
}

TEST(simd_matmul_2x3_3x2) {
    auto tA = CTensor::Zeros({2, 3});
    auto tB = CTensor::Zeros({3, 2});
    for (int64_t i = 0; i < 6; i++)
        tA.fFlat(i) = (float)(i + 1);
    for (int64_t i = 0; i < 6; i++)
        tB.fFlat(i) = (float)(i + 7);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 2, "shape[1]");
    CheckClose(tC.fAt({0, 0}), 58.0f);
    CheckClose(tC.fAt({0, 1}), 64.0f);
    CheckClose(tC.fAt({1, 0}), 139.0f);
    CheckClose(tC.fAt({1, 1}), 154.0f);
}

TEST(simd_matmul_1x1) {
    auto tA = CTensor::Fill({1, 1}, 3.0f);
    auto tB = CTensor::Fill({1, 1}, 5.0f);
    auto tC = SM::Matmul(tA, tB);
    CheckClose(tC.fAt({0, 0}), 15.0f);
}

TEST(simd_matmul_rectangular) {
    auto tA = CTensor::Fill({1, 4}, 2.0f);
    auto tB = CTensor::Fill({4, 1}, 3.0f);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 1, "shape[0]");
    Check(tC.m_lShape[1] == 1, "shape[1]");
    CheckClose(tC.fAt({0, 0}), 24.0f);
}

TEST(simd_matmul_zeros) {
    auto tA = CTensor::Zeros({3, 3});
    auto tB = CTensor::Fill({3, 3}, 5.0f);
    auto tC = SM::Matmul(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 0.0f);
}

TEST(simd_matmul_wide) {
    // test avx path with N > 16
    auto tA = CTensor::Fill({4, 8}, 1.0f);
    auto tB = CTensor::Fill({8, 20}, 1.0f);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 4, "shape[0]");
    Check(tC.m_lShape[1] == 20, "shape[1]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 8.0f);
}

TEST(simd_matmul_micro_m_exact) {
    // exactly 6 rows (MICRO_M) to test even grouping
    auto tA = CTensor::Fill({6, 10}, 2.0f);
    auto tB = CTensor::Fill({10, 16}, 0.5f);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 6, "shape[0]");
    Check(tC.m_lShape[1] == 16, "shape[1]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 10.0f);
}

TEST(simd_matmul_micro_m_remainder) {
    // 7 rows = 1 group of 6 + 1 leftover
    auto tA = CTensor::Fill({7, 10}, 2.0f);
    auto tB = CTensor::Fill({10, 16}, 0.5f);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 7, "shape[0]");
    Check(tC.m_lShape[1] == 16, "shape[1]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 10.0f);
}

TEST(simd_matmul_col_tail) {
    // N=13 tests 8-wide block + scalar tail
    auto tA = CTensor::Fill({4, 5}, 1.0f);
    auto tB = CTensor::Fill({5, 13}, 1.0f);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 4, "shape[0]");
    Check(tC.m_lShape[1] == 13, "shape[1]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 5.0f);
}

TEST(simd_matmul_tile_k) {
    // K > 128 to exercise tile_k loop
    auto tA = CTensor::Fill({4, 200}, 0.1f);
    auto tB = CTensor::Fill({200, 8}, 0.1f);
    auto tC = SM::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 4, "shape[0]");
    Check(tC.m_lShape[1] == 8, "shape[1]");
    // each element = 200 * 0.1 * 0.1 = 2.0
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 2.0f, 1e-2f);
}

TEST(simd_matmul_agrees_with_manual) {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    auto tA = CTensor::Zeros({2, 2});
    auto tB = CTensor::Zeros({2, 2});
    tA.fAt({0, 0}) = 1.0f;
    tA.fAt({0, 1}) = 2.0f;
    tA.fAt({1, 0}) = 3.0f;
    tA.fAt({1, 1}) = 4.0f;
    tB.fAt({0, 0}) = 5.0f;
    tB.fAt({0, 1}) = 6.0f;
    tB.fAt({1, 0}) = 7.0f;
    tB.fAt({1, 1}) = 8.0f;
    auto tC = SM::Matmul(tA, tB);
    CheckClose(tC.fAt({0, 0}), 19.0f);
    CheckClose(tC.fAt({0, 1}), 22.0f);
    CheckClose(tC.fAt({1, 0}), 43.0f);
    CheckClose(tC.fAt({1, 1}), 50.0f);
}

// --- simd matvec
TEST(simd_matvec_basic) {
    auto tMat = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto tVec = CTensor::Fill({3}, 1.0f);
    auto tOut = SM::Matvec(tMat, tVec);
    Check(tOut.m_iNdim == 1, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    CheckClose(tOut.fFlat(0), 6.0f);
    CheckClose(tOut.fFlat(1), 15.0f);
}

TEST(simd_matvec_identity) {
    auto tMat = CTensor::Zeros({3, 3});
    tMat.fAt({0, 0}) = 1.0f;
    tMat.fAt({1, 1}) = 1.0f;
    tMat.fAt({2, 2}) = 1.0f;
    auto tVec = CTensor::Zeros({3});
    tVec.fFlat(0) = 7.0f;
    tVec.fFlat(1) = 8.0f;
    tVec.fFlat(2) = 9.0f;
    auto tOut = SM::Matvec(tMat, tVec);
    CheckClose(tOut.fFlat(0), 7.0f);
    CheckClose(tOut.fFlat(1), 8.0f);
    CheckClose(tOut.fFlat(2), 9.0f);
}

TEST(simd_matvec_single_row) {
    auto tMat = CTensor::Zeros({1, 4});
    for (int64_t i = 0; i < 4; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto tVec = CTensor::Fill({4}, 2.0f);
    auto tOut = SM::Matvec(tMat, tVec);
    Check(tOut.m_lShape[0] == 1, "shape[0]");
    CheckClose(tOut.fFlat(0), 20.0f);
}

TEST(simd_matvec_large_k) {
    // K > 32 to exercise unrolled dot kernel
    auto tMat = CTensor::Fill({2, 64}, 1.0f);
    auto tVec = CTensor::Fill({64}, 1.0f);
    auto tOut = SM::Matvec(tMat, tVec);
    CheckClose(tOut.fFlat(0), 64.0f);
    CheckClose(tOut.fFlat(1), 64.0f);
}

TEST(simd_matvec_non_multiple_of_8) {
    auto tMat = CTensor::Fill({3, 13}, 2.0f);
    auto tVec = CTensor::Fill({13}, 3.0f);
    auto tOut = SM::Matvec(tMat, tVec);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 78.0f); // 13 * 2 * 3
}
// >>>s_end(simd_matrix)

// <<<s_start(simd_dot)
// --- simd dot product
TEST(simd_dot_basic) {
    auto tA = CTensor::Zeros({3});
    auto tB = CTensor::Zeros({3});
    tA.fFlat(0) = 1.0f;
    tA.fFlat(1) = 2.0f;
    tA.fFlat(2) = 3.0f;
    tB.fFlat(0) = 4.0f;
    tB.fFlat(1) = 5.0f;
    tB.fFlat(2) = 6.0f;
    float fD = SM::fDot(tA, tB);
    CheckClose(fD, 32.0f);
}

TEST(simd_dot_orthogonal) {
    auto tA = CTensor::Zeros({2});
    auto tB = CTensor::Zeros({2});
    tA.fFlat(0) = 1.0f;
    tA.fFlat(1) = 0.0f;
    tB.fFlat(0) = 0.0f;
    tB.fFlat(1) = 1.0f;
    float fD = SM::fDot(tA, tB);
    CheckClose(fD, 0.0f);
}

TEST(simd_dot_self) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = 2.0f;
    t.fFlat(2) = 3.0f;
    float fD = SM::fDot(t, t);
    CheckClose(fD, 14.0f);
}

TEST(simd_dot_zeros) {
    auto tA = CTensor::Zeros({5});
    auto tB = CTensor::Fill({5}, 99.0f);
    float fD = SM::fDot(tA, tB);
    CheckClose(fD, 0.0f);
}

TEST(simd_dot_large) {
    // > 32 to exercise the 4-accumulator unrolled path
    auto tA = CTensor::Fill({64}, 1.0f);
    auto tB = CTensor::Fill({64}, 2.0f);
    float fD = SM::fDot(tA, tB);
    CheckClose(fD, 128.0f);
}

TEST(simd_dot_non_multiple_of_32) {
    auto tA = CTensor::Fill({37}, 1.0f);
    auto tB = CTensor::Fill({37}, 3.0f);
    float fD = SM::fDot(tA, tB);
    CheckClose(fD, 111.0f);
}
// >>>s_end(simd_dot)

// <<<s_start(simd_sum)
// --- simd sum
TEST(simd_sum_basic) {
    auto t = CTensor::Fill({3, 4}, 2.0f);
    float fS = SM::fSum(t);
    CheckClose(fS, 24.0f);
}

TEST(simd_sum_varied) {
    auto t = CTensor::Zeros({5});
    for (int64_t i = 0; i < 5; i++)
        t.fFlat(i) = (float)(i + 1);
    float fS = SM::fSum(t);
    CheckClose(fS, 15.0f);
}

TEST(simd_sum_large) {
    auto t = CTensor::Fill({256}, 1.0f);
    float fS = SM::fSum(t);
    CheckClose(fS, 256.0f);
}

TEST(simd_sum_non_multiple_of_8) {
    auto t = CTensor::Fill({11}, 3.0f);
    float fS = SM::fSum(t);
    CheckClose(fS, 33.0f);
}

TEST(simd_sum_zeros) {
    auto t = CTensor::Zeros({100});
    float fS = SM::fSum(t);
    CheckClose(fS, 0.0f);
}

TEST(simd_sum_negative) {
    auto t = CTensor::Fill({4}, -2.5f);
    float fS = SM::fSum(t);
    CheckClose(fS, -10.0f);
}
// >>>s_end(simd_sum)

// <<<s_start(simd_activations)
// --- simd activations
TEST(simd_relu_positive) {
    auto t = CTensor::Fill({16}, 3.0f);
    auto tR = SM::Relu(t);
    for (int64_t i = 0; i < tR.lNumel(); i++)
        CheckClose(tR.fFlat(i), 3.0f);
}

TEST(simd_relu_negative) {
    auto t = CTensor::Fill({16}, -3.0f);
    auto tR = SM::Relu(t);
    for (int64_t i = 0; i < tR.lNumel(); i++)
        CheckClose(tR.fFlat(i), 0.0f);
}

TEST(simd_relu_mixed) {
    auto t = CTensor::Zeros({5});
    t.fFlat(0) = -2.0f;
    t.fFlat(1) = 0.0f;
    t.fFlat(2) = 3.0f;
    t.fFlat(3) = -1.0f;
    t.fFlat(4) = 5.0f;
    auto tR = SM::Relu(t);
    CheckClose(tR.fFlat(0), 0.0f);
    CheckClose(tR.fFlat(1), 0.0f);
    CheckClose(tR.fFlat(2), 3.0f);
    CheckClose(tR.fFlat(3), 0.0f);
    CheckClose(tR.fFlat(4), 5.0f);
}

TEST(simd_relu_large) {
    auto t = CTensor::Zeros({33});
    for (int64_t i = 0; i < 33; i++)
        t.fFlat(i) = (float)(i - 16);
    auto tR = SM::Relu(t);
    for (int64_t i = 0; i < 33; i++) {
        float fExpected = (float)(i - 16);
        CheckClose(tR.fFlat(i), fExpected > 0.0f ? fExpected : 0.0f);
    }
}

TEST(simd_relu_preserves_shape) {
    auto t = CTensor::Rand({3, 4, 5});
    auto tR = SM::Relu(t);
    Check(tR.m_iNdim == 3, "ndim");
    Check(tR.m_lShape[0] == 3, "shape[0]");
    Check(tR.m_lShape[1] == 4, "shape[1]");
    Check(tR.m_lShape[2] == 5, "shape[2]");
}

TEST(simd_silu_zero) {
    auto t = CTensor::Zeros({1});
    auto tS = SM::Silu(t);
    CheckClose(tS.fFlat(0), 0.0f);
}

TEST(simd_silu_positive) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = 1.0f;
    auto tS = SM::Silu(t);
    float fExpected = 1.0f / (1.0f + std::exp(-1.0f));
    CheckClose(tS.fFlat(0), fExpected, 1e-3f);
}

TEST(simd_silu_large_positive) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = 10.0f;
    auto tS = SM::Silu(t);
    CheckClose(tS.fFlat(0), 10.0f, 1e-3f);
}

TEST(simd_silu_negative) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = -5.0f;
    auto tS = SM::Silu(t);
    float fExpected = -5.0f / (1.0f + std::exp(5.0f));
    CheckClose(tS.fFlat(0), fExpected, 1e-3f);
}

TEST(simd_silu_large_avx_path) {
    // > 8 elements to exercise avx fast exp approx
    auto t = CTensor::Zeros({16});
    for (int64_t i = 0; i < 16; i++)
        t.fFlat(i) = (float)(i - 8);
    auto tS = SM::Silu(t);
    for (int64_t i = 0; i < 16; i++) {
        float fX = (float)(i - 8);
        float fExpected = fX / (1.0f + std::exp(-fX));
        CheckClose(tS.fFlat(i), fExpected, 5e-2f);
    }
}

TEST(simd_silu_does_not_modify_input) {
    auto t = CTensor::Fill({8}, 2.0f);
    auto tS = SM::Silu(t);
    CheckClose(t.fFlat(0), 2.0f);
}
// >>>s_end(simd_activations)

// <<<s_start(simd_normalization)
// --- simd normalization
TEST(simd_rmsnorm_basic) {
    auto tX = CTensor::Fill({2, 4}, 1.0f);
    auto tW = CTensor::Fill({4}, 1.0f);
    auto tOut = SM::RmsNorm(tX, tW);
    Check(tOut.m_iNdim == 2, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    Check(tOut.m_lShape[1] == 4, "shape[1]");
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 1.0f, 1e-3f);
}

TEST(simd_rmsnorm_weight_scaling) {
    auto tX = CTensor::Fill({1, 4}, 1.0f);
    auto tW = CTensor::Fill({4}, 2.0f);
    auto tOut = SM::RmsNorm(tX, tW);
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 2.0f, 1e-3f);
}

TEST(simd_rmsnorm_varied_input) {
    auto tX = CTensor::Zeros({1, 4});
    tX.fFlat(0) = 1.0f;
    tX.fFlat(1) = 2.0f;
    tX.fFlat(2) = 3.0f;
    tX.fFlat(3) = 4.0f;
    auto tW = CTensor::Fill({4}, 1.0f);
    auto tOut = SM::RmsNorm(tX, tW);
    float fRms = std::sqrt(7.5f);
    CheckClose(tOut.fFlat(0), 1.0f / fRms, 1e-3f);
    CheckClose(tOut.fFlat(1), 2.0f / fRms, 1e-3f);
    CheckClose(tOut.fFlat(2), 3.0f / fRms, 1e-3f);
    CheckClose(tOut.fFlat(3), 4.0f / fRms, 1e-3f);
}

TEST(simd_rmsnorm_3d) {
    auto tX = CTensor::Fill({2, 3, 4}, 1.0f);
    auto tW = CTensor::Fill({4}, 1.0f);
    auto tOut = SM::RmsNorm(tX, tW);
    Check(tOut.m_iNdim == 3, "ndim");
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 1.0f, 1e-3f);
}

TEST(simd_rmsnorm_large_dim) {
    // > 16 to exercise both avx unroll loops
    auto tX = CTensor::Fill({2, 32}, 2.0f);
    auto tW = CTensor::Fill({32}, 1.0f);
    auto tOut = SM::RmsNorm(tX, tW);
    // rms of all 2s = 2, so output = 2/2 * 1 = 1
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 1.0f, 1e-3f);
}

TEST(simd_rmsnorm_non_multiple_of_8) {
    auto tX = CTensor::Fill({1, 13}, 1.0f);
    auto tW = CTensor::Fill({13}, 1.0f);
    auto tOut = SM::RmsNorm(tX, tW);
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 1.0f, 1e-3f);
}

TEST(simd_softmax_1d) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = 2.0f;
    t.fFlat(2) = 3.0f;
    auto tS = SM::Softmax(t);
    Check(tS.m_iNdim == 1, "ndim");
    Check(tS.m_lShape[0] == 3, "shape[0]");
    float fSum = 0.0f;
    for (int64_t i = 0; i < 3; i++)
        fSum += tS.fFlat(i);
    CheckClose(fSum, 1.0f, 1e-5f);
    Check(tS.fFlat(0) < tS.fFlat(1), "monotonic 0<1");
    Check(tS.fFlat(1) < tS.fFlat(2), "monotonic 1<2");
}

TEST(simd_softmax_uniform) {
    auto t = CTensor::Fill({4}, 5.0f);
    auto tS = SM::Softmax(t);
    for (int64_t i = 0; i < 4; i++)
        CheckClose(tS.fFlat(i), 0.25f, 1e-5f);
}

TEST(simd_softmax_2d) {
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tS = SM::Softmax(t);
    for (int64_t r = 0; r < 2; r++) {
        float fSum = 0.0f;
        for (int64_t c = 0; c < 3; c++)
            fSum += tS.fAt({r, c});
        CheckClose(fSum, 1.0f, 1e-5f);
    }
}

TEST(simd_softmax_large_values) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1000.0f;
    t.fFlat(1) = 1001.0f;
    t.fFlat(2) = 1002.0f;
    auto tS = SM::Softmax(t);
    float fSum = 0.0f;
    for (int64_t i = 0; i < 3; i++) {
        Check(!std::isnan(tS.fFlat(i)), "softmax should not produce NaN");
        Check(!std::isinf(tS.fFlat(i)), "softmax should not produce Inf");
        fSum += tS.fFlat(i);
    }
    CheckClose(fSum, 1.0f, 1e-5f);
}

TEST(simd_softmax_preserves_shape) {
    auto t = CTensor::Rand({3, 4, 5});
    auto tS = SM::Softmax(t);
    Check(tS.m_iNdim == 3, "ndim");
    Check(tS.m_lShape[0] == 3, "shape[0]");
    Check(tS.m_lShape[1] == 4, "shape[1]");
    Check(tS.m_lShape[2] == 5, "shape[2]");
}

TEST(simd_softmax_large_dim) {
    // > 8 to exercise avx max path
    auto t = CTensor::Zeros({1, 20});
    for (int64_t i = 0; i < 20; i++)
        t.fFlat(i) = (float)i;
    auto tS = SM::Softmax(t);
    float fSum = 0.0f;
    for (int64_t i = 0; i < 20; i++) {
        Check(!std::isnan(tS.fFlat(i)), "no NaN");
        fSum += tS.fFlat(i);
    }
    CheckClose(fSum, 1.0f, 1e-5f);
    // last element should be largest
    Check(tS.fFlat(19) > tS.fFlat(0), "largest should be last");
}
// >>>s_end(simd_normalization)

// <<<s_start(simd_consistency)
// --- simd vs scalar consistency
TEST(simd_add_matches_scalar_add) {
    auto tA = CTensor::Rand({64});
    auto tB = CTensor::Rand({64});
    auto tSimd = SM::Add(tA, tB);
    // manual scalar add
    for (int64_t i = 0; i < 64; i++)
        CheckClose(tSimd.fFlat(i), tA.fFlat(i) + tB.fFlat(i), 1e-5f);
}

TEST(simd_sub_matches_scalar_sub) {
    auto tA = CTensor::Rand({64});
    auto tB = CTensor::Rand({64});
    auto tSimd = SM::Sub(tA, tB);
    for (int64_t i = 0; i < 64; i++)
        CheckClose(tSimd.fFlat(i), tA.fFlat(i) - tB.fFlat(i), 1e-5f);
}

TEST(simd_mul_matches_scalar_mul) {
    auto tA = CTensor::Rand({64});
    auto tB = CTensor::Rand({64});
    auto tSimd = SM::Mul(tA, tB);
    for (int64_t i = 0; i < 64; i++)
        CheckClose(tSimd.fFlat(i), tA.fFlat(i) * tB.fFlat(i), 1e-5f);
}

TEST(simd_dot_matches_scalar_dot) {
    auto tA = CTensor::Rand({100});
    auto tB = CTensor::Rand({100});
    float fSimd = SM::fDot(tA, tB);
    float fScalar = 0.0f;
    for (int64_t i = 0; i < 100; i++)
        fScalar += tA.fFlat(i) * tB.fFlat(i);
    CheckClose(fSimd, fScalar, 1e-3f);
}

TEST(simd_sum_matches_scalar_sum) {
    auto t = CTensor::Rand({100});
    float fSimd = SM::fSum(t);
    float fScalar = 0.0f;
    for (int64_t i = 0; i < 100; i++)
        fScalar += t.fFlat(i);
    CheckClose(fSimd, fScalar, 1e-3f);
}
// >>>s_end(simd_consistency)

// <<<s_start(simd_benches)
// --- simd benchmarks
TEST(bench_simd_add) {
    auto tA = CTensor::Rand({512, 512});
    auto tB = CTensor::Rand({512, 512});
    Bench("op   add 512x512", 1000, [&]() {
        auto tC = OP::Add(tA, tB);
        (void)tC;
    });
    Bench("simd add 512x512", 1000, [&]() {
        auto tC = SM::Add(tA, tB);
        (void)tC;
    });
}

TEST(bench_simd_mul) {
    auto tA = CTensor::Rand({512, 512});
    auto tB = CTensor::Rand({512, 512});
    Bench("op   mul 512x512", 1000, [&]() {
        auto tC = OP::Mul(tA, tB);
        (void)tC;
    });
    Bench("simd mul 512x512", 1000, [&]() {
        auto tC = SM::Mul(tA, tB);
        (void)tC;
    });
}

TEST(bench_simd_matmul_small) {
    auto tA = CTensor::Rand({64, 64});
    auto tB = CTensor::Rand({64, 64});
    Bench("op   matmul 64x64", 500, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
    Bench("simd matmul 64x64", 500, [&]() {
        auto tC = SM::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_simd_matmul_medium) {
    auto tA = CTensor::Rand({256, 256});
    auto tB = CTensor::Rand({256, 256});
    Bench("op   matmul 256x256", 50, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
    Bench("simd matmul 256x256", 50, [&]() {
        auto tC = SM::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_simd_matmul_large) {
    auto tA = CTensor::Rand({512, 512});
    auto tB = CTensor::Rand({512, 512});
    Bench("op   matmul 512x512", 50, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
    Bench("simd matmul 512x512", 50, [&]() {
        auto tC = SM::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_simd_matmul_even_larger) {
    auto tA = CTensor::Rand({1024, 1024});
    auto tB = CTensor::Rand({1024, 1024});
    Bench("op   matmul 1024x1024", 50, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
    Bench("simd matmul 1024x1024", 50, [&]() {
        auto tC = SM::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_simd_matvec) {
    auto tMat = CTensor::Rand({512, 512});
    auto tVec = CTensor::Rand({512});
    Bench("op   matvec 512x512", 1000, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
    Bench("simd matvec 512x512", 1000, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_simd_matvec_large) {
    auto tMat = CTensor::Rand({1024, 1024});
    auto tVec = CTensor::Rand({1024});
    Bench("op   matvec 1024x1024", 1000, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
    Bench("simd matvec 1024x1024", 1000, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_simd_matvec_ever_larger) {
    auto tMat = CTensor::Rand({4096, 4096});
    auto tVec = CTensor::Rand({4096});
    Bench("op   matvec 4096x4096", 1000, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
    Bench("simd matvec 4096x4096", 1000, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
}

// TEST(bench_clone) {
//     auto t = CTensor::Rand({256, 256});
//     volatile float kys;
//
//     Bench("clone 256x256", 5000, [&]() {
//         auto tC = t.Clone();
//         kys = tC.fFlat(0);
//     });
// }

TEST(bench_simd_dot) {
    auto tA = CTensor::Rand({1024 * 1024});
    auto tB = CTensor::Rand({1024 * 1024});
    volatile float kys;
    Bench("op   dot 1M", 1000, [&]() {
        float fD = OP::fDot(tA, tB);
        kys = fD;
    });
    Bench("simd dot 1M", 1000, [&]() {
        float fD = SM::fDot(tA, tB);
        (void)fD;
    });
}

TEST(bench_simd_relu) {
    auto t = CTensor::Rand({512, 512});
    Bench("op   relu 512x512", 1000, [&]() {
        auto tR = OP::Relu(t);
        (void)tR;
    });
    Bench("simd relu 512x512", 1000, [&]() {
        auto tR = SM::Relu(t);
        (void)tR;
    });
}

TEST(bench_simd_silu) {
    auto t = CTensor::Rand({512, 512});
    Bench("op   silu 512x512", 1000, [&]() {
        auto tS = OP::Silu(t);
        (void)tS;
    });
    Bench("simd silu 512x512", 1000, [&]() {
        auto tS = SM::Silu(t);
        (void)tS;
    });
}

TEST(bench_simd_softmax) {
    auto t = CTensor::Rand({64, 512});
    Bench("op   softmax 64x512", 500, [&]() {
        auto tS = OP::Softmax(t);
        (void)tS;
    });
    Bench("simd softmax 64x512", 500, [&]() {
        auto tS = SM::Softmax(t);
        (void)tS;
    });
}

TEST(bench_simd_rmsnorm) {
    auto tX = CTensor::Rand({64, 512});
    auto tW = CTensor::Rand({512});
    Bench("op   rmsnorm 64x512", 1000, [&]() {
        auto tO = OP::RmsNorm(tX, tW);
        (void)tO;
    });
    Bench("simd rmsnorm 64x512", 1000, [&]() {
        auto tO = SM::RmsNorm(tX, tW);
        (void)tO;
    });
}
// >>>s_end(simd_benches)
