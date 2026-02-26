// Created by Unium on 24.02.26

#pragma once

#include "../Tensor/mtTnOps_.hpp"
#include "../Tensor/mtTnQnt8.hpp"
#include "../Tensor/mtTnSimd.hpp"
#include "../Tensor/mtTnTnsr.hpp"
#include "tsTsTstf.hpp"

#include <cmath>
#include <vector>

using namespace MT;

// <<<s_start(q8_quantize)
// --- q8 quantization
TEST(q8_quantize_basic) {
    auto tMat = CTensor::Fill({2, 4}, 1.0f);
    auto qMat = Q8::Quantize(tMat);
    Check(qMat.lRows == 2, "rows");
    Check(qMat.lCols == 4, "cols");
    Check((int64_t)qMat.viData.size() == 8, "data size");
    Check((int64_t)qMat.vfScale.size() == 2, "scale size");
}

TEST(q8_quantize_scale_correct) {
    auto tMat = CTensor::Zeros({1, 4});
    tMat.fFlat(0) = 127.0f;
    tMat.fFlat(1) = -127.0f;
    tMat.fFlat(2) = 0.0f;
    tMat.fFlat(3) = 63.5f;
    auto qMat = Q8::Quantize(tMat);
    // absmax = 127, scale = 127/127 = 1.0
    CheckClose(qMat.vfScale[0], 1.0f);
    Check(qMat.viData[0] == 127, "max val quantized to 127");
    Check(qMat.viData[1] == -127, "min val quantized to -127");
    Check(qMat.viData[2] == 0, "zero stays zero");
    Check(qMat.viData[3] == 64, "63.5 rounds to 64");
}

TEST(q8_quantize_per_row_scales) {
    auto tMat = CTensor::Zeros({2, 3});
    // row 0: absmax = 10
    tMat.fAt({0, 0}) = 10.0f;
    tMat.fAt({0, 1}) = -5.0f;
    tMat.fAt({0, 2}) = 0.0f;
    // row 1: absmax = 254
    tMat.fAt({1, 0}) = 254.0f;
    tMat.fAt({1, 1}) = 127.0f;
    tMat.fAt({1, 2}) = -254.0f;
    auto qMat = Q8::Quantize(tMat);
    CheckClose(qMat.vfScale[0], 10.0f / 127.0f, 1e-5f);
    CheckClose(qMat.vfScale[1], 254.0f / 127.0f, 1e-5f);
    // row 0: 10 / (10/127) = 127
    Check(qMat.viData[0] == 127, "row0 max -> 127");
    // row 1: 254 / (254/127) = 127, -254 --> -127
    Check(qMat.viData[3] == 127, "row1 max -> 127");
    Check(qMat.viData[5] == -127, "row1 min -> -127");
}

TEST(q8_quantize_zeros) {
    auto tMat = CTensor::Zeros({3, 5});
    auto qMat = Q8::Quantize(tMat);
    for (int64_t i = 0; i < qMat.lNumel(); i++)
        Check(qMat.viData[i] == 0, "zero input -> zero quantized");
}

TEST(q8_quantize_uniform) {
    auto tMat = CTensor::Fill({4, 8}, 3.0f);
    auto qMat = Q8::Quantize(tMat);
    for (int64_t r = 0; r < 4; r++) {
        CheckClose(qMat.vfScale[r], 3.0f / 127.0f, 1e-6f);
        for (int64_t c = 0; c < 8; c++)
            Check(qMat.viData[r * 8 + c] == 127, "uniform -> all 127");
    }
}

TEST(q8_quantize_negative_uniform) {
    auto tMat = CTensor::Fill({2, 4}, -5.0f);
    auto qMat = Q8::Quantize(tMat);
    for (int64_t r = 0; r < 2; r++) {
        CheckClose(qMat.vfScale[r], 5.0f / 127.0f, 1e-6f);
        for (int64_t c = 0; c < 4; c++)
            Check(qMat.viData[r * 4 + c] == -127, "negative uniform -> all -127");
    }
}

TEST(q8_quantize_large_matrix) {
    auto tMat = CTensor::Rand({64, 128});
    auto qMat = Q8::Quantize(tMat);
    Check(qMat.lRows == 64, "rows");
    Check(qMat.lCols == 128, "cols");
    for (int64_t i = 0; i < qMat.lNumel(); i++) {
        Check(qMat.viData[i] >= -128, "lower bound");
        Check(qMat.viData[i] <= 127, "upper bound");
    }
    for (int64_t r = 0; r < qMat.lRows; r++)
        Check(qMat.vfScale[r] > 0.0f, "scale positive");
}

TEST(q8_quantize_numel) {
    auto tMat = CTensor::Rand({7, 13});
    auto qMat = Q8::Quantize(tMat);
    Check(qMat.lNumel() == 91, "numel = 7*13");
}

TEST(q8_quantize_non_multiple_of_8) {
    auto tMat = CTensor::Fill({1, 13}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    Check(qMat.lCols == 13, "cols");
    for (int64_t c = 0; c < 13; c++)
        Check(qMat.viData[c] == 127, "all max");
}
// >>>s_end(q8_quantize)

// <<<s_start(q8_matvec_scalar)
// --- q8 scalar matvec
TEST(q8_matvec_basic) {
    auto tMat = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({3}, 1.0f);
    auto tOut = Q8::Matvec(qMat, tVec);
    Check(tOut.m_iNdim == 1, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    // row0: 1+2+3=6, row1: 4+5+6=15
    // quantization error expected
    CheckClose(tOut.fFlat(0), 6.0f, 0.5f);
    CheckClose(tOut.fFlat(1), 15.0f, 0.5f);
}

TEST(q8_matvec_identity_like) {
    auto tMat = CTensor::Zeros({3, 3});
    tMat.fAt({0, 0}) = 1.0f;
    tMat.fAt({1, 1}) = 1.0f;
    tMat.fAt({2, 2}) = 1.0f;
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Zeros({3});
    tVec.fFlat(0) = 7.0f;
    tVec.fFlat(1) = 8.0f;
    tVec.fFlat(2) = 9.0f;
    auto tOut = Q8::Matvec(qMat, tVec);
    CheckClose(tOut.fFlat(0), 7.0f, 0.5f);
    CheckClose(tOut.fFlat(1), 8.0f, 0.5f);
    CheckClose(tOut.fFlat(2), 9.0f, 0.5f);
}

TEST(q8_matvec_single_row) {
    auto tMat = CTensor::Zeros({1, 4});
    for (int64_t i = 0; i < 4; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({4}, 2.0f);
    auto tOut = Q8::Matvec(qMat, tVec);
    Check(tOut.m_lShape[0] == 1, "shape[0]");
    // 2*(1+2+3+4) = 20
    CheckClose(tOut.fFlat(0), 20.0f, 1.0f);
}

TEST(q8_matvec_zero_matrix) {
    auto tMat = CTensor::Zeros({4, 5});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({5}, 99.0f);
    auto tOut = Q8::Matvec(qMat, tVec);
    for (int64_t i = 0; i < 4; i++)
        CheckClose(tOut.fFlat(i), 0.0f, 1e-5f);
}

TEST(q8_matvec_zero_vector) {
    auto tMat = CTensor::Fill({3, 4}, 5.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Zeros({4});
    auto tOut = Q8::Matvec(qMat, tVec);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 0.0f, 1e-5f);
}

TEST(q8_matvec_uniform) {
    auto tMat = CTensor::Fill({3, 8}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({8}, 3.0f);
    auto tOut = Q8::Matvec(qMat, tVec);
    // each row: 8 * 2 * 3 = 48
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 48.0f, 1.0f);
}

TEST(q8_matvec_non_multiple_of_4) {
    auto tMat = CTensor::Fill({2, 7}, 1.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({7}, 2.0f);
    auto tOut = Q8::Matvec(qMat, tVec);
    for (int64_t i = 0; i < 2; i++)
        CheckClose(tOut.fFlat(i), 14.0f, 0.5f);
}
// >>>s_end(q8_matvec_scalar)

// <<<s_start(q8_matvec_simd)
// --- q8 simd matvec
TEST(q8_simd_matvec_basic) {
    auto tMat = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({3}, 1.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    Check(tOut.m_iNdim == 1, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    CheckClose(tOut.fFlat(0), 6.0f, 0.5f);
    CheckClose(tOut.fFlat(1), 15.0f, 0.5f);
}

TEST(q8_simd_matvec_identity_like) {
    auto tMat = CTensor::Zeros({3, 3});
    tMat.fAt({0, 0}) = 1.0f;
    tMat.fAt({1, 1}) = 1.0f;
    tMat.fAt({2, 2}) = 1.0f;
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Zeros({3});
    tVec.fFlat(0) = 7.0f;
    tVec.fFlat(1) = 8.0f;
    tVec.fFlat(2) = 9.0f;
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    CheckClose(tOut.fFlat(0), 7.0f, 0.5f);
    CheckClose(tOut.fFlat(1), 8.0f, 0.5f);
    CheckClose(tOut.fFlat(2), 9.0f, 0.5f);
}

TEST(q8_simd_matvec_single_row) {
    auto tMat = CTensor::Zeros({1, 4});
    for (int64_t i = 0; i < 4; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({4}, 2.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    CheckClose(tOut.fFlat(0), 20.0f, 1.0f);
}

TEST(q8_simd_matvec_zero_matrix) {
    auto tMat = CTensor::Zeros({4, 5});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({5}, 99.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 4; i++)
        CheckClose(tOut.fFlat(i), 0.0f, 1e-5f);
}

TEST(q8_simd_matvec_zero_vector) {
    auto tMat = CTensor::Fill({3, 4}, 5.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Zeros({4});
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 0.0f, 1e-5f);
}

TEST(q8_simd_matvec_uniform) {
    auto tMat = CTensor::Fill({3, 8}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({8}, 3.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 48.0f, 1.0f);
}

TEST(q8_simd_matvec_large_k) {
    auto tMat = CTensor::Fill({2, 64}, 1.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({64}, 1.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    CheckClose(tOut.fFlat(0), 64.0f, 1.0f);
    CheckClose(tOut.fFlat(1), 64.0f, 1.0f);
}

TEST(q8_simd_matvec_non_multiple_of_8) {
    auto tMat = CTensor::Fill({3, 13}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({13}, 3.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 78.0f, 2.0f);
}

TEST(q8_simd_matvec_4row_exact) {
    auto tMat = CTensor::Fill({4, 16}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({16}, 0.5f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 4; i++)
        CheckClose(tOut.fFlat(i), 16.0f, 0.5f);
}

TEST(q8_simd_matvec_4row_remainder) {
    auto tMat = CTensor::Fill({5, 16}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({16}, 0.5f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 5; i++)
        CheckClose(tOut.fFlat(i), 16.0f, 0.5f);
}

TEST(q8_simd_matvec_7_rows) {
    auto tMat = CTensor::Fill({7, 32}, 1.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({32}, 1.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 7; i++)
        CheckClose(tOut.fFlat(i), 32.0f, 1.0f);
}

TEST(q8_simd_matvec_large_matrix) {
    auto tMat = CTensor::Rand({128, 256});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({256});
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    Check(tOut.m_iNdim == 1, "ndim");
    Check(tOut.m_lShape[0] == 128, "shape[0]");
    for (int64_t i = 0; i < 128; i++) {
        Check(!std::isnan(tOut.fFlat(i)), "no NaN");
        Check(!std::isinf(tOut.fFlat(i)), "no Inf");
    }
}
// >>>s_end(q8_matvec_simd)

// <<<s_start(q8_consistency)
// --- q8 consistency checks
TEST(q8_scalar_vs_simd_agree) {
    auto tMat = CTensor::Rand({32, 64});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({64});
    auto tScalar = Q8::Matvec(qMat, tVec);
    auto tSimd = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 32; i++)
        CheckClose(tScalar.fFlat(i), tSimd.fFlat(i), 1e-3f);
}

TEST(q8_scalar_vs_simd_agree_large) {
    auto tMat = CTensor::Rand({128, 256});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({256});
    auto tScalar = Q8::Matvec(qMat, tVec);
    auto tSimd = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 128; i++)
        CheckClose(tScalar.fFlat(i), tSimd.fFlat(i), 1e-3f);
}

TEST(q8_scalar_vs_simd_agree_non_aligned) {
    auto tMat = CTensor::Rand({17, 37});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({37});
    auto tScalar = Q8::Matvec(qMat, tVec);
    auto tSimd = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 17; i++)
        CheckClose(tScalar.fFlat(i), tSimd.fFlat(i), 1e-3f);
}

TEST(q8_vs_f32_matvec_close) {
    auto tMat = CTensor::Rand({16, 32});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({32});
    auto tQ8 = Q8::MatvecSimd(qMat, tVec);
    auto tF32 = SM::Matvec(tMat, tVec);
    for (int64_t i = 0; i < 16; i++)
        CheckClose(tQ8.fFlat(i), tF32.fFlat(i), 0.1f);
}

TEST(q8_vs_f32_matvec_close_large) {
    auto tMat = CTensor::Rand({64, 256});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({256});
    auto tQ8 = Q8::MatvecSimd(qMat, tVec);
    auto tF32 = SM::Matvec(tMat, tVec);
    float fMaxErr = 0.0f;
    for (int64_t i = 0; i < 64; i++) {
        float fErr = std::fabs(tQ8.fFlat(i) - tF32.fFlat(i));
        if (fErr > fMaxErr)
            fMaxErr = fErr;
    }
    Check(fMaxErr < 1.0f, "max error < 1.0 for rand matrix");
}

TEST(q8_vs_f32_matvec_uniform_exact) {
    auto tMat = CTensor::Fill({8, 16}, 1.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({16}, 1.0f);
    auto tQ8 = Q8::MatvecSimd(qMat, tVec);
    auto tF32 = SM::Matvec(tMat, tVec);
    for (int64_t i = 0; i < 8; i++)
        CheckClose(tQ8.fFlat(i), tF32.fFlat(i), 0.5f);
}

TEST(q8_roundtrip_quality) {
    auto tMat = CTensor::Rand({4, 8});
    auto qMat = Q8::Quantize(tMat);
    const float *pfOrig = tMat.pfData();
    for (int64_t r = 0; r < 4; r++) {
        for (int64_t c = 0; c < 8; c++) {
            float fOrig = pfOrig[r * 8 + c];
            float fDeq = (float)qMat.viData[r * 8 + c] * qMat.vfScale[r];
            CheckClose(fOrig, fDeq, qMat.vfScale[r] * 0.6f);
        }
    }
}
// >>>s_end(q8_consistency)

// <<<s_start(q8_edge)
// --- q8 edge cases
TEST(q8_matvec_single_element) {
    auto tMat = CTensor::Fill({1, 1}, 3.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({1}, 5.0f);
    auto tScalar = Q8::Matvec(qMat, tVec);
    auto tSimd = Q8::MatvecSimd(qMat, tVec);
    CheckClose(tScalar.fFlat(0), 15.0f, 0.5f);
    CheckClose(tSimd.fFlat(0), 15.0f, 0.5f);
}

TEST(q8_matvec_negative_values) {
    auto tMat = CTensor::Fill({2, 4}, -3.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({4}, 2.0f);
    auto tScalar = Q8::Matvec(qMat, tVec);
    auto tSimd = Q8::MatvecSimd(qMat, tVec);
    CheckClose(tScalar.fFlat(0), -24.0f, 1.0f);
    CheckClose(tScalar.fFlat(1), -24.0f, 1.0f);
    CheckClose(tSimd.fFlat(0), -24.0f, 1.0f);
    CheckClose(tSimd.fFlat(1), -24.0f, 1.0f);
}

TEST(q8_matvec_large_values) {
    auto tMat = CTensor::Fill({2, 4}, 1000.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({4}, 1.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    // each row: 4 * 1000 = 4000
    for (int64_t i = 0; i < 2; i++)
        CheckClose(tOut.fFlat(i), 4000.0f, 50.0f);
}

TEST(q8_matvec_mixed_signs) {
    auto tMat = CTensor::Zeros({1, 4});
    tMat.fFlat(0) = 5.0f;
    tMat.fFlat(1) = -5.0f;
    tMat.fFlat(2) = 5.0f;
    tMat.fFlat(3) = -5.0f;
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({4}, 1.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    // 5 - 5 + 5 - 5 = 0
    CheckClose(tOut.fFlat(0), 0.0f, 0.5f);
}

TEST(q8_does_not_modify_input) {
    auto tMat = CTensor::Fill({4, 8}, 2.0f);
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Fill({8}, 3.0f);
    auto tOut = Q8::MatvecSimd(qMat, tVec);
    for (int64_t i = 0; i < 8; i++)
        CheckClose(tVec.fFlat(i), 3.0f);
    Check(qMat.lRows == 4, "rows unchanged");
    Check(qMat.lCols == 8, "cols unchanged");
}
// >>>s_end(q8_edge)

// <<<s_start(q8_benches)
// --- q8 benchmarks
TEST(bench_q8_quantize) {
    auto tMat = CTensor::Rand({512, 512});
    Bench("q8  quantize 512x512", 500, [&]() {
        auto qMat = Q8::Quantize(tMat);
        (void)qMat;
    });
}

TEST(bench_q8_quantize_large) {
    auto tMat = CTensor::Rand({4096, 4096});
    Bench("q8  quantize 4096x4096", 50, [&]() {
        auto qMat = Q8::Quantize(tMat);
        (void)qMat;
    });
}

TEST(bench_q8_matvec_scalar) {
    auto tMat = CTensor::Rand({512, 512});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({512});
    Bench("q8  matvec scalar 512", 1000, [&]() {
        auto tO = Q8::Matvec(qMat, tVec);
        (void)tO;
    });
}

TEST(bench_q8_matvec_simd_small) {
    auto tMat = CTensor::Rand({512, 512});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({512});
    Bench("q8  matvec simd 512", 1000, [&]() {
        auto tO = Q8::MatvecSimd(qMat, tVec);
        (void)tO;
    });
    Bench("f32 matvec simd 512", 1000, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
    Bench("f32 matvec op   512", 1000, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_q8_matvec_simd_medium) {
    auto tMat = CTensor::Rand({1024, 1024});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({1024});
    Bench("q8  matvec simd 1024", 1000, [&]() {
        auto tO = Q8::MatvecSimd(qMat, tVec);
        (void)tO;
    });
    Bench("f32 matvec simd 1024", 1000, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_q8_matvec_simd_large) {
    auto tMat = CTensor::Rand({4096, 4096});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({4096});
    Bench("q8  matvec simd 4096", 500, [&]() {
        auto tO = Q8::MatvecSimd(qMat, tVec);
        (void)tO;
    });
    Bench("f32 matvec simd 4096", 500, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
    Bench("f32 matvec op   4096", 500, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_q8_matvec_simd_xlarge) {
    auto tMat = CTensor::Rand({8192, 4096});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({4096});
    Bench("q8  matvec simd 8192x4096", 200, [&]() {
        auto tO = Q8::MatvecSimd(qMat, tVec);
        (void)tO;
    });
    Bench("f32 matvec simd 8192x4096", 200, [&]() {
        auto tO = SM::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_q8_scalar_vs_simd) {
    auto tMat = CTensor::Rand({2048, 2048});
    auto qMat = Q8::Quantize(tMat);
    auto tVec = CTensor::Rand({2048});
    Bench("q8  matvec scalar 2048", 200, [&]() {
        auto tO = Q8::Matvec(qMat, tVec);
        (void)tO;
    });
    Bench("q8  matvec simd   2048", 200, [&]() {
        auto tO = Q8::MatvecSimd(qMat, tVec);
        (void)tO;
    });
}

TEST(bench_q8_memory_advantage) {
    // q8: 4096*4096 * 1 byte + 4096 * 4 bytes = ~16.0 MB
    // f32: 4096*4096 * 4 bytes = ~64.0 MB
    auto tMat = CTensor::Rand({4096, 4096});
    auto qMat = Q8::Quantize(tMat);
    int64_t lF32Bytes = 4096LL * 4096 * 4;
    int64_t lQ8Bytes = 4096LL * 4096 * 1 + 4096 * 4;
    float fRatio = (float)lF32Bytes / (float)lQ8Bytes;
    std::cout << "           f32: " << lF32Bytes / (1024 * 1024) << " MB, q8: " << lQ8Bytes / (1024 * 1024)
              << " MB, ratio: " << fRatio << "x" << std::endl;
    Check(fRatio > 3.5f, "q8 should be ~4x smaller");
}
// >>>s_end(q8_benches)
