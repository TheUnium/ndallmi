// Created by Unium on 21.02.26

#pragma once

#include "../Tensor/mtTnTnsr.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// <<<s_start(tests)
// --- tensor tests
// --- constructors
TEST(default_constructor) {
    CTensor t;
    Check(t.m_iNdim == 0, "ndim should be 0");
    Check(t.m_pData == nullptr, "data should be null");
    Check(t.m_bOwnsData == false, "should not own data");
    Check(t.lNumel() == 0, "numel should be 0");
}

TEST(init_list_constructor) {
    auto t = CTensor({3, 4});
    Check(t.m_iNdim == 2, "ndim");
    Check(t.m_lShape[0] == 3, "shape[0]");
    Check(t.m_lShape[1] == 4, "shape[1]");
    Check(t.lNumel() == 12, "numel");
    Check(t.m_bOwnsData == true, "should own data");
    Check(t.m_pData != nullptr, "data should not be null");
}

TEST(vector_constructor) {
    std::vector<int64_t> vlShape = {2, 3, 4};
    CTensor t(vlShape);
    Check(t.m_iNdim == 3, "ndim");
    Check(t.m_lShape[0] == 2, "shape[0]");
    Check(t.m_lShape[1] == 3, "shape[1]");
    Check(t.m_lShape[2] == 4, "shape[2]");
    Check(t.lNumel() == 24, "numel");
}

TEST(1d_constructor) {
    auto t = CTensor({10});
    Check(t.m_iNdim == 1, "ndim");
    Check(t.m_lShape[0] == 10, "shape[0]");
    Check(t.lNumel() == 10, "numel");
}

TEST(high_dim_constructor) {
    auto t = CTensor({2, 3, 4, 5});
    Check(t.m_iNdim == 4, "ndim");
    Check(t.lNumel() == 120, "numel");
}

TEST(move_constructor) {
    auto tA = CTensor::Fill({3, 3}, 5.0f);
    void *pOrigData = tA.m_pData;
    CTensor tB(std::move(tA));

    Check(tB.m_pData == pOrigData, "should steal data pointer");
    Check(tB.m_iNdim == 2, "ndim");
    Check(tB.m_lShape[0] == 3, "shape[0]");
    Check(tB.m_lShape[1] == 3, "shape[1]");
    Check(tB.m_bOwnsData == true, "should own data");
    CheckClose(tB.fAt({1, 1}), 5.0f);

    Check(tA.m_pData == nullptr, "source data should be null");
    Check(tA.m_bOwnsData == false, "source should not own data");
    Check(tA.m_iNdim == 0, "source ndim should be 0");
}

TEST(move_assignment) {
    auto tA = CTensor::Fill({2, 2}, 3.0f);
    auto tB = CTensor::Fill({4, 4}, 9.0f);
    void *pOrigA = tA.m_pData;

    tB = std::move(tA);

    Check(tB.m_pData == pOrigA, "should steal data pointer");
    Check(tB.m_iNdim == 2, "ndim");
    Check(tB.m_lShape[0] == 2, "shape[0]");
    Check(tB.m_lShape[1] == 2, "shape[1]");
    CheckClose(tB.fAt({0, 0}), 3.0f);

    Check(tA.m_pData == nullptr, "source should be null");
}

TEST(move_self_assignment) {
    auto t = CTensor::Fill({2, 2}, 7.0f);
    auto *pSelf = &t;
    *pSelf = std::move(t);
    // should not crash, data should survive
    Check(t.m_pData != nullptr, "self move should preserve data");
}

// --- factory methods
TEST(zeros) {
    auto t = CTensor::Zeros({2, 3});
    Check(t.m_iNdim == 2, "ndim");
    Check(t.lNumel() == 6, "numel");
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 0.0f);
}

TEST(zeros_vector) {
    std::vector<int64_t> vlShape = {4, 5};
    auto t = CTensor::Zeros(vlShape);
    Check(t.lNumel() == 20, "numel");
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 0.0f);
}

TEST(ones) {
    auto t = CTensor::Ones({3, 3});
    Check(t.lNumel() == 9, "numel");
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 1.0f);
}

TEST(ones_vector) {
    std::vector<int64_t> vlShape = {2, 2, 2};
    auto t = CTensor::Ones(vlShape);
    Check(t.lNumel() == 8, "numel");
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 1.0f);
}

TEST(fill) {
    auto t = CTensor::Fill({2, 4}, 3.14f);
    Check(t.lNumel() == 8, "numel");
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 3.14f);
}

TEST(fill_negative) {
    auto t = CTensor::Fill({5}, -2.5f);
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), -2.5f);
}

TEST(fill_vector) {
    std::vector<int64_t> vlShape = {3, 3};
    auto t = CTensor::Fill(vlShape, 42.0f);
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 42.0f);
}

TEST(rand) {
    auto t = CTensor::Rand({100});
    Check(t.lNumel() == 100, "numel");
    bool bHasNonZero = false;
    bool bAllInRange = true;
    for (int64_t i = 0; i < t.lNumel(); i++) {
        float fV = t.fFlat(i);
        if (fV != 0.0f)
            bHasNonZero = true;
        if (fV < 0.0f || fV >= 1.0f)
            bAllInRange = false;
    }
    Check(bHasNonZero, "rand should produce non-zero values");
    Check(bAllInRange, "rand values should be in [0, 1)");
}

TEST(rand_vector) {
    std::vector<int64_t> vlShape = {10, 10};
    auto t = CTensor::Rand(vlShape);
    Check(t.lNumel() == 100, "numel");
    for (int64_t i = 0; i < t.lNumel(); i++) {
        float fV = t.fFlat(i);
        Check(fV >= 0.0f && fV < 1.0f, "rand out of range");
    }
}

TEST(rand_not_all_same) {
    auto t = CTensor::Rand({50});
    float fFirst = t.fFlat(0);
    bool bDifferent = false;
    for (int64_t i = 1; i < t.lNumel(); i++) {
        if (t.fFlat(i) != fFirst) {
            bDifferent = true;
            break;
        }
    }
    Check(bDifferent, "rand should produce varying values");
}

// --- element access
TEST(fAt_2d) {
    auto t = CTensor::Zeros({3, 4});
    t.fAt({0, 0}) = 1.0f;
    t.fAt({1, 2}) = 42.0f;
    t.fAt({2, 3}) = -7.0f;
    CheckClose(t.fAt({0, 0}), 1.0f);
    CheckClose(t.fAt({1, 2}), 42.0f);
    CheckClose(t.fAt({2, 3}), -7.0f);
    CheckClose(t.fAt({0, 1}), 0.0f);
}

TEST(fAt_1d) {
    auto t = CTensor::Zeros({5});
    t.fAt({3}) = 99.0f;
    CheckClose(t.fAt({3}), 99.0f);
    CheckClose(t.fAt({0}), 0.0f);
}

TEST(fAt_3d) {
    auto t = CTensor::Zeros({2, 3, 4});
    t.fAt({1, 2, 3}) = 123.0f;
    t.fAt({0, 0, 0}) = -1.0f;
    CheckClose(t.fAt({1, 2, 3}), 123.0f);
    CheckClose(t.fAt({0, 0, 0}), -1.0f);
    CheckClose(t.fAt({0, 1, 1}), 0.0f);
}

TEST(fFlat_readwrite) {
    auto t = CTensor::Zeros({4, 4});
    for (int64_t i = 0; i < t.lNumel(); i++)
        t.fFlat(i) = (float)i;
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), (float)i);
}

TEST(fAt_vs_fFlat_consistency) {
    auto t = CTensor::Zeros({3, 4});
    // r major should be fAt({ r, c }) == fFlat(r * 4 + c)
    for (int64_t r = 0; r < 3; r++) {
        for (int64_t c = 0; c < 4; c++) {
            float fVal = (float)(r * 10 + c);
            t.fAt({r, c}) = fVal;
        }
    }
    for (int64_t r = 0; r < 3; r++) {
        for (int64_t c = 0; c < 4; c++) {
            float fExpected = (float)(r * 10 + c);
            CheckClose(t.fFlat(r * 4 + c), fExpected);
        }
    }
}

TEST(pfData_access) {
    auto t = CTensor::Fill({3}, 5.0f);
    float *pf = t.pfData();
    Check(pf != nullptr, "pfData should not be null");
    CheckClose(pf[0], 5.0f);
    CheckClose(pf[1], 5.0f);
    CheckClose(pf[2], 5.0f);
    pf[1] = 77.0f;
    CheckClose(t.fFlat(1), 77.0f);
}

TEST(pfData_const) {
    auto t = CTensor::Fill({4}, 2.0f);
    const CTensor &tRef = t;
    const float *pf = tRef.pfData();
    CheckClose(pf[0], 2.0f);
    CheckClose(pf[3], 2.0f);
}

// --- clone
TEST(clone_deep_copy) {
    auto tA = CTensor::Fill({2, 2}, 7.0f);
    auto tB = tA.Clone();
    tB.fAt({0, 0}) = 999.0f;
    CheckClose(tA.fAt({0, 0}), 7.0f, 1e-5f);
    CheckClose(tB.fAt({0, 0}), 999.0f);
}

TEST(clone_preserves_shape) {
    auto tA = CTensor::Rand({3, 4, 5});
    auto tB = tA.Clone();
    Check(tB.m_iNdim == 3, "ndim");
    Check(tB.m_lShape[0] == 3, "shape[0]");
    Check(tB.m_lShape[1] == 4, "shape[1]");
    Check(tB.m_lShape[2] == 5, "shape[2]");
    Check(tB.m_bOwnsData == true, "clone should own data");
    Check(tB.m_pData != tA.m_pData, "clone should have different data pointer");
}

TEST(clone_preserves_values) {
    auto tA = CTensor::Rand({10});
    auto tB = tA.Clone();
    CheckTensorsClose(tA, tB);
}

// --- strides
TEST(strides_row_major_2d) {
    auto t = CTensor({3, 4});
    // row-major: stride[0] = 4, stride[1] = 1
    Check(t.m_lStride[0] == 4, "stride[0]");
    Check(t.m_lStride[1] == 1, "stride[1]");
}

TEST(strides_row_major_3d) {
    auto t = CTensor({2, 3, 4});
    // stride[0] = 12, stride[1] = 4, stride[2] = 1
    Check(t.m_lStride[0] == 12, "stride[0]");
    Check(t.m_lStride[1] == 4, "stride[1]");
    Check(t.m_lStride[2] == 1, "stride[2]");
}

TEST(strides_1d) {
    auto t = CTensor({7});
    Check(t.m_lStride[0] == 1, "stride[0]");
}

TEST(strides_4d) {
    auto t = CTensor({2, 3, 4, 5});
    Check(t.m_lStride[0] == 60, "stride[0]");
    Check(t.m_lStride[1] == 20, "stride[1]");
    Check(t.m_lStride[2] == 5, "stride[2]");
    Check(t.m_lStride[3] == 1, "stride[3]");
}

// --- contiguous
TEST(is_contiguous_fresh) {
    auto t = CTensor::Zeros({3, 4});
    Check(t.bIsContiguous(), "fresh tensor should be contiguous");
}

TEST(is_contiguous_1d) {
    auto t = CTensor::Zeros({10});
    Check(t.bIsContiguous(), "1d tensor should be contiguous");
}

TEST(is_contiguous_default) {
    CTensor t;
    Check(t.bIsContiguous(), "empty tensor should be contiguous");
}

TEST(contiguous_already) {
    auto t = CTensor::Fill({3, 4}, 2.0f);
    auto tC = t.Contiguous();
    // should return a non-owning view since already contiguous
    Check(tC.m_pData == t.m_pData, "contiguous of contiguous should share data");
    Check(tC.m_bOwnsData == false, "should not own data (view)");
    CheckTensorsClose(t, tC);
}

// --- reshape
TEST(reshape_basic) {
    auto t = CTensor::Zeros({3, 4});
    for (int64_t i = 0; i < 12; i++)
        t.fFlat(i) = (float)i;

    auto tR = t.Reshape({4, 3});
    Check(tR.m_iNdim == 2, "ndim");
    Check(tR.m_lShape[0] == 4, "shape[0]");
    Check(tR.m_lShape[1] == 3, "shape[1]");
    Check(tR.lNumel() == 12, "numel");
    // data should be shared (v)
    Check(tR.m_pData == t.m_pData, "reshape should share data");
    Check(tR.m_bOwnsData == false, "reshape should not own data");
}

TEST(reshape_1d_to_2d) {
    auto t = CTensor::Zeros({12});
    for (int64_t i = 0; i < 12; i++)
        t.fFlat(i) = (float)i;

    auto tR = t.Reshape({3, 4});
    Check(tR.m_iNdim == 2, "ndim");
    CheckClose(tR.fAt({0, 0}), 0.0f);
    CheckClose(tR.fAt({0, 3}), 3.0f);
    CheckClose(tR.fAt({1, 0}), 4.0f);
    CheckClose(tR.fAt({2, 3}), 11.0f);
}

TEST(reshape_infer_dim) {
    auto t = CTensor::Zeros({2, 3, 4});
    auto tR = t.Reshape({6, -1});
    Check(tR.m_lShape[0] == 6, "shape[0]");
    Check(tR.m_lShape[1] == 4, "shape[1] should be inferred as 4");
    Check(tR.lNumel() == 24, "numel");
}

TEST(reshape_infer_first_dim) {
    auto t = CTensor::Zeros({24});
    auto tR = t.Reshape({-1, 6});
    Check(tR.m_lShape[0] == 4, "shape[0] inferred");
    Check(tR.m_lShape[1] == 6, "shape[1]");
}

TEST(reshape_flatten) {
    auto t = CTensor::Rand({2, 3, 4});
    auto tR = t.Reshape({-1});
    Check(tR.m_iNdim == 1, "ndim");
    Check(tR.m_lShape[0] == 24, "shape[0]");
}

TEST(reshape_preserves_values) {
    auto t = CTensor::Rand({3, 4});
    auto tR = t.Reshape({12});
    for (int64_t i = 0; i < 12; i++)
        CheckClose(tR.fFlat(i), t.fFlat(i));
}

// --- view
TEST(view_basic) {
    auto t = CTensor::Rand({6});
    auto tV = t.View({2, 3});
    Check(tV.m_iNdim == 2, "ndim");
    Check(tV.m_lShape[0] == 2, "shape[0]");
    Check(tV.m_lShape[1] == 3, "shape[1]");
    Check(tV.m_pData == t.m_pData, "view should share data");
}

// --- transpose
TEST(transpose_2d) {
    auto t = CTensor::Zeros({3, 4});
    for (int64_t r = 0; r < 3; r++)
        for (int64_t c = 0; c < 4; c++)
            t.fAt({r, c}) = (float)(r * 10 + c);

    auto tT = t.Transpose(0, 1);
    Check(tT.m_iNdim == 2, "ndim");
    Check(tT.m_lShape[0] == 4, "shape[0]");
    Check(tT.m_lShape[1] == 3, "shape[1]");
    Check(tT.m_pData == t.m_pData, "transpose should share data");
    Check(tT.m_bOwnsData == false, "transpose should not own data");

    // tT[c, r] should equal t[r, c]
    CheckClose(tT.fAt({0, 0}), 0.0f);
    CheckClose(tT.fAt({1, 0}), 1.0f);
    CheckClose(tT.fAt({0, 2}), 20.0f);
    CheckClose(tT.fAt({3, 2}), 23.0f);
}

TEST(transpose_not_contiguous) {
    auto t = CTensor::Zeros({3, 4});
    auto tT = t.Transpose(0, 1);
    Check(!tT.bIsContiguous(), "transposed 2d should not be contiguous");
}

TEST(transpose_strides) {
    auto t = CTensor({3, 4});
    auto tT = t.Transpose(0, 1);
    // og - stride = [4, 1] | transposed - stride = [1, 4]
    Check(tT.m_lStride[0] == 1, "transposed stride[0]");
    Check(tT.m_lStride[1] == 4, "transposed stride[1]");
}

TEST(transpose_3d) {
    auto t = CTensor::Zeros({2, 3, 4});
    for (int64_t i = 0; i < t.lNumel(); i++)
        t.fFlat(i) = (float)i;

    auto tT = t.Transpose(0, 2);
    Check(tT.m_lShape[0] == 4, "shape[0]");
    Check(tT.m_lShape[1] == 3, "shape[1]");
    Check(tT.m_lShape[2] == 2, "shape[2]");
}

TEST(transpose_same_dim) {
    auto t = CTensor::Rand({3, 4});
    auto tT = t.Transpose(0, 0);
    // transposing same dim is identity
    Check(tT.m_lShape[0] == 3, "shape[0]");
    Check(tT.m_lShape[1] == 4, "shape[1]");
    Check(tT.bIsContiguous(), "same-dim transpose should stay contiguous");
}

// --- contiguous from non-contiguous
TEST(contiguous_from_transpose) {
    auto t = CTensor::Zeros({3, 4});
    for (int64_t r = 0; r < 3; r++)
        for (int64_t c = 0; c < 4; c++)
            t.fAt({r, c}) = (float)(r * 10 + c);

    auto tT = t.Transpose(0, 1); // 4x3, non cont
    Check(!tT.bIsContiguous(), "transposed should not be contiguous");

    auto tC = tT.Contiguous(); // 4x3, cont copy
    Check(tC.bIsContiguous(), "contiguous copy should be contiguous");
    Check(tC.m_bOwnsData == true, "contiguous copy should own data");
    Check(tC.m_lShape[0] == 4, "shape[0]");
    Check(tC.m_lShape[1] == 3, "shape[1]");

    // vals tC[c, r] == t[r, c]
    for (int64_t c = 0; c < 4; c++) {
        for (int64_t r = 0; r < 3; r++) {
            float fExpected = (float)(r * 10 + c);
            CheckClose(tC.fAt({c, r}), fExpected);
        }
    }
}

TEST(contiguous_data_independence) {
    auto t = CTensor::Fill({2, 3}, 5.0f);
    auto tT = t.Transpose(0, 1);
    auto tC = tT.Contiguous();
    // modify original bc cont copy should be unaffected
    t.fAt({0, 0}) = 999.0f;
    CheckClose(tC.fAt({0, 0}), 5.0f, 1e-5f);
}

// --- dtype utils
TEST(type_size) {
    Check(nGetTypeSize(EType::F32) == 4, "f32 size");
    Check(nGetTypeSize(EType::F16) == 2, "f16 size");
    Check(nGetTypeSize(EType::I32) == 4, "i32 size");
    Check(nGetTypeSize(EType::I8) == 1, "i8 size");
}

TEST(type_name) {
    Check(std::string(szGetTypeName(EType::F32)) == "f32", "f32 name");
    Check(std::string(szGetTypeName(EType::F16)) == "f16", "f16 name");
    Check(std::string(szGetTypeName(EType::I32)) == "i32", "i32 name");
    Check(std::string(szGetTypeName(EType::I8)) == "i8", "i8 name");
}

// --- numel edge cases
TEST(numel_large) {
    auto t = CTensor({100, 100, 100});
    Check(t.lNumel() == 1000000, "numel large");
}

TEST(numel_1x1) {
    auto t = CTensor({1, 1});
    Check(t.lNumel() == 1, "numel 1x1");
}

TEST(numel_single) {
    auto t = CTensor({1});
    Check(t.lNumel() == 1, "numel single");
}

// --- data size
TEST(data_size_f32) {
    auto t = CTensor({3, 4}, EType::F32);
    Check(t.m_iDataSize == 12 * 4, "data size f32");
}

// --- print
TEST(print_smoke_1d) {
    auto t = CTensor::Rand({5});
    t.Print("smoke_1d");
}

TEST(print_smoke_2d) {
    auto t = CTensor::Rand({3, 4});
    t.Print("smoke_2d");
}

TEST(print_smoke_3d) {
    auto t = CTensor::Rand({2, 3, 4});
    t.Print("smoke_3d");
}

TEST(print_smoke_empty) {
    CTensor t;
    t.Print("smoke_empty");
}

TEST(print_shape_smoke) {
    auto t = CTensor::Rand({2, 3, 4});
    t.PrintShape("shape_smoke");
}

TEST(print_large_truncated) {
    auto t = CTensor::Rand({100});
    t.Print("large_truncated");
}

TEST(print_large_2d_truncated) {
    auto t = CTensor::Rand({20, 20});
    t.Print("large_2d_truncated");
}

// --- reshape + transpose combined
TEST(reshape_then_transpose) {
    auto t = CTensor::Zeros({12});
    for (int64_t i = 0; i < 12; i++)
        t.fFlat(i) = (float)i;

    auto tR = t.Reshape({3, 4});
    auto tT = tR.Transpose(0, 1);
    Check(tT.m_lShape[0] == 4, "shape[0]");
    Check(tT.m_lShape[1] == 3, "shape[1]");
    // tT[c, r] = tR[r, c] = flat[r*4 + c]
    CheckClose(tT.fAt({2, 1}), 6.0f); // tR[1,2] = flat[6]
}

TEST(transpose_then_contiguous_then_reshape) {
    auto t = CTensor::Zeros({3, 4});
    for (int64_t r = 0; r < 3; r++)
        for (int64_t c = 0; c < 4; c++)
            t.fAt({r, c}) = (float)(r * 4 + c);

    auto tT = t.Transpose(0, 1);  // 4x3
    auto tC = tT.Contiguous();    // 4x3 cont
    auto tR = tC.Reshape({2, 6}); // 2x6
    Check(tR.m_lShape[0] == 2, "shape[0]");
    Check(tR.m_lShape[1] == 6, "shape[1]");
    Check(tR.bIsContiguous(), "reshaped contiguous should be contiguous");
}

// --- view shares data
TEST(view_writes_reflect) {
    auto t = CTensor::Zeros({12});
    for (int64_t i = 0; i < 12; i++)
        t.fFlat(i) = (float)i;

    auto tV = t.View({3, 4});
    tV.fAt({1, 0}) = 999.0f;
    CheckClose(t.fFlat(4), 999.0f, 1e-5f);
}

// --- misc
TEST(check_tensors_close_pass) {
    auto tA = CTensor::Fill({2, 3}, 1.0f);
    auto tB = CTensor::Fill({2, 3}, 1.0f);
    CheckTensorsClose(tA, tB);
}
// >>>s_end(tests)

// <<<s_start(benches)
// --- benchmarks
TEST(bench_alloc_small) {
    Bench("alloc 64x64", 10000, []() {
        auto t = CTensor::Zeros({64, 64});
        (void)t;
    });
}

TEST(bench_alloc_medium) {
    Bench("alloc 512x512", 1000, []() {
        auto t = CTensor::Zeros({512, 512});
        (void)t;
    });
}

TEST(bench_clone) {
    auto t = CTensor::Rand({256, 256});
    Bench("clone 256x256", 5000, [&]() {
        auto tC = t.Clone();
        (void)tC;
    });
}

TEST(bench_transpose_contiguous) {
    auto t = CTensor::Rand({256, 256});
    Bench("transpose+contiguous 256x256", 1000, [&]() {
        auto tT = t.Transpose(0, 1);
        auto tC = tT.Contiguous();
        (void)tC;
    });
}

TEST(bench_fill) {
    Bench("fill 1024x1024", 200, []() {
        auto t = CTensor::Fill({1024, 1024}, 3.14f);
        (void)t;
    });
}

TEST(bench_rand) {
    Bench("rand 1024x1024", 100, []() {
        auto t = CTensor::Rand({1024, 1024});
        (void)t;
    });
}

TEST(bench_flat_access) {
    auto t = CTensor::Rand({1024, 1024});
    Bench("flat read 1M", 100, [&]() {
        float fSum = 0.0f;
        for (int64_t i = 0; i < t.lNumel(); i++)
            fSum += t.fFlat(i);
        (void)fSum;
    });
}
// >>>s_end(benches)
