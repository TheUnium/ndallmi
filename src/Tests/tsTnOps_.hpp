// Created by Unium on 21.02.26

#pragma once

#include "../Tensor/mtTnOps_.hpp"
#include "../Tensor/mtTnTnsr.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// <<<s_start(element_wb)
// --- element wise binary
TEST(add_basic) {
    auto tA = CTensor::Fill({3, 4}, 2.0f);
    auto tB = CTensor::Fill({3, 4}, 3.0f);
    auto tC = OP::Add(tA, tB);
    Check(tC.m_iNdim == 2, "ndim");
    Check(tC.m_lShape[0] == 3, "shape[0]");
    Check(tC.m_lShape[1] == 4, "shape[1]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 5.0f);
}

TEST(add_varied) {
    auto tA = CTensor::Zeros({4});
    auto tB = CTensor::Zeros({4});
    for (int64_t i = 0; i < 4; i++) {
        tA.fFlat(i) = (float)i;
        tB.fFlat(i) = (float)(10 + i);
    }
    auto tC = OP::Add(tA, tB);
    CheckClose(tC.fFlat(0), 10.0f);
    CheckClose(tC.fFlat(1), 12.0f);
    CheckClose(tC.fFlat(2), 14.0f);
    CheckClose(tC.fFlat(3), 16.0f);
}

TEST(sub_basic) {
    auto tA = CTensor::Fill({2, 3}, 10.0f);
    auto tB = CTensor::Fill({2, 3}, 3.0f);
    auto tC = OP::Sub(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 7.0f);
}

TEST(sub_negative_result) {
    auto tA = CTensor::Fill({5}, 1.0f);
    auto tB = CTensor::Fill({5}, 5.0f);
    auto tC = OP::Sub(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), -4.0f);
}

TEST(mul_basic) {
    auto tA = CTensor::Fill({2, 2}, 3.0f);
    auto tB = CTensor::Fill({2, 2}, 4.0f);
    auto tC = OP::Mul(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 12.0f);
}

TEST(mul_by_zero) {
    auto tA = CTensor::Fill({3}, 99.0f);
    auto tB = CTensor::Zeros({3});
    auto tC = OP::Mul(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 0.0f);
}

TEST(div_basic) {
    auto tA = CTensor::Fill({4}, 12.0f);
    auto tB = CTensor::Fill({4}, 3.0f);
    auto tC = OP::Div(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 4.0f);
}

TEST(div_fractional) {
    auto tA = CTensor::Fill({2}, 1.0f);
    auto tB = CTensor::Fill({2}, 3.0f);
    auto tC = OP::Div(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 1.0f / 3.0f);
}

TEST(add_does_not_modify_inputs) {
    auto tA = CTensor::Fill({3}, 2.0f);
    auto tB = CTensor::Fill({3}, 5.0f);
    auto tC = OP::Add(tA, tB);
    CheckClose(tA.fFlat(0), 2.0f);
    CheckClose(tB.fFlat(0), 5.0f);
}

TEST(elementwise_1d) {
    auto tA = CTensor::Zeros({5});
    auto tB = CTensor::Zeros({5});
    for (int64_t i = 0; i < 5; i++) {
        tA.fFlat(i) = (float)i;
        tB.fFlat(i) = (float)(i * 2);
    }
    auto tC = OP::Add(tA, tB);
    for (int64_t i = 0; i < 5; i++)
        CheckClose(tC.fFlat(i), (float)(i * 3));
}

TEST(elementwise_3d) {
    auto tA = CTensor::Fill({2, 3, 4}, 1.0f);
    auto tB = CTensor::Fill({2, 3, 4}, 2.0f);
    auto tC = OP::Add(tA, tB);
    Check(tC.m_iNdim == 3, "ndim");
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 3, "shape[1]");
    Check(tC.m_lShape[2] == 4, "shape[2]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 3.0f);
}
// >>>s_end(element_wb)

// <<<s_start(broadcast)
// --- broadcast ops
TEST(add_broadcast_scalar_to_vector) {
    auto tA = CTensor::Zeros({4});
    for (int64_t i = 0; i < 4; i++)
        tA.fFlat(i) = (float)i;
    auto tB = CTensor::Fill({1}, 10.0f);
    auto tC = OP::AddBroadcast(tA, tB);
    Check(tC.m_lShape[0] == 4, "shape[0]");
    CheckClose(tC.fFlat(0), 10.0f);
    CheckClose(tC.fFlat(1), 11.0f);
    CheckClose(tC.fFlat(2), 12.0f);
    CheckClose(tC.fFlat(3), 13.0f);
}

TEST(add_broadcast_row_to_matrix) {
    auto tA = CTensor::Fill({3, 4}, 1.0f);
    auto tB = CTensor::Zeros({1, 4});
    for (int64_t i = 0; i < 4; i++)
        tB.fFlat(i) = (float)(i * 10);
    auto tC = OP::AddBroadcast(tA, tB);
    Check(tC.m_lShape[0] == 3, "shape[0]");
    Check(tC.m_lShape[1] == 4, "shape[1]");
    for (int64_t r = 0; r < 3; r++) {
        for (int64_t c = 0; c < 4; c++) {
            CheckClose(tC.fAt({r, c}), 1.0f + (float)(c * 10));
        }
    }
}

TEST(add_broadcast_col_to_matrix) {
    auto tA = CTensor::Fill({3, 4}, 1.0f);
    auto tB = CTensor::Zeros({3, 1});
    for (int64_t i = 0; i < 3; i++)
        tB.fFlat(i) = (float)(i * 100);
    auto tC = OP::AddBroadcast(tA, tB);
    Check(tC.m_lShape[0] == 3, "shape[0]");
    Check(tC.m_lShape[1] == 4, "shape[1]");
    for (int64_t r = 0; r < 3; r++) {
        for (int64_t c = 0; c < 4; c++) {
            CheckClose(tC.fAt({r, c}), 1.0f + (float)(r * 100));
        }
    }
}

TEST(sub_broadcast) {
    auto tA = CTensor::Fill({2, 3}, 10.0f);
    auto tB = CTensor::Fill({1, 3}, 3.0f);
    auto tC = OP::SubBroadcast(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 7.0f);
}

TEST(mul_broadcast) {
    auto tA = CTensor::Fill({2, 3}, 5.0f);
    auto tB = CTensor::Fill({1, 3}, 2.0f);
    auto tC = OP::MulBroadcast(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 10.0f);
}

TEST(div_broadcast) {
    auto tA = CTensor::Fill({2, 3}, 12.0f);
    auto tB = CTensor::Fill({1, 3}, 4.0f);
    auto tC = OP::DivBroadcast(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 3.0f);
}

TEST(broadcast_different_ndim) {
    // (3, 4) + (4) --> (3, 4)
    auto tA = CTensor::Fill({3, 4}, 1.0f);
    auto tB = CTensor::Zeros({4});
    for (int64_t i = 0; i < 4; i++)
        tB.fFlat(i) = (float)i;
    auto tC = OP::AddBroadcast(tA, tB);
    Check(tC.m_iNdim == 2, "ndim");
    Check(tC.m_lShape[0] == 3, "shape[0]");
    Check(tC.m_lShape[1] == 4, "shape[1]");
    for (int64_t r = 0; r < 3; r++) {
        for (int64_t c = 0; c < 4; c++) {
            CheckClose(tC.fAt({r, c}), 1.0f + (float)c);
        }
    }
}

TEST(broadcast_both_expand) {
    // (3, 1) + (1, 4) --> (3, 4)
    auto tA = CTensor::Zeros({3, 1});
    auto tB = CTensor::Zeros({1, 4});
    for (int64_t i = 0; i < 3; i++)
        tA.fFlat(i) = (float)(i * 10);
    for (int64_t i = 0; i < 4; i++)
        tB.fFlat(i) = (float)i;
    auto tC = OP::AddBroadcast(tA, tB);
    Check(tC.m_lShape[0] == 3, "shape[0]");
    Check(tC.m_lShape[1] == 4, "shape[1]");
    for (int64_t r = 0; r < 3; r++) {
        for (int64_t c = 0; c < 4; c++) {
            CheckClose(tC.fAt({r, c}), (float)(r * 10 + c));
        }
    }
}

TEST(broadcast_same_shape_fallback) {
    auto tA = CTensor::Fill({2, 3}, 1.0f);
    auto tB = CTensor::Fill({2, 3}, 2.0f);
    auto tC = OP::AddBroadcast(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 3.0f);
}

TEST(broadcast_3d) {
    // (2, 3, 1) + (1, 1, 4) --> (2, 3, 4)
    auto tA = CTensor::Fill({2, 3, 1}, 5.0f);
    auto tB = CTensor::Fill({1, 1, 4}, 3.0f);
    auto tC = OP::AddBroadcast(tA, tB);
    Check(tC.m_iNdim == 3, "ndim");
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 3, "shape[1]");
    Check(tC.m_lShape[2] == 4, "shape[2]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 8.0f);
}
// >>>s_end(broadcast)

// <<<s_start(scalar)
// --- scalar ops
TEST(add_scalar_basic) {
    auto tA = CTensor::Fill({3, 3}, 5.0f);
    auto tB = OP::AddScalar(tA, 10.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 15.0f);
}

TEST(add_scalar_negative) {
    auto tA = CTensor::Fill({4}, 3.0f);
    auto tB = OP::AddScalar(tA, -5.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), -2.0f);
}

TEST(add_scalar_zero) {
    auto tA = CTensor::Fill({3}, 7.0f);
    auto tB = OP::AddScalar(tA, 0.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 7.0f);
}

TEST(mul_scalar_basic) {
    auto tA = CTensor::Fill({2, 2}, 3.0f);
    auto tB = OP::MulScalar(tA, 4.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 12.0f);
}

TEST(mul_scalar_zero) {
    auto tA = CTensor::Fill({5}, 99.0f);
    auto tB = OP::MulScalar(tA, 0.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), 0.0f);
}

TEST(mul_scalar_negative) {
    auto tA = CTensor::Fill({3}, 2.0f);
    auto tB = OP::MulScalar(tA, -3.0f);
    for (int64_t i = 0; i < tB.lNumel(); i++)
        CheckClose(tB.fFlat(i), -6.0f);
}

TEST(scalar_does_not_modify_input) {
    auto tA = CTensor::Fill({3}, 5.0f);
    auto tB = OP::AddScalar(tA, 10.0f);
    CheckClose(tA.fFlat(0), 5.0f);
}
// >>>s_end(scalar)

// <<<s_start(matrix)
// --- matmul
TEST(matmul_identity) {
    // [1 0 | 0 1] x [a b | c d] = [a b | c d]
    auto tI = CTensor::Zeros({2, 2});
    tI.fAt({0, 0}) = 1.0f;
    tI.fAt({1, 1}) = 1.0f;
    auto tA = CTensor::Zeros({2, 2});
    tA.fAt({0, 0}) = 1.0f;
    tA.fAt({0, 1}) = 2.0f;
    tA.fAt({1, 0}) = 3.0f;
    tA.fAt({1, 1}) = 4.0f;
    auto tC = OP::Matmul(tI, tA);
    CheckClose(tC.fAt({0, 0}), 1.0f);
    CheckClose(tC.fAt({0, 1}), 2.0f);
    CheckClose(tC.fAt({1, 0}), 3.0f);
    CheckClose(tC.fAt({1, 1}), 4.0f);
}

TEST(matmul_2x3_3x2) {
    auto tA = CTensor::Zeros({2, 3});
    auto tB = CTensor::Zeros({3, 2});
    // a = [[1, 2, 3], [4, 5, 6]]
    for (int64_t i = 0; i < 6; i++)
        tA.fFlat(i) = (float)(i + 1);
    // b = [[7, 8], [9, 10], [11, 12]]
    for (int64_t i = 0; i < 6; i++)
        tB.fFlat(i) = (float)(i + 7);
    auto tC = OP::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 2, "shape[1]");
    // c[0,0] = 1*7 + 2*9 + 3*11 = 58
    CheckClose(tC.fAt({0, 0}), 58.0f);
    // c[0,1] = 1*8 + 2*10 + 3*12 = 64
    CheckClose(tC.fAt({0, 1}), 64.0f);
    // c[1,0] = 4*7 + 5*9 + 6*11 = 139
    CheckClose(tC.fAt({1, 0}), 139.0f);
    // c[1,1] = 4*8 + 5*10 + 6*12 = 154
    CheckClose(tC.fAt({1, 1}), 154.0f);
}

TEST(matmul_1x1) {
    auto tA = CTensor::Fill({1, 1}, 3.0f);
    auto tB = CTensor::Fill({1, 1}, 5.0f);
    auto tC = OP::Matmul(tA, tB);
    CheckClose(tC.fAt({0, 0}), 15.0f);
}

TEST(matmul_rectangular) {
    auto tA = CTensor::Fill({1, 4}, 2.0f);
    auto tB = CTensor::Fill({4, 1}, 3.0f);
    auto tC = OP::Matmul(tA, tB);
    Check(tC.m_lShape[0] == 1, "shape[0]");
    Check(tC.m_lShape[1] == 1, "shape[1]");
    CheckClose(tC.fAt({0, 0}), 24.0f); // 4 * 2 * 3
}

TEST(matmul_zeros) {
    auto tA = CTensor::Zeros({3, 3});
    auto tB = CTensor::Fill({3, 3}, 5.0f);
    auto tC = OP::Matmul(tA, tB);
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 0.0f);
}

// --- bmm
TEST(bmm_basic) {
    auto tA = CTensor::Fill({2, 3, 4}, 1.0f);
    auto tB = CTensor::Fill({2, 4, 2}, 1.0f);
    auto tC = OP::Bmm(tA, tB);
    Check(tC.m_iNdim == 3, "ndim");
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 3, "shape[1]");
    Check(tC.m_lShape[2] == 2, "shape[2]");
    for (int64_t i = 0; i < tC.lNumel(); i++)
        CheckClose(tC.fFlat(i), 4.0f);
}

TEST(bmm_batch_independence) {
    auto tA = CTensor::Zeros({2, 2, 2});
    auto tB = CTensor::Zeros({2, 2, 2});
    // idn
    tA.fAt({0, 0, 0}) = 1.0f;
    tA.fAt({0, 1, 1}) = 1.0f;
    tB.fAt({0, 0, 0}) = 5.0f;
    tB.fAt({0, 0, 1}) = 6.0f;
    tB.fAt({0, 1, 0}) = 7.0f;
    tB.fAt({0, 1, 1}) = 8.0f;
    // all twos
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++) {
            tA.fAt({1, r, c}) = 2.0f;
            tB.fAt({1, r, c}) = 3.0f;
        }
    auto tC = OP::Bmm(tA, tB);
    // idn * b = b
    CheckClose(tC.fAt({0, 0, 0}), 5.0f);
    CheckClose(tC.fAt({0, 0, 1}), 6.0f);
    CheckClose(tC.fAt({0, 1, 0}), 7.0f);
    CheckClose(tC.fAt({0, 1, 1}), 8.0f);
    // 2 * 2 * 3 * 2 = 12e
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 2; c++)
            CheckClose(tC.fAt({1, r, c}), 12.0f);
}

// --- matvec
TEST(matvec_basic) {
    auto tMat = CTensor::Zeros({2, 3});
    // [[1, 2, 3], [4, 5, 6]]
    for (int64_t i = 0; i < 6; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto tVec = CTensor::Fill({3}, 1.0f);
    auto tOut = OP::Matvec(tMat, tVec);
    Check(tOut.m_iNdim == 1, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    CheckClose(tOut.fFlat(0), 6.0f);  // 1 + 2 + 3
    CheckClose(tOut.fFlat(1), 15.0f); // 4 + 5 +  6
}

TEST(matvec_identity) {
    auto tMat = CTensor::Zeros({3, 3});
    tMat.fAt({0, 0}) = 1.0f;
    tMat.fAt({1, 1}) = 1.0f;
    tMat.fAt({2, 2}) = 1.0f;
    auto tVec = CTensor::Zeros({3});
    tVec.fFlat(0) = 7.0f;
    tVec.fFlat(1) = 8.0f;
    tVec.fFlat(2) = 9.0f;
    auto tOut = OP::Matvec(tMat, tVec);
    CheckClose(tOut.fFlat(0), 7.0f);
    CheckClose(tOut.fFlat(1), 8.0f);
    CheckClose(tOut.fFlat(2), 9.0f);
}

TEST(matvec_single_row) {
    auto tMat = CTensor::Zeros({1, 4});
    for (int64_t i = 0; i < 4; i++)
        tMat.fFlat(i) = (float)(i + 1);
    auto tVec = CTensor::Fill({4}, 2.0f);
    auto tOut = OP::Matvec(tMat, tVec);
    Check(tOut.m_lShape[0] == 1, "shape[0]");
    CheckClose(tOut.fFlat(0), 20.0f); // 2 * (1 + 2 + 3 + 4)
}
// >>>s_end(matrix)

// <<<s_start(reduction)
// --- reductions
TEST(sum_all) {
    auto t = CTensor::Fill({3, 4}, 2.0f);
    auto tS = OP::Sum(t);
    Check(tS.m_lShape[0] == 1, "shape");
    CheckClose(tS.fFlat(0), 24.0f);
}

TEST(sum_all_varied) {
    auto t = CTensor::Zeros({5});
    for (int64_t i = 0; i < 5; i++)
        t.fFlat(i) = (float)(i + 1);
    auto tS = OP::Sum(t);
    CheckClose(tS.fFlat(0), 15.0f);
}

TEST(sum_dim0) {
    // [[1, 2, 3], [4, 5, 6]] sum dim 0 -----> [5 ,7, 9]
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)(i + 1);
    auto tS = OP::Sum(t, 0);
    Check(tS.m_iNdim == 1, "ndim");
    Check(tS.m_lShape[0] == 3, "shape[0]");
    CheckClose(tS.fFlat(0), 5.0f);
    CheckClose(tS.fFlat(1), 7.0f);
    CheckClose(tS.fFlat(2), 9.0f);
}

TEST(sum_dim1) {
    // [[1, 2, 3], [4, 5, 6]] sum dim 1 ---> [6, 15]
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)(i + 1);
    auto tS = OP::Sum(t, 1);
    Check(tS.m_iNdim == 1, "ndim");
    Check(tS.m_lShape[0] == 2, "shape[0]");
    CheckClose(tS.fFlat(0), 6.0f);
    CheckClose(tS.fFlat(1), 15.0f);
}

TEST(sum_3d_dim1) {
    auto t = CTensor::Fill({2, 3, 4}, 1.0f);
    auto tS = OP::Sum(t, 1);
    Check(tS.m_iNdim == 2, "ndim");
    Check(tS.m_lShape[0] == 2, "shape[0]");
    Check(tS.m_lShape[1] == 4, "shape[1]");
    for (int64_t i = 0; i < tS.lNumel(); i++)
        CheckClose(tS.fFlat(i), 3.0f);
}

TEST(mean_all) {
    auto t = CTensor::Fill({4}, 8.0f);
    auto tM = OP::Mean(t);
    CheckClose(tM.fFlat(0), 8.0f);
}

TEST(mean_all_varied) {
    auto t = CTensor::Zeros({4});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = 2.0f;
    t.fFlat(2) = 3.0f;
    t.fFlat(3) = 4.0f;
    auto tM = OP::Mean(t);
    CheckClose(tM.fFlat(0), 2.5f);
}

TEST(mean_dim0) {
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)(i + 1);
    auto tM = OP::Mean(t, 0);
    Check(tM.m_lShape[0] == 3, "shape[0]");
    CheckClose(tM.fFlat(0), 2.5f); // (1+4)/2
    CheckClose(tM.fFlat(1), 3.5f); // (2+5)/2
    CheckClose(tM.fFlat(2), 4.5f); // (3+6)/2
}

TEST(max_all) {
    auto t = CTensor::Zeros({5});
    t.fFlat(0) = 3.0f;
    t.fFlat(1) = 7.0f;
    t.fFlat(2) = 1.0f;
    t.fFlat(3) = 9.0f;
    t.fFlat(4) = 2.0f;
    auto tM = OP::Max(t);
    CheckClose(tM.fFlat(0), 9.0f);
}

TEST(max_dim0) {
    auto t = CTensor::Zeros({2, 3});
    // [[1, 5, 3], [4, 2, 6]]
    t.fAt({0, 0}) = 1.0f;
    t.fAt({0, 1}) = 5.0f;
    t.fAt({0, 2}) = 3.0f;
    t.fAt({1, 0}) = 4.0f;
    t.fAt({1, 1}) = 2.0f;
    t.fAt({1, 2}) = 6.0f;
    auto tM = OP::Max(t, 0);
    Check(tM.m_lShape[0] == 3, "shape[0]");
    CheckClose(tM.fFlat(0), 4.0f);
    CheckClose(tM.fFlat(1), 5.0f);
    CheckClose(tM.fFlat(2), 6.0f);
}

TEST(max_dim1) {
    auto t = CTensor::Zeros({2, 3});
    t.fAt({0, 0}) = 1.0f;
    t.fAt({0, 1}) = 5.0f;
    t.fAt({0, 2}) = 3.0f;
    t.fAt({1, 0}) = 4.0f;
    t.fAt({1, 1}) = 2.0f;
    t.fAt({1, 2}) = 6.0f;
    auto tM = OP::Max(t, 1);
    Check(tM.m_lShape[0] == 2, "shape[0]");
    CheckClose(tM.fFlat(0), 5.0f);
    CheckClose(tM.fFlat(1), 6.0f);
}

TEST(max_negative_values) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = -5.0f;
    t.fFlat(1) = -1.0f;
    t.fFlat(2) = -10.0f;
    auto tM = OP::Max(t);
    CheckClose(tM.fFlat(0), -1.0f);
}

TEST(argmax_flat) {
    auto t = CTensor::Zeros({5});
    t.fFlat(0) = 3.0f;
    t.fFlat(1) = 7.0f;
    t.fFlat(2) = 1.0f;
    t.fFlat(3) = 9.0f;
    t.fFlat(4) = 2.0f;
    auto tA = OP::Argmaxxing(t);
    Check(tA.m_eType == EType::I32, "dtype");
    int32_t iIdx = static_cast<int32_t *>(tA.m_pData)[0];
    Check(iIdx == 3, "argmax index");
}

TEST(argmax_dim1) {
    auto t = CTensor::Zeros({2, 3});
    t.fAt({0, 0}) = 1.0f;
    t.fAt({0, 1}) = 5.0f;
    t.fAt({0, 2}) = 3.0f;
    t.fAt({1, 0}) = 4.0f;
    t.fAt({1, 1}) = 2.0f;
    t.fAt({1, 2}) = 6.0f;
    auto tA = OP::Argmaxxing(t, 1);
    Check(tA.m_eType == EType::I32, "dtype");
    Check(tA.m_lShape[0] == 2, "shape[0]");
    int32_t *piData = static_cast<int32_t *>(tA.m_pData);
    Check(piData[0] == 1, "argmax row 0");
    Check(piData[1] == 2, "argmax row 1");
}

TEST(argmax_dim0) {
    auto t = CTensor::Zeros({3, 2});
    t.fAt({0, 0}) = 1.0f;
    t.fAt({0, 1}) = 9.0f;
    t.fAt({1, 0}) = 5.0f;
    t.fAt({1, 1}) = 2.0f;
    t.fAt({2, 0}) = 3.0f;
    t.fAt({2, 1}) = 4.0f;
    auto tA = OP::Argmaxxing(t, 0);
    int32_t *piData = static_cast<int32_t *>(tA.m_pData);
    Check(piData[0] == 1, "argmax col 0");
    Check(piData[1] == 0, "argmax col 1");
}
// >>>s_end(reduction)

// <<<s_start(activation)
// --- activations
TEST(relu_positive) {
    auto t = CTensor::Fill({4}, 3.0f);
    auto tR = OP::Relu(t);
    for (int64_t i = 0; i < tR.lNumel(); i++)
        CheckClose(tR.fFlat(i), 3.0f);
}

TEST(relu_negative) {
    auto t = CTensor::Fill({4}, -3.0f);
    auto tR = OP::Relu(t);
    for (int64_t i = 0; i < tR.lNumel(); i++)
        CheckClose(tR.fFlat(i), 0.0f);
}

TEST(relu_mixed) {
    auto t = CTensor::Zeros({5});
    t.fFlat(0) = -2.0f;
    t.fFlat(1) = 0.0f;
    t.fFlat(2) = 3.0f;
    t.fFlat(3) = -1.0f;
    t.fFlat(4) = 5.0f;
    auto tR = OP::Relu(t);
    CheckClose(tR.fFlat(0), 0.0f);
    CheckClose(tR.fFlat(1), 0.0f);
    CheckClose(tR.fFlat(2), 3.0f);
    CheckClose(tR.fFlat(3), 0.0f);
    CheckClose(tR.fFlat(4), 5.0f);
}

TEST(silu_zero) {
    auto t = CTensor::Zeros({1});
    auto tS = OP::Silu(t);
    CheckClose(tS.fFlat(0), 0.0f);
}

TEST(silu_positive) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = 1.0f;
    auto tS = OP::Silu(t);
    float fExpected = 1.0f / (1.0f + std::exp(-1.0f));
    CheckClose(tS.fFlat(0), fExpected);
}

TEST(silu_large_positive) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = 10.0f;
    auto tS = OP::Silu(t);
    // silu(10) ~ 10 * 1 = ~10 (9.99954 smth smth)
    CheckClose(tS.fFlat(0), 10.0f, 1e-3f);
}

TEST(silu_negative) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = -5.0f;
    auto tS = OP::Silu(t);
    float fExpected = -5.0f / (1.0f + std::exp(5.0f));
    CheckClose(tS.fFlat(0), fExpected, 1e-4f);
}

TEST(gelu_zero) {
    auto t = CTensor::Zeros({1});
    auto tG = OP::Gelu(t);
    CheckClose(tG.fFlat(0), 0.0f, 1e-4f);
}

TEST(gelu_positive) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = 1.0f;
    auto tG = OP::Gelu(t);
    // gelu(1) ~ 0.841344smthsmth (see stuff.py)
    CheckClose(tG.fFlat(0), 0.8413f, 1e-3f);
}

TEST(gelu_negative) {
    auto t = CTensor::Zeros({1});
    t.fFlat(0) = -1.0f;
    auto tG = OP::Gelu(t);
    // gelu(-1) ~ -0.15smthsmth (see stuff.py)
    CheckClose(tG.fFlat(0), -0.1586f, 1e-3f);
}

TEST(softmax_1d) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = 2.0f;
    t.fFlat(2) = 3.0f;
    auto tS = OP::Softmax(t);
    Check(tS.m_iNdim == 1, "ndim");
    Check(tS.m_lShape[0] == 3, "shape[0]");
    float fSum = 0.0f;
    for (int64_t i = 0; i < 3; i++)
        fSum += tS.fFlat(i);
    CheckClose(fSum, 1.0f, 1e-5f);
    Check(tS.fFlat(0) < tS.fFlat(1), "monotonic 0<1");
    Check(tS.fFlat(1) < tS.fFlat(2), "monotonic 1<2");
}

TEST(softmax_uniform) {
    auto t = CTensor::Fill({4}, 5.0f);
    auto tS = OP::Softmax(t);
    for (int64_t i = 0; i < 4; i++)
        CheckClose(tS.fFlat(i), 0.25f, 1e-5f);
}

TEST(softmax_2d_last_dim) {
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tS = OP::Softmax(t, -1);
    for (int64_t r = 0; r < 2; r++) {
        float fSum = 0.0f;
        for (int64_t c = 0; c < 3; c++)
            fSum += tS.fAt({r, c});
        CheckClose(fSum, 1.0f, 1e-5f);
    }
}

TEST(softmax_2d_dim0) {
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tS = OP::Softmax(t, 0);
    for (int64_t c = 0; c < 3; c++) {
        float fSum = 0.0f;
        for (int64_t r = 0; r < 2; r++)
            fSum += tS.fAt({r, c});
        CheckClose(fSum, 1.0f, 1e-5f);
    }
}

TEST(softmax_large_values) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1000.0f;
    t.fFlat(1) = 1001.0f;
    t.fFlat(2) = 1002.0f;
    auto tS = OP::Softmax(t);
    float fSum = 0.0f;
    for (int64_t i = 0; i < 3; i++) {
        Check(!std::isnan(tS.fFlat(i)), "softmax should not produce NaN");
        Check(!std::isinf(tS.fFlat(i)), "softmax should not produce Inf");
        fSum += tS.fFlat(i);
    }
    CheckClose(fSum, 1.0f, 1e-5f);
}

TEST(softmax_preserves_shape) {
    auto t = CTensor::Rand({3, 4, 5});
    auto tS = OP::Softmax(t, -1);
    Check(tS.m_iNdim == 3, "ndim");
    Check(tS.m_lShape[0] == 3, "shape[0]");
    Check(tS.m_lShape[1] == 4, "shape[1]");
    Check(tS.m_lShape[2] == 5, "shape[2]");
}
// >>>s_end(activation)

// <<<s_start(normalization)
// --- normalization
TEST(rmsnorm_basic) {
    auto tX = CTensor::Fill({2, 4}, 1.0f);
    auto tW = CTensor::Fill({4}, 1.0f);
    auto tOut = OP::RmsNorm(tX, tW);
    Check(tOut.m_iNdim == 2, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    Check(tOut.m_lShape[1] == 4, "shape[1]");
    // rms of all ones = 1 so o = ~1?
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 1.0f, 1e-3f);
}

TEST(rmsnorm_weight_scaling) {
    auto tX = CTensor::Fill({1, 4}, 1.0f);
    auto tW = CTensor::Fill({4}, 2.0f);
    auto tOut = OP::RmsNorm(tX, tW);
    // rms = 1 o = 1 * 1/1 * 2 = 2
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 2.0f, 1e-3f);
}

TEST(rmsnorm_varied_input) {
    auto tX = CTensor::Zeros({1, 4});
    tX.fFlat(0) = 1.0f;
    tX.fFlat(1) = 2.0f;
    tX.fFlat(2) = 3.0f;
    tX.fFlat(3) = 4.0f;
    auto tW = CTensor::Fill({4}, 1.0f);
    auto tOut = OP::RmsNorm(tX, tW);
    // rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
    float fRms = std::sqrt(7.5f);
    CheckClose(tOut.fFlat(0), 1.0f / fRms, 1e-3f);
    CheckClose(tOut.fFlat(1), 2.0f / fRms, 1e-3f);
    CheckClose(tOut.fFlat(2), 3.0f / fRms, 1e-3f);
    CheckClose(tOut.fFlat(3), 4.0f / fRms, 1e-3f);
}

TEST(rmsnorm_3d) {
    auto tX = CTensor::Fill({2, 3, 4}, 1.0f);
    auto tW = CTensor::Fill({4}, 1.0f);
    auto tOut = OP::RmsNorm(tX, tW);
    Check(tOut.m_iNdim == 3, "ndim");
    for (int64_t i = 0; i < tOut.lNumel(); i++)
        CheckClose(tOut.fFlat(i), 1.0f, 1e-3f);
}
// >>>s_end(normalization)

// <<<s_start(unary)
// --- unary math
TEST(neg_basic) {
    auto t = CTensor::Zeros({4});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = -2.0f;
    t.fFlat(2) = 0.0f;
    t.fFlat(3) = 3.5f;
    auto tN = OP::Neg(t);
    CheckClose(tN.fFlat(0), -1.0f);
    CheckClose(tN.fFlat(1), 2.0f);
    CheckClose(tN.fFlat(2), 0.0f);
    CheckClose(tN.fFlat(3), -3.5f);
}

TEST(sqrt_basic) {
    auto t = CTensor::Zeros({4});
    t.fFlat(0) = 0.0f;
    t.fFlat(1) = 1.0f;
    t.fFlat(2) = 4.0f;
    t.fFlat(3) = 9.0f;
    auto tS = OP::Sqrt(t);
    CheckClose(tS.fFlat(0), 0.0f);
    CheckClose(tS.fFlat(1), 1.0f);
    CheckClose(tS.fFlat(2), 2.0f);
    CheckClose(tS.fFlat(3), 3.0f);
}

TEST(rsqrt_basic) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = 4.0f;
    t.fFlat(2) = 16.0f;
    auto tR = OP::Rsqrt(t);
    CheckClose(tR.fFlat(0), 1.0f);
    CheckClose(tR.fFlat(1), 0.5f);
    CheckClose(tR.fFlat(2), 0.25f);
}

TEST(abs_basic) {
    auto t = CTensor::Zeros({4});
    t.fFlat(0) = -3.0f;
    t.fFlat(1) = 0.0f;
    t.fFlat(2) = 5.0f;
    t.fFlat(3) = -0.5f;
    auto tA = OP::Abs(t);
    CheckClose(tA.fFlat(0), 3.0f);
    CheckClose(tA.fFlat(1), 0.0f);
    CheckClose(tA.fFlat(2), 5.0f);
    CheckClose(tA.fFlat(3), 0.5f);
}

TEST(exp_basic) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 0.0f;
    t.fFlat(1) = 1.0f;
    t.fFlat(2) = -1.0f;
    auto tE = OP::Exp(t);
    CheckClose(tE.fFlat(0), 1.0f);
    CheckClose(tE.fFlat(1), std::exp(1.0f));
    CheckClose(tE.fFlat(2), std::exp(-1.0f));
}

TEST(log_basic) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = std::exp(1.0f);
    t.fFlat(2) = std::exp(2.0f);
    auto tL = OP::Log(t);
    CheckClose(tL.fFlat(0), 0.0f);
    CheckClose(tL.fFlat(1), 1.0f, 1e-4f);
    CheckClose(tL.fFlat(2), 2.0f, 1e-4f);
}

TEST(exp_log_roundtrip) {
    auto t = CTensor::Zeros({4});
    t.fFlat(0) = 0.5f;
    t.fFlat(1) = 1.0f;
    t.fFlat(2) = 2.0f;
    t.fFlat(3) = 3.0f;
    auto tE = OP::Exp(t);
    auto tL = OP::Log(tE);
    for (int64_t i = 0; i < 4; i++)
        CheckClose(tL.fFlat(i), t.fFlat(i), 1e-4f);
}

TEST(clamp_basic) {
    auto t = CTensor::Zeros({5});
    t.fFlat(0) = -10.0f;
    t.fFlat(1) = -1.0f;
    t.fFlat(2) = 0.5f;
    t.fFlat(3) = 1.0f;
    t.fFlat(4) = 10.0f;
    auto tC = OP::Clamp(t, 0.0f, 1.0f);
    CheckClose(tC.fFlat(0), 0.0f);
    CheckClose(tC.fFlat(1), 0.0f);
    CheckClose(tC.fFlat(2), 0.5f);
    CheckClose(tC.fFlat(3), 1.0f);
    CheckClose(tC.fFlat(4), 1.0f);
}

TEST(clamp_no_effect) {
    auto t = CTensor::Fill({3}, 5.0f);
    auto tC = OP::Clamp(t, 0.0f, 10.0f);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tC.fFlat(i), 5.0f);
}

TEST(clamp_all_below) {
    auto t = CTensor::Fill({3}, -5.0f);
    auto tC = OP::Clamp(t, 0.0f, 1.0f);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tC.fFlat(i), 0.0f);
}

TEST(clamp_all_above) {
    auto t = CTensor::Fill({3}, 100.0f);
    auto tC = OP::Clamp(t, 0.0f, 1.0f);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tC.fFlat(i), 1.0f);
}

TEST(unary_preserves_shape) {
    auto t = CTensor::Rand({3, 4, 5});
    auto tN = OP::Neg(t);
    Check(tN.m_iNdim == 3, "ndim");
    Check(tN.m_lShape[0] == 3, "shape[0]");
    Check(tN.m_lShape[1] == 4, "shape[1]");
    Check(tN.m_lShape[2] == 5, "shape[2]");
}

TEST(unary_does_not_modify_input) {
    auto t = CTensor::Fill({3}, 5.0f);
    auto tN = OP::Neg(t);
    CheckClose(t.fFlat(0), 5.0f);
}
// >>>s_end(unary)

// <<<s_start(dot)
// --- dot product
TEST(dot_basic) {
    auto tA = CTensor::Zeros({3});
    auto tB = CTensor::Zeros({3});
    tA.fFlat(0) = 1.0f;
    tA.fFlat(1) = 2.0f;
    tA.fFlat(2) = 3.0f;
    tB.fFlat(0) = 4.0f;
    tB.fFlat(1) = 5.0f;
    tB.fFlat(2) = 6.0f;
    float fD = OP::fDot(tA, tB);
    CheckClose(fD, 32.0f);
}

TEST(dot_orthogonal) {
    auto tA = CTensor::Zeros({2});
    auto tB = CTensor::Zeros({2});
    tA.fFlat(0) = 1.0f;
    tA.fFlat(1) = 0.0f;
    tB.fFlat(0) = 0.0f;
    tB.fFlat(1) = 1.0f;
    float fD = OP::fDot(tA, tB);
    CheckClose(fD, 0.0f);
}

TEST(dot_self) {
    auto t = CTensor::Zeros({3});
    t.fFlat(0) = 1.0f;
    t.fFlat(1) = 2.0f;
    t.fFlat(2) = 3.0f;
    float fD = OP::fDot(t, t);
    CheckClose(fD, 14.0f);
}

TEST(dot_zeros) {
    auto tA = CTensor::Zeros({5});
    auto tB = CTensor::Fill({5}, 99.0f);
    float fD = OP::fDot(tA, tB);
    CheckClose(fD, 0.0f);
}
// >>>s_end(dot)

// <<<s_start(index)
// --- indexing/slicing
TEST(slice_row_2d) {
    auto t = CTensor::Zeros({3, 4});
    for (int64_t r = 0; r < 3; r++)
        for (int64_t c = 0; c < 4; c++)
            t.fAt({r, c}) = (float)(r * 10 + c);

    auto tR = OP::SliceRow(t, 1);
    Check(tR.m_iNdim == 1, "ndim");
    Check(tR.m_lShape[0] == 4, "shape[0]");
    Check(tR.m_bOwnsData == false, "should be view");
    CheckClose(tR.fFlat(0), 10.0f);
    CheckClose(tR.fFlat(1), 11.0f);
    CheckClose(tR.fFlat(2), 12.0f);
    CheckClose(tR.fFlat(3), 13.0f);
}

TEST(slice_row_3d) {
    auto t = CTensor::Zeros({2, 3, 4});
    for (int64_t i = 0; i < t.lNumel(); i++)
        t.fFlat(i) = (float)i;

    auto tR = OP::SliceRow(t, 1); // shape (3, 4) starts at flat[12]
    Check(tR.m_iNdim == 2, "ndim");
    Check(tR.m_lShape[0] == 3, "shape[0]");
    Check(tR.m_lShape[1] == 4, "shape[1]");
    CheckClose(tR.fAt({0, 0}), 12.0f);
}

TEST(slice_row_first) {
    auto t = CTensor::Zeros({3, 2});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tR = OP::SliceRow(t, 0);
    CheckClose(tR.fFlat(0), 0.0f);
    CheckClose(tR.fFlat(1), 1.0f);
}

TEST(slice_row_last) {
    auto t = CTensor::Zeros({3, 2});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tR = OP::SliceRow(t, 2);
    CheckClose(tR.fFlat(0), 4.0f);
    CheckClose(tR.fFlat(1), 5.0f);
}

TEST(slice_row_is_view) {
    auto t = CTensor::Fill({3, 2}, 5.0f);
    auto tR = OP::SliceRow(t, 1);
    tR.fFlat(0) = 999.0f;
    CheckClose(t.fAt({1, 0}), 999.0f);
}

TEST(slice_range_basic) {
    auto t = CTensor::Zeros({5, 3});
    for (int64_t i = 0; i < t.lNumel(); i++)
        t.fFlat(i) = (float)i;

    auto tR = OP::SliceRange(t, 1, 4); // rows 1 2 3
    Check(tR.m_iNdim == 2, "ndim");
    Check(tR.m_lShape[0] == 3, "shape[0]");
    Check(tR.m_lShape[1] == 3, "shape[1]");
    Check(tR.m_bOwnsData == false, "should be view");
    CheckClose(tR.fAt({0, 0}), 3.0f);  // row 1
    CheckClose(tR.fAt({2, 2}), 11.0f); // row 3 col 2
}

TEST(slice_range_full) {
    auto t = CTensor::Fill({3, 2}, 7.0f);
    auto tR = OP::SliceRange(t, 0, 3);
    Check(tR.m_lShape[0] == 3, "shape[0]");
    for (int64_t i = 0; i < tR.lNumel(); i++)
        CheckClose(tR.fFlat(i), 7.0f);
}

TEST(slice_range_single) {
    auto t = CTensor::Zeros({4, 2});
    for (int64_t i = 0; i < 8; i++)
        t.fFlat(i) = (float)i;
    auto tR = OP::SliceRange(t, 2, 3); // just row 2
    Check(tR.m_lShape[0] == 1, "shape[0]");
    CheckClose(tR.fAt({0, 0}), 4.0f);
    CheckClose(tR.fAt({0, 1}), 5.0f);
}

TEST(slice_range_is_view) {
    auto t = CTensor::Fill({4, 2}, 1.0f);
    auto tR = OP::SliceRange(t, 1, 3);
    tR.fAt({0, 0}) = 999.0f;
    CheckClose(t.fAt({1, 0}), 999.0f);
}

TEST(gather_basic) {
    auto tA = CTensor::Zeros({4, 3});
    for (int64_t r = 0; r < 4; r++)
        for (int64_t c = 0; c < 3; c++)
            tA.fAt({r, c}) = (float)(r * 10 + c);

    CTensor tIdx({2}, EType::I32);
    static_cast<int32_t *>(tIdx.m_pData)[0] = 1;
    static_cast<int32_t *>(tIdx.m_pData)[1] = 3;

    auto tOut = OP::Gather(tA, tIdx);
    Check(tOut.m_iNdim == 2, "ndim");
    Check(tOut.m_lShape[0] == 2, "shape[0]");
    Check(tOut.m_lShape[1] == 3, "shape[1]");
    // row 1 [10, 11, 12]
    CheckClose(tOut.fAt({0, 0}), 10.0f);
    CheckClose(tOut.fAt({0, 1}), 11.0f);
    CheckClose(tOut.fAt({0, 2}), 12.0f);
    // row 3 [30, 31, 32]
    CheckClose(tOut.fAt({1, 0}), 30.0f);
    CheckClose(tOut.fAt({1, 1}), 31.0f);
    CheckClose(tOut.fAt({1, 2}), 32.0f);
}

TEST(gather_single) {
    auto tA = CTensor::Zeros({3, 2});
    for (int64_t i = 0; i < 6; i++)
        tA.fFlat(i) = (float)i;

    CTensor tIdx({1}, EType::I32);
    static_cast<int32_t *>(tIdx.m_pData)[0] = 2;

    auto tOut = OP::Gather(tA, tIdx);
    Check(tOut.m_lShape[0] == 1, "shape[0]");
    CheckClose(tOut.fAt({0, 0}), 4.0f);
    CheckClose(tOut.fAt({0, 1}), 5.0f);
}

TEST(gather_repeated_index) {
    auto tA = CTensor::Zeros({3, 2});
    for (int64_t i = 0; i < 6; i++)
        tA.fFlat(i) = (float)i;

    CTensor tIdx({3}, EType::I32);
    int32_t *piIdx = static_cast<int32_t *>(tIdx.m_pData);
    piIdx[0] = 0;
    piIdx[1] = 0;
    piIdx[2] = 0;

    auto tOut = OP::Gather(tA, tIdx);
    Check(tOut.m_lShape[0] == 3, "shape[0]");
    for (int64_t r = 0; r < 3; r++) {
        CheckClose(tOut.fAt({r, 0}), 0.0f);
        CheckClose(tOut.fAt({r, 1}), 1.0f);
    }
}

TEST(gather_is_copy) {
    auto tA = CTensor::Fill({3, 2}, 5.0f);
    CTensor tIdx({1}, EType::I32);
    static_cast<int32_t *>(tIdx.m_pData)[0] = 1;
    auto tOut = OP::Gather(tA, tIdx);
    tOut.fAt({0, 0}) = 999.0f;
    CheckClose(tA.fAt({1, 0}), 5.0f);
}
// >>>s_end(index)

// <<<s_start(construction)
// --- construction/manipulation
TEST(concat_dim0) {
    auto tA = CTensor::Fill({2, 3}, 1.0f);
    auto tB = CTensor::Fill({3, 3}, 2.0f);
    std::vector<const CTensor *> vtPtrs = {&tA, &tB};
    auto tC = OP::Concat(vtPtrs, 0);
    Check(tC.m_iNdim == 2, "ndim");
    Check(tC.m_lShape[0] == 5, "shape[0]");
    Check(tC.m_lShape[1] == 3, "shape[1]");
    for (int64_t r = 0; r < 2; r++)
        for (int64_t c = 0; c < 3; c++)
            CheckClose(tC.fAt({r, c}), 1.0f);
    for (int64_t r = 2; r < 5; r++)
        for (int64_t c = 0; c < 3; c++)
            CheckClose(tC.fAt({r, c}), 2.0f);
}

TEST(concat_dim1) {
    auto tA = CTensor::Fill({2, 3}, 1.0f);
    auto tB = CTensor::Fill({2, 4}, 2.0f);
    std::vector<const CTensor *> vtPtrs = {&tA, &tB};
    auto tC = OP::Concat(vtPtrs, 1);
    Check(tC.m_lShape[0] == 2, "shape[0]");
    Check(tC.m_lShape[1] == 7, "shape[1]");
    for (int64_t r = 0; r < 2; r++) {
        for (int64_t c = 0; c < 3; c++)
            CheckClose(tC.fAt({r, c}), 1.0f);
        for (int64_t c = 3; c < 7; c++)
            CheckClose(tC.fAt({r, c}), 2.0f);
    }
}

TEST(concat_three_tensors) {
    auto tA = CTensor::Fill({1, 2}, 1.0f);
    auto tB = CTensor::Fill({1, 2}, 2.0f);
    auto tC = CTensor::Fill({1, 2}, 3.0f);
    std::vector<const CTensor *> vtPtrs = {&tA, &tB, &tC};
    auto tOut = OP::Concat(vtPtrs, 0);
    Check(tOut.m_lShape[0] == 3, "shape[0]");
    CheckClose(tOut.fAt({0, 0}), 1.0f);
    CheckClose(tOut.fAt({1, 0}), 2.0f);
    CheckClose(tOut.fAt({2, 0}), 3.0f);
}

TEST(concat_negative_dim) {
    auto tA = CTensor::Fill({2, 3}, 1.0f);
    auto tB = CTensor::Fill({2, 4}, 2.0f);
    std::vector<const CTensor *> vtPtrs = {&tA, &tB};
    auto tC = OP::Concat(vtPtrs, -1);
    Check(tC.m_lShape[1] == 7, "shape[1]");
}

TEST(repeat_basic) {
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tR = OP::Repeat(t, 0, 3);
    Check(tR.m_lShape[0] == 6, "shape[0]");
    Check(tR.m_lShape[1] == 3, "shape[1]");
    // rows 0, 1 == rows 2, 3 == rows 4, 5
    for (int64_t rep = 0; rep < 3; rep++) {
        for (int64_t r = 0; r < 2; r++) {
            for (int64_t c = 0; c < 3; c++) {
                CheckClose(tR.fAt({rep * 2 + r, c}), t.fAt({r, c}));
            }
        }
    }
}

TEST(repeat_dim1) {
    auto t = CTensor::Zeros({2, 3});
    for (int64_t i = 0; i < 6; i++)
        t.fFlat(i) = (float)i;
    auto tR = OP::Repeat(t, 1, 2);
    Check(tR.m_lShape[0] == 2, "shape[0]");
    Check(tR.m_lShape[1] == 6, "shape[1]");
}

TEST(repeat_once) {
    auto t = CTensor::Fill({2, 3}, 5.0f);
    auto tR = OP::Repeat(t, 0, 1);
    Check(tR.m_lShape[0] == 2, "shape[0]");
    for (int64_t i = 0; i < tR.lNumel(); i++)
        CheckClose(tR.fFlat(i), 5.0f);
}

TEST(arange_basic) {
    auto t = OP::Arange(0.0f, 5.0f);
    Check(t.m_iNdim == 1, "ndim");
    Check(t.m_lShape[0] == 5, "shape[0]");
    for (int64_t i = 0; i < 5; i++)
        CheckClose(t.fFlat(i), (float)i);
}

TEST(arange_with_step) {
    auto t = OP::Arange(0.0f, 10.0f, 2.0f);
    Check(t.m_lShape[0] == 5, "shape[0]");
    CheckClose(t.fFlat(0), 0.0f);
    CheckClose(t.fFlat(1), 2.0f);
    CheckClose(t.fFlat(2), 4.0f);
    CheckClose(t.fFlat(3), 6.0f);
    CheckClose(t.fFlat(4), 8.0f);
}

TEST(arange_fractional_step) {
    auto t = OP::Arange(0.0f, 1.0f, 0.25f);
    Check(t.m_lShape[0] == 4, "shape[0]");
    CheckClose(t.fFlat(0), 0.0f);
    CheckClose(t.fFlat(1), 0.25f);
    CheckClose(t.fFlat(2), 0.5f);
    CheckClose(t.fFlat(3), 0.75f);
}

TEST(arange_nonzero_start) {
    auto t = OP::Arange(5.0f, 8.0f);
    Check(t.m_lShape[0] == 3, "shape[0]");
    CheckClose(t.fFlat(0), 5.0f);
    CheckClose(t.fFlat(1), 6.0f);
    CheckClose(t.fFlat(2), 7.0f);
}

TEST(trimask_basic) {
    auto t = OP::TriMask(3);
    Check(t.m_iNdim == 2, "ndim");
    Check(t.m_lShape[0] == 3, "shape[0]");
    Check(t.m_lShape[1] == 3, "shape[1]");
    // diagonal and below = 0
    CheckClose(t.fAt({0, 0}), 0.0f);
    CheckClose(t.fAt({1, 0}), 0.0f);
    CheckClose(t.fAt({1, 1}), 0.0f);
    CheckClose(t.fAt({2, 0}), 0.0f);
    CheckClose(t.fAt({2, 1}), 0.0f);
    CheckClose(t.fAt({2, 2}), 0.0f);
    // above diagonal = fill
    Check(t.fAt({0, 1}) < -1e8f, "above diag should be large negative");
    Check(t.fAt({0, 2}) < -1e8f, "above diag should be large negative");
    Check(t.fAt({1, 2}) < -1e8f, "above diag should be large negative");
}

TEST(trimask_custom_fill) {
    auto t = OP::TriMask(2, -42.0f);
    CheckClose(t.fAt({0, 0}), 0.0f);
    CheckClose(t.fAt({0, 1}), -42.0f);
    CheckClose(t.fAt({1, 0}), 0.0f);
    CheckClose(t.fAt({1, 1}), 0.0f);
}

TEST(trimask_1x1) {
    auto t = OP::TriMask(1);
    Check(t.lNumel() == 1, "numel");
    CheckClose(t.fFlat(0), 0.0f);
}

TEST(where_basic) {
    auto tCond = CTensor::Zeros({4});
    tCond.fFlat(0) = 1.0f;  // true
    tCond.fFlat(1) = 0.0f;  // false
    tCond.fFlat(2) = -1.0f; // false
    tCond.fFlat(3) = 0.5f;  // true
    auto tA = CTensor::Fill({4}, 10.0f);
    auto tB = CTensor::Fill({4}, 20.0f);
    auto tOut = OP::Where(tCond, tA, tB);
    CheckClose(tOut.fFlat(0), 10.0f);
    CheckClose(tOut.fFlat(1), 20.0f);
    CheckClose(tOut.fFlat(2), 20.0f);
    CheckClose(tOut.fFlat(3), 10.0f);
}

TEST(where_all_true) {
    auto tCond = CTensor::Fill({3}, 1.0f);
    auto tA = CTensor::Fill({3}, 5.0f);
    auto tB = CTensor::Fill({3}, 9.0f);
    auto tOut = OP::Where(tCond, tA, tB);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 5.0f);
}

TEST(where_all_false) {
    auto tCond = CTensor::Fill({3}, -1.0f);
    auto tA = CTensor::Fill({3}, 5.0f);
    auto tB = CTensor::Fill({3}, 9.0f);
    auto tOut = OP::Where(tCond, tA, tB);
    for (int64_t i = 0; i < 3; i++)
        CheckClose(tOut.fFlat(i), 9.0f);
}

TEST(where_2d) {
    auto tCond = CTensor::Zeros({2, 2});
    tCond.fAt({0, 0}) = 1.0f;
    tCond.fAt({0, 1}) = -1.0f;
    tCond.fAt({1, 0}) = -1.0f;
    tCond.fAt({1, 1}) = 1.0f;
    auto tA = CTensor::Fill({2, 2}, 100.0f);
    auto tB = CTensor::Fill({2, 2}, 200.0f);
    auto tOut = OP::Where(tCond, tA, tB);
    CheckClose(tOut.fAt({0, 0}), 100.0f);
    CheckClose(tOut.fAt({0, 1}), 200.0f);
    CheckClose(tOut.fAt({1, 0}), 200.0f);
    CheckClose(tOut.fAt({1, 1}), 100.0f);
}
// >>>s_end(construction)

// <<<s_start(ipo)
// --- in place ops
TEST(copy_into_basic) {
    auto tDst = CTensor::Zeros({4, 3});
    auto tSrc = CTensor::Fill({2, 3}, 7.0f);
    OP::CopyInto(tDst, tSrc, 1);
    // rows 0 should still be 0
    for (int64_t c = 0; c < 3; c++)
        CheckClose(tDst.fAt({0, c}), 0.0f);
    // rows 1, 2 should be 7
    for (int64_t r = 1; r < 3; r++)
        for (int64_t c = 0; c < 3; c++)
            CheckClose(tDst.fAt({r, c}), 7.0f);
    // row 3 should be 0
    for (int64_t c = 0; c < 3; c++)
        CheckClose(tDst.fAt({3, c}), 0.0f);
}

TEST(copy_into_start) {
    auto tDst = CTensor::Zeros({3, 2});
    auto tSrc = CTensor::Fill({1, 2}, 5.0f);
    OP::CopyInto(tDst, tSrc, 0);
    CheckClose(tDst.fAt({0, 0}), 5.0f);
    CheckClose(tDst.fAt({0, 1}), 5.0f);
    CheckClose(tDst.fAt({1, 0}), 0.0f);
}

TEST(copy_into_end) {
    auto tDst = CTensor::Zeros({3, 2});
    auto tSrc = CTensor::Fill({1, 2}, 9.0f);
    OP::CopyInto(tDst, tSrc, 2);
    CheckClose(tDst.fAt({0, 0}), 0.0f);
    CheckClose(tDst.fAt({2, 0}), 9.0f);
    CheckClose(tDst.fAt({2, 1}), 9.0f);
}

TEST(copy_into_overwrites) {
    auto tDst = CTensor::Fill({3, 2}, 1.0f);
    auto tSrc = CTensor::Fill({2, 2}, 99.0f);
    OP::CopyInto(tDst, tSrc, 0);
    CheckClose(tDst.fAt({0, 0}), 99.0f);
    CheckClose(tDst.fAt({1, 0}), 99.0f);
    CheckClose(tDst.fAt({2, 0}), 1.0f);
}

TEST(fill_inplace_basic) {
    auto t = CTensor::Rand({3, 4});
    OP::FillInplace(t, 42.0f);
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 42.0f);
}

TEST(fill_inplace_zero) {
    auto t = CTensor::Fill({5}, 99.0f);
    OP::FillInplace(t, 0.0f);
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), 0.0f);
}

TEST(fill_inplace_negative) {
    auto t = CTensor::Zeros({3});
    OP::FillInplace(t, -7.5f);
    for (int64_t i = 0; i < t.lNumel(); i++)
        CheckClose(t.fFlat(i), -7.5f);
}
// >>>s_end(ipo)

// <<<s_start(benches)
// --- ops benchmarks
TEST(bench_add_elementwise) {
    auto tA = CTensor::Rand({512, 512});
    auto tB = CTensor::Rand({512, 512});
    Bench("add 512x512", 1000, [&]() {
        auto tC = OP::Add(tA, tB);
        (void)tC;
    });
}

TEST(bench_matmul_small) {
    auto tA = CTensor::Rand({64, 64});
    auto tB = CTensor::Rand({64, 64});
    Bench("matmul 64x64", 500, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_matmul_medium) {
    auto tA = CTensor::Rand({256, 256});
    auto tB = CTensor::Rand({256, 256});
    Bench("matmul 256x256", 50, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_matmul_large) {
    auto tA = CTensor::Rand({512, 512});
    auto tB = CTensor::Rand({512, 512});
    Bench("matmul 512x512", 50, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_matmul_even_larger) {
    auto tA = CTensor::Rand({1024, 1024});
    auto tB = CTensor::Rand({1024, 1024});
    Bench("matmul 1024x1024", 50, [&]() {
        auto tC = OP::Matmul(tA, tB);
        (void)tC;
    });
}

TEST(bench_softmax) {
    auto t = CTensor::Rand({64, 512});
    Bench("softmax 64x512", 500, [&]() {
        auto tS = OP::Softmax(t, -1);
        (void)tS;
    });
}

TEST(bench_rmsnorm) {
    auto tX = CTensor::Rand({64, 512});
    auto tW = CTensor::Rand({512});
    Bench("rmsnorm 64x512", 1000, [&]() {
        auto tO = OP::RmsNorm(tX, tW);
        (void)tO;
    });
}

TEST(bench_broadcast_add) {
    auto tA = CTensor::Rand({256, 256});
    auto tB = CTensor::Rand({1, 256});
    Bench("broadcast add 256x256 + 1x256", 500, [&]() {
        auto tC = OP::AddBroadcast(tA, tB);
        (void)tC;
    });
}

TEST(bench_matvec) {
    auto tMat = CTensor::Rand({512, 512});
    auto tVec = CTensor::Rand({512});
    Bench("matvec 512x512", 1000, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
}

TEST(bench_matvec_large) {
    auto tMat = CTensor::Rand({1024, 1024});
    auto tVec = CTensor::Rand({1024});
    Bench("matvec 1024x1024", 1000, [&]() {
        auto tO = OP::Matvec(tMat, tVec);
        (void)tO;
    });
}
// >>>s_end(benches)
