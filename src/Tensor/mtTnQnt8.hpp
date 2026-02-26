// Created by Unium on 24.02.26

#pragma once

#include "mtTnTnsr.hpp"
#include <cstdint>
#include <vector>

namespace MT {
namespace Q8 {
struct SQMatrix {
    int64_t lRows = 0;
    int64_t lCols = 0;
    std::vector<int8_t> viData; // [rows * cols] row-major
    std::vector<float> vfScale; // [rows] per-row scale

    /*---------------------------------------------------------
     * FN: lNumel
     * DESC: returns total number of elements
     * PARMS: none
     * AUTH: unium (24.02.26)
     *-------------------------------------------------------*/
    auto lNumel() const -> int64_t { return lRows * lCols; }
};

/*---------------------------------------------------------
 * FN: Quantize
 * DESC: quantizes a f32 tensor 2d to int8 with pr absmax
 *       scaling, each row: scale = max(|row|) / 127,
 *       q[i] = round(x[i] / scale)
 * PARMS: tMat (f32 matrix [rows, cols])
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto Quantize(const CTensor &tMat) -> SQMatrix;

/*---------------------------------------------------------
 * FN: Matvec
 * DESC: matvec
 * PARMS: qMat (quantized matrix), tVec (f32 vector)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto Matvec(const SQMatrix &qMat, const CTensor &tVec) -> CTensor;

/*---------------------------------------------------------
 * FN: MatvecSimd
 * DESC: simd matvec
 * PARMS: qMat (quantized matrix), tVec (f32 vector)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto MatvecSimd(const SQMatrix &qMat, const CTensor &tVec) -> CTensor;
} // namespace Q8
} // namespace MT
