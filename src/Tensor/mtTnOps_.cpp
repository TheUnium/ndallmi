// Created by Unium on 12.02.26

#include "mtTnOps_.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

namespace MT {
namespace OP {
// <<<s_start(helpers)
// --- helpers
/*---------------------------------------------------------
 * FN: tLike
 * DESC: creates an uninitialized tensor with same
 *       shape/dtype
 * PARMS: tA (tensor to mimic)
 * AUTH: Rajendra (14.02.26)
 *-------------------------------------------------------*/
static auto tLike(const CTensor &tA) -> CTensor {
    return CTensor(std::vector<int64_t>(tA.m_lShape, tA.m_lShape + tA.m_iNdim), tA.m_eType);
}

using BinaryFn = float (*)(float, float);

/*---------------------------------------------------------
 * FN: tElementwise
 * DESC: applies a binary function element wise to two
 *       tensors
 * PARMS: tA (first), tB (second), pfnOp (operation)
 * AUTH: Rajendra (14.02.26)
 *-------------------------------------------------------*/
static auto tElementwise(const CTensor &tA, const CTensor &tB, BinaryFn pfnOp) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == tB.m_iNdim && "[op:tew] shape mismatch for elementwise op");
    for (int i = 0; i < tA.m_iNdim; i++) {
        assert(tA.m_lShape[i] == tB.m_lShape[i] && "[op:tew] shape mismatch");
    }
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    CTensor tOut = tLike(tA);
    const float *__restrict__ pfA = tA.pfData();
    const float *__restrict__ pfB = tB.pfData();
    float *__restrict__ pfO = tOut.pfData();
    const int64_t lN = tA.lNumel();

    for (int64_t i = 0; i < lN; i++) {
        pfO[i] = pfnOp(pfA[i], pfB[i]);
    }
    return tOut;
}

using UnaryFn = float (*)(float);

/*---------------------------------------------------------
 * FN: tUnarywise
 * DESC: applies a unary function element wise to a tensor
 * PARMS: tA (input), pfnOp (operation)
 * AUTH: Rajendra (14.02.26)
 *-------------------------------------------------------*/
static auto tUnarywise(const CTensor &tA, UnaryFn pfnOp) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *__restrict__ pfA = tA.pfData();
    float *__restrict__ pfO = tOut.pfData();
    const int64_t lN = tA.lNumel();
    for (int64_t i = 0; i < lN; i++) {
        pfO[i] = pfnOp(pfA[i]);
    }
    return tOut;
}

// <<<s_start(helpers_broadcasting)
// --- broadcasting helpers
/*---------------------------------------------------------
 * FN: BroadcastShapes
 * DESC: computes output shape for basting. shapes are
 *       right sided and dims must be equal or 1.
 * PARMS: tA (first), tB (second),
 *        viOutShape (output shape), iOutNdim (output ndim),
 *        vlStrideA/B (broadcast strides for A and B)
 * AUTH: Rajendra (14.02.26)
 *-------------------------------------------------------*/
static void BroadcastShapes(const CTensor &tA, const CTensor &tB, int64_t *viOutShape, int &iOutNdim,
                            int64_t *vlStrideA, int64_t *vlStrideB) {
    iOutNdim = std::max(tA.m_iNdim, tB.m_iNdim);
    assert(iOutNdim <= mmDims);

    int64_t lShapeA[mmDims], lShapeB[mmDims];
    int64_t lOrigStrideA[mmDims], lOrigStrideB[mmDims];

    for (int i = 0; i < iOutNdim; i++) {
        int iIdxA = i - (iOutNdim - tA.m_iNdim);
        int iIdxB = i - (iOutNdim - tB.m_iNdim);

        lShapeA[i] = (iIdxA >= 0) ? tA.m_lShape[iIdxA] : 1;
        lShapeB[i] = (iIdxB >= 0) ? tB.m_lShape[iIdxB] : 1;
        lOrigStrideA[i] = (iIdxA >= 0) ? tA.m_lStride[iIdxA] : 0;
        lOrigStrideB[i] = (iIdxB >= 0) ? tB.m_lStride[iIdxB] : 0;

        assert((lShapeA[i] == lShapeB[i] || lShapeA[i] == 1 || lShapeB[i] == 1) &&
               "[mt:broadcast:hs] incompatible shapes");

        viOutShape[i] = std::max(lShapeA[i], lShapeB[i]);
        vlStrideA[i] = (lShapeA[i] == 1) ? 0 : lOrigStrideA[i];
        vlStrideB[i] = (lShapeB[i] == 1) ? 0 : lOrigStrideB[i];
    }
}

/*---------------------------------------------------------
 * FN: tBroadcastOp
 * DESC: applies a binary op with broadcasting
 * PARMS: tA (first), tB (second), pfnOp (operation)
 * AUTH: Rajendra (14.02.26)
 *-------------------------------------------------------*/

static auto tBroadcastOp(const CTensor &tA, const CTensor &tB, BinaryFn pfnOp) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    if (tA.m_iNdim == tB.m_iNdim && tA.bIsContiguous() && tB.bIsContiguous()) {
        bool bSameShape = true;
        for (int i = 0; i < tA.m_iNdim; i++) {
            if (tA.m_lShape[i] != tB.m_lShape[i]) {
                bSameShape = false;
                break;
            }
        }
        if (bSameShape) {
            return tElementwise(tA, tB, pfnOp);
        }
    }

    int64_t lOutShape[mmDims];
    int64_t lStrideA[mmDims], lStrideB[mmDims];
    int iOutNdim;

    BroadcastShapes(tA, tB, lOutShape, iOutNdim, lStrideA, lStrideB);

    CTensor tOut(std::vector<int64_t>(lOutShape, lOutShape + iOutNdim));

    const float *__restrict__ pfA = tA.pfData();
    const float *__restrict__ pfB = tB.pfData();
    float *__restrict__ pfO = tOut.pfData();

    const int64_t lN = tOut.lNumel();

    int64_t lSuffix[mmDims];
    lSuffix[iOutNdim - 1] = 1;
    for (int d = iOutNdim - 2; d >= 0; d--) {
        lSuffix[d] = lSuffix[d + 1] * lOutShape[d + 1];
    }

    const int iLastDim = iOutNdim - 1;
    const int64_t lInnerSize = lOutShape[iLastDim];
    const bool bInnerContiguous = (lStrideA[iLastDim] != 0) && (lStrideB[iLastDim] != 0);
    const bool bInnerBcastA = (lStrideA[iLastDim] == 0);
    const bool bInnerBcastB = (lStrideB[iLastDim] == 0);

    if (iOutNdim == 1) {
        const int64_t lSA = lStrideA[0];
        const int64_t lSB = lStrideB[0];
        for (int64_t i = 0; i < lN; i++) {
            pfO[i] = pfnOp(pfA[i * lSA], pfB[i * lSB]);
        }
        return tOut;
    }

    const int64_t lOuterN = lN / lInnerSize;

    for (int64_t outer = 0; outer < lOuterN; outer++) {
        int64_t lOffA = 0, lOffB = 0;
        int64_t lTmp = outer;
        for (int d = 0; d < iLastDim; d++) {
            int64_t lSuffixD = lSuffix[d] / lInnerSize;
            int64_t lIdx = lTmp / lSuffixD;
            lTmp %= lSuffixD;
            lOffA += lIdx * lStrideA[d];
            lOffB += lIdx * lStrideB[d];
        }

        float *__restrict__ pfORow = pfO + outer * lInnerSize;

        if (bInnerContiguous) {
            const float *__restrict__ pfAR = pfA + lOffA;
            const float *__restrict__ pfBR = pfB + lOffB;
            const int64_t lSA = lStrideA[iLastDim];
            const int64_t lSB = lStrideB[iLastDim];
            if (lSA == 1 && lSB == 1) {
                for (int64_t i = 0; i < lInnerSize; i++) {
                    pfORow[i] = pfnOp(pfAR[i], pfBR[i]);
                }
            } else {
                for (int64_t i = 0; i < lInnerSize; i++) {
                    pfORow[i] = pfnOp(pfAR[i * lSA], pfBR[i * lSB]);
                }
            }
        } else if (bInnerBcastA) {
            const float fAVal = pfA[lOffA];
            const float *__restrict__ pfBR = pfB + lOffB;
            const int64_t lSB = lStrideB[iLastDim];
            if (lSB == 1) {
                for (int64_t i = 0; i < lInnerSize; i++) {
                    pfORow[i] = pfnOp(fAVal, pfBR[i]);
                }
            } else {
                for (int64_t i = 0; i < lInnerSize; i++) {
                    pfORow[i] = pfnOp(fAVal, pfBR[i * lSB]);
                }
            }
        } else if (bInnerBcastB) {
            const float *__restrict__ pfAR = pfA + lOffA;
            const float fBVal = pfB[lOffB];
            const int64_t lSA = lStrideA[iLastDim];
            if (lSA == 1) {
                for (int64_t i = 0; i < lInnerSize; i++) {
                    pfORow[i] = pfnOp(pfAR[i], fBVal);
                }
            } else {
                for (int64_t i = 0; i < lInnerSize; i++) {
                    pfORow[i] = pfnOp(pfAR[i * lSA], fBVal);
                }
            }
        } else {
            const float fAVal = pfA[lOffA];
            const float fBVal = pfB[lOffB];
            const float fResult = pfnOp(fAVal, fBVal);
            for (int64_t i = 0; i < lInnerSize; i++) {
                pfORow[i] = fResult;
            }
        }
    }

    return tOut;
}
// >>>s_end(helpers_broadcasting)
// >>>s_end(helpers)

// <<<s_start(element_wb)
// --- element wise binary
/*---------------------------------------------------------
 * FN: Add
 * DESC: element wise add of two same shape tensoir
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Add(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tElementwise(tA, tB, [](float fX, float fY) { return fX + fY; });
}

/*---------------------------------------------------------
 * FN: Sub
 * DESC: element wise subtraction of two same shape tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Sub(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tElementwise(tA, tB, [](float fX, float fY) { return fX - fY; });
}

/*---------------------------------------------------------
 * FN: Mul
 * DESC: element wise multiplication of two same shape tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Mul(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tElementwise(tA, tB, [](float fX, float fY) { return fX * fY; });
}

/*---------------------------------------------------------
 * FN: Div
 * DESC: element wise division of two same shape tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (12.02.26 R: 21.02.26)
 *-------------------------------------------------------*/
auto Div(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tElementwise(tA, tB, [](float fX, float fY) { return fX / fY; });
}
// >>>s_end(element_wb)

// <<<s_start(broadcast)
// --- binary broadcast stuff
/*---------------------------------------------------------
 * FN: AddBroadcast
 * DESC: element wise add with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto AddBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tBroadcastOp(tA, tB, [](float fX, float fY) { return fX + fY; });
}

/*---------------------------------------------------------
 * FN: SubBroadcast
 * DESC: element wise sub with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto SubBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tBroadcastOp(tA, tB, [](float fX, float fY) { return fX - fY; });
}

/*---------------------------------------------------------
 * FN: MulBroadcast
 * DESC: element wise mul with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto MulBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tBroadcastOp(tA, tB, [](float fX, float fY) { return fX * fY; });
}

/*---------------------------------------------------------
 * FN: DivBroadcast
 * DESC: element wise div with bcasting
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (13.02.26 R: 19.02.26)
 *-------------------------------------------------------*/
auto DivBroadcast(const CTensor &tA, const CTensor &tB) -> CTensor {
    return tBroadcastOp(tA, tB, [](float fX, float fY) { return fX / fY; });
}
// >>>s_end(broadcast)

// <<<s_start(scalar)
// --- scalar opers
/*---------------------------------------------------------
 * FN: AddScalar
 * DESC: adds a scalar to every element
 * PARMS: tA (tensor), fS (scalar)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto AddScalar(const CTensor &tA, float fS) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *__restrict__ pfA = tA.pfData();
    float *__restrict__ pfO = tOut.pfData();
    const int64_t lN = tA.lNumel();
    for (int64_t i = 0; i < lN; i++)
        pfO[i] = pfA[i] + fS;
    return tOut;
}

/*---------------------------------------------------------
 * FN: MulScalar
 * DESC: multiplies every element by a scalar
 * PARMS: tA (tensor), fS (scalar)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto MulScalar(const CTensor &tA, float fS) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *__restrict__ pfA = tA.pfData();
    float *__restrict__ pfO = tOut.pfData();
    const int64_t lN = tA.lNumel();
    for (int64_t i = 0; i < lN; i++)
        pfO[i] = pfA[i] * fS;
    return tOut;
}
// >>>s_end(scalar)

// <<<s_start(matrix)
// --- matrix mult
/*---------------------------------------------------------
 * FN: Matmul
 * DESC: 2d matrix multiply
 *       [i.e., (m, k) x (k, n) --> (m, n)]
 * PARMS: tA (left matrix), tB (right matrix)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Matmul(const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == 2 && tB.m_iNdim == 2);
    assert(tA.m_lShape[1] == tB.m_lShape[0] && "[op:matmul] matmul dimension mismatch");
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    const int64_t lM = tA.m_lShape[0];
    const int64_t lK = tA.m_lShape[1];
    const int64_t lN = tB.m_lShape[1];

    CTensor tOut = CTensor::Zeros({lM, lN});
    const float *__restrict__ pfA = tA.pfData();
    const float *__restrict__ pfB = tB.pfData();
    float *__restrict__ pfO = tOut.pfData();

    for (int64_t i = 0; i < lM; i++) {
        float *__restrict__ pfORow = pfO + i * lN;
        for (int64_t k = 0; k < lK; k++) {
            const float fAik = pfA[i * lK + k];
            const float *__restrict__ pfBRow = pfB + k * lN;
            for (int64_t j = 0; j < lN; j++) {
                pfORow[j] += fAik * pfBRow[j];
            }
        }
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Bmm
 * DESC: batched matrix multiply
 *       [i.e., (b, m, k) x (b, k, n) --> (b, m, n)]
 * PARMS: tA (left batch), tB (right batch)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Bmm(const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == 3 && tB.m_iNdim == 3);
    assert(tA.m_lShape[0] == tB.m_lShape[0] && "[op:bmm] batch size mismatch");
    assert(tA.m_lShape[2] == tB.m_lShape[1] && "[op:bmm] bmm inner dimension mismatch");
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    const int64_t lB = tA.m_lShape[0];
    const int64_t lM = tA.m_lShape[1];
    const int64_t lK = tA.m_lShape[2];
    const int64_t lN = tB.m_lShape[2];

    CTensor tOut = CTensor::Zeros({lB, lM, lN});
    const float *__restrict__ pfA = tA.pfData();
    const float *__restrict__ pfB = tB.pfData();
    float *__restrict__ pfO = tOut.pfData();

    for (int64_t b = 0; b < lB; b++) {
        const float *__restrict__ pfBa = pfA + b * lM * lK;
        const float *__restrict__ pfBb = pfB + b * lK * lN;
        float *__restrict__ pfBo = pfO + b * lM * lN;

        for (int64_t i = 0; i < lM; i++) {
            float *__restrict__ pfORow = pfBo + i * lN;
            for (int64_t k = 0; k < lK; k++) {
                const float fAik = pfBa[i * lK + k];
                const float *__restrict__ pfBRow = pfBb + k * lN;
                for (int64_t j = 0; j < lN; j++) {
                    pfORow[j] += fAik * pfBRow[j];
                }
            }
        }
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Matvec
 * DESC: matrix-vector multiply
 *       [i.e., (m, k) x (k) -> (m)]
 * PARMS: tMat (matrix), tVec (vector)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Matvec(const CTensor &tMat, const CTensor &tVec) -> CTensor {
    assert(tMat.m_eType == EType::F32 && tVec.m_eType == EType::F32);
    assert(tMat.m_iNdim == 2 && tVec.m_iNdim == 1);
    assert(tMat.m_lShape[1] == tVec.m_lShape[0] && "[op:matvec] matvec dimension mismatch");
    assert(tMat.bIsContiguous() && tVec.bIsContiguous());

    const int64_t lM = tMat.m_lShape[0];
    const int64_t lK = tMat.m_lShape[1];

    CTensor tOut = CTensor::Zeros({lM});
    const float *__restrict__ pfMat = tMat.pfData();
    const float *__restrict__ pfVec = tVec.pfData();
    float *__restrict__ pfO = tOut.pfData();

    // yay ilp magic
    for (int64_t i = 0; i < lM; i++) {
        const float *__restrict__ pfRow = pfMat + i * lK;
        float fSum0 = 0.0f, fSum1 = 0.0f, fSum2 = 0.0f, fSum3 = 0.0f;
        int64_t k = 0;
        const int64_t lK4 = lK - (lK % 4);
        for (; k < lK4; k += 4) {
            fSum0 += pfRow[k] * pfVec[k];
            fSum1 += pfRow[k + 1] * pfVec[k + 1];
            fSum2 += pfRow[k + 2] * pfVec[k + 2];
            fSum3 += pfRow[k + 3] * pfVec[k + 3];
        }
        float fSum = fSum0 + fSum1 + fSum2 + fSum3;
        for (; k < lK; k++) {
            fSum += pfRow[k] * pfVec[k];
        }
        pfO[i] = fSum;
    }

    return tOut;
}
// >>>s_end(matrix)

// <<<s_start(reduction)
// --- reductions
/*---------------------------------------------------------
 * FN: Sum
 * DESC: sum elements (o: dimension)
 * PARMS: tA (tensor), iDim (dimension, -1 for all)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Sum(const CTensor &tA, int iDim) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());

    if (iDim == -1) {
        CTensor tOut({1});
        const float *__restrict__ pfA = tA.pfData();
        const int64_t lN = tA.lNumel();
        // more ilp magic!!!
        float fSum0 = 0.0f, fSum1 = 0.0f, fSum2 = 0.0f, fSum3 = 0.0f;
        int64_t i = 0;
        const int64_t lN4 = lN - (lN % 4);
        for (; i < lN4; i += 4) {
            fSum0 += pfA[i];
            fSum1 += pfA[i + 1];
            fSum2 += pfA[i + 2];
            fSum3 += pfA[i + 3];
        }
        float fSum = fSum0 + fSum1 + fSum2 + fSum3;
        for (; i < lN; i++)
            fSum += pfA[i];
        tOut.pfData()[0] = fSum;
        return tOut;
    }

    assert(iDim >= 0 && iDim < tA.m_iNdim);

    std::vector<int64_t> vlOutShape;
    for (int i = 0; i < tA.m_iNdim; i++) {
        if (i != iDim)
            vlOutShape.push_back(tA.m_lShape[i]);
    }
    if (vlOutShape.empty())
        vlOutShape.push_back(1);

    CTensor tOut = CTensor::Zeros(vlOutShape);

    int64_t lOuter = 1, lInner = 1;
    for (int i = 0; i < iDim; i++)
        lOuter *= tA.m_lShape[i];
    for (int i = iDim + 1; i < tA.m_iNdim; i++)
        lInner *= tA.m_lShape[i];
    const int64_t lDimSize = tA.m_lShape[iDim];

    const float *__restrict__ pfA = tA.pfData();
    float *__restrict__ pfO = tOut.pfData();

    if (lInner == 1) {
        // reduc last dim or cont inner=1
        // sum along cont run
        for (int64_t o = 0; o < lOuter; o++) {
            const float *__restrict__ pfRow = pfA + o * lDimSize;
            float fSum = 0.0f;
            for (int64_t d = 0; d < lDimSize; d++) {
                fSum += pfRow[d];
            }
            pfO[o] = fSum;
        }
    } else {
        for (int64_t o = 0; o < lOuter; o++) {
            float *__restrict__ pfORow = pfO + o * lInner;
            for (int64_t d = 0; d < lDimSize; d++) {
                const float *__restrict__ pfSrc = pfA + (o * lDimSize + d) * lInner;
                for (int64_t in = 0; in < lInner; in++) {
                    pfORow[in] += pfSrc[in];
                }
            }
        }
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Mean
 * DESC: mean of elements (o: dimension)
 * PARMS: tA (tensor), iDim (dimension, -1 for all)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Mean(const CTensor &tA, int iDim) -> CTensor {
    if (iDim == -1) {
        CTensor tS = Sum(tA, -1);
        return MulScalar(tS, 1.0f / (float)tA.lNumel());
    }
    CTensor tS = Sum(tA, iDim);
    return MulScalar(tS, 1.0f / (float)tA.m_lShape[iDim]);
}

/*---------------------------------------------------------
 * FN: Max
 * DESC: max of elements (o: dimension)
 * PARMS: tA (tensor), iDim (dimension, -1 for all)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Max(const CTensor &tA, int iDim) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());

    if (iDim == -1) {
        CTensor tOut({1});
        const float *__restrict__ pfA = tA.pfData();
        const int64_t lN = tA.lNumel();
        float fMax = pfA[0];
        for (int64_t i = 1; i < lN; i++) {
            if (pfA[i] > fMax)
                fMax = pfA[i];
        }
        tOut.pfData()[0] = fMax;
        return tOut;
    }

    assert(iDim >= 0 && iDim < tA.m_iNdim);

    std::vector<int64_t> vlOutShape;
    for (int i = 0; i < tA.m_iNdim; i++) {
        if (i != iDim)
            vlOutShape.push_back(tA.m_lShape[i]);
    }
    if (vlOutShape.empty())
        vlOutShape.push_back(1);

    CTensor tOut = CTensor::Zeros(vlOutShape);

    int64_t lOuter = 1, lInner = 1;
    for (int i = 0; i < iDim; i++)
        lOuter *= tA.m_lShape[i];
    for (int i = iDim + 1; i < tA.m_iNdim; i++)
        lInner *= tA.m_lShape[i];
    const int64_t lDimSize = tA.m_lShape[iDim];

    const float *__restrict__ pfA = tA.pfData();
    float *__restrict__ pfO = tOut.pfData();

    if (lInner == 1) {
        for (int64_t o = 0; o < lOuter; o++) {
            const float *__restrict__ pfRow = pfA + o * lDimSize;
            float fMax = pfRow[0];
            for (int64_t d = 1; d < lDimSize; d++) {
                if (pfRow[d] > fMax)
                    fMax = pfRow[d];
            }
            pfO[o] = fMax;
        }
    } else {
        for (int64_t o = 0; o < lOuter; o++) {
            float *__restrict__ pfORow = pfO + o * lInner;
            const float *__restrict__ pfFirst = pfA + o * lDimSize * lInner;
            std::memcpy(pfORow, pfFirst, lInner * sizeof(float));
            for (int64_t d = 1; d < lDimSize; d++) {
                const float *__restrict__ pfSrc = pfA + (o * lDimSize + d) * lInner;
                for (int64_t in = 0; in < lInner; in++) {
                    if (pfSrc[in] > pfORow[in])
                        pfORow[in] = pfSrc[in];
                }
            }
        }
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Argmaxxing
 * DESC: index of max element along dimension
 * PARMS: tA (tensor), iDim (dimension, -1 for flat)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Argmaxxing(const CTensor &tA, int iDim) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());

    if (iDim == -1) {
        // flat argmax
        CTensor tOut({1}, EType::I32);
        const float *pfA = tA.pfData();
        int32_t iMaxIdx = 0;
        float fMax = pfA[0];
        for (int64_t i = 1; i < tA.lNumel(); i++) {
            if (pfA[i] > fMax) {
                fMax = pfA[i];
                iMaxIdx = (int32_t)i;
            }
        }
        static_cast<int32_t *>(tOut.m_pData)[0] = iMaxIdx;
        return tOut;
    }

    assert(iDim >= 0 && iDim < tA.m_iNdim);

    std::vector<int64_t> vlOutShape;
    for (int i = 0; i < tA.m_iNdim; i++) {
        if (i != iDim)
            vlOutShape.push_back(tA.m_lShape[i]);
    }
    if (vlOutShape.empty())
        vlOutShape.push_back(1);

    CTensor tOut(vlOutShape, EType::I32);

    int64_t lOuter = 1, lInner = 1;
    for (int i = 0; i < iDim; i++)
        lOuter *= tA.m_lShape[i];
    for (int i = iDim + 1; i < tA.m_iNdim; i++)
        lInner *= tA.m_lShape[i];
    int64_t lDimSize = tA.m_lShape[iDim];

    const float *pfA = tA.pfData();
    int32_t *piO = static_cast<int32_t *>(tOut.m_pData);

    for (int64_t o = 0; o < lOuter; o++) {
        for (int64_t in = 0; in < lInner; in++) {
            float fMax = pfA[(o * lDimSize + 0) * lInner + in];
            int32_t iMaxIdx = 0;
            for (int64_t d = 1; d < lDimSize; d++) {
                float fVal = pfA[(o * lDimSize + d) * lInner + in];
                if (fVal > fMax) {
                    fMax = fVal;
                    iMaxIdx = (int32_t)d;
                }
            }
            piO[o * lInner + in] = iMaxIdx;
        }
    }

    return tOut;
}
// >>>s_end(reductions)

// <<<s_start(activiation)
// --- activations
/*---------------------------------------------------------
 * FN: Relu
 * DESC: rectified linear unit, max(0, x)
 * PARMS: tA (input tensor)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Relu(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return fX > 0.0f ? fX : 0.0f; });
}

/*---------------------------------------------------------
 * FN: Silu
 * DESC: sigmoid linear unit, x * sigmoid(x)
 * PARMS: tA (input tensor)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Silu(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return fX / (1.0f + std::exp(-fX)); });
}

/*---------------------------------------------------------
 * FN: Gelu
 * DESC: gaussian error linear unit (approximate)
 * PARMS: tA (input tensor)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Gelu(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) {
        return 0.5f * fX * (1.0f + std::tanh(std::sqrt(2.0f / (float)M_PI) * (fX + 0.044715f * fX * fX * fX)));
    });
}

/*---------------------------------------------------------
 * FN: Softmax
 * DESC: softmax along a dimension (default last)
 * PARMS: tA (input tensor), iDim (dimension)
 * AUTH: unium (13.02.26)
 *-------------------------------------------------------*/
auto Softmax(const CTensor &tA, int iDim) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());

    if (iDim == -1)
        iDim = tA.m_iNdim - 1;
    assert(iDim >= 0 && iDim < tA.m_iNdim);

    CTensor tOut = tLike(tA);

    int64_t lOuter = 1, lInner = 1;
    for (int i = 0; i < iDim; i++)
        lOuter *= tA.m_lShape[i];
    for (int i = iDim + 1; i < tA.m_iNdim; i++)
        lInner *= tA.m_lShape[i];
    const int64_t lDimSize = tA.m_lShape[iDim];

    const float *__restrict__ pfA = tA.pfData();
    float *__restrict__ pfO = tOut.pfData();

    if (lInner == 1) {
        for (int64_t o = 0; o < lOuter; o++) {
            const float *__restrict__ pfRow = pfA + o * lDimSize;
            float *__restrict__ pfORow = pfO + o * lDimSize;

            float fMax = pfRow[0];
            for (int64_t d = 1; d < lDimSize; d++) {
                if (pfRow[d] > fMax)
                    fMax = pfRow[d];
            }

            float fSumExp = 0.0f;
            for (int64_t d = 0; d < lDimSize; d++) {
                float fE = std::exp(pfRow[d] - fMax);
                pfORow[d] = fE;
                fSumExp += fE;
            }

            const float fInvSum = 1.0f / fSumExp;
            for (int64_t d = 0; d < lDimSize; d++) {
                pfORow[d] *= fInvSum;
            }
        }
    } else {
        for (int64_t o = 0; o < lOuter; o++) {
            for (int64_t in = 0; in < lInner; in++) {
                float fMax = pfA[(o * lDimSize + 0) * lInner + in];
                for (int64_t d = 1; d < lDimSize; d++) {
                    float fV = pfA[(o * lDimSize + d) * lInner + in];
                    if (fV > fMax)
                        fMax = fV;
                }

                float fSumExp = 0.0f;
                for (int64_t d = 0; d < lDimSize; d++) {
                    int64_t lIdx = (o * lDimSize + d) * lInner + in;
                    float fE = std::exp(pfA[lIdx] - fMax);
                    pfO[lIdx] = fE;
                    fSumExp += fE;
                }

                const float fInvSum = 1.0f / fSumExp;
                for (int64_t d = 0; d < lDimSize; d++) {
                    int64_t lIdx = (o * lDimSize + d) * lInner + in;
                    pfO[lIdx] *= fInvSum;
                }
            }
        }
    }

    return tOut;
}
// >>>s_end(activiation)

// <<<s_start(normalization)
// --- literally just rms norm
/*---------------------------------------------------------
 * FN: RmsNorm
 * DESC: root mean square normalization with learned weight
 * PARMS: tX (input), tW (weight), fEps (epsilon)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto RmsNorm(const CTensor &tX, const CTensor &tW, float fEps) -> CTensor {
    assert(tX.m_eType == EType::F32 && tW.m_eType == EType::F32);
    assert(tX.bIsContiguous() && tW.bIsContiguous());
    assert(tW.m_iNdim == 1);

    const int64_t lNormDim = tX.m_lShape[tX.m_iNdim - 1];
    assert(tW.m_lShape[0] == lNormDim);

    CTensor tOut = tLike(tX);
    const float *__restrict__ pfX = tX.pfData();
    const float *__restrict__ pfW = tW.pfData();
    float *__restrict__ pfO = tOut.pfData();

    const int64_t lRows = tX.lNumel() / lNormDim;
    const float fInvDim = 1.0f / (float)lNormDim;

    for (int64_t r = 0; r < lRows; r++) {
        const float *__restrict__ pfRow = pfX + r * lNormDim;
        float *__restrict__ pfOutRow = pfO + r * lNormDim;

        float fSS0 = 0.0f, fSS1 = 0.0f, fSS2 = 0.0f, fSS3 = 0.0f;
        int64_t i = 0;
        const int64_t lN4 = lNormDim - (lNormDim % 4);
        for (; i < lN4; i += 4) {
            fSS0 += pfRow[i] * pfRow[i];
            fSS1 += pfRow[i + 1] * pfRow[i + 1];
            fSS2 += pfRow[i + 2] * pfRow[i + 2];
            fSS3 += pfRow[i + 3] * pfRow[i + 3];
        }
        float fSS = fSS0 + fSS1 + fSS2 + fSS3;
        for (; i < lNormDim; i++) {
            fSS += pfRow[i] * pfRow[i];
        }

        const float fRms = 1.0f / std::sqrt(fSS * fInvDim + fEps);

        for (i = 0; i < lNormDim; i++) {
            pfOutRow[i] = pfRow[i] * fRms * pfW[i];
        }
    }

    return tOut;
}
// >>>s_end(normalization)

// <<<s_start(unary)
// --- unary maths
/*---------------------------------------------------------
 * FN: Neg
 * DESC: negates every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Neg(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return -fX; });
}

/*---------------------------------------------------------
 * FN: Sqrt
 * DESC: square root of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Sqrt(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return std::sqrt(fX); });
}

/*---------------------------------------------------------
 * FN: Rsqrt
 * DESC: reciprocal square root (1/sqrt) of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Rsqrt(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return 1.0f / std::sqrt(fX); });
}

/*---------------------------------------------------------
 * FN: Abs
 * DESC: absolute value of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Abs(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return std::fabs(fX); });
}

/*---------------------------------------------------------
 * FN: Exp
 * DESC: e^x of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Exp(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return std::exp(fX); });
}

/*---------------------------------------------------------
 * FN: Log
 * DESC: natural log of every element
 * PARMS: tA (input tensor)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Log(const CTensor &tA) -> CTensor {
    return tUnarywise(tA, [](float fX) { return std::log(fX); });
}

/*---------------------------------------------------------
 * FN: Clamp
 * DESC: clamps every element to [fMin, fMax]
 * PARMS: tA (input), fMin (lower bound), fMax (upper bound)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Clamp(const CTensor &tA, float fMin, float fMax) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();
    for (int64_t i = 0; i < lN; i++) {
        float fV = pfA[i];
        if (fV < fMin)
            fV = fMin;
        if (fV > fMax)
            fV = fMax;
        pfO[i] = fV;
    }
    return tOut;
}
// s_end(unary)

// <<<s_start(dot)
// --- dot product
/*---------------------------------------------------------
 * FN: fDot
 * DESC: dot product of two 1D tensors
 * PARMS: tA (first vector), tB (second vector)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto fDot(const CTensor &tA, const CTensor &tB) -> float {
    assert(tA.m_iNdim == 1 && tB.m_iNdim == 1);
    assert(tA.m_lShape[0] == tB.m_lShape[0]);
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    const float *__restrict__ pfA = tA.pfData();
    const float *__restrict__ pfB = tB.pfData();
    const int64_t lN = tA.m_lShape[0];

    float fSum0 = 0.0f, fSum1 = 0.0f, fSum2 = 0.0f, fSum3 = 0.0f;
    int64_t i = 0;
    const int64_t lN4 = lN - (lN % 4);
    for (; i < lN4; i += 4) {
        fSum0 += pfA[i] * pfB[i];
        fSum1 += pfA[i + 1] * pfB[i + 1];
        fSum2 += pfA[i + 2] * pfB[i + 2];
        fSum3 += pfA[i + 3] * pfB[i + 3];
    }
    float fSum = fSum0 + fSum1 + fSum2 + fSum3;
    for (; i < lN; i++) {
        fSum += pfA[i] * pfB[i];
    }
    return fSum;
}
// >>>s_end(dot)

// <<<s_start(index)
// --- index/slicing
/*---------------------------------------------------------
 * FN: SliceRow
 * DESC: extracts a single row from dim 0 as a view
 *       (does not copy data)
 * PARMS: tA (tensor), lIdx (row index)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto SliceRow(const CTensor &tA, int64_t lIdx) -> CTensor {
    assert(tA.m_iNdim >= 1);
    assert(lIdx >= 0 && lIdx < tA.m_lShape[0]);

    CTensor tOut;
    if (tA.m_iNdim == 1) {
        tOut.m_iNdim = 1;
        tOut.m_lShape[0] = 1;
        tOut.m_lStride[0] = 1;
    } else {
        tOut.m_iNdim = tA.m_iNdim - 1;
        for (int i = 0; i < tOut.m_iNdim; i++) {
            tOut.m_lShape[i] = tA.m_lShape[i + 1];
            tOut.m_lStride[i] = tA.m_lStride[i + 1];
        }
    }

    tOut.m_eType = tA.m_eType;
    tOut.m_pData = static_cast<char *>(tA.m_pData) + lIdx * tA.m_lStride[0] * nGetTypeSize(tA.m_eType);
    tOut.m_iDataSize = tOut.lNumel() * nGetTypeSize(tA.m_eType);
    tOut.m_bOwnsData = false;

    return tOut;
}

/*---------------------------------------------------------
 * FN: SliceRange
 * DESC: extracts rows [lStart, lEnd) from dim 0 as a view
 * PARMS: tA (tensor), lStart (start idx), lEnd (end idx)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto SliceRange(const CTensor &tA, int64_t lStart, int64_t lEnd) -> CTensor {
    assert(tA.m_iNdim >= 1);
    assert(lStart >= 0 && lStart < lEnd && lEnd <= tA.m_lShape[0]);

    CTensor tOut;
    tOut.m_iNdim = tA.m_iNdim;
    tOut.m_lShape[0] = lEnd - lStart;
    for (int i = 1; i < tA.m_iNdim; i++) {
        tOut.m_lShape[i] = tA.m_lShape[i];
    }
    for (int i = 0; i < tA.m_iNdim; i++) {
        tOut.m_lStride[i] = tA.m_lStride[i];
    }

    tOut.m_eType = tA.m_eType;
    tOut.m_pData = static_cast<char *>(tA.m_pData) + lStart * tA.m_lStride[0] * nGetTypeSize(tA.m_eType);
    tOut.m_iDataSize = tOut.lNumel() * nGetTypeSize(tA.m_eType);
    tOut.m_bOwnsData = false;

    return tOut;
}

/*---------------------------------------------------------
 * FN: Gather
 * DESC: gathers rows from tA using indices in tIdx
 *       tIdx is 1D int32, output shape = (len(tIdx), ...)
 *       used for token embedding lookup
 * PARMS: tA (source), tIdx (index tensor, I32)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Gather(const CTensor &tA, const CTensor &tIdx) -> CTensor {
    assert(tA.m_eType == EType::F32);
    assert(tIdx.m_eType == EType::I32 && tIdx.m_iNdim == 1);
    assert(tA.bIsContiguous() && tIdx.bIsContiguous());

    int64_t lNumIdx = tIdx.m_lShape[0];
    int64_t lRowSize = tA.lNumel() / tA.m_lShape[0];

    std::vector<int64_t> vlOutShape;
    vlOutShape.push_back(lNumIdx);
    for (int i = 1; i < tA.m_iNdim; i++) {
        vlOutShape.push_back(tA.m_lShape[i]);
    }

    CTensor tOut(vlOutShape);
    const float *pfA = tA.pfData();
    const int32_t *piIdx = static_cast<const int32_t *>(tIdx.m_pData);
    float *pfO = tOut.pfData();

    for (int64_t i = 0; i < lNumIdx; i++) {
        int32_t iRow = piIdx[i];
        assert(iRow >= 0 && iRow < tA.m_lShape[0] && "[op:gather] gather index out of bounds");
        std::memcpy(pfO + i * lRowSize, pfA + iRow * lRowSize, lRowSize * sizeof(float));
    }

    return tOut;
}
// >>>s_end(index)

// <<<s_start(construction)
// --- construction/manipulation
/*---------------------------------------------------------
 * FN: Concat
 * DESC: concatenates tensors along a dimension
 * PARMS: vtTensors (vector of tensor pointers), iDim (dim)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Concat(const std::vector<const CTensor *> &vtTensors, int iDim) -> CTensor {
    assert(!vtTensors.empty());
    const CTensor &tFirst = *vtTensors[0];
    assert(tFirst.m_eType == EType::F32);

    if (iDim < 0)
        iDim = tFirst.m_iNdim + iDim;
    assert(iDim >= 0 && iDim < tFirst.m_iNdim);

    int64_t lConcatSize = 0;
    for (auto *ptT : vtTensors) {
        assert(ptT->m_eType == EType::F32 && ptT->bIsContiguous());
        assert(ptT->m_iNdim == tFirst.m_iNdim);
        for (int i = 0; i < tFirst.m_iNdim; i++) {
            if (i != iDim) {
                assert(ptT->m_lShape[i] == tFirst.m_lShape[i] && "[op:concat] concat shape mismatch");
            }
        }
        lConcatSize += ptT->m_lShape[iDim];
    }

    std::vector<int64_t> vlOutShape;
    for (int i = 0; i < tFirst.m_iNdim; i++) {
        vlOutShape.push_back(i == iDim ? lConcatSize : tFirst.m_lShape[i]);
    }

    CTensor tOut(vlOutShape);
    float *__restrict__ pfO = tOut.pfData();

    int64_t lOuter = 1;
    for (int i = 0; i < iDim; i++)
        lOuter *= tFirst.m_lShape[i];
    int64_t lInner = 1;
    for (int i = iDim + 1; i < tFirst.m_iNdim; i++)
        lInner *= tFirst.m_lShape[i];

    const int64_t lOutConcatDim = lConcatSize;
    std::vector<int64_t> vlColOff(vtTensors.size());

    // <<<ignore(allman)
    {
        int64_t lOff = 0;
        for (size_t t = 0; t < vtTensors.size(); t++) {
            vlColOff[t] = lOff;
            lOff += vtTensors[t]->m_lShape[iDim];
        }
    }
    // >>>ignore

    for (int64_t o = 0; o < lOuter; o++) {
        for (size_t t = 0; t < vtTensors.size(); t++) {
            const CTensor *ptT = vtTensors[t];
            const int64_t lThisDim = ptT->m_lShape[iDim];
            const int64_t lChunkSize = lThisDim * lInner;
            const float *__restrict__ pfSrc = ptT->pfData() + o * lChunkSize;
            float *__restrict__ pfDst = pfO + o * lOutConcatDim * lInner + vlColOff[t] * lInner;
            std::memcpy(pfDst, pfSrc, lChunkSize * sizeof(float));
        }
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Repeat
 * DESC: repeats tensor along a dimension
 * PARMS: tA (tensor), iDim (dimension), iCount (repeats)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Repeat(const CTensor &tA, int iDim, int64_t iCount) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    assert(iDim >= 0 && iDim < tA.m_iNdim);
    assert(iCount >= 1);

    if (iCount == 1)
        return tA.Clone();

    std::vector<const CTensor *> vtPtrs(iCount, &tA);
    return Concat(vtPtrs, iDim);
}

/*---------------------------------------------------------
 * FN: Arange
 * DESC: creates a 1D tensor [fStart, fStart+1, ..., fEnd-1]
 * PARMS: fStart (start val), fEnd (end val), fStep (step)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Arange(float fStart, float fEnd, float fStep) -> CTensor {
    assert(fStep != 0.0f);
    assert((fEnd - fStart) / fStep > 0.0f && "[op:arange] empty range");

    int64_t lN = (int64_t)std::ceil((fEnd - fStart) / fStep);
    CTensor tOut({lN});
    float *pfO = tOut.pfData();

    for (int64_t i = 0; i < lN; i++) {
        pfO[i] = fStart + (float)i * fStep;
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: TriMask
 * DESC: creates an (iSize x iSize) upper triangular mask
 *       with fFillVal above diagonal, 0 on and below
 *       used for causal attention masking
 * PARMS: iSize (matrix size), fFillVal (mask value)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto TriMask(int64_t iSize, float fFillVal) -> CTensor {
    CTensor tOut = CTensor::Zeros({iSize, iSize});
    float *pfO = tOut.pfData();

    for (int64_t r = 0; r < iSize; r++) {
        for (int64_t c = r + 1; c < iSize; c++) {
            pfO[r * iSize + c] = fFillVal;
        }
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Where
 * DESC: element-wise conditional: out[i] = cond[i] ? tA[i] : tB[i]
 *       condition is true when value > 0
 * PARMS: tCond (condition), tA (true vals), tB (false vals)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
auto Where(const CTensor &tCond, const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tCond.m_eType == EType::F32 && tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tCond.m_iNdim == tA.m_iNdim && tA.m_iNdim == tB.m_iNdim);
    for (int i = 0; i < tA.m_iNdim; i++) {
        assert(tCond.m_lShape[i] == tA.m_lShape[i]);
        assert(tA.m_lShape[i] == tB.m_lShape[i]);
    }
    assert(tCond.bIsContiguous() && tA.bIsContiguous() && tB.bIsContiguous());

    CTensor tOut = tLike(tA);
    const float *pfC = tCond.pfData();
    const float *pfA = tA.pfData();
    const float *pfB = tB.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    for (int64_t i = 0; i < lN; i++) {
        pfO[i] = (pfC[i] > 0.0f) ? pfA[i] : pfB[i];
    }

    return tOut;
}
// >>>s_end(construction)

// <<<s_start(ipo)
// --- in place opers
/*---------------------------------------------------------
 * FN: CopyInto
 * DESC: copies tSrc data into tDst at row offset
 *       used for KV cache updates without allocation
 * PARMS: tDst (destination), tSrc (source), lRowOffset (row)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
void CopyInto(CTensor &tDst, const CTensor &tSrc, int64_t lRowOffset) {
    assert(tDst.m_eType == EType::F32 && tSrc.m_eType == EType::F32);
    assert(tDst.bIsContiguous() && tSrc.bIsContiguous());
    assert(tDst.m_iNdim == tSrc.m_iNdim);

    for (int i = 1; i < tDst.m_iNdim; i++) {
        assert(tDst.m_lShape[i] == tSrc.m_lShape[i] && "[op:copyinto] shape mismatch");
    }

    assert(lRowOffset >= 0);
    assert(lRowOffset + tSrc.m_lShape[0] <= tDst.m_lShape[0] && "[op:copyinto] out of bounds");

    const int64_t lRowSize = tSrc.lNumel() / tSrc.m_lShape[0];
    const int64_t lBytes = tSrc.lNumel() * sizeof(float);

    float *__restrict__ pfDst = tDst.pfData() + lRowOffset * lRowSize;
    const float *__restrict__ pfSrc = tSrc.pfData();

    std::memcpy(pfDst, pfSrc, lBytes);
}

/*---------------------------------------------------------
 * FN: FillInplace
 * DESC: fills entire tensor with a value in place
 * PARMS: tA (tensor to fill), fVal (value)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
void FillInplace(CTensor &tA, float fVal) {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    float *pfA = tA.pfData();
    int64_t lN = tA.lNumel();

    if (fVal == 0.0f) {
        std::memset(pfA, 0, lN * sizeof(float));
    } else {
        for (int64_t i = 0; i < lN; i++) {
            pfA[i] = fVal;
        }
    }
}
// >>>s_end(ipo)
} // namespace OP
} // namespace MT
