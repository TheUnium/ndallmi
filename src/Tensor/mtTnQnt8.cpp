// Created by Unium on 24.02.26

#include "mtTnQnt8.hpp"
#include "../Thread/mtThPool.hpp"
#include <cassert>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MT_X86 1
#include <immintrin.h>
#else
#define MT_X86 0
#endif

#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace MT {
namespace Q8 {

// <<<s_start(utils)
// --- utils
#if MT_X86 && defined(__AVX2__)
/*---------------------------------------------------------
 * FN: fHsumAvx
 * DESC: horizontal sum of all 8 floats in __m256
 * PARMS: v (avx reg)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static inline auto fHsumAvx(__m256 v) -> float {
    __m128 vLo = _mm256_castps256_ps128(v);
    __m128 vHi = _mm256_extractf128_ps(v, 1);
    __m128 vSum = _mm_add_ps(vLo, vHi);
    __m128 vShuf = _mm_movehdup_ps(vSum);
    __m128 vS = _mm_add_ps(vSum, vShuf);
    vShuf = _mm_movehl_ps(vShuf, vS);
    vS = _mm_add_ss(vS, vShuf);
    return _mm_cvtss_f32(vS);
}

/*---------------------------------------------------------
 * FN: vCvtI8ToF32x8
 * DESC: converts 8 packed int8 values to __m256 float
 *       loads 8 bytes from piSrc, sign-extends to int32,
 *       converts to float
 * PARMS: piSrc (int8 source pointer)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static inline auto vCvtI8ToF32x8(const int8_t *piSrc) -> __m256 {
    __m128i vQ8 = _mm_loadl_epi64((const __m128i *)piSrc);
    __m128i vQ16 = _mm_cvtepi8_epi16(vQ8);
    __m256i vQ32 = _mm256_cvtepi16_epi32(vQ16);
    return _mm256_cvtepi32_ps(vQ32);
}
#endif
// >>>s_end(utils)

// <<<s_start(quantize)
// --- quantization
/*---------------------------------------------------------
 * FN: Quantize
 * DESC: quantizes a f32 tensor 2d to int8 with pr absmax
 *       scaling, each row: scale = max(|row|) / 127,
 *       q[i] = round(x[i] / scale)
 * PARMS: tMat (f32 matrix [rows, cols])
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto Quantize(const CTensor &tMat) -> SQMatrix {
    assert(tMat.m_eType == EType::F32);
    assert(tMat.m_iNdim == 2);
    assert(tMat.bIsContiguous());

    SQMatrix qMat;
    qMat.lRows = tMat.m_lShape[0];
    qMat.lCols = tMat.m_lShape[1];
    qMat.viData.resize(qMat.lRows * qMat.lCols);
    qMat.vfScale.resize(qMat.lRows);

    const float *pfSrc = tMat.pfData();

    for (int64_t r = 0; r < qMat.lRows; r++) {
        const float *pfRow = pfSrc + r * qMat.lCols;

        float fAbsMax = 0.0f;
        int64_t c = 0;
#if MT_X86 && defined(__AVX2__)
        __m256 vAbsMax = _mm256_setzero_ps();
        __m256 vSignMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for (; c + 32 <= qMat.lCols; c += 32) {
            vAbsMax = _mm256_max_ps(vAbsMax, _mm256_and_ps(_mm256_loadu_ps(pfRow + c), vSignMask));
            vAbsMax = _mm256_max_ps(vAbsMax, _mm256_and_ps(_mm256_loadu_ps(pfRow + c + 8), vSignMask));
            vAbsMax = _mm256_max_ps(vAbsMax, _mm256_and_ps(_mm256_loadu_ps(pfRow + c + 16), vSignMask));
            vAbsMax = _mm256_max_ps(vAbsMax, _mm256_and_ps(_mm256_loadu_ps(pfRow + c + 24), vSignMask));
        }
        for (; c + 8 <= qMat.lCols; c += 8) {
            vAbsMax = _mm256_max_ps(vAbsMax, _mm256_and_ps(_mm256_loadu_ps(pfRow + c), vSignMask));
        }
        __m128 vHi = _mm256_extractf128_ps(vAbsMax, 1);
        __m128 vLo = _mm256_castps256_ps128(vAbsMax);
        __m128 vM = _mm_max_ps(vLo, vHi);
        vM = _mm_max_ps(vM, _mm_movehl_ps(vM, vM));
        vM = _mm_max_ss(vM, _mm_movehdup_ps(vM));
        fAbsMax = _mm_cvtss_f32(vM);
#endif
        for (; c < qMat.lCols; c++) {
            float fAbs = std::fabs(pfRow[c]);
            if (fAbs > fAbsMax)
                fAbsMax = fAbs;
        }

        float fScale = fAbsMax / 127.0f;
        if (fScale < 1e-10f)
            fScale = 1e-10f;

        qMat.vfScale[r] = fScale;

        float fInvScale = 1.0f / fScale;
        int8_t *piRow = qMat.viData.data() + r * qMat.lCols;

        c = 0;
#if MT_X86 && defined(__AVX2__)
        __m256 vInvScale = _mm256_set1_ps(fInvScale);
        __m256 vMin = _mm256_set1_ps(-128.0f);
        __m256 vMax = _mm256_set1_ps(127.0f);
        for (; c + 8 <= qMat.lCols; c += 8) {
            __m256 vVal = _mm256_mul_ps(_mm256_loadu_ps(pfRow + c), vInvScale);
            vVal = _mm256_round_ps(vVal, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            vVal = _mm256_max_ps(vVal, vMin);
            vVal = _mm256_min_ps(vVal, vMax);
            __m256i vI32 = _mm256_cvtps_epi32(vVal);
            // pack i32 -> i16 --> i8
            __m128i vLo32 = _mm256_castsi256_si128(vI32);
            __m128i vHi32 = _mm256_extracti128_si256(vI32, 1);
            __m128i vI16 = _mm_packs_epi32(vLo32, vHi32);
            __m128i vI8 = _mm_packs_epi16(vI16, vI16);
            _mm_storel_epi64((__m128i *)(piRow + c), vI8);
        }
#endif
        for (; c < qMat.lCols; c++) {
            float fVal = pfRow[c] * fInvScale;
            int32_t iVal = (int32_t)std::round(fVal);
            if (iVal > 127)
                iVal = 127;
            if (iVal < -128)
                iVal = -128;
            piRow[c] = (int8_t)iVal;
        }
    }

    return qMat;
}
// >>>s_end(quantize)

// <<<s_start(scalar)
// --- scalar matvec
/*---------------------------------------------------------
 * FN: Matvec
 * DESC: matvec
 * PARMS: qMat (quantized matrix), tVec (f32 vector)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto Matvec(const SQMatrix &qMat, const CTensor &tVec) -> CTensor {
    assert(tVec.m_eType == EType::F32);
    assert(tVec.m_iNdim == 1);
    assert(tVec.m_lShape[0] == qMat.lCols);
    assert(tVec.bIsContiguous());

    CTensor tOut = CTensor::Zeros({qMat.lRows});
    const float *RESTRICT pfVec = tVec.pfData();
    float *RESTRICT pfOut = tOut.pfData();
    const int8_t *piData = qMat.viData.data();

    for (int64_t r = 0; r < qMat.lRows; r++) {
        const int8_t *RESTRICT piRow = piData + r * qMat.lCols;
        float fSum0 = 0.0f, fSum1 = 0.0f, fSum2 = 0.0f, fSum3 = 0.0f;
        int64_t c = 0;
        const int64_t lC4 = qMat.lCols - (qMat.lCols % 4);
        for (; c < lC4; c += 4) {
            fSum0 += (float)piRow[c] * pfVec[c];
            fSum1 += (float)piRow[c + 1] * pfVec[c + 1];
            fSum2 += (float)piRow[c + 2] * pfVec[c + 2];
            fSum3 += (float)piRow[c + 3] * pfVec[c + 3];
        }
        float fSum = fSum0 + fSum1 + fSum2 + fSum3;
        for (; c < qMat.lCols; c++)
            fSum += (float)piRow[c] * pfVec[c];
        pfOut[r] = fSum * qMat.vfScale[r];
    }

    return tOut;
}
// >>>s_end(scalar)

// <<<s_start(simd)
// --- SIMD + threaded matvec
#if MT_X86 && defined(__AVX2__)
/*---------------------------------------------------------
 * FN: KernelQ8MatvecRows
 * DESC: penis
 * PARMS: piData (int8 matrix), pfVec (f32 vector),
 *        pfScale (per-row scales), pfOut (f32 output),
 *        lK (cols), lRowStart, lRowEnd
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static void KernelQ8MatvecRows(const int8_t *RESTRICT piData, const float *RESTRICT pfVec,
                               const float *RESTRICT pfScale, float *RESTRICT pfOut, int64_t lK, int64_t lRowStart,
                               int64_t lRowEnd) {
    int64_t i = lRowStart;

    // process 4 rows at a time
    // (8 rows like f32 kernel would need 8x int8->f32 conversions
    //  per k-step which thrashes the converter pipeline.
    //  4 rows is the sweet spot for q8 since each row needs
    //  a separate cvt chain: load i8 -> cvtepi8_epi16 -> cvtepi16_epi32 -> cvtepi32_ps)
    for (; i + 4 <= lRowEnd; i += 4) {
        const int8_t *piR0 = piData + (i + 0) * lK;
        const int8_t *piR1 = piData + (i + 1) * lK;
        const int8_t *piR2 = piData + (i + 2) * lK;
        const int8_t *piR3 = piData + (i + 3) * lK;

        __m256 vAcc0 = _mm256_setzero_ps();
        __m256 vAcc1 = _mm256_setzero_ps();
        __m256 vAcc2 = _mm256_setzero_ps();
        __m256 vAcc3 = _mm256_setzero_ps();

        int64_t k = 0;

        // ur2: process 16 elements per iteration
        for (; k + 16 <= lK; k += 16) {
            __m256 vV0 = _mm256_loadu_ps(pfVec + k);
            __m256 vV1 = _mm256_loadu_ps(pfVec + k + 8);

            // chunk 0 (first 8 elements)
            vAcc0 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR0 + k), vV0, vAcc0);
            vAcc1 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR1 + k), vV0, vAcc1);
            vAcc2 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR2 + k), vV0, vAcc2);
            vAcc3 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR3 + k), vV0, vAcc3);

            // chunk 1 (next 8 elements)
            vAcc0 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR0 + k + 8), vV1, vAcc0);
            vAcc1 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR1 + k + 8), vV1, vAcc1);
            vAcc2 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR2 + k + 8), vV1, vAcc2);
            vAcc3 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR3 + k + 8), vV1, vAcc3);
        }

        // clean up remaining k in blocks of 8
        for (; k + 8 <= lK; k += 8) {
            __m256 vV = _mm256_loadu_ps(pfVec + k);
            vAcc0 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR0 + k), vV, vAcc0);
            vAcc1 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR1 + k), vV, vAcc1);
            vAcc2 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR2 + k), vV, vAcc2);
            vAcc3 = _mm256_fmadd_ps(vCvtI8ToF32x8(piR3 + k), vV, vAcc3);
        }

        // horizontal sum happens EXACTLY ONCE per row
        float f0 = fHsumAvx(vAcc0);
        float f1 = fHsumAvx(vAcc1);
        float f2 = fHsumAvx(vAcc2);
        float f3 = fHsumAvx(vAcc3);

        // scalar tail
        for (; k < lK; k++) {
            float fV = pfVec[k];
            f0 += (float)piR0[k] * fV;
            f1 += (float)piR1[k] * fV;
            f2 += (float)piR2[k] * fV;
            f3 += (float)piR3[k] * fV;
        }

        // direct write with scale
        pfOut[i + 0] = f0 * pfScale[i + 0];
        pfOut[i + 1] = f1 * pfScale[i + 1];
        pfOut[i + 2] = f2 * pfScale[i + 2];
        pfOut[i + 3] = f3 * pfScale[i + 3];
    }

    // remaining rows: single row kernel
    for (; i < lRowEnd; i++) {
        const int8_t *piRow = piData + i * lK;
        __m256 vAcc = _mm256_setzero_ps();
        int64_t k = 0;
        for (; k + 8 <= lK; k += 8)
            vAcc = _mm256_fmadd_ps(vCvtI8ToF32x8(piRow + k), _mm256_loadu_ps(pfVec + k), vAcc);
        float fS = fHsumAvx(vAcc);
        for (; k < lK; k++)
            fS += (float)piRow[k] * pfVec[k];
        pfOut[i] = fS * pfScale[i];
    }
}
#endif

/*---------------------------------------------------------
 * FN: MatvecSimd
 * DESC: simd matvec
 * PARMS: qMat (quantized matrix), tVec (f32 vector)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto MatvecSimd(const SQMatrix &qMat, const CTensor &tVec) -> CTensor {
    assert(tVec.m_eType == EType::F32);
    assert(tVec.m_iNdim == 1);
    assert(tVec.m_lShape[0] == qMat.lCols);
    assert(tVec.bIsContiguous());

    CTensor tOut({qMat.lRows});
    const float *pfVec = tVec.pfData();
    float *pfOut = tOut.pfData();
    const int8_t *piData = qMat.viData.data();
    const float *pfScale = qMat.vfScale.data();
    int64_t lK = qMat.lCols;
    int64_t lM = qMat.lRows;

#if MT_X86 && defined(__AVX2__)
    int64_t lMatBytes = lM * lK; // int8 so 1b each
    int iActualThreads = 1;

    if (lMatBytes > 4 * 1024 * 1024) {
        int iThreads = TH::GetGlobalPool().iNumThreads();
        iActualThreads = std::min((int64_t)iThreads, lM / 16);
        if (iActualThreads < 1)
            iActualThreads = 1;
    }

    if (iActualThreads <= 1) {
        KernelQ8MatvecRows(piData, pfVec, pfScale, pfOut, lK, 0, lM);
    } else {
        int64_t lChunk = (lM + iActualThreads - 1) / iActualThreads;
        lChunk = ((lChunk + 3) / 4) * 4;
        TH::GetGlobalPool().ParallelFor(iActualThreads, [&](int64_t lThreadStart, int64_t lThreadEnd) {
            for (int64_t t = lThreadStart; t < lThreadEnd; t++) {
                int64_t lRowStart = t * lChunk;
                int64_t lRowEnd = std::min(lRowStart + lChunk, lM);
                if (lRowStart < lRowEnd) {
                    KernelQ8MatvecRows(piData, pfVec, pfScale, pfOut, lK, lRowStart, lRowEnd);
                }
            }
        });
    }
#else
    TH::ParFor(lM, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t r = lStart; r < lEnd; r++) {
            const int8_t *piRow = piData + r * lK;
            float fDot = 0.0f;
            for (int64_t c = 0; c < lK; c++)
                fDot += (float)piRow[c] * pfVec[c];
            pfOut[r] = fDot * pfScale[r];
        }
    });
#endif

    return tOut;
}
// >>>s_end(simd)
} // namespace Q8
} // namespace MT
