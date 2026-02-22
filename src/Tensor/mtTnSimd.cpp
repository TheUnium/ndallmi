// Created by Unium on 22.02.26

#include "mtTnSimd.hpp"
#include "../Thread/mtThPool.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define MT_X86 1
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#elif defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#define MT_X86 0
#endif

namespace MT {
namespace SM {
// <<<s_start(utils)
// --- cpu detection stuff
#if MT_X86
static void CpuId(int iLeaf, int viRegs[4]) {
#if defined(__GNUC__) || defined(__clang__)
    __cpuid(iLeaf, viRegs[0], viRegs[1], viRegs[2], viRegs[3]);
#elif defined(_MSC_VER)
    __cpuid(viRegs, iLeaf);
#endif
}

static void CpuIdEx(int iLeaf, int iSub, int viRegs[4]) {
#if defined(__GNUC__) || defined(__clang__)
    __cpuid_count(iLeaf, iSub, viRegs[0], viRegs[1], viRegs[2], viRegs[3]);
#elif defined(_MSC_VER)
    __cpuidex(viRegs, iLeaf, iSub);
#endif
}
#endif

/*---------------------------------------------------------
 * FN: bHasAvx2
 * DESC: checks if the cpu supports avx2 instructions
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto bHasAvx2() -> bool {
#if MT_X86
    static int s_iCached = -1;
    if (s_iCached >= 0)
        return s_iCached != 0;
    int viRegs[4] = {0};
    CpuId(0, viRegs);
    if (viRegs[0] >= 7) {
        CpuIdEx(7, 0, viRegs);
        s_iCached = (viRegs[1] & (1 << 5)) ? 1 : 0;
    } else {
        s_iCached = 0;
    }
    return s_iCached != 0;
#else
    return false;
#endif
}

/*---------------------------------------------------------
 * FN: bHasAvx512
 * DESC: checks if the cpu supports avx512f instructions
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto bHasAvx512() -> bool {
#if MT_X86
    static int s_iCached = -1;
    if (s_iCached >= 0)
        return s_iCached != 0;
    int viRegs[4] = {0};
    CpuId(0, viRegs);
    if (viRegs[0] >= 7) {
        CpuIdEx(7, 0, viRegs);
        s_iCached = (viRegs[1] & (1 << 16)) ? 1 : 0;
    } else {
        s_iCached = 0;
    }
    return s_iCached != 0;
#else
    return false;
#endif
}

/*---------------------------------------------------------
 * FN: PCF
 * DESC: prints detected simd and threading info to stdout
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void PCF() {
    std::cout << "  SIMD capabilities:" << std::endl;
#if MT_X86
    std::cout << "    x86 arch:  yes" << std::endl;
    std::cout << "    AVX2:      " << (bHasAvx2() ? "yes" : "no") << std::endl;
    std::cout << "    AVX-512F:  " << (bHasAvx512() ? "yes" : "no") << std::endl;
#else
    std::cout << "    x86 arch:  no (scalar fallback)" << std::endl;
#endif
    std::cout << "    Threads:   " << TH::GetGlobalPool().iNumThreads() << std::endl;
}

/*---------------------------------------------------------
 * FN: tLike
 * DESC: creates an uninitd tensor with same shape/dtype
 * PARMS: tA (tensor to mimic)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
static auto tLike(const CTensor &tA) -> CTensor {
    return CTensor(std::vector<int64_t>(tA.m_lShape, tA.m_lShape + tA.m_iNdim), tA.m_eType);
}

/*---------------------------------------------------------
 * FN: fHsumAvx
 * DESC: horizontal sum of all 8 flopats in __m256
 * PARMS: v (avx reg)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
#if MT_X86 && defined(__AVX2__)
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
#endif
// >>>s_end(utils)

// <<<s_start(sk)
// --- simd kernel
// <<<s_start(skelement)
// --- simd kernel for element wise opers
/*---------------------------------------------------------
 * FN: KernelAdd
 * DESC: simd add kernel for flat range [lStart, lEnd]
 * PARMS: pfA, pfB, pfO, lStart, lEnd
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
static void KernelAdd(const float *pfA, const float *pfB, float *pfO, int64_t lStart, int64_t lEnd) {
    int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
    for (; i + 32 <= lEnd; i += 32) {
        _mm256_storeu_ps(pfO + i, _mm256_add_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i)));
        _mm256_storeu_ps(pfO + i + 8, _mm256_add_ps(_mm256_loadu_ps(pfA + i + 8), _mm256_loadu_ps(pfB + i + 8)));
        _mm256_storeu_ps(pfO + i + 16, _mm256_add_ps(_mm256_loadu_ps(pfA + i + 16), _mm256_loadu_ps(pfB + i + 16)));
        _mm256_storeu_ps(pfO + i + 24, _mm256_add_ps(_mm256_loadu_ps(pfA + i + 24), _mm256_loadu_ps(pfB + i + 24)));
    }
    for (; i + 8 <= lEnd; i += 8) {
        _mm256_storeu_ps(pfO + i, _mm256_add_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i)));
    }
#endif
    for (; i < lEnd; i++)
        pfO[i] = pfA[i] + pfB[i];
}

/*---------------------------------------------------------
 * FN: KernelSub
 * DESC: simd sub kernel for flat range [lStart, lEnd]
 * PARMS: pfA, pfB, pfO, lStart, lEnd
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
static void KernelSub(const float *pfA, const float *pfB, float *pfO, int64_t lStart, int64_t lEnd) {
    int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
    for (; i + 32 <= lEnd; i += 32) {
        _mm256_storeu_ps(pfO + i, _mm256_sub_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i)));
        _mm256_storeu_ps(pfO + i + 8, _mm256_sub_ps(_mm256_loadu_ps(pfA + i + 8), _mm256_loadu_ps(pfB + i + 8)));
        _mm256_storeu_ps(pfO + i + 16, _mm256_sub_ps(_mm256_loadu_ps(pfA + i + 16), _mm256_loadu_ps(pfB + i + 16)));
        _mm256_storeu_ps(pfO + i + 24, _mm256_sub_ps(_mm256_loadu_ps(pfA + i + 24), _mm256_loadu_ps(pfB + i + 24)));
    }
    for (; i + 8 <= lEnd; i += 8) {
        _mm256_storeu_ps(pfO + i, _mm256_sub_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i)));
    }
#endif
    for (; i < lEnd; i++)
        pfO[i] = pfA[i] - pfB[i];
}

/*---------------------------------------------------------
 * FN: KernelMul
 * DESC: simd mul kernel for flat range [lStart, lEnd]
 * PARMS: pfA, pfB, pfO, lStart, lEnd
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
static void KernelMul(const float *pfA, const float *pfB, float *pfO, int64_t lStart, int64_t lEnd) {
    int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
    for (; i + 32 <= lEnd; i += 32) {
        _mm256_storeu_ps(pfO + i, _mm256_mul_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i)));
        _mm256_storeu_ps(pfO + i + 8, _mm256_mul_ps(_mm256_loadu_ps(pfA + i + 8), _mm256_loadu_ps(pfB + i + 8)));
        _mm256_storeu_ps(pfO + i + 16, _mm256_mul_ps(_mm256_loadu_ps(pfA + i + 16), _mm256_loadu_ps(pfB + i + 16)));
        _mm256_storeu_ps(pfO + i + 24, _mm256_mul_ps(_mm256_loadu_ps(pfA + i + 24), _mm256_loadu_ps(pfB + i + 24)));
    }
    for (; i + 8 <= lEnd; i += 8) {
        _mm256_storeu_ps(pfO + i, _mm256_mul_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i)));
    }
#endif
    for (; i < lEnd; i++)
        pfO[i] = pfA[i] * pfB[i];
}

/*---------------------------------------------------------
 * FN: KernelDot
 * DESC: simd dot prod for flat range [lStart, lEnd]
 * PARMS: pfA, pfB, lStart, lEnd
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
static auto KernelDot(const float *pfA, const float *pfB, int64_t lStart, int64_t lEnd) -> float {
    float fSum = 0.0f;
    int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
    __m256 vAcc0 = _mm256_setzero_ps();
    __m256 vAcc1 = _mm256_setzero_ps();
    __m256 vAcc2 = _mm256_setzero_ps();
    __m256 vAcc3 = _mm256_setzero_ps();
    for (; i + 32 <= lEnd; i += 32) {
        vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i), vAcc0);
        vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfA + i + 8), _mm256_loadu_ps(pfB + i + 8), vAcc1);
        vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfA + i + 16), _mm256_loadu_ps(pfB + i + 16), vAcc2);
        vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfA + i + 24), _mm256_loadu_ps(pfB + i + 24), vAcc3);
    }
    for (; i + 8 <= lEnd; i += 8) {
        vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfA + i), _mm256_loadu_ps(pfB + i), vAcc0);
    }
    fSum = fHsumAvx(_mm256_add_ps(_mm256_add_ps(vAcc0, vAcc1), _mm256_add_ps(vAcc2, vAcc3)));
#endif
    for (; i < lEnd; i++)
        fSum += pfA[i] * pfB[i];
    return fSum;
}
// >>>s_end(element)

// <<<s_start(mmk)
// --- matmul micro kernel
/*---------------------------------------------------------
 * FN: KernelMatmulBlock
 * DESC: computes a block of c+=a*b for row range
 *       [lStart, lEnd]
 *       - outer loop over k in tiles of tile_k
 *         (cache B panel)
 *       - inner loop over rows in groups of micro_m
 *       - each micro row group processes n in 16/8 wide
 *         strips
 *      multiple rows share b loads = cache resues
 * PARMS: pfA, pfB, pfC, lK, lN, lRowStart, lRowEnd
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
static void KernelMatmulBlock(const float *pfA, const float *pfB, float *pfC, int64_t lK, int64_t lN, int64_t lRowStart,
                              int64_t lRowEnd) {
    constexpr int64_t TILE_K = 256;
    constexpr int64_t TILE_N = 128;
    constexpr int64_t MICRO_M = 6;

    for (int64_t i = lRowStart; i < lRowEnd; i++) {
        for (int64_t j = 0; j < lN; j++) {
            pfC[i * lN + j] = 0.0f;
        }
    }

    for (int64_t kk = 0; kk < lK; kk += TILE_K) {
        int64_t lKEnd = std::min(kk + TILE_K, lK);

        for (int64_t jjj = 0; jjj < lN; jjj += TILE_N) {
            int64_t lNEnd = std::min(jjj + TILE_N, lN);

            int64_t ii = lRowStart;
            for (; ii + MICRO_M <= lRowEnd; ii += MICRO_M) {
                int64_t jj = jjj;

#if MT_X86 && defined(__AVX2__)
                for (; jj + 16 <= lNEnd; jj += 16) {
                    __m256 vC[MICRO_M][2];
                    for (int mi = 0; mi < MICRO_M; mi++) {
                        vC[mi][0] = _mm256_loadu_ps(pfC + (ii + mi) * lN + jj);
                        vC[mi][1] = _mm256_loadu_ps(pfC + (ii + mi) * lN + jj + 8);
                    }

                    for (int64_t k = kk; k < lKEnd; k++) {
                        __m256 vB0 = _mm256_loadu_ps(pfB + k * lN + jj);
                        __m256 vB1 = _mm256_loadu_ps(pfB + k * lN + jj + 8);

                        for (int mi = 0; mi < MICRO_M; mi++) {
                            __m256 vA = _mm256_set1_ps(pfA[(ii + mi) * lK + k]);
                            vC[mi][0] = _mm256_fmadd_ps(vA, vB0, vC[mi][0]);
                            vC[mi][1] = _mm256_fmadd_ps(vA, vB1, vC[mi][1]);
                        }
                    }

                    for (int mi = 0; mi < MICRO_M; mi++) {
                        _mm256_storeu_ps(pfC + (ii + mi) * lN + jj, vC[mi][0]);
                        _mm256_storeu_ps(pfC + (ii + mi) * lN + jj + 8, vC[mi][1]);
                    }
                }

                for (; jj + 8 <= lNEnd; jj += 8) {
                    __m256 vC[MICRO_M];
                    for (int mi = 0; mi < MICRO_M; mi++) {
                        vC[mi] = _mm256_loadu_ps(pfC + (ii + mi) * lN + jj);
                    }

                    for (int64_t k = kk; k < lKEnd; k++) {
                        __m256 vB0 = _mm256_loadu_ps(pfB + k * lN + jj);
                        for (int mi = 0; mi < MICRO_M; mi++) {
                            __m256 vA = _mm256_set1_ps(pfA[(ii + mi) * lK + k]);
                            vC[mi] = _mm256_fmadd_ps(vA, vB0, vC[mi]);
                        }
                    }

                    for (int mi = 0; mi < MICRO_M; mi++) {
                        _mm256_storeu_ps(pfC + (ii + mi) * lN + jj, vC[mi]);
                    }
                }
#endif

                if (jj < lNEnd) {
                    for (int mi = 0; mi < MICRO_M; mi++) {
                        for (int64_t k = kk; k < lKEnd; k++) {
                            float fA = pfA[(ii + mi) * lK + k];
                            for (int64_t j = jj; j < lNEnd; j++) {
                                pfC[(ii + mi) * lN + j] += fA * pfB[k * lN + j];
                            }
                        }
                    }
                }
            }

            for (; ii < lRowEnd; ii++) {
                int64_t jj = jjj;

#if MT_X86 && defined(__AVX2__)
                for (; jj + 16 <= lNEnd; jj += 16) {
                    __m256 vC0 = _mm256_loadu_ps(pfC + ii * lN + jj);
                    __m256 vC1 = _mm256_loadu_ps(pfC + ii * lN + jj + 8);

                    for (int64_t k = kk; k < lKEnd; k++) {
                        __m256 vA = _mm256_set1_ps(pfA[ii * lK + k]);
                        vC0 = _mm256_fmadd_ps(vA, _mm256_loadu_ps(pfB + k * lN + jj), vC0);
                        vC1 = _mm256_fmadd_ps(vA, _mm256_loadu_ps(pfB + k * lN + jj + 8), vC1);
                    }

                    _mm256_storeu_ps(pfC + ii * lN + jj, vC0);
                    _mm256_storeu_ps(pfC + ii * lN + jj + 8, vC1);
                }

                for (; jj + 8 <= lNEnd; jj += 8) {
                    __m256 vC0 = _mm256_loadu_ps(pfC + ii * lN + jj);

                    for (int64_t k = kk; k < lKEnd; k++) {
                        __m256 vA = _mm256_set1_ps(pfA[ii * lK + k]);
                        vC0 = _mm256_fmadd_ps(vA, _mm256_loadu_ps(pfB + k * lN + jj), vC0);
                    }

                    _mm256_storeu_ps(pfC + ii * lN + jj, vC0);
                }
#endif

                for (; jj < lNEnd; jj++) {
                    float fSum = pfC[ii * lN + jj];
                    for (int64_t k = kk; k < lKEnd; k++) {
                        fSum += pfA[ii * lK + k] * pfB[k * lN + jj];
                    }
                    pfC[ii * lN + jj] = fSum;
                }
            }
        }
    }
}

/*---------------------------------------------------------
 * FN: KernelMatvecRows
 * DESC: computes rows [lStart, lEnd] of y=m*x using simd
 *       dot prod per row
 * PARMS: pfM, pfV, pfO, lK, lStart, lEnd
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
static void KernelMatvecRows(const float *RESTRICT pfM, const float *RESTRICT pfV, float *RESTRICT pfO, int64_t lK,
                             int64_t lRowStart, int64_t lRowEnd) {
    int64_t i = lRowStart;
#if MT_X86 && defined(__AVX2__)
    for (; i + 8 <= lRowEnd; i += 8) {
        const float *pfR0 = pfM + (i + 0) * lK;
        const float *pfR1 = pfM + (i + 1) * lK;
        const float *pfR2 = pfM + (i + 2) * lK;
        const float *pfR3 = pfM + (i + 3) * lK;
        const float *pfR4 = pfM + (i + 4) * lK;
        const float *pfR5 = pfM + (i + 5) * lK;
        const float *pfR6 = pfM + (i + 6) * lK;
        const float *pfR7 = pfM + (i + 7) * lK;

        __m256 vAcc0 = _mm256_setzero_ps();
        __m256 vAcc1 = _mm256_setzero_ps();
        __m256 vAcc2 = _mm256_setzero_ps();
        __m256 vAcc3 = _mm256_setzero_ps();
        __m256 vAcc4 = _mm256_setzero_ps();
        __m256 vAcc5 = _mm256_setzero_ps();
        __m256 vAcc6 = _mm256_setzero_ps();
        __m256 vAcc7 = _mm256_setzero_ps();

        int64_t k = 0;

        // ur2 process 16 elm per loop iter
        for (; k + 16 <= lK; k += 16) {
            __m256 vV0 = _mm256_loadu_ps(pfV + k);
            __m256 vV1 = _mm256_loadu_ps(pfV + k + 8);

            // c0 (first 8 elements)
            vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR0 + k), vV0, vAcc0);
            vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR1 + k), vV0, vAcc1);
            vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR2 + k), vV0, vAcc2);
            vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR3 + k), vV0, vAcc3);
            vAcc4 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR4 + k), vV0, vAcc4);
            vAcc5 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR5 + k), vV0, vAcc5);
            vAcc6 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR6 + k), vV0, vAcc6);
            vAcc7 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR7 + k), vV0, vAcc7);

            // c1 (next 8 elements)
            vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR0 + k + 8), vV1, vAcc0);
            vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR1 + k + 8), vV1, vAcc1);
            vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR2 + k + 8), vV1, vAcc2);
            vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR3 + k + 8), vV1, vAcc3);
            vAcc4 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR4 + k + 8), vV1, vAcc4);
            vAcc5 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR5 + k + 8), vV1, vAcc5);
            vAcc6 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR6 + k + 8), vV1, vAcc6);
            vAcc7 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR7 + k + 8), vV1, vAcc7);
        }

        // clean up remaining k in blocks of 8
        for (; k + 8 <= lK; k += 8) {
            __m256 vV = _mm256_loadu_ps(pfV + k);
            vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR0 + k), vV, vAcc0);
            vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR1 + k), vV, vAcc1);
            vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR2 + k), vV, vAcc2);
            vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR3 + k), vV, vAcc3);
            vAcc4 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR4 + k), vV, vAcc4);
            vAcc5 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR5 + k), vV, vAcc5);
            vAcc6 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR6 + k), vV, vAcc6);
            vAcc7 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR7 + k), vV, vAcc7);
        }

        // horizontal sum happens EXACTLY ONCE per row!!!
        float f0 = fHsumAvx(vAcc0);
        float f1 = fHsumAvx(vAcc1);
        float f2 = fHsumAvx(vAcc2);
        float f3 = fHsumAvx(vAcc3);
        float f4 = fHsumAvx(vAcc4);
        float f5 = fHsumAvx(vAcc5);
        float f6 = fHsumAvx(vAcc6);
        float f7 = fHsumAvx(vAcc7);

        // sr
        for (; k < lK; k++) {
            float fV = pfV[k];
            f0 += pfR0[k] * fV;
            f1 += pfR1[k] * fV;
            f2 += pfR2[k] * fV;
            f3 += pfR3[k] * fV;
            f4 += pfR4[k] * fV;
            f5 += pfR5[k] * fV;
            f6 += pfR6[k] * fV;
            f7 += pfR7[k] * fV;
        }

        // direct write 1ce
        pfO[i + 0] = f0;
        pfO[i + 1] = f1;
        pfO[i + 2] = f2;
        pfO[i + 3] = f3;
        pfO[i + 4] = f4;
        pfO[i + 5] = f5;
        pfO[i + 6] = f6;
        pfO[i + 7] = f7;
    }

    // process rem rs in chunks of 4
    for (; i + 4 <= lRowEnd; i += 4) {
        const float *pfR0 = pfM + (i + 0) * lK;
        const float *pfR1 = pfM + (i + 1) * lK;
        const float *pfR2 = pfM + (i + 2) * lK;
        const float *pfR3 = pfM + (i + 3) * lK;

        __m256 vAcc0 = _mm256_setzero_ps();
        __m256 vAcc1 = _mm256_setzero_ps();
        __m256 vAcc2 = _mm256_setzero_ps();
        __m256 vAcc3 = _mm256_setzero_ps();

        int64_t k = 0;
        for (; k + 16 <= lK; k += 16) {
            __m256 vV0 = _mm256_loadu_ps(pfV + k);
            __m256 vV1 = _mm256_loadu_ps(pfV + k + 8);

            vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR0 + k), vV0, vAcc0);
            vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR1 + k), vV0, vAcc1);
            vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR2 + k), vV0, vAcc2);
            vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR3 + k), vV0, vAcc3);

            vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR0 + k + 8), vV1, vAcc0);
            vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR1 + k + 8), vV1, vAcc1);
            vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR2 + k + 8), vV1, vAcc2);
            vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR3 + k + 8), vV1, vAcc3);
        }

        for (; k + 8 <= lK; k += 8) {
            __m256 vV = _mm256_loadu_ps(pfV + k);
            vAcc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR0 + k), vV, vAcc0);
            vAcc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR1 + k), vV, vAcc1);
            vAcc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR2 + k), vV, vAcc2);
            vAcc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pfR3 + k), vV, vAcc3);
        }

        float f0 = fHsumAvx(vAcc0);
        float f1 = fHsumAvx(vAcc1);
        float f2 = fHsumAvx(vAcc2);
        float f3 = fHsumAvx(vAcc3);

        for (; k < lK; k++) {
            float fV = pfV[k];
            f0 += pfR0[k] * fV;
            f1 += pfR1[k] * fV;
            f2 += pfR2[k] * fV;
            f3 += pfR3[k] * fV;
        }

        pfO[i + 0] = f0;
        pfO[i + 1] = f1;
        pfO[i + 2] = f2;
        pfO[i + 3] = f3;
    }
#endif
    for (; i < lRowEnd; i++) {
        const float *pfRow = pfM + i * lK;
        float fS = 0.0f;
        int64_t k = 0;
#if MT_X86 && defined(__AVX2__)
        __m256 vAcc = _mm256_setzero_ps();
        for (; k + 8 <= lK; k += 8)
            vAcc = _mm256_fmadd_ps(_mm256_loadu_ps(pfRow + k), _mm256_loadu_ps(pfV + k), vAcc);
        fS = fHsumAvx(vAcc);
#endif
        for (; k < lK; k++)
            fS += pfRow[k] * pfV[k];
        pfO[i] = fS;
    }
}
// >>>s_end(mmk)
// >>>s_end(sk)

// <<<s_start(element)
// --- element wise opers
/*---------------------------------------------------------
 * FN: Add
 * DESC: [DEPRECATED] performs element wise addition of two
 *       tensors. naive add is ~same. check benchmarks.
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Add(const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == tB.m_iNdim);
    for (int i = 0; i < tA.m_iNdim; i++)
        assert(tA.m_lShape[i] == tB.m_lShape[i]);
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    const float *pfB = tB.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    if (lN < 500000) {
        KernelAdd(pfA, pfB, pfO, 0, lN);
    } else {
        TH::ParFor(lN, [&](int64_t s, int64_t e) { KernelAdd(pfA, pfB, pfO, s, e); });
    }
    return tOut;
}

/*---------------------------------------------------------
 * FN: Sub
 * DESC: performs element wise subtraction of two tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Sub(const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == tB.m_iNdim);
    for (int i = 0; i < tA.m_iNdim; i++)
        assert(tA.m_lShape[i] == tB.m_lShape[i]);
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    const float *pfB = tB.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    if (lN < 500000) {
        KernelSub(pfA, pfB, pfO, 0, lN);
    } else {
        TH::ParFor(lN, [&](int64_t s, int64_t e) { KernelSub(pfA, pfB, pfO, s, e); });
    }
    return tOut;
}

/*---------------------------------------------------------
 * FN: Mul
 * DESC: [DEPRECATED] performs element wise multiplication
 *       of two tensors. naive mul is ~same. check benchmarks.
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Mul(const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == tB.m_iNdim);
    for (int i = 0; i < tA.m_iNdim; i++)
        assert(tA.m_lShape[i] == tB.m_lShape[i]);
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    const float *pfB = tB.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    if (lN < 500000) {
        KernelMul(pfA, pfB, pfO, 0, lN);
    } else {
        TH::ParFor(lN, [&](int64_t s, int64_t e) { KernelMul(pfA, pfB, pfO, s, e); });
    }
    return tOut;
}

/*---------------------------------------------------------
 * FN: AddScalar
 * DESC: adds a scalar value to every element in the tensor
 * PARMS: tA (tensor), fS (scalar value)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto AddScalar(const CTensor &tA, float fS) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    auto fnKernel = [&](int64_t lStart, int64_t lEnd) {
        int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
        __m256 vS = _mm256_set1_ps(fS);
        for (; i + 32 <= lEnd; i += 32) {
            _mm256_storeu_ps(pfO + i, _mm256_add_ps(_mm256_loadu_ps(pfA + i), vS));
            _mm256_storeu_ps(pfO + i + 8, _mm256_add_ps(_mm256_loadu_ps(pfA + i + 8), vS));
            _mm256_storeu_ps(pfO + i + 16, _mm256_add_ps(_mm256_loadu_ps(pfA + i + 16), vS));
            _mm256_storeu_ps(pfO + i + 24, _mm256_add_ps(_mm256_loadu_ps(pfA + i + 24), vS));
        }
        for (; i + 8 <= lEnd; i += 8)
            _mm256_storeu_ps(pfO + i, _mm256_add_ps(_mm256_loadu_ps(pfA + i), vS));
#endif
        for (; i < lEnd; i++)
            pfO[i] = pfA[i] + fS;
    };

    if (lN < 500000) {
        fnKernel(0, lN);
    } else {
        TH::ParFor(lN, fnKernel);
    }
    return tOut;
}

/*---------------------------------------------------------
 * FN: MulScalar
 * DESC: multiplies every element in the tensor by a scalar
 * PARMS: tA (tensor), fS (scalar value)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto MulScalar(const CTensor &tA, float fS) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    auto fnKernel = [&](int64_t lStart, int64_t lEnd) {
        int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
        __m256 vS = _mm256_set1_ps(fS);
        for (; i + 32 <= lEnd; i += 32) {
            _mm256_storeu_ps(pfO + i, _mm256_mul_ps(_mm256_loadu_ps(pfA + i), vS));
            _mm256_storeu_ps(pfO + i + 8, _mm256_mul_ps(_mm256_loadu_ps(pfA + i + 8), vS));
            _mm256_storeu_ps(pfO + i + 16, _mm256_mul_ps(_mm256_loadu_ps(pfA + i + 16), vS));
            _mm256_storeu_ps(pfO + i + 24, _mm256_mul_ps(_mm256_loadu_ps(pfA + i + 24), vS));
        }
        for (; i + 8 <= lEnd; i += 8)
            _mm256_storeu_ps(pfO + i, _mm256_mul_ps(_mm256_loadu_ps(pfA + i), vS));
#endif
        for (; i < lEnd; i++)
            pfO[i] = pfA[i] * fS;
    };

    if (lN < 500000) {
        fnKernel(0, lN);
    } else {
        TH::ParFor(lN, fnKernel);
    }
    return tOut;
}
// >>>s_end(element)

// <<<s_start(matrix)
// --- matrix opers
/*---------------------------------------------------------
 * FN: Matmul
 * DESC: performs matrix multiplication between two tensors
 * PARMS: tA (matrix A), tB (matrix B)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Matmul(const CTensor &tA, const CTensor &tB) -> CTensor {
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.m_iNdim == 2 && tB.m_iNdim == 2);
    assert(tA.m_lShape[1] == tB.m_lShape[0]);
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    int64_t lM = tA.m_lShape[0];
    int64_t lK = tA.m_lShape[1];
    int64_t lN = tB.m_lShape[1];

    CTensor tOut({lM, lN});
    const float *pfA = tA.pfData();
    const float *pfB = tB.pfData();
    float *pfO = tOut.pfData();

    int iThreads = TH::GetGlobalPool().iNumThreads();
    int64_t lWorkSize = lM * lN * lK;
    int iActualThreads = (lWorkSize < 100000 || lM < 4) ? 1 : std::min((int64_t)iThreads, lM);

    if (iActualThreads <= 1) {
        KernelMatmulBlock(pfA, pfB, pfO, lK, lN, 0, lM);
    } else {
        int64_t lChunk = (lM + iActualThreads - 1) / iActualThreads;

        TH::GetGlobalPool().ParallelFor(iActualThreads, [&](int64_t lThreadStart, int64_t lThreadEnd) {
            for (int64_t t = lThreadStart; t < lThreadEnd; t++) {
                int64_t lRowStart = t * lChunk;
                int64_t lRowEnd = std::min(lRowStart + lChunk, lM);
                if (lRowStart < lRowEnd) {
                    KernelMatmulBlock(pfA, pfB, pfO, lK, lN, lRowStart, lRowEnd);
                }
            }
        });
    }

    return tOut;
}

/*---------------------------------------------------------
 * FN: Matvec
 * DESC: performs matrix vector multiplicstion
 * PARMS: tMat (matrix), tVec (vector)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Matvec(const CTensor &tMat, const CTensor &tVec) -> CTensor {
    assert(tMat.m_eType == EType::F32 && tVec.m_eType == EType::F32);
    assert(tMat.m_iNdim == 2 && tVec.m_iNdim == 1);
    assert(tMat.m_lShape[1] == tVec.m_lShape[0]);
    assert(tMat.bIsContiguous() && tVec.bIsContiguous());
    int64_t lM = tMat.m_lShape[0];
    int64_t lK = tMat.m_lShape[1];
    CTensor tOut({lM});
    const float *pfM = tMat.pfData();
    const float *pfV = tVec.pfData();
    float *pfO = tOut.pfData();

    int64_t lMatBytes = lM * lK * (int64_t)sizeof(float);
    int iActualThreads = 1;

    if (lMatBytes > 4 * 1024 * 1024) {
        int iThreads = TH::GetGlobalPool().iNumThreads();
        // iActualThreads = std::min(4, iThreads);
        iActualThreads = (int)std::min((int64_t)iActualThreads, lM / 64);
        if (iActualThreads < 1)
            iActualThreads = 1;
    }

    if (iActualThreads <= 1) {
        KernelMatvecRows(pfM, pfV, pfO, lK, 0, lM);
    } else {
        int64_t lChunk = (lM + iActualThreads - 1) / iActualThreads;
        lChunk = ((lChunk + 7) / 8) * 8;

        TH::GetGlobalPool().ParallelFor(iActualThreads, [&](int64_t lThreadStart, int64_t lThreadEnd) {
            for (int64_t t = lThreadStart; t < lThreadEnd; t++) {
                int64_t lRowStart = t * lChunk;
                int64_t lRowEnd = std::min(lRowStart + lChunk, lM);
                if (lRowStart < lRowEnd) {
                    KernelMatvecRows(pfM, pfV, pfO, lK, lRowStart, lRowEnd);
                }
            }
        });
    }
    return tOut;
}
// >>>s_end(matrix)

// <<<s_start(reductions)
// --- reductions
/*---------------------------------------------------------
 * FN: fDot
 * DESC: calculates the dot product of two tensors
 * PARMS: tA (first tensor), tB (second tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto fDot(const CTensor &tA, const CTensor &tB) -> float {
    assert(tA.m_iNdim == 1 && tB.m_iNdim == 1);
    assert(tA.m_lShape[0] == tB.m_lShape[0]);
    assert(tA.m_eType == EType::F32 && tB.m_eType == EType::F32);
    assert(tA.bIsContiguous() && tB.bIsContiguous());

    const float *pfA = tA.pfData();
    const float *pfB = tB.pfData();
    int64_t lN = tA.m_lShape[0];

    if (lN < 500000) {
        return KernelDot(pfA, pfB, 0, lN);
    }

    int iThreads = TH::GetGlobalPool().iNumThreads();
    std::vector<float> vfPartial(iThreads * 16, 0.0f); // cl

    TH::ParFor(lN, [&](int64_t lStart, int64_t lEnd) {
        int64_t lChunk = (lN + iThreads - 1) / iThreads;
        int iThread = (lChunk > 0) ? (int)(lStart / lChunk) : 0;
        if (iThread >= iThreads)
            iThread = iThreads - 1;
        vfPartial[iThread * 16] = KernelDot(pfA, pfB, lStart, lEnd);
    });

    float fTotal = 0.0f;
    for (int i = 0; i < iThreads; i++)
        fTotal += vfPartial[i * 16];
    return fTotal;
}

/*---------------------------------------------------------
 * FN: fSum
 * DESC: calculates the sum of all elements in the tensor
 * PARMS: tA (tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto fSum(const CTensor &tA) -> float {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());

    const float *pfA = tA.pfData();
    int64_t lN = tA.lNumel();

    auto fnSumRange = [&](int64_t lStart, int64_t lEnd) -> float {
        float fS = 0.0f;
        int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
        __m256 vAcc0 = _mm256_setzero_ps();
        __m256 vAcc1 = _mm256_setzero_ps();
        for (; i + 16 <= lEnd; i += 16) {
            vAcc0 = _mm256_add_ps(vAcc0, _mm256_loadu_ps(pfA + i));
            vAcc1 = _mm256_add_ps(vAcc1, _mm256_loadu_ps(pfA + i + 8));
        }
        for (; i + 8 <= lEnd; i += 8)
            vAcc0 = _mm256_add_ps(vAcc0, _mm256_loadu_ps(pfA + i));
        fS = fHsumAvx(_mm256_add_ps(vAcc0, vAcc1));
#endif
        for (; i < lEnd; i++)
            fS += pfA[i];
        return fS;
    };

    if (lN < 500000) {
        return fnSumRange(0, lN);
    }

    int iThreads = TH::GetGlobalPool().iNumThreads();
    std::vector<float> vfPartial(iThreads * 16, 0.0f);

    TH::ParFor(lN, [&](int64_t lStart, int64_t lEnd) {
        int64_t lChunk = (lN + iThreads - 1) / iThreads;
        int iThread = (lChunk > 0) ? (int)(lStart / lChunk) : 0;
        if (iThread >= iThreads)
            iThread = iThreads - 1;
        vfPartial[iThread * 16] = fnSumRange(lStart, lEnd);
    });

    float fTotal = 0.0f;
    for (int i = 0; i < iThreads; i++)
        fTotal += vfPartial[i * 16];
    return fTotal;
}
// >>>s_end(reductions)

// <<<s_start(activations)
// --- activations
/*---------------------------------------------------------
 * FN: Relu
 * DESC: applies the rectified linear unit activation function
 * PARMS: tA (input tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Relu(const CTensor &tA) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    auto fnKernel = [&](int64_t lStart, int64_t lEnd) {
        int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
        __m256 vZero = _mm256_setzero_ps();
        for (; i + 32 <= lEnd; i += 32) {
            _mm256_storeu_ps(pfO + i, _mm256_max_ps(_mm256_loadu_ps(pfA + i), vZero));
            _mm256_storeu_ps(pfO + i + 8, _mm256_max_ps(_mm256_loadu_ps(pfA + i + 8), vZero));
            _mm256_storeu_ps(pfO + i + 16, _mm256_max_ps(_mm256_loadu_ps(pfA + i + 16), vZero));
            _mm256_storeu_ps(pfO + i + 24, _mm256_max_ps(_mm256_loadu_ps(pfA + i + 24), vZero));
        }
        for (; i + 8 <= lEnd; i += 8)
            _mm256_storeu_ps(pfO + i, _mm256_max_ps(_mm256_loadu_ps(pfA + i), vZero));
#endif
        for (; i < lEnd; i++)
            pfO[i] = pfA[i] > 0.0f ? pfA[i] : 0.0f;
    };

    if (lN < 500000) {
        fnKernel(0, lN);
    } else {
        TH::ParFor(lN, fnKernel);
    }
    return tOut;
}

/*---------------------------------------------------------
 * FN: Silu
 * DESC: applies the sigmoid linear unit activation function
 * PARMS: tA (input tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Silu(const CTensor &tA) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    float *pfO = tOut.pfData();
    int64_t lN = tA.lNumel();

    auto fnKernel = [&](int64_t lStart, int64_t lEnd) {
        int64_t i = lStart;
#if MT_X86 && defined(__AVX2__)
        __m256 vOne = _mm256_set1_ps(1.0f);
        __m256 vNegOne = _mm256_set1_ps(-1.0f);
        __m256 vA = _mm256_set1_ps(12102203.2f);
        __m256 vB = _mm256_set1_ps(1064866805.0f);
        __m256 vClampLo = _mm256_set1_ps(-80.0f);
        __m256 vClampHi = _mm256_set1_ps(80.0f);
        for (; i + 8 <= lEnd; i += 8) {
            __m256 vX = _mm256_loadu_ps(pfA + i);
            __m256 vNegX = _mm256_mul_ps(vX, vNegOne);
            vNegX = _mm256_max_ps(vNegX, vClampLo);
            vNegX = _mm256_min_ps(vNegX, vClampHi);
            __m256 vExp = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(vNegX, vA), vB)));
            _mm256_storeu_ps(pfO + i, _mm256_mul_ps(vX, _mm256_div_ps(vOne, _mm256_add_ps(vOne, vExp))));
        }
#endif
        for (; i < lEnd; i++)
            pfO[i] = pfA[i] / (1.0f + std::exp(-pfA[i]));
    };

    if (lN < 500000) {
        fnKernel(0, lN);
    } else {
        TH::ParFor(lN, fnKernel);
    }
    return tOut;
}
// >>>s_end(activations)

// <<<s_start(normalization)
// --- normalization
/*---------------------------------------------------------
 * FN: RmsNorm
 * DESC: applies root mean square normalization
 * PARMS: tX (input), tW (weights), fEps (epsilon)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto RmsNorm(const CTensor &tX, const CTensor &tW, float fEps) -> CTensor {
    assert(tX.m_eType == EType::F32 && tW.m_eType == EType::F32);
    assert(tX.bIsContiguous() && tW.bIsContiguous());
    assert(tW.m_iNdim == 1);

    int64_t lNormDim = tX.m_lShape[tX.m_iNdim - 1];
    assert(tW.m_lShape[0] == lNormDim);

    CTensor tOut = tLike(tX);
    const float *pfX = tX.pfData();
    const float *pfW = tW.pfData();
    float *pfO = tOut.pfData();
    int64_t lRows = tX.lNumel() / lNormDim;

    auto fnProcessRows = [&](int64_t lStart, int64_t lEnd) {
        for (int64_t r = lStart; r < lEnd; r++) {
            const float *pfRow = pfX + r * lNormDim;
            float *pfOutRow = pfO + r * lNormDim;

            float fSS = 0.0f;
            int64_t j = 0;
#if MT_X86 && defined(__AVX2__)
            __m256 vAcc0 = _mm256_setzero_ps();
            __m256 vAcc1 = _mm256_setzero_ps();
            for (; j + 16 <= lNormDim; j += 16) {
                __m256 v0 = _mm256_loadu_ps(pfRow + j);
                __m256 v1 = _mm256_loadu_ps(pfRow + j + 8);
                vAcc0 = _mm256_fmadd_ps(v0, v0, vAcc0);
                vAcc1 = _mm256_fmadd_ps(v1, v1, vAcc1);
            }
            for (; j + 8 <= lNormDim; j += 8) {
                __m256 v0 = _mm256_loadu_ps(pfRow + j);
                vAcc0 = _mm256_fmadd_ps(v0, v0, vAcc0);
            }
            fSS = fHsumAvx(_mm256_add_ps(vAcc0, vAcc1));
#endif
            for (; j < lNormDim; j++)
                fSS += pfRow[j] * pfRow[j];

            float fScale = 1.0f / std::sqrt(fSS / (float)lNormDim + fEps);
            j = 0;
#if MT_X86 && defined(__AVX2__)
            __m256 vScale = _mm256_set1_ps(fScale);
            for (; j + 16 <= lNormDim; j += 16) {
                _mm256_storeu_ps(pfOutRow + j, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(pfRow + j), vScale),
                                                             _mm256_loadu_ps(pfW + j)));
                _mm256_storeu_ps(pfOutRow + j + 8, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(pfRow + j + 8), vScale),
                                                                 _mm256_loadu_ps(pfW + j + 8)));
            }
            for (; j + 8 <= lNormDim; j += 8) {
                _mm256_storeu_ps(pfOutRow + j, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(pfRow + j), vScale),
                                                             _mm256_loadu_ps(pfW + j)));
            }
#endif
            for (; j < lNormDim; j++)
                pfOutRow[j] = pfRow[j] * fScale * pfW[j];
        }
    };

    if (lRows < 256) {
        fnProcessRows(0, lRows);
    } else {
        TH::ParFor(lRows, fnProcessRows);
    }
    return tOut;
}

/*---------------------------------------------------------
 * FN: Softmax
 * DESC: apolies the softmax function to the input tensor
 * PARMS: tA (input tensor)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto Softmax(const CTensor &tA) -> CTensor {
    assert(tA.m_eType == EType::F32 && tA.bIsContiguous());
    assert(tA.m_iNdim >= 1);

    CTensor tOut = tLike(tA);
    const float *pfA = tA.pfData();
    float *pfO = tOut.pfData();

    int64_t lDimSize = tA.m_lShape[tA.m_iNdim - 1];
    int64_t lRows = tA.lNumel() / lDimSize;

    auto fnProcessRows = [&](int64_t lStart, int64_t lEnd) {
        for (int64_t r = lStart; r < lEnd; r++) {
            const float *pfRow = pfA + r * lDimSize;
            float *pfOutRow = pfO + r * lDimSize;

            // find max
            float fMax = pfRow[0];
            int64_t j = 1;
#if MT_X86 && defined(__AVX2__)
            if (lDimSize >= 8) {
                __m256 vMax = _mm256_loadu_ps(pfRow);
                j = 8;
                for (; j + 8 <= lDimSize; j += 8)
                    vMax = _mm256_max_ps(vMax, _mm256_loadu_ps(pfRow + j));
                __m128 vHi = _mm256_extractf128_ps(vMax, 1);
                __m128 vLo = _mm256_castps256_ps128(vMax);
                __m128 vM = _mm_max_ps(vLo, vHi);
                vM = _mm_max_ps(vM, _mm_movehl_ps(vM, vM));
                vM = _mm_max_ss(vM, _mm_movehdup_ps(vM));
                fMax = _mm_cvtss_f32(vM);
            }
#endif
            for (; j < lDimSize; j++)
                if (pfRow[j] > fMax)
                    fMax = pfRow[j];

            // exp and sum
            float fSumExp = 0.0f;
            j = 0;
#if MT_X86 && defined(__AVX2__)
            {
                __m256 vMax256 = _mm256_set1_ps(fMax);
                __m256 vExpA = _mm256_set1_ps(12102203.2f);
                __m256 vExpB = _mm256_set1_ps(1064866805.0f);
                __m256 vClampLo = _mm256_set1_ps(-80.0f);
                __m256 vClampHi = _mm256_set1_ps(80.0f);
                __m256 vSumAcc = _mm256_setzero_ps();

                for (; j + 8 <= lDimSize; j += 8) {
                    __m256 vX = _mm256_sub_ps(_mm256_loadu_ps(pfRow + j), vMax256);
                    vX = _mm256_max_ps(vX, vClampLo);
                    vX = _mm256_min_ps(vX, vClampHi);
                    __m256 vE = _mm256_castsi256_ps(_mm256_cvtps_epi32(_mm256_add_ps(_mm256_mul_ps(vX, vExpA), vExpB)));
                    _mm256_storeu_ps(pfOutRow + j, vE);
                    vSumAcc = _mm256_add_ps(vSumAcc, vE);
                }
                fSumExp = fHsumAvx(vSumAcc);
            }
#endif
            for (; j < lDimSize; j++) {
                float fE = std::exp(pfRow[j] - fMax);
                pfOutRow[j] = fE;
                fSumExp += fE;
            }

            // normalize
            j = 0;
            float fInvSum = 1.0f / fSumExp;
#if MT_X86 && defined(__AVX2__)
            __m256 vInvSum = _mm256_set1_ps(fInvSum);
            for (; j + 16 <= lDimSize; j += 16) {
                _mm256_storeu_ps(pfOutRow + j, _mm256_mul_ps(_mm256_loadu_ps(pfOutRow + j), vInvSum));
                _mm256_storeu_ps(pfOutRow + j + 8, _mm256_mul_ps(_mm256_loadu_ps(pfOutRow + j + 8), vInvSum));
            }
            for (; j + 8 <= lDimSize; j += 8)
                _mm256_storeu_ps(pfOutRow + j, _mm256_mul_ps(_mm256_loadu_ps(pfOutRow + j), vInvSum));
#endif
            for (; j < lDimSize; j++)
                pfOutRow[j] *= fInvSum;
        }
    };

    if (lRows < 256) {
        fnProcessRows(0, lRows);
    } else {
        TH::ParFor(lRows, fnProcessRows);
    }
    return tOut;
}
// >>>s_end(normalization)
} // namespace SM
} // namespace MT
