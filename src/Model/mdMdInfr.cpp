// Created by Unium on 25.02.26

#include "mdMdInfr.hpp"
#include "../Tensor/mtTnOps_.hpp"
#include "../Tensor/mtTnQnt8.hpp"
#include "../Tensor/mtTnSimd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace MT;

namespace MD {

// <<<s_start(construction)
// --- construction / reset
/*---------------------------------------------------------
 * FN: CInferState (ctor)
 * DESC: initializes inference state from a loaded model
 *       uses explicit head_dim from config for kv cache
 *       sizing
 * PARMS: psModel (pointer to loaded model)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
CInferState::CInferState(const SModel *psModel) : m_psModel(psModel), m_iPos(0) {

    const auto &sCfg = m_psModel->sConfig;
    int32_t iKvDim = sCfg.iNKvHeads * sCfg.iHeadDim;

    m_vKvCache.resize(sCfg.iNLayers);
    for (int32_t i = 0; i < sCfg.iNLayers; i++) {
        m_vKvCache[i].tK = CTensor::Zeros({(int64_t)sCfg.iMaxSeqLen, (int64_t)iKvDim});
        m_vKvCache[i].tV = CTensor::Zeros({(int64_t)sCfg.iMaxSeqLen, (int64_t)iKvDim});
        m_vKvCache[i].iLen = 0;
    }

    m_tX = CTensor::Zeros({(int64_t)sCfg.iDim});
    m_lRngState = 42;
}

/*---------------------------------------------------------
 * FN: Reset
 * DESC: resets kv cache and pos to 0
 * PARMS: none
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
void CInferState::Reset() {
    m_iPos = 0;
    m_viHistory.clear();
    for (auto &sKv : m_vKvCache) {
        OP::FillInplace(sKv.tK, 0.0f);
        OP::FillInplace(sKv.tV, 0.0f);
        sKv.iLen = 0;
    }
}
// >>>s_end(construction)

// <<<s_start(rng)
// --- rng
/*---------------------------------------------------------
 * FN: fRandFloat
 * DESC: returns random float in [0, 1) using xorshift64
 * PARMS: none
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
auto CInferState::fRandFloat() -> float {
    m_lRngState ^= m_lRngState << 13;
    m_lRngState ^= m_lRngState >> 7;
    m_lRngState ^= m_lRngState << 17;
    return (float)(m_lRngState & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}
// >>>s_end(rng)

// <<<s_start(helpers)
// --- bias + nope helpers
/*---------------------------------------------------------
 * FN: AddBias
 * DESC: adds bias vector to tensor in place no-op if bias
 *       has no data (model doesnt use biases)
 * PARMS: tOut (tensor to modify), tBias (bias vector)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
void CInferState::AddBias(CTensor &tOut, const CTensor &tBias) {
    if (tBias.lNumel() == 0 || tBias.m_pData == nullptr)
        return;

    float *pfOut = tOut.pfData();
    const float *pfBias = tBias.pfData();
    int64_t lN = tOut.lNumel();

    for (int64_t i = 0; i < lN; i++)
        pfOut[i] += pfBias[i];
}

/*---------------------------------------------------------
 * FN: bLayerUsesRope
 * DESC: returns true if the given layer should apply rope
 *       viNoRopeLayers contains layer indices that skip rope
 * PARMS: iLayer (layer index)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
auto CInferState::bLayerUsesRope(int32_t iLayer) const -> bool {
    const auto &viNoRope = m_psModel->sConfig.viNoRopeLayers;
    if (viNoRope.empty())
        return true;
    if (iLayer < (int32_t)viNoRope.size())
        return (viNoRope[iLayer] == 0);
    return true;
}
// >>>s_end(helpers)

// <<<s_start(rope)
// --- RoPE
/*---------------------------------------------------------
 * FN: ApplyRope
 * DESC: applies rotary position embeddings in-place
 *       uses configs head_dim for the rotation dimension
 * PARMS: tQ (query [n_heads * head_dim]), tK (key [n_kv_heads * head_dim]), iPos (current position)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
void CInferState::ApplyRope(CTensor &tQ, CTensor &tK, int32_t iPos) {
    const auto &sCfg = m_psModel->sConfig;
    int32_t iHeadDim = sCfg.iHeadDim;
    int32_t iHalf = iHeadDim / 2;
    float fTheta = sCfg.fRopeTheta;

    float *pfQ = tQ.pfData();
    float *pfK = tK.pfData();

    // q heads
    for (int32_t iH = 0; iH < sCfg.iNHeads; iH++) {
        float *pfQh = pfQ + iH * iHeadDim;
        for (int32_t i = 0; i < iHalf; i++) {
            float fFreq = 1.0f / std::pow(fTheta, (float)(2 * i) / (float)iHeadDim);
            float fAngle = (float)iPos * fFreq;
            float fCos = std::cos(fAngle);
            float fSin = std::sin(fAngle);

            float fReal = pfQh[i];
            float fImag = pfQh[i + iHalf];

            // --- llama rot
            // out[i]      = x[i]*cos - x[i+half]*sin
            // out[i+half] = x[i]*sin + x[i+half]*cos
            pfQh[i] = fReal * fCos - fImag * fSin;
            pfQh[i + iHalf] = fReal * fSin + fImag * fCos;
        }
    }

    // k heads
    for (int32_t iH = 0; iH < sCfg.iNKvHeads; iH++) {
        float *pfKh = pfK + iH * iHeadDim;
        for (int32_t i = 0; i < iHalf; i++) {
            float fFreq = 1.0f / std::pow(fTheta, (float)(2 * i) / (float)iHeadDim);
            float fAngle = (float)iPos * fFreq;
            float fCos = std::cos(fAngle);
            float fSin = std::sin(fAngle);

            float fReal = pfKh[i];
            float fImag = pfKh[i + iHalf];

            pfKh[i] = fReal * fCos - fImag * fSin;
            pfKh[i + iHalf] = fReal * fSin + fImag * fCos;
        }
    }
}
// >>>s_end(rope)

// <<<s_start(attention)
// --- attention
/*---------------------------------------------------------
 * FN: ForwardAttention
 * DESC: runs grouped query attention for one layer
 * PARMS: iLayer (layer index), tNormed (rms-normed input [dim])
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
auto CInferState::ForwardAttention(int32_t iLayer, const CTensor &tNormed) -> CTensor {
    const auto &sCfg = m_psModel->sConfig;
    const auto &sW = m_psModel->sWeights;
    const auto &sLyr = sW.vLayers[iLayer];
    auto &sKv = m_vKvCache[iLayer];

    int32_t iNHeads = sCfg.iNHeads;
    int32_t iNKvH = sCfg.iNKvHeads;
    int32_t iHeadDim = sCfg.iHeadDim;
    int32_t iKvDim = iNKvH * iHeadDim;
    int32_t iGrpSize = iNHeads / iNKvH;
    int32_t iQDim = iNHeads * iHeadDim;

    CTensor tQ, tK, tV;

    if (sW.bQuantized) {
        const auto &sQ = sW.vLayersQ8[iLayer];
        tQ = Q8::MatvecSimd(sQ.qWq, tNormed);
        tK = Q8::MatvecSimd(sQ.qWk, tNormed);
        tV = Q8::MatvecSimd(sQ.qWv, tNormed);
    } else {
        tQ = SM::Matvec(sLyr.tWq, tNormed);
        tK = SM::Matvec(sLyr.tWk, tNormed);
        tV = SM::Matvec(sLyr.tWv, tNormed);
    }

    AddBias(tQ, sLyr.tBq);
    AddBias(tK, sLyr.tBk);
    AddBias(tV, sLyr.tBv);

    if (bLayerUsesRope(iLayer))
        ApplyRope(tQ, tK, m_iPos);

    int32_t iPos = m_iPos;
    std::memcpy(sKv.tK.pfData() + (int64_t)iPos * iKvDim, tK.pfData(), iKvDim * sizeof(float));
    std::memcpy(sKv.tV.pfData() + (int64_t)iPos * iKvDim, tV.pfData(), iKvDim * sizeof(float));
    sKv.iLen = iPos + 1;

    int32_t iSeqLen = sKv.iLen;

    auto tOut = CTensor::Zeros({(int64_t)iQDim});
    float *pfOut = tOut.pfData();

    float fScale = 1.0f / std::sqrt((float)iHeadDim);

    for (int32_t iH = 0; iH < iNHeads; iH++) {
        int32_t iKvHead = iH / iGrpSize;
        float *pfQh = tQ.pfData() + iH * iHeadDim;

        std::vector<float> vfScores(iSeqLen);

        // dot product with all cached keys
        for (int32_t iT = 0; iT < iSeqLen; iT++) {
            float *pfKt = sKv.tK.pfData() + (int64_t)iT * iKvDim + iKvHead * iHeadDim;
            float fDot = 0.0f;
            for (int32_t i = 0; i < iHeadDim; i++)
                fDot += pfQh[i] * pfKt[i];
            vfScores[iT] = fDot * fScale;
        }

        float fMax = *std::max_element(vfScores.begin(), vfScores.end());
        float fSum = 0.0f;
        for (int32_t iT = 0; iT < iSeqLen; iT++) {
            vfScores[iT] = std::exp(vfScores[iT] - fMax);
            fSum += vfScores[iT];
        }
        float fInvSum = 1.0f / (fSum + 1e-10f);
        for (int32_t iT = 0; iT < iSeqLen; iT++)
            vfScores[iT] *= fInvSum;

        // weighted sum of values
        float *pfOutH = pfOut + iH * iHeadDim;
        for (int32_t i = 0; i < iHeadDim; i++)
            pfOutH[i] = 0.0f;

        for (int32_t iT = 0; iT < iSeqLen; iT++) {
            float fW = vfScores[iT];
            if (fW < 1e-8f)
                continue;
            float *pfVt = sKv.tV.pfData() + (int64_t)iT * iKvDim + iKvHead * iHeadDim;
            for (int32_t i = 0; i < iHeadDim; i++)
                pfOutH[i] += fW * pfVt[i];
        }
    }

    CTensor tAttnOut;
    if (sW.bQuantized)
        tAttnOut = Q8::MatvecSimd(sW.vLayersQ8[iLayer].qWo, tOut);
    else
        tAttnOut = SM::Matvec(sLyr.tWo, tOut);

    AddBias(tAttnOut, sLyr.tBo);

    return tAttnOut;
}
// >>>s_end(attention)

// <<<s_start(ffn)
// --- feed-forward
/*---------------------------------------------------------
 * FN: ForwardFfn
 * DESC: runs swiglu ff for one layer
 * PARMS: iLayer (layer index), tNormed (rms-normed input [dim])
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
auto CInferState::ForwardFfn(int32_t iLayer, const CTensor &tNormed) -> CTensor {
    const auto &sW = m_psModel->sWeights;
    const auto &sLyr = sW.vLayers[iLayer];

    CTensor tGate, tUp;

    if (sW.bQuantized) {
        const auto &sQ = sW.vLayersQ8[iLayer];
        tGate = Q8::MatvecSimd(sQ.qWGate, tNormed);
        tUp = Q8::MatvecSimd(sQ.qWUp, tNormed);
    } else {
        tGate = SM::Matvec(sLyr.tWGate, tNormed);
        tUp = SM::Matvec(sLyr.tWUp, tNormed);
    }

    AddBias(tGate, sLyr.tBGate);
    AddBias(tUp, sLyr.tBUp);

    // silu(gate) * up
    float *pfGate = tGate.pfData();
    float *pfUp = tUp.pfData();
    int64_t lHidden = tGate.lNumel();
    for (int64_t i = 0; i < lHidden; i++) {
        float fX = pfGate[i];
        pfGate[i] = fX / (1.0f + std::exp(-fX));
    }
    for (int64_t i = 0; i < lHidden; i++)
        pfGate[i] *= pfUp[i];

    // down proj
    CTensor tDown;
    if (sW.bQuantized)
        tDown = Q8::MatvecSimd(sW.vLayersQ8[iLayer].qWDown, tGate);
    else
        tDown = SM::Matvec(sLyr.tWDown, tGate);

    AddBias(tDown, sLyr.tBDown);

    return tDown;
}
// >>>s_end(ffn)

// <<<s_start(forward)
// --- forward pass
/*---------------------------------------------------------
 * FN: ForwardLayer
 * DESC: runs one transformer layer (attn + ffn sublayers)
 * PARMS: iLayer (layer index)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
void CInferState::ForwardLayer(int32_t iLayer) {
    const auto &sCfg = m_psModel->sConfig;
    const auto &sLyr = m_psModel->sWeights.vLayers[iLayer];

    int64_t lDim = (int64_t)sCfg.iDim;

    // attention sl
    {
        auto tXview = m_tX.Reshape({1, lDim});
        auto tNormed = SM::RmsNorm(tXview, sLyr.tAttnNorm, sCfg.fRmsEps);
        auto tNormedFlat = tNormed.Reshape({lDim}).Contiguous();

        auto tAttnOut = ForwardAttention(iLayer, tNormedFlat);

        float *pfX = m_tX.pfData();
        float *pfA = tAttnOut.pfData();
        for (int64_t i = 0; i < lDim; i++)
            pfX[i] += pfA[i];
    }

    // ffn sl
    {
        auto tXview = m_tX.Reshape({1, lDim});
        auto tNormed = SM::RmsNorm(tXview, sLyr.tFfnNorm, sCfg.fRmsEps);
        auto tNormedFlat = tNormed.Reshape({lDim}).Contiguous();

        auto tFfnOut = ForwardFfn(iLayer, tNormedFlat);

        float *pfX = m_tX.pfData();
        float *pfF = tFfnOut.pfData();
        for (int64_t i = 0; i < lDim; i++)
            pfX[i] += pfF[i];
    }
}

/*---------------------------------------------------------
 * FN: Forward
 * DESC: runs a single forward pass for one token at the
 *       current position & returns logits [vocab_size]
 * PARMS: iToken (input token id)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
auto CInferState::Forward(int32_t iToken) -> CTensor {
    const auto &sCfg = m_psModel->sConfig;
    const auto &sW = m_psModel->sWeights;
    int64_t lDim = (int64_t)sCfg.iDim;

    const float *pfEmbed = sW.tTokEmbed.pfData() + (int64_t)iToken * lDim;
    std::memcpy(m_tX.pfData(), pfEmbed, lDim * sizeof(float));

    for (int32_t iL = 0; iL < sCfg.iNLayers; iL++)
        ForwardLayer(iL);

    auto tXview = m_tX.Reshape({1, lDim});
    auto tNormed = SM::RmsNorm(tXview, sW.tOutputNorm, sCfg.fRmsEps);
    auto tNormedFlat = tNormed.Reshape({lDim}).Contiguous();

    auto tLogits = SM::Matvec(sW.tOutputProj, tNormedFlat);

    m_viHistory.push_back(iToken);
    m_iPos++;
    return tLogits;
}
// >>>s_end(forward)

// <<<s_start(sampling)
// --- sampling
/*---------------------------------------------------------
 * FN: iSample
 * DESC: samples next token from logits using temperature,
 *       top k, top p, and repetition penalty
 * PARMS: tLogits (vocab-sized logits tensor),
 *        sCfg (sampler config)
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
auto CInferState::iSample(const CTensor &tLogits, const SSamplerConfig &sCfg) -> int32_t {
    int64_t lVocab = tLogits.lNumel();
    const float *pfLogits = tLogits.pfData();

    std::vector<float> vfLogits(pfLogits, pfLogits + lVocab);

    // repp over last 64 toks
    if (sCfg.fRepPenalty > 1.0f && !m_viHistory.empty()) {
        int32_t iStart = (int32_t)m_viHistory.size() - 64;
        if (iStart < 0)
            iStart = 0;
        for (int32_t iH = iStart; iH < (int32_t)m_viHistory.size(); iH++) {
            int32_t iTok = m_viHistory[iH];
            if (iTok >= 0 && iTok < (int32_t)lVocab) {
                if (vfLogits[iTok] > 0.0f)
                    vfLogits[iTok] /= sCfg.fRepPenalty;
                else
                    vfLogits[iTok] *= sCfg.fRepPenalty;
            }
        }
    }

    // greedy
    if (sCfg.fTemperature < 1e-6f) {
        int32_t iBest = 0;
        float fBest = vfLogits[0];
        for (int64_t i = 1; i < lVocab; i++) {
            if (vfLogits[i] > fBest) {
                fBest = vfLogits[i];
                iBest = (int32_t)i;
            }
        }
        m_viHistory.push_back(iBest);
        return iBest;
    }

    struct SPair {
        int32_t iIdx;
        float fVal;
    };

    std::vector<SPair> vPairs(lVocab);
    for (int64_t i = 0; i < lVocab; i++) {
        vPairs[i].iIdx = (int32_t)i;
        vPairs[i].fVal = vfLogits[i] / sCfg.fTemperature;
    }

    // top k
    int32_t iK = sCfg.iTopK;
    if (iK > 0 && iK < (int32_t)lVocab) {
        std::partial_sort(vPairs.begin(), vPairs.begin() + iK, vPairs.end(),
                          [](const SPair &a, const SPair &b) { return a.fVal > b.fVal; });
        vPairs.resize(iK);
    } else {
        std::sort(vPairs.begin(), vPairs.end(), [](const SPair &a, const SPair &b) { return a.fVal > b.fVal; });
    }

    // softmax
    float fMax = vPairs[0].fVal;
    float fSum = 0.0f;
    for (auto &p : vPairs) {
        p.fVal = std::exp(p.fVal - fMax);
        fSum += p.fVal;
    }
    for (auto &p : vPairs)
        p.fVal /= fSum;

    // top p
    if (sCfg.fTopP > 0.0f && sCfg.fTopP < 1.0f) {
        float fCumul = 0.0f;
        int32_t iCutoff = (int32_t)vPairs.size();
        for (int32_t i = 0; i < (int32_t)vPairs.size(); i++) {
            fCumul += vPairs[i].fVal;
            if (fCumul >= sCfg.fTopP) {
                iCutoff = i + 1;
                break;
            }
        }
        vPairs.resize(iCutoff);

        fSum = 0.0f;
        for (auto &p : vPairs)
            fSum += p.fVal;
        for (auto &p : vPairs)
            p.fVal /= fSum;
    }

    float fR = fRandFloat();
    float fCumul = 0.0f;
    for (const auto &p : vPairs) {
        fCumul += p.fVal;
        if (fR < fCumul) {
            m_viHistory.push_back(p.iIdx);
            return p.iIdx;
        }
    }

    int32_t iResult = vPairs.back().iIdx;
    m_viHistory.push_back(iResult);
    return iResult;
}
// >>>s_end(sampling)
} // namespace MD
