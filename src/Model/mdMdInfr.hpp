// Created by Unium on 25.02.26

#pragma once

#include "../Tensor/mtTnTnsr.hpp"
#include "mdMdLoad.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace MD {
struct SKvCache {
    MT::CTensor tK; // [max_seq, kv_dim]
    MT::CTensor tV; // [max_seq, kv_dim]
    int32_t iLen = 0;
};

struct SSamplerConfig {
    float fTemperature = 0.7f;
    float fTopP = 0.9f;
    int32_t iTopK = 40;
    int32_t iSeed = 42;
    float fRepPenalty = 1.2f;
};

class CInferState {
public:
    /*---------------------------------------------------------
     * FN: CInferState (ctor)
     * DESC: initializes inference state from a loaded model
     *       uses explicit head_dim from config for kv cache
     *       sizing
     * PARMS: psModel (pointer to loaded model)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    explicit CInferState(const SModel *psModel);

    /*---------------------------------------------------------
     * FN: Forward
     * DESC: runs a single forward pass for one token at the
     *       current position & returns logits [vocab_size]
     * PARMS: iToken (input token id)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto Forward(int32_t iToken) -> MT::CTensor;

    /*---------------------------------------------------------
     * FN: iSample
     * DESC: samples next token from logits using temperature,
     *       top k, top p, and repetition penalty
     * PARMS: tLogits (vocab-sized logits tensor),
     *        sCfg (sampler config)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto iSample(const MT::CTensor &tLogits, const SSamplerConfig &sCfg) -> int32_t;

    /*---------------------------------------------------------
     * FN: Reset
     * DESC: resets kv cache and pos to 0
     * PARMS: none
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    void Reset();

    /*---------------------------------------------------------
     * FN: iPos
     * DESC: returns current sequence position
     * PARMS: none
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto iPos() const -> int32_t { return m_iPos; }

private:
    const SModel *m_psModel;
    int32_t m_iPos = 0;
    std::vector<SKvCache> m_vKvCache;
    std::vector<int32_t> m_viHistory;

    MT::CTensor m_tX; // [dim] current hidden state
    uint64_t m_lRngState = 0;

    /*---------------------------------------------------------
     * FN: fRandFloat
     * DESC: returns random float in [0, 1) using xorshift64
     * PARMS: none
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto fRandFloat() -> float;

    /*---------------------------------------------------------
     * FN: ApplyRope
     * DESC: applies rotary position embeddings in-place
     *       uses configs head_dim for the rotation dimension
     * PARMS: tQ (query [n_heads * head_dim]), tK (key [n_kv_heads * head_dim]), iPos (current position)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    void ApplyRope(MT::CTensor &tQ, MT::CTensor &tK, int32_t iPos);

    /*---------------------------------------------------------
     * FN: AddBias
     * DESC: adds bias vector to tensor in place no-op if bias
     *       has no data (model doesnt use biases)
     * PARMS: tOut (tensor to modify), tBias (bias vector)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    void AddBias(MT::CTensor &tOut, const MT::CTensor &tBias);

    /*---------------------------------------------------------
     * FN: bLayerUsesRope
     * DESC: returns true if the given layer should apply rope
     *       viNoRopeLayers contains layer indices that skip rope
     * PARMS: iLayer (layer index)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto bLayerUsesRope(int32_t iLayer) const -> bool;

    /*---------------------------------------------------------
     * FN: ForwardLayer
     * DESC: runs one transformer layer (attn + ffn sublayers)
     * PARMS: iLayer (layer index)
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    void ForwardLayer(int32_t iLayer);

    /*---------------------------------------------------------
     * FN: ForwardAttention
     * DESC: runs grouped query attention for one layer
     * PARMS: iLayer (layer index), tNormed (rms-normed input [dim])
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto ForwardAttention(int32_t iLayer, const MT::CTensor &tNormed) -> MT::CTensor;

    /*---------------------------------------------------------
     * FN: ForwardFfn
     * DESC: runs swiglu ff for one layer
     * PARMS: iLayer (layer index), tNormed (rms-normed input [dim])
     * AUTH: unium (25.02.26)
     *-------------------------------------------------------*/
    auto ForwardFfn(int32_t iLayer, const MT::CTensor &tNormed) -> MT::CTensor;
};

} // namespace MD
