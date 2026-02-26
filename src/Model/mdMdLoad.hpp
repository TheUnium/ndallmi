// Created by Unium on 24.02.26

#pragma once

#include "../Tensor/mtTnQnt8.hpp"
#include "../Tensor/mtTnTnsr.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace MD {
struct SModelConfig {
    int32_t iVocabSize = 0;
    int32_t iDim = 0;
    int32_t iHiddenDim = 0;
    int32_t iNLayers = 0;
    int32_t iNHeads = 0;
    int32_t iNKvHeads = 0;
    int32_t iHeadDim = 0;
    int32_t iMaxSeqLen = 2048;
    float fRmsEps = 1e-5f;
    float fRopeTheta = 10000.0f;
    bool bHasAttnBias = false;
    bool bHasMlpBias = false;
    std::string szArchitecture;

    // nope: pl flag 1=skip rope, 0=apply rope
    // empty means apply rope to all layers

    // tbh, rpoe should probably be applied to every layer anyways bc a lot
    // of config.json files have leftover stuff from training/testing
    // but just in case some models somehow use it for some reason im
    // keeping it
    //
    // eg: when testing on the older version (llmcpp) i remember smollm3-3b
    // was outputting total gibberish bc the config.json had rope applied to
    // only every 4th layer (ie: [0,0,0,1,0,0,0,1]) and when i ignored the
    // data given in config.json it became coherent again.
    std::vector<int32_t> viNoRopeLayers;
};

struct SLayerWeightsQ8 {
    MT::Q8::SQMatrix qWq;
    MT::Q8::SQMatrix qWk;
    MT::Q8::SQMatrix qWv;
    MT::Q8::SQMatrix qWo;
    MT::Q8::SQMatrix qWGate;
    MT::Q8::SQMatrix qWUp;
    MT::Q8::SQMatrix qWDown;
};

struct SLayerWeights {
    MT::CTensor tAttnNorm;
    MT::CTensor tWq;
    MT::CTensor tWk;
    MT::CTensor tWv;
    MT::CTensor tWo;
    MT::CTensor tBq;
    MT::CTensor tBk;
    MT::CTensor tBv;
    MT::CTensor tBo;
    MT::CTensor tFfnNorm;
    MT::CTensor tWGate;
    MT::CTensor tWUp;
    MT::CTensor tWDown;
    MT::CTensor tBGate;
    MT::CTensor tBUp;
    MT::CTensor tBDown;
};

struct SModelWeights {
    MT::CTensor tTokEmbed;
    MT::CTensor tOutputNorm;
    MT::CTensor tOutputProj;

    std::vector<SLayerWeights> vLayers;
    std::vector<SLayerWeightsQ8> vLayersQ8;

    bool bTiedEmbeddings = false;
    bool bQuantized = false;
};

struct SModel {
    SModelConfig sConfig;
    SModelWeights sWeights;
};

/*---------------------------------------------------------
 * FN: bLoadModel
 * DESC: loads a model from a huggingface directory containing
 *       config.json and safetensors file(s)
 * PARMS: szDirPath (path to model directory), sModel (output)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto bLoadModel(const std::string &szDirPath, SModel &sModel) -> bool;

/*---------------------------------------------------------
 * FN: QuantizeModel
 * DESC: quantizes all linear layer weights to int8 and
 *       frees the f32 copies to save memory
 * PARMS: sModel (model to quantize)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
void QuantizeModel(SModel &sModel);

/*---------------------------------------------------------
 * FN: PrintModelInfo
 * DESC: prints model config and weight shapes to stdout
 * PARMS: sModel (loaded model)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
void PrintModelInfo(const SModel &sModel);

} // namespace MD
