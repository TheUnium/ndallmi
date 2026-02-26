// Created by Unium on 24.02.26

#include "mdMdLoad.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace MD {

// <<<s_start(json)
// --- json helpers
/*---------------------------------------------------------
 * FN: szReadFileToString
 * DESC: reads file into a str
 * PARMS: szPath (file path)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto szReadFileToString(const std::string &szPath) -> std::string {
    std::ifstream ifs(szPath, std::ios::binary);
    if (!ifs.is_open())
        return "";
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

/*---------------------------------------------------------
 * FN: szExtractJsonString
 * DESC: extracts a str val for a key from json
 *       NOTE: this is very retarded and only works for
 *       flat config.json fields
 * PARMS: szJson (json text), szKey (key name)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto szExtractJsonString(const std::string &szJson, const std::string &szKey) -> std::string {
    std::string szSearch = "\"" + szKey + "\"";
    size_t lPos = szJson.find(szSearch);
    if (lPos == std::string::npos)
        return "";

    lPos = szJson.find(':', lPos + szSearch.size());
    if (lPos == std::string::npos)
        return "";
    lPos++;

    while (lPos < szJson.size() &&
           (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' || szJson[lPos] == '\r'))
        lPos++;

    if (lPos >= szJson.size() || szJson[lPos] != '"')
        return "";

    lPos++;
    size_t lEnd = szJson.find('"', lPos);
    if (lEnd == std::string::npos)
        return "";
    return szJson.substr(lPos, lEnd - lPos);
}

/*---------------------------------------------------------
 * FN: iExtractJsonInt
 * DESC: extracts a int val for a key from json
 * PARMS: szJson (json text), szKey (key name), iDefault (fallback)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto iExtractJsonInt(const std::string &szJson, const std::string &szKey, int32_t iDefault = 0) -> int32_t {
    std::string szSearch = "\"" + szKey + "\"";
    size_t lPos = szJson.find(szSearch);
    if (lPos == std::string::npos)
        return iDefault;

    lPos = szJson.find(':', lPos + szSearch.size());
    if (lPos == std::string::npos)
        return iDefault;
    lPos++;

    while (lPos < szJson.size() &&
           (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' || szJson[lPos] == '\r'))
        lPos++;

    if (lPos >= szJson.size())
        return iDefault;

    std::string szNum;
    if (szJson[lPos] == '-') {
        szNum += '-';
        lPos++;
    }
    while (lPos < szJson.size() && szJson[lPos] >= '0' && szJson[lPos] <= '9')
        szNum += szJson[lPos++];

    if (szNum.empty() || szNum == "-")
        return iDefault;
    return std::stoi(szNum);
}

/*---------------------------------------------------------
 * FN: fExtractJsonFloat
 * DESC: extracts a float value for a given key from json
 * PARMS: szJson (json text), szKey (key name), fDefault (fallback)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto fExtractJsonFloat(const std::string &szJson, const std::string &szKey, float fDefault = 0.0f) -> float {
    std::string szSearch = "\"" + szKey + "\"";
    size_t lPos = szJson.find(szSearch);
    if (lPos == std::string::npos)
        return fDefault;

    lPos = szJson.find(':', lPos + szSearch.size());
    if (lPos == std::string::npos)
        return fDefault;
    lPos++;

    while (lPos < szJson.size() &&
           (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' || szJson[lPos] == '\r'))
        lPos++;

    if (lPos >= szJson.size())
        return fDefault;

    std::string szNum;
    while (lPos < szJson.size() &&
           (szJson[lPos] == '-' || szJson[lPos] == '.' || szJson[lPos] == 'e' || szJson[lPos] == 'E' ||
            szJson[lPos] == '+' || (szJson[lPos] >= '0' && szJson[lPos] <= '9')))
        szNum += szJson[lPos++];

    if (szNum.empty())
        return fDefault;
    return std::stof(szNum);
}

/*---------------------------------------------------------
 * FN: bExtractJsonBool
 * DESC: extracts a boolean value for a given key from json
 * PARMS: szJson (json text), szKey (key name), bDefault (fallback)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto bExtractJsonBool(const std::string &szJson, const std::string &szKey, bool bDefault = false) -> bool {
    std::string szSearch = "\"" + szKey + "\"";
    size_t lPos = szJson.find(szSearch);
    if (lPos == std::string::npos)
        return bDefault;

    lPos = szJson.find(':', lPos + szSearch.size());
    if (lPos == std::string::npos)
        return bDefault;
    lPos++;

    while (lPos < szJson.size() &&
           (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' || szJson[lPos] == '\r'))
        lPos++;

    if (lPos + 4 <= szJson.size() && szJson.compare(lPos, 4, "true") == 0)
        return true;
    if (lPos + 5 <= szJson.size() && szJson.compare(lPos, 5, "false") == 0)
        return false;
    return bDefault;
}

/*---------------------------------------------------------
 * FN: szExtractJsonArrayFirst
 * DESC: extracts the first string element from a json array
 * PARMS: szJson (json text), szKey (key name)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto szExtractJsonArrayFirst(const std::string &szJson, const std::string &szKey) -> std::string {
    std::string szSearch = "\"" + szKey + "\"";
    size_t lPos = szJson.find(szSearch);
    if (lPos == std::string::npos)
        return "";

    lPos = szJson.find('[', lPos);
    if (lPos == std::string::npos)
        return "";

    lPos = szJson.find('"', lPos);
    if (lPos == std::string::npos)
        return "";
    lPos++;

    size_t lEnd = szJson.find('"', lPos);
    if (lEnd == std::string::npos)
        return "";
    return szJson.substr(lPos, lEnd - lPos);
}

/*---------------------------------------------------------
 * FN: viExtractJsonIntArray
 * DESC: extracts an array of integers for a given key.
 *       returns empty vector if key not found
 * PARMS: szJson (json text), szKey (key name)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto viExtractJsonIntArray(const std::string &szJson, const std::string &szKey) -> std::vector<int32_t> {
    std::vector<int32_t> viResult;

    std::string szSearch = "\"" + szKey + "\"";
    size_t lPos = szJson.find(szSearch);
    if (lPos == std::string::npos)
        return viResult;

    lPos = szJson.find(':', lPos + szSearch.size());
    if (lPos == std::string::npos)
        return viResult;
    lPos++;

    while (lPos < szJson.size() &&
           (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' || szJson[lPos] == '\r'))
        lPos++;

    if (lPos >= szJson.size() || szJson[lPos] != '[')
        return viResult;
    lPos++;

    while (lPos < szJson.size() && szJson[lPos] != ']') {
        while (lPos < szJson.size() && (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' ||
                                        szJson[lPos] == '\r' || szJson[lPos] == ','))
            lPos++;

        if (lPos >= szJson.size() || szJson[lPos] == ']')
            break;

        std::string szNum;
        if (szJson[lPos] == '-') {
            szNum += '-';
            lPos++;
        }
        while (lPos < szJson.size() && szJson[lPos] >= '0' && szJson[lPos] <= '9')
            szNum += szJson[lPos++];

        if (!szNum.empty() && szNum != "-")
            viResult.push_back(std::stoi(szNum));
    }

    return viResult;
}
// >>>s_end(json)

// <<<s_start(config)
// --- config loading
/*---------------------------------------------------------
 * FN: bLoadConfig
 * DESC: parses config.json and fills SModelConfig
 * PARMS: szPath (path to config.json), sConfig (output)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto bLoadConfig(const std::string &szPath, SModelConfig &sConfig) -> bool {
    std::string szJson = szReadFileToString(szPath);
    if (szJson.empty()) {
        std::cerr << "  error: cannot read " << szPath << std::endl;
        return false;
    }

    sConfig.szArchitecture = szExtractJsonArrayFirst(szJson, "architectures");
    if (sConfig.szArchitecture.empty())
        sConfig.szArchitecture = szExtractJsonString(szJson, "model_type");

    sConfig.iVocabSize = iExtractJsonInt(szJson, "vocab_size", 32000);
    sConfig.iDim = iExtractJsonInt(szJson, "hidden_size", 4096);
    sConfig.iNLayers = iExtractJsonInt(szJson, "num_hidden_layers", 32);
    sConfig.iNHeads = iExtractJsonInt(szJson, "num_attention_heads", 32);
    sConfig.iNKvHeads = iExtractJsonInt(szJson, "num_key_value_heads", 0);
    sConfig.iMaxSeqLen = iExtractJsonInt(szJson, "max_position_embeddings", 2048);

    sConfig.iHiddenDim = iExtractJsonInt(szJson, "intermediate_size", 0);
    if (sConfig.iHiddenDim == 0)
        sConfig.iHiddenDim = 4 * sConfig.iDim;

    sConfig.fRmsEps = fExtractJsonFloat(szJson, "rms_norm_eps", 1e-5f);
    sConfig.fRopeTheta = fExtractJsonFloat(szJson, "rope_theta", 10000.0f);

    if (sConfig.iNKvHeads <= 0)
        sConfig.iNKvHeads = sConfig.iNHeads;

    sConfig.iHeadDim = iExtractJsonInt(szJson, "head_dim", 0);
    if (sConfig.iHeadDim <= 0)
        sConfig.iHeadDim = sConfig.iDim / sConfig.iNHeads;

    sConfig.bHasAttnBias = bExtractJsonBool(szJson, "attention_bias", false);
    sConfig.bHasMlpBias = bExtractJsonBool(szJson, "mlp_bias", false);

    sConfig.viNoRopeLayers.clear();

    int32_t iNopeCount = 0;
    for (int32_t v : sConfig.viNoRopeLayers) {
        if (v != 0)
            iNopeCount++;
    }

    std::cout << "  config loaded from " << szPath << std::endl;
    std::cout << "  head_dim=" << sConfig.iHeadDim << " attn_bias=" << (sConfig.bHasAttnBias ? "yes" : "no")
              << " mlp_bias=" << (sConfig.bHasMlpBias ? "yes" : "no") << " rope_theta=" << sConfig.fRopeTheta
              << " rms_eps=" << sConfig.fRmsEps << std::endl;

    if (!sConfig.viNoRopeLayers.empty()) {
        std::cout << "  NoPE: " << iNopeCount << "/" << sConfig.iNLayers << " layers skip RoPE (3:1 NoPE ratio)"
                  << std::endl;
    }

    return true;
}
// >>>s_end(config)

// <<<s_start(safetensors)
// --- safetensors parsing
struct SSafeTensorInfo {
    std::string szDtype;
    std::vector<int64_t> vlShape;
    int64_t lDataStart = 0;
    int64_t lDataEnd = 0;
};

/*---------------------------------------------------------
 * FN: lParseJsonNumber
 * DESC: parses a number from json at given position
 * PARMS: szJson, lPos (in/out)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto lParseJsonNumber(const std::string &szJson, size_t &lPos) -> int64_t {
    std::string szNum;
    if (lPos < szJson.size() && szJson[lPos] == '-') {
        szNum += '-';
        lPos++;
    }
    while (lPos < szJson.size() && szJson[lPos] >= '0' && szJson[lPos] <= '9')
        szNum += szJson[lPos++];
    return szNum.empty() ? 0 : std::stoll(szNum);
}

/*---------------------------------------------------------
 * FN: szParseJsonString
 * DESC: parses a quoted string from json at given position
 * PARMS: szJson, lPos (in/out)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto szParseJsonString(const std::string &szJson, size_t &lPos) -> std::string {
    if (lPos >= szJson.size() || szJson[lPos] != '"')
        return "";
    lPos++;
    std::string szResult;
    while (lPos < szJson.size() && szJson[lPos] != '"') {
        if (szJson[lPos] == '\\' && lPos + 1 < szJson.size()) {
            lPos++;
            szResult += szJson[lPos];
        } else {
            szResult += szJson[lPos];
        }
        lPos++;
    }
    if (lPos < szJson.size())
        lPos++;
    return szResult;
}

/*---------------------------------------------------------
 * FN: SkipWs
 * DESC: advances position past whitespace
 * PARMS: szJson, lPos (in/out)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static void SkipWs(const std::string &szJson, size_t &lPos) {
    while (lPos < szJson.size() &&
           (szJson[lPos] == ' ' || szJson[lPos] == '\t' || szJson[lPos] == '\n' || szJson[lPos] == '\r'))
        lPos++;
}

/*---------------------------------------------------------
 * FN: SkipJsonValue
 * DESC: skips over any json value at position
 * PARMS: szJson, lPos (in/out)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static void SkipJsonValue(const std::string &szJson, size_t &lPos) {
    SkipWs(szJson, lPos);
    if (lPos >= szJson.size())
        return;

    char c = szJson[lPos];
    if (c == '"') {
        szParseJsonString(szJson, lPos);
    } else if (c == '{') {
        lPos++;
        int iDepth = 1;
        while (lPos < szJson.size() && iDepth > 0) {
            if (szJson[lPos] == '"')
                szParseJsonString(szJson, lPos);
            else {
                if (szJson[lPos] == '{')
                    iDepth++;
                else if (szJson[lPos] == '}')
                    iDepth--;
                lPos++;
            }
        }
    } else if (c == '[') {
        lPos++;
        int iDepth = 1;
        while (lPos < szJson.size() && iDepth > 0) {
            if (szJson[lPos] == '"')
                szParseJsonString(szJson, lPos);
            else {
                if (szJson[lPos] == '[')
                    iDepth++;
                else if (szJson[lPos] == ']')
                    iDepth--;
                lPos++;
            }
        }
    } else {
        while (lPos < szJson.size() && szJson[lPos] != ',' && szJson[lPos] != '}' && szJson[lPos] != ']' &&
               szJson[lPos] != ' ' && szJson[lPos] != '\n' && szJson[lPos] != '\r' && szJson[lPos] != '\t')
            lPos++;
    }
}

/*---------------------------------------------------------
 * FN: ParseTensorInfo
 * DESC: parses a single tensors info object from the header
 * PARMS: szJson, lPos (in/out)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto ParseTensorInfo(const std::string &szJson, size_t &lPos) -> SSafeTensorInfo {
    SSafeTensorInfo sInfo;

    SkipWs(szJson, lPos);
    if (lPos >= szJson.size() || szJson[lPos] != '{')
        return sInfo;
    lPos++;

    while (lPos < szJson.size() && szJson[lPos] != '}') {
        SkipWs(szJson, lPos);
        if (szJson[lPos] == '}')
            break;
        if (szJson[lPos] == ',') {
            lPos++;
            continue;
        }

        std::string szFieldKey = szParseJsonString(szJson, lPos);
        SkipWs(szJson, lPos);
        if (lPos < szJson.size() && szJson[lPos] == ':')
            lPos++;
        SkipWs(szJson, lPos);

        if (szFieldKey == "dtype") {
            sInfo.szDtype = szParseJsonString(szJson, lPos);
        } else if (szFieldKey == "shape") {
            if (lPos < szJson.size() && szJson[lPos] == '[') {
                lPos++;
                while (lPos < szJson.size() && szJson[lPos] != ']') {
                    SkipWs(szJson, lPos);
                    if (szJson[lPos] == ',' || szJson[lPos] == ']') {
                        if (szJson[lPos] == ',')
                            lPos++;
                        continue;
                    }
                    sInfo.vlShape.push_back(lParseJsonNumber(szJson, lPos));
                    SkipWs(szJson, lPos);
                    if (lPos < szJson.size() && szJson[lPos] == ',')
                        lPos++;
                }
                if (lPos < szJson.size())
                    lPos++;
            }
        } else if (szFieldKey == "data_offsets") {
            if (lPos < szJson.size() && szJson[lPos] == '[') {
                lPos++;
                SkipWs(szJson, lPos);
                sInfo.lDataStart = lParseJsonNumber(szJson, lPos);
                SkipWs(szJson, lPos);
                if (lPos < szJson.size() && szJson[lPos] == ',')
                    lPos++;
                SkipWs(szJson, lPos);
                sInfo.lDataEnd = lParseJsonNumber(szJson, lPos);
                SkipWs(szJson, lPos);
                if (lPos < szJson.size() && szJson[lPos] == ']')
                    lPos++;
            }
        } else {
            SkipJsonValue(szJson, lPos);
        }
    }
    if (lPos < szJson.size())
        lPos++;

    return sInfo;
}

/*---------------------------------------------------------
 * FN: ParseSafetensorsHeader
 * DESC: parses the full safetensors json header into a map
 * PARMS: szHeader (json header text)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto ParseSafetensorsHeader(const std::string &szHeader) -> std::unordered_map<std::string, SSafeTensorInfo> {

    std::unordered_map<std::string, SSafeTensorInfo> mTensors;
    size_t lPos = 0;

    SkipWs(szHeader, lPos);
    if (lPos >= szHeader.size() || szHeader[lPos] != '{')
        return mTensors;
    lPos++;

    while (lPos < szHeader.size() && szHeader[lPos] != '}') {
        SkipWs(szHeader, lPos);
        if (szHeader[lPos] == '}')
            break;
        if (szHeader[lPos] == ',') {
            lPos++;
            continue;
        }

        std::string szTensorName = szParseJsonString(szHeader, lPos);
        SkipWs(szHeader, lPos);
        if (lPos < szHeader.size() && szHeader[lPos] == ':')
            lPos++;
        SkipWs(szHeader, lPos);

        if (szTensorName == "__metadata__") {
            SkipJsonValue(szHeader, lPos);
            continue;
        }

        SSafeTensorInfo sInfo = ParseTensorInfo(szHeader, lPos);
        mTensors[szTensorName] = sInfo;
    }

    return mTensors;
}
// >>>s_end(safetensors)

// <<<s_start(dtype)
// --- dtype conversion
/*---------------------------------------------------------
 * FN: fBF16ToF32
 * DESC: converts a single bfloat16 value to float32
 * PARMS: u16 (raw bfloat16 bits)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto fBF16ToF32(uint16_t u16) -> float {
    uint32_t u32 = (uint32_t)u16 << 16;
    float fVal;
    std::memcpy(&fVal, &u32, sizeof(float));
    return fVal;
}

/*---------------------------------------------------------
 * FN: fFP16ToF32
 * DESC: converts a single float16 value to float32
 * PARMS: u16 (raw float16 bits)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto fFP16ToF32(uint16_t u16) -> float {
    uint32_t iSign = (u16 >> 15) & 0x1;
    uint32_t iExp = (u16 >> 10) & 0x1F;
    uint32_t iMant = u16 & 0x3FF;

    if (iExp == 0) {
        if (iMant == 0) {
            uint32_t u32 = iSign << 31;
            float fVal;
            std::memcpy(&fVal, &u32, sizeof(float));
            return fVal;
        }
        float fVal = std::ldexp((float)iMant, -24);
        return iSign ? -fVal : fVal;
    }

    if (iExp == 0x1F) {
        uint32_t u32 = (iSign << 31) | 0x7F800000 | (iMant << 13);
        float fVal;
        std::memcpy(&fVal, &u32, sizeof(float));
        return fVal;
    }

    uint32_t u32 = (iSign << 31) | ((iExp - 15 + 127) << 23) | (iMant << 13);
    float fVal;
    std::memcpy(&fVal, &u32, sizeof(float));
    return fVal;
}

/*---------------------------------------------------------
 * FN: LoadTensorFromData
 * DESC: creates a CTensor from raw safetensors data with
 *       dtype conversion to f32 if needed
 * PARMS: pData (pointer into buffer), sInfo (tensor metadata)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto LoadTensorFromData(const uint8_t *pData, const SSafeTensorInfo &sInfo) -> MT::CTensor {
    std::vector<int64_t> vlShape = sInfo.vlShape;
    if (vlShape.empty())
        vlShape.push_back(1);

    MT::CTensor tResult(vlShape);
    int64_t lNumel = tResult.lNumel();
    float *pfDst = tResult.pfData();

    const uint8_t *pSrc = pData + sInfo.lDataStart;

    if (sInfo.szDtype == "F32") {
        std::memcpy(pfDst, pSrc, lNumel * sizeof(float));
    } else if (sInfo.szDtype == "BF16") {
        const uint16_t *pU16 = reinterpret_cast<const uint16_t *>(pSrc);
        for (int64_t i = 0; i < lNumel; i++)
            pfDst[i] = fBF16ToF32(pU16[i]);
    } else if (sInfo.szDtype == "F16") {
        const uint16_t *pU16 = reinterpret_cast<const uint16_t *>(pSrc);
        for (int64_t i = 0; i < lNumel; i++)
            pfDst[i] = fFP16ToF32(pU16[i]);
    } else {
        std::cerr << "  warning: unsupported dtype '" << sInfo.szDtype << "', filling zeros" << std::endl;
    }

    return tResult;
}
// >>>s_end(dtype)

// <<<s_start(fileload)
// --- safetensors file loading
struct SSafetensorsFile {
    std::unordered_map<std::string, SSafeTensorInfo> mTensors;
    std::vector<uint8_t> vData;
};

/*---------------------------------------------------------
 * FN: bLoadSafetensorsFile
 * DESC: loads and parses a single .safetensors file
 * PARMS: szPath (file path), sFile (output)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto bLoadSafetensorsFile(const std::string &szPath, SSafetensorsFile &sFile) -> bool {
    std::ifstream ifs(szPath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "  error: cannot open " << szPath << std::endl;
        return false;
    }

    uint64_t lHeaderSize = 0;
    ifs.read(reinterpret_cast<char *>(&lHeaderSize), 8);
    if (!ifs.good()) {
        std::cerr << "  error: cannot read header size from " << szPath << std::endl;
        return false;
    }

    if (lHeaderSize > 100 * 1024 * 1024) {
        std::cerr << "  error: header too large (" << lHeaderSize << " bytes)" << std::endl;
        return false;
    }

    std::string szHeader(lHeaderSize, '\0');
    ifs.read(&szHeader[0], lHeaderSize);
    if (!ifs.good()) {
        std::cerr << "  error: cannot read header from " << szPath << std::endl;
        return false;
    }

    sFile.mTensors = ParseSafetensorsHeader(szHeader);

    auto lDataStart = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    auto lFileSize = ifs.tellg();
    int64_t lDataSize = (int64_t)lFileSize - (int64_t)lDataStart;

    if (lDataSize > 0) {
        sFile.vData.resize(lDataSize);
        ifs.seekg(lDataStart);
        ifs.read(reinterpret_cast<char *>(sFile.vData.data()), lDataSize);
    }

    std::cout << "  loaded " << szPath << " (" << sFile.mTensors.size() << " tensors, " << (lDataSize / (1024 * 1024))
              << " MB data)" << std::endl;

    return true;
}
// >>>s_end(fileload)

// <<<s_start(weightmap)
// --- weight name mapping
/*---------------------------------------------------------
 * FN: szLayerKey
 * DESC: builds a huggingface style layer weight key
 * PARMS: iLayer (layer index), szSuffix
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto szLayerKey(int32_t iLayer, const std::string &szSuffix) -> std::string {
    return "model.layers." + std::to_string(iLayer) + "." + szSuffix;
}

/*---------------------------------------------------------
 * FN: bFindAndLoadTensor
 * DESC: looks up a tensor by name across all loaded safetensors
 *       files and returns it as a CTensor
 * PARMS: vFiles (loaded files), szName (tensor name),
 *        tOut (output tensor), bRequired (warn if missing)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto bFindAndLoadTensor(const std::vector<SSafetensorsFile> &vFiles, const std::string &szName,
                               MT::CTensor &tOut, bool bRequired = true) -> bool {
    for (const auto &sFile : vFiles) {
        auto it = sFile.mTensors.find(szName);
        if (it != sFile.mTensors.end()) {
            tOut = LoadTensorFromData(sFile.vData.data(), it->second);
            return true;
        }
    }

    if (bRequired)
        std::cerr << "  warning: tensor '" << szName << "' not found" << std::endl;
    return false;
}
// >>>s_end(weightmap)

// <<<s_start(api)
// --- public api
/*---------------------------------------------------------
 * FN: bLoadModel
 * DESC: loads a model from a huggingface directory containing
 *       config.json and safetensors file(s)
 * PARMS: szDirPath (path to model directory), sModel (output)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
auto bLoadModel(const std::string &szDirPath, SModel &sModel) -> bool {
    std::cout << "  loading model from: " << szDirPath << std::endl;

    std::string szConfigPath = szDirPath + "/config.json";
    if (!bLoadConfig(szConfigPath, sModel.sConfig))
        return false;

    const auto &sCfg = sModel.sConfig;
    std::cout << "  architecture: " << sCfg.szArchitecture << std::endl;
    std::cout << "  dim=" << sCfg.iDim << " layers=" << sCfg.iNLayers << " heads=" << sCfg.iNHeads
              << " kv_heads=" << sCfg.iNKvHeads << " head_dim=" << sCfg.iHeadDim << " vocab=" << sCfg.iVocabSize
              << std::endl;

    std::vector<std::string> vszFiles;
    for (const auto &entry : fs::directory_iterator(szDirPath)) {
        if (entry.path().extension() == ".safetensors")
            vszFiles.push_back(entry.path().string());
    }

    if (vszFiles.empty()) {
        std::cerr << "  error: no .safetensors files found in " << szDirPath << std::endl;
        return false;
    }

    std::sort(vszFiles.begin(), vszFiles.end());
    std::cout << "  found " << vszFiles.size() << " safetensors file(s)" << std::endl;

    std::vector<SSafetensorsFile> vFiles(vszFiles.size());
    for (size_t i = 0; i < vszFiles.size(); i++) {
        if (!bLoadSafetensorsFile(vszFiles[i], vFiles[i]))
            return false;
    }

    size_t lTotalTensors = 0;
    for (const auto &f : vFiles)
        lTotalTensors += f.mTensors.size();
    std::cout << "  total tensors in file(s): " << lTotalTensors << std::endl;

    auto &sW = sModel.sWeights;
    if (!bFindAndLoadTensor(vFiles, "model.embed_tokens.weight", sW.tTokEmbed))
        return false;
    std::cout << "  loaded embed_tokens: [" << sW.tTokEmbed.m_lShape[0] << ", " << sW.tTokEmbed.m_lShape[1] << "]"
              << std::endl;

    if (!bFindAndLoadTensor(vFiles, "model.norm.weight", sW.tOutputNorm))
        return false;

    if (!bFindAndLoadTensor(vFiles, "lm_head.weight", sW.tOutputProj, false)) {
        std::cout << "  lm_head not found, using tied embeddings" << std::endl;
        sW.bTiedEmbeddings = true;
        sW.tOutputProj = MT::CTensor();
        sW.tOutputProj.m_iNdim = sW.tTokEmbed.m_iNdim;
        for (int d = 0; d < sW.tTokEmbed.m_iNdim; d++) {
            sW.tOutputProj.m_lShape[d] = sW.tTokEmbed.m_lShape[d];
            sW.tOutputProj.m_lStride[d] = sW.tTokEmbed.m_lStride[d];
        }
        sW.tOutputProj.m_pData = sW.tTokEmbed.m_pData;
        sW.tOutputProj.m_iDataSize = sW.tTokEmbed.m_iDataSize;
        sW.tOutputProj.m_bOwnsData = false;
        sW.tOutputProj.m_eType = sW.tTokEmbed.m_eType;
    }
    sW.vLayers.resize(sCfg.iNLayers);

    for (int32_t iL = 0; iL < sCfg.iNLayers; iL++) {
        auto &sLyr = sW.vLayers[iL];

        bool bOk = true;
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "input_layernorm.weight"), sLyr.tAttnNorm);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.q_proj.weight"), sLyr.tWq);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.k_proj.weight"), sLyr.tWk);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.v_proj.weight"), sLyr.tWv);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.o_proj.weight"), sLyr.tWo);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "post_attention_layernorm.weight"), sLyr.tFfnNorm);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "mlp.gate_proj.weight"), sLyr.tWGate);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "mlp.up_proj.weight"), sLyr.tWUp);
        bOk &= bFindAndLoadTensor(vFiles, szLayerKey(iL, "mlp.down_proj.weight"), sLyr.tWDown);

        if (!bOk) {
            std::cerr << "  error: missing weights for layer " << iL << std::endl;
            return false;
        }
        if (sCfg.bHasAttnBias) {
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.q_proj.bias"), sLyr.tBq, false);
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.k_proj.bias"), sLyr.tBk, false);
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.v_proj.bias"), sLyr.tBv, false);
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "self_attn.o_proj.bias"), sLyr.tBo, false);
        }
        if (sCfg.bHasMlpBias) {
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "mlp.gate_proj.bias"), sLyr.tBGate, false);
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "mlp.up_proj.bias"), sLyr.tBUp, false);
            bFindAndLoadTensor(vFiles, szLayerKey(iL, "mlp.down_proj.bias"), sLyr.tBDown, false);
        }

        bool bUsesRope = true;
        if (iL < (int32_t)sCfg.viNoRopeLayers.size())
            bUsesRope = (sCfg.viNoRopeLayers[iL] == 0);

        std::cout << "  loaded layer " << iL << "/" << sCfg.iNLayers << " (Wq: [" << sLyr.tWq.m_lShape[0] << ","
                  << sLyr.tWq.m_lShape[1] << "]"
                  << " Wk: [" << sLyr.tWk.m_lShape[0] << "," << sLyr.tWk.m_lShape[1] << "]"
                  << (bUsesRope ? " RoPE" : " NoPE") << ")" << std::endl;
    }

    std::cout << "  model loaded successfully!" << std::endl;
    return true;
}

/*---------------------------------------------------------
 * FN: PrintModelInfo
 * DESC: prints model config and weight shapes to stdout
 * PARMS: sModel (loaded model)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
void PrintModelInfo(const SModel &sModel) {
    const auto &sCfg = sModel.sConfig;
    const auto &sW = sModel.sWeights;

    std::cout << "--- model info" << std::endl;
    std::cout << "      architecture:   " << sCfg.szArchitecture << std::endl;
    std::cout << "      vocab_size:     " << sCfg.iVocabSize << std::endl;
    std::cout << "      hidden_size:    " << sCfg.iDim << std::endl;
    std::cout << "      intermediate:   " << sCfg.iHiddenDim << std::endl;
    std::cout << "      num_layers:     " << sCfg.iNLayers << std::endl;
    std::cout << "      num_heads:      " << sCfg.iNHeads << std::endl;
    std::cout << "      num_kv_heads:   " << sCfg.iNKvHeads << std::endl;
    std::cout << "      head_dim:       " << sCfg.iHeadDim << std::endl;
    std::cout << "      max_seq_len:    " << sCfg.iMaxSeqLen << std::endl;
    std::cout << "      rms_norm_eps:   " << sCfg.fRmsEps << std::endl;
    std::cout << "      rope_theta:     " << sCfg.fRopeTheta << std::endl;
    std::cout << "      attn_bias:      " << (sCfg.bHasAttnBias ? "yes" : "no") << std::endl;
    std::cout << "      mlp_bias:       " << (sCfg.bHasMlpBias ? "yes" : "no") << std::endl;
    std::cout << "      tied_embed:     " << (sW.bTiedEmbeddings ? "yes" : "no") << std::endl;

    if (!sCfg.viNoRopeLayers.empty()) {
        int32_t iNopeCount = 0;
        for (int32_t v : sCfg.viNoRopeLayers) {
            if (v != 0)
                iNopeCount++;
        }
        std::cout << "      NoPE layers:    " << iNopeCount << "/" << sCfg.iNLayers << std::endl;
    }

    int64_t lParams = 0;
    lParams += sW.tTokEmbed.lNumel();
    lParams += sW.tOutputNorm.lNumel();
    if (!sW.bTiedEmbeddings)
        lParams += sW.tOutputProj.lNumel();

    for (const auto &sLyr : sW.vLayers) {
        lParams += sLyr.tAttnNorm.lNumel();
        lParams += sLyr.tWq.lNumel();
        lParams += sLyr.tWk.lNumel();
        lParams += sLyr.tWv.lNumel();
        lParams += sLyr.tWo.lNumel();
        lParams += sLyr.tFfnNorm.lNumel();
        lParams += sLyr.tWGate.lNumel();
        lParams += sLyr.tWUp.lNumel();
        lParams += sLyr.tWDown.lNumel();
        if (sLyr.tBq.lNumel() > 0)
            lParams += sLyr.tBq.lNumel();
        if (sLyr.tBk.lNumel() > 0)
            lParams += sLyr.tBk.lNumel();
        if (sLyr.tBv.lNumel() > 0)
            lParams += sLyr.tBv.lNumel();
        if (sLyr.tBo.lNumel() > 0)
            lParams += sLyr.tBo.lNumel();
        if (sLyr.tBGate.lNumel() > 0)
            lParams += sLyr.tBGate.lNumel();
        if (sLyr.tBUp.lNumel() > 0)
            lParams += sLyr.tBUp.lNumel();
        if (sLyr.tBDown.lNumel() > 0)
            lParams += sLyr.tBDown.lNumel();
    }

    double dParamsM = (double)lParams / 1e6;
    double dSizeMb = (double)(lParams * 4) / (1024.0 * 1024.0);
    std::cout << "      total params:   " << lParams << " (" << dParamsM << "M)" << std::endl;
    std::cout << "      f32 size:       " << dSizeMb << " MB" << std::endl;

    std::cout << std::endl;
    std::cout << "      weight shapes:" << std::endl;
    std::cout << "          embed_tokens:   [" << sW.tTokEmbed.m_lShape[0] << ", " << sW.tTokEmbed.m_lShape[1] << "]"
              << std::endl;
    std::cout << "      output_norm:    [" << sW.tOutputNorm.m_lShape[0] << "]" << std::endl;
    std::cout << "      output_proj:    [" << sW.tOutputProj.m_lShape[0] << ", " << sW.tOutputProj.m_lShape[1] << "]"
              << (sW.bTiedEmbeddings ? " (tied)" : "") << std::endl;

    if (!sW.vLayers.empty()) {
        const auto &sL0 = sW.vLayers[0];
        std::cout << "      layer 0:" << std::endl;
        std::cout << "          attn_norm:  [" << sL0.tAttnNorm.m_lShape[0] << "]" << std::endl;
        std::cout << "          wq:         [" << sL0.tWq.m_lShape[0] << ", " << sL0.tWq.m_lShape[1] << "]"
                  << std::endl;
        std::cout << "          wk:         [" << sL0.tWk.m_lShape[0] << ", " << sL0.tWk.m_lShape[1] << "]"
                  << std::endl;
        std::cout << "          wv:         [" << sL0.tWv.m_lShape[0] << ", " << sL0.tWv.m_lShape[1] << "]"
                  << std::endl;
        std::cout << "          wo:         [" << sL0.tWo.m_lShape[0] << ", " << sL0.tWo.m_lShape[1] << "]"
                  << std::endl;
        if (sL0.tBq.lNumel() > 0)
            std::cout << "          bq:         [" << sL0.tBq.m_lShape[0] << "]" << std::endl;
        std::cout << "          ffn_norm:   [" << sL0.tFfnNorm.m_lShape[0] << "]" << std::endl;
        std::cout << "          gate_proj:  [" << sL0.tWGate.m_lShape[0] << ", " << sL0.tWGate.m_lShape[1] << "]"
                  << std::endl;
        std::cout << "          up_proj:    [" << sL0.tWUp.m_lShape[0] << ", " << sL0.tWUp.m_lShape[1] << "]"
                  << std::endl;
        std::cout << "          down_proj:  [" << sL0.tWDown.m_lShape[0] << ", " << sL0.tWDown.m_lShape[1] << "]"
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "      embed_tokens sample values:" << std::endl;
    int64_t lShow = std::min((int64_t)5, sW.tTokEmbed.m_lShape[1]);
    for (int64_t r = 0; r < std::min((int64_t)3, sW.tTokEmbed.m_lShape[0]); r++) {
        std::cout << "          row " << r << ": [";
        for (int64_t c = 0; c < lShow; c++) {
            if (c > 0)
                std::cout << ", ";
            printf("%.6f", sW.tTokEmbed.fAt({r, c}));
        }
        std::cout << ", ...]" << std::endl;
    }
}

/*---------------------------------------------------------
 * FN: QuantizeModel
 * DESC: quantizes all linear layer weights to int8 and
 *       frees the f32 copies. norms, biases, embeddings
 *       stay f32.
 * PARMS: sModel (model to quantize)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
void QuantizeModel(SModel &sModel) {
    auto &sW = sModel.sWeights;
    int32_t iNLayers = sModel.sConfig.iNLayers;

    std::cout << "  quantizing " << iNLayers << " layers to int8..." << std::endl;

    sW.vLayersQ8.resize(iNLayers);

    int64_t lF32Bytes = 0;
    int64_t lQ8Bytes = 0;

    for (int32_t iL = 0; iL < iNLayers; iL++) {
        auto &sLyr = sW.vLayers[iL];
        auto &sQ = sW.vLayersQ8[iL];

        sQ.qWq = MT::Q8::Quantize(sLyr.tWq);
        sQ.qWk = MT::Q8::Quantize(sLyr.tWk);
        sQ.qWv = MT::Q8::Quantize(sLyr.tWv);
        sQ.qWo = MT::Q8::Quantize(sLyr.tWo);
        sQ.qWGate = MT::Q8::Quantize(sLyr.tWGate);
        sQ.qWUp = MT::Q8::Quantize(sLyr.tWUp);
        sQ.qWDown = MT::Q8::Quantize(sLyr.tWDown);

        lF32Bytes += sLyr.tWq.lNumel() * 4;
        lF32Bytes += sLyr.tWk.lNumel() * 4;
        lF32Bytes += sLyr.tWv.lNumel() * 4;
        lF32Bytes += sLyr.tWo.lNumel() * 4;
        lF32Bytes += sLyr.tWGate.lNumel() * 4;
        lF32Bytes += sLyr.tWUp.lNumel() * 4;
        lF32Bytes += sLyr.tWDown.lNumel() * 4;

        lQ8Bytes += sQ.qWq.lNumel() + sQ.qWq.lRows * 4;
        lQ8Bytes += sQ.qWk.lNumel() + sQ.qWk.lRows * 4;
        lQ8Bytes += sQ.qWv.lNumel() + sQ.qWv.lRows * 4;
        lQ8Bytes += sQ.qWo.lNumel() + sQ.qWo.lRows * 4;
        lQ8Bytes += sQ.qWGate.lNumel() + sQ.qWGate.lRows * 4;
        lQ8Bytes += sQ.qWUp.lNumel() + sQ.qWUp.lRows * 4;
        lQ8Bytes += sQ.qWDown.lNumel() + sQ.qWDown.lRows * 4;

        sLyr.tWq = MT::CTensor();
        sLyr.tWk = MT::CTensor();
        sLyr.tWv = MT::CTensor();
        sLyr.tWo = MT::CTensor();
        sLyr.tWGate = MT::CTensor();
        sLyr.tWUp = MT::CTensor();
        sLyr.tWDown = MT::CTensor();

        if ((iL + 1) % 8 == 0 || iL == iNLayers - 1)
            std::cout << "  quantized layer " << iL + 1 << "/" << iNLayers << std::endl;
    }

    sW.bQuantized = true;
    double dF32Mb = (double)lF32Bytes / (1024.0 * 1024.0);
    double dQ8Mb = (double)lQ8Bytes / (1024.0 * 1024.0);
    std::cout << "  quantization complete: " << dF32Mb << " MB -> " << dQ8Mb << " MB (" << (dF32Mb / dQ8Mb)
              << "x reduction)" << std::endl;
}
// >>>s_end(api)
} // namespace MD
