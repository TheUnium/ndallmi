// Created by Unium on 24.02.26

#include "mdMdLoad.hpp"
#include "../Thread/mtThPool.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace MD {

// <<<s_start(mmap)
// --- memory-mapped file
/*---------------------------------------------------------
 * FN: SMappedFile
 * DESC: raii wrapper for mm file io to avoid copying entire
 *       saftetensor thing into ram
 * AUTH: unium (25.02.26)
 *-------------------------------------------------------*/
struct SMappedFile {
    const uint8_t *pData = nullptr;
    int64_t lSize = 0;

#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = nullptr;
#else
    int iFd = -1;
#endif

    SMappedFile() = default;
    SMappedFile(const SMappedFile &) = delete;
    SMappedFile &operator=(const SMappedFile &) = delete;

    SMappedFile(SMappedFile &&o) noexcept : pData(o.pData), lSize(o.lSize) {
#ifdef _WIN32
        hFile = o.hFile;
        hMapping = o.hMapping;
        o.hFile = INVALID_HANDLE_VALUE;
        o.hMapping = nullptr;
#else
        iFd = o.iFd;
        o.iFd = -1;
#endif
        o.pData = nullptr;
        o.lSize = 0;
    }

    SMappedFile &operator=(SMappedFile &&o) noexcept {
        if (this != &o) {
            Close();
            pData = o.pData;
            lSize = o.lSize;
#ifdef _WIN32
            hFile = o.hFile;
            hMapping = o.hMapping;
            o.hFile = INVALID_HANDLE_VALUE;
            o.hMapping = nullptr;
#else
            iFd = o.iFd;
            o.iFd = -1;
#endif
            o.pData = nullptr;
            o.lSize = 0;
        }
        return *this;
    }

    ~SMappedFile() { Close(); }

    void Close() {
#ifdef _WIN32
        if (pData) {
            UnmapViewOfFile(pData);
            pData = nullptr;
        }
        if (hMapping) {
            CloseHandle(hMapping);
            hMapping = nullptr;
        }
        if (hFile != INVALID_HANDLE_VALUE) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
        }
#else
        if (pData && lSize > 0) {
            munmap(const_cast<uint8_t *>(pData), lSize);
            pData = nullptr;
        }
        if (iFd >= 0) {
            close(iFd);
            iFd = -1;
        }
#endif
        lSize = 0;
    }

    auto bOpen(const std::string &szPath) -> bool {
#ifdef _WIN32
        hFile = CreateFileA(szPath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (hFile == INVALID_HANDLE_VALUE)
            return false;

        LARGE_INTEGER liSize;
        if (!GetFileSizeEx(hFile, &liSize)) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
            return false;
        }
        lSize = liSize.QuadPart;

        hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMapping) {
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
            return false;
        }

        pData = (const uint8_t *)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        if (!pData) {
            CloseHandle(hMapping);
            hMapping = nullptr;
            CloseHandle(hFile);
            hFile = INVALID_HANDLE_VALUE;
            return false;
        }
        return true;
#else
        iFd = open(szPath.c_str(), O_RDONLY);
        if (iFd < 0)
            return false;

        struct stat st;
        if (fstat(iFd, &st) != 0) {
            close(iFd);
            iFd = -1;
            return false;
        }
        lSize = st.st_size;

        void *pMapped = mmap(nullptr, lSize, PROT_READ, MAP_PRIVATE, iFd, 0);
        if (pMapped == MAP_FAILED) {
            close(iFd);
            iFd = -1;
            return false;
        }

        madvise(pMapped, lSize, MADV_SEQUENTIAL);
        pData = (const uint8_t *)pMapped;
        return true;
#endif
    }
};
// >>>s_end(mmap)

// <<<s_start(json)
// --- json helpers
/*---------------------------------------------------------
 * FN: szReadFileToString
 * DESC: reads file into a str
 * PARMS: szPath (file path)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto szReadFileToString(const std::string &szPath) -> std::string {
    std::ifstream ifs(szPath, std::ios::binary | std::ios::ate);
    if (!ifs.is_open())
        return "";
    auto lSize = ifs.tellg();
    ifs.seekg(0);
    std::string szBuf(lSize, '\0');
    ifs.read(&szBuf[0], lSize);
    return szBuf;
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
        mTensors[szTensorName] = std::move(sInfo);
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
static inline auto fBF16ToF32(uint16_t u16) -> float {
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
static inline auto fFP16ToF32(uint16_t u16) -> float {
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
 * FN: LoadTensorFromMmap
 * DESC: creates a CTensor from mmapd safetensors data with
 *       parallel dtype conversion to f32 if needed
 * PARMS: pFileData (pointer to data section start), sInfo (tensor metadata)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto LoadTensorFromMmap(const uint8_t *pFileData, const SSafeTensorInfo &sInfo) -> MT::CTensor {
    std::vector<int64_t> vlShape = sInfo.vlShape;
    if (vlShape.empty())
        vlShape.push_back(1);

    MT::CTensor tResult(vlShape);
    int64_t lNumel = tResult.lNumel();
    float *pfDst = tResult.pfData();

    const uint8_t *pSrc = pFileData + sInfo.lDataStart;

    if (sInfo.szDtype == "F32") {
        std::memcpy(pfDst, pSrc, lNumel * sizeof(float));
    } else if (sInfo.szDtype == "BF16") {
        const uint16_t *pU16 = reinterpret_cast<const uint16_t *>(pSrc);
        MT::TH::ParFor(lNumel, [pfDst, pU16](int64_t lStart, int64_t lEnd) {
            for (int64_t i = lStart; i < lEnd; i++) {
                uint32_t u32 = (uint32_t)pU16[i] << 16;
                std::memcpy(&pfDst[i], &u32, sizeof(float));
            }
        });
    } else if (sInfo.szDtype == "F16") {
        const uint16_t *pU16 = reinterpret_cast<const uint16_t *>(pSrc);
        MT::TH::ParFor(lNumel, [pfDst, pU16](int64_t lStart, int64_t lEnd) {
            for (int64_t i = lStart; i < lEnd; i++)
                pfDst[i] = fFP16ToF32(pU16[i]);
        });
    } else {
        std::cerr << "  warning: unsupported dtype '" << sInfo.szDtype << "', filling zeros" << std::endl;
    }

    return tResult;
}
// >>>s_end(dtype)

// <<<s_start(fileload)
// --- safetensors file loading with mmap
struct SSafetensorsFile {
    std::unordered_map<std::string, SSafeTensorInfo> mTensors;
    SMappedFile sMmap;
    const uint8_t *pDataSection = nullptr;
    int64_t lDataSize = 0;
};

/*---------------------------------------------------------
 * FN: bLoadSafetensorsFile
 * DESC: loads and parses a single .safetensors file
 * PARMS: szPath (file path), sFile (output)
 * AUTH: unium (24.02.26)
 *-------------------------------------------------------*/
static auto bLoadSafetensorsFile(const std::string &szPath, SSafetensorsFile &sFile) -> bool {
    if (!sFile.sMmap.bOpen(szPath)) {
        std::cerr << "  error: cannot mmap " << szPath << std::endl;
        return false;
    }

    if (sFile.sMmap.lSize < 8) {
        std::cerr << "  error: file too small " << szPath << std::endl;
        return false;
    }

    uint64_t lHeaderSize = 0;
    std::memcpy(&lHeaderSize, sFile.sMmap.pData, 8);

    if (lHeaderSize > 100 * 1024 * 1024) {
        std::cerr << "  error: header too large (" << lHeaderSize << " bytes)" << std::endl;
        return false;
    }

    if ((int64_t)(8 + lHeaderSize) > sFile.sMmap.lSize) {
        std::cerr << "  error: header exceeds file size" << std::endl;
        return false;
    }

    std::string szHeader((const char *)(sFile.sMmap.pData + 8), lHeaderSize);
    sFile.mTensors = ParseSafetensorsHeader(szHeader);

    sFile.pDataSection = sFile.sMmap.pData + 8 + lHeaderSize;
    sFile.lDataSize = sFile.sMmap.lSize - 8 - (int64_t)lHeaderSize;

    std::cout << "  mmap'd " << szPath << " (" << sFile.mTensors.size() << " tensors, "
              << (sFile.lDataSize / (1024 * 1024)) << " MB data)" << std::endl;

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
            tOut = LoadTensorFromMmap(sFile.pDataSection, it->second);
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

    {
        std::atomic<bool> bAnyFailed{false};
        std::mutex mtxPrint;

        if (vszFiles.size() > 1) {
            std::vector<std::thread> vThreads;
            for (size_t i = 0; i < vszFiles.size(); i++) {
                vThreads.emplace_back([&, i]() {
                    if (!bLoadSafetensorsFile(vszFiles[i], vFiles[i]))
                        bAnyFailed.store(true);
                });
            }
            for (auto &t : vThreads)
                t.join();
        } else {
            if (!bLoadSafetensorsFile(vszFiles[0], vFiles[0]))
                bAnyFailed.store(true);
        }

        if (bAnyFailed.load())
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

    struct SLayerLoadJob {
        int32_t iLayer;
        bool bOk = true;
    };

    std::vector<SLayerLoadJob> vJobs(sCfg.iNLayers);
    for (int32_t i = 0; i < sCfg.iNLayers; i++)
        vJobs[i].iLayer = i;

    std::mutex mtxLog;

    MT::TH::ParFor((int64_t)sCfg.iNLayers, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t iIdx = lStart; iIdx < lEnd; iIdx++) {
            int32_t iL = (int32_t)iIdx;
            auto &sLyr = sW.vLayers[iL];
            auto &sJob = vJobs[iL];

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

            sJob.bOk = bOk;

            if ((iL + 1) % 8 == 0 || iL == sCfg.iNLayers - 1) {
                std::lock_guard<std::mutex> lock(mtxLog);
                std::cout << "  loaded layer " << iL + 1 << "/" << sCfg.iNLayers << std::endl;
            }
        }
    });

    for (int32_t iL = 0; iL < sCfg.iNLayers; iL++) {
        if (!vJobs[iL].bOk) {
            std::cerr << "  error: missing weights for layer " << iL << std::endl;
            return false;
        }
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

    struct SQuantStats {
        int64_t lF32Bytes = 0;
        int64_t lQ8Bytes = 0;
    };

    int iNumThreads = MT::TH::GetGlobalPool().iNumThreads();
    std::vector<SQuantStats> vStats(iNumThreads);

    std::mutex mtxLog;

    MT::TH::ParFor((int64_t)iNLayers, [&](int64_t lStart, int64_t lEnd) {
        int iTid = 0;

        {
            int64_t lChunk = ((int64_t)iNLayers + iNumThreads - 1) / iNumThreads;
            if (lChunk > 0)
                iTid = (int)(lStart / lChunk);
            if (iTid >= iNumThreads)
                iTid = iNumThreads - 1;
        }
        auto &sStats = vStats[iTid];

        for (int64_t iIdx = lStart; iIdx < lEnd; iIdx++) {
            int32_t iL = (int32_t)iIdx;
            auto &sLyr = sW.vLayers[iL];
            auto &sQ = sW.vLayersQ8[iL];

            sQ.qWq = MT::Q8::Quantize(sLyr.tWq);
            sQ.qWk = MT::Q8::Quantize(sLyr.tWk);
            sQ.qWv = MT::Q8::Quantize(sLyr.tWv);
            sQ.qWo = MT::Q8::Quantize(sLyr.tWo);
            sQ.qWGate = MT::Q8::Quantize(sLyr.tWGate);
            sQ.qWUp = MT::Q8::Quantize(sLyr.tWUp);
            sQ.qWDown = MT::Q8::Quantize(sLyr.tWDown);

            sStats.lF32Bytes += sLyr.tWq.lNumel() * 4;
            sStats.lF32Bytes += sLyr.tWk.lNumel() * 4;
            sStats.lF32Bytes += sLyr.tWv.lNumel() * 4;
            sStats.lF32Bytes += sLyr.tWo.lNumel() * 4;
            sStats.lF32Bytes += sLyr.tWGate.lNumel() * 4;
            sStats.lF32Bytes += sLyr.tWUp.lNumel() * 4;
            sStats.lF32Bytes += sLyr.tWDown.lNumel() * 4;

            sStats.lQ8Bytes += sQ.qWq.lNumel() + sQ.qWq.lRows * 4;
            sStats.lQ8Bytes += sQ.qWk.lNumel() + sQ.qWk.lRows * 4;
            sStats.lQ8Bytes += sQ.qWv.lNumel() + sQ.qWv.lRows * 4;
            sStats.lQ8Bytes += sQ.qWo.lNumel() + sQ.qWo.lRows * 4;
            sStats.lQ8Bytes += sQ.qWGate.lNumel() + sQ.qWGate.lRows * 4;
            sStats.lQ8Bytes += sQ.qWUp.lNumel() + sQ.qWUp.lRows * 4;
            sStats.lQ8Bytes += sQ.qWDown.lNumel() + sQ.qWDown.lRows * 4;

            sLyr.tWq = MT::CTensor();
            sLyr.tWk = MT::CTensor();
            sLyr.tWv = MT::CTensor();
            sLyr.tWo = MT::CTensor();
            sLyr.tWGate = MT::CTensor();
            sLyr.tWUp = MT::CTensor();
            sLyr.tWDown = MT::CTensor();

            if ((iL + 1) % 8 == 0 || iL == iNLayers - 1) {
                std::lock_guard<std::mutex> lock(mtxLog);
                std::cout << "  quantized layer " << iL + 1 << "/" << iNLayers << std::endl;
            }
        }
    });

    int64_t lF32Bytes = 0, lQ8Bytes = 0;
    for (const auto &s : vStats) {
        lF32Bytes += s.lF32Bytes;
        lQ8Bytes += s.lQ8Bytes;
    }

    sW.bQuantized = true;
    double dF32Mb = (double)lF32Bytes / (1024.0 * 1024.0);
    double dQ8Mb = (double)lQ8Bytes / (1024.0 * 1024.0);
    std::cout << "  quantization complete: " << dF32Mb << " MB -> " << dQ8Mb << " MB (" << (dF32Mb / dQ8Mb)
              << "x reduction)" << std::endl;
}
// >>>s_end(api)
} // namespace MD
