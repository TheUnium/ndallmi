// Created by Unium on 23.02.26

// TODO: std::string_view would be faster!!
//       substr triggers heap alloc for every single char so its p slow
//       but thats for later me, this uses substr a bunch of times tho

#include "tkTkBpe_.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>

namespace TK {
/*---------------------------------------------------------
 * FN: CBpeTokenizer
 * DESC: default constructor
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
CBpeTokenizer::CBpeTokenizer() : m_iBosId(0), m_iEosId(0), m_iUnkId(0) {}

/*---------------------------------------------------------
 * FN: ~CBpeTokenizer
 * DESC: destructor
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
CBpeTokenizer::~CBpeTokenizer() = default;

// <<<s_start(helpers)
// --- helper stuff
/*---------------------------------------------------------
 * FN: EnsureVocabSize
 * DESC: grows the vocab vector if needed to fit iId
 * PARMS: iId (token ID to ensure space for)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::EnsureVocabSize(int32_t iId) {
    if (iId < 0)
        return;
    if ((int32_t)m_vVocab.size() <= iId) {
        m_vVocab.resize(iId + 1, {"", 0.0f, false, false});
    }
}

/*---------------------------------------------------------
 * FN: lPackPair
 * DESC: packs two 32b token ids into a single 64b key for
 *       merge lookup
 * PARMS: iLeft (left token ID), iRight (right token ID)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::lPackPair(int32_t iLeft, int32_t iRight) -> int64_t {
    return ((int64_t)(uint32_t)iLeft << 32) | (int64_t)(uint32_t)iRight;
}
// >>>s_end(helpers)

/*---------------------------------------------------------
 * FN: BuildGptByteMap
 * DESC: builds gpt2 styled byte to unicode mapping tables
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::BuildGptByteMap() {
    std::vector<int> viBs;
    std::vector<int> viCs;

    for (int i = 33; i <= 126; i++) {
        viBs.push_back(i);
        viCs.push_back(i);
    }
    for (int i = 161; i <= 172; i++) {
        viBs.push_back(i);
        viCs.push_back(i);
    }
    for (int i = 174; i <= 255; i++) {
        viBs.push_back(i);
        viCs.push_back(i);
    }

    int iN = 0;
    for (int b = 0; b < 256; b++) {
        bool bFound = false;
        for (int x : viBs) {
            if (x == b) {
                bFound = true;
                break;
            }
        }
        if (!bFound) {
            viBs.push_back(b);
            viCs.push_back(256 + iN);
            iN++;
        }
    }

    for (size_t i = 0; i < viBs.size(); i++) {
        int iByte = viBs[i];
        int iCodepoint = viCs[i];

        std::string szUtf8;
        if (iCodepoint < 0x80) {
            szUtf8 += (char)iCodepoint;
        } else if (iCodepoint < 0x800) {
            szUtf8 += (char)(0xC0 | (iCodepoint >> 6));
            szUtf8 += (char)(0x80 | (iCodepoint & 0x3F));
        } else {
            szUtf8 += (char)(0xE0 | (iCodepoint >> 12));
            szUtf8 += (char)(0x80 | ((iCodepoint >> 6) & 0x3F));
            szUtf8 += (char)(0x80 | (iCodepoint & 0x3F));
        }

        m_vszByteToGpt[iByte] = szUtf8;
        m_mGptToByte[szUtf8] = (uint8_t)iByte;
    }
}

/*---------------------------------------------------------
 * FN: szTextToGptBytes
 * DESC: converts raw text bytes to gpt2 unicode encoding
 * PARMS: szIn (raw input text)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::szTextToGptBytes(const std::string &szIn) const -> std::string {
    std::string szOut;
    szOut.reserve(szIn.size() * 2);
    for (unsigned char c : szIn) {
        szOut += m_vszByteToGpt[c];
    }
    return szOut;
}

/*---------------------------------------------------------
 * FN: szGptBytesToText
 * DESC: converts gpt2 unicode encoding back to raw bytes
 * PARMS: szIn (GPT-encoded input)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::szGptBytesToText(const std::string &szIn) const -> std::string {
    std::string szOut;
    szOut.reserve(szIn.size());

    size_t i = 0;
    while (i < szIn.size()) {
        unsigned char c = (unsigned char)szIn[i];
        int iLen = 1;
        if ((c & 0x80) == 0)
            iLen = 1;
        else if ((c & 0xE0) == 0xC0)
            iLen = 2;
        else if ((c & 0xF0) == 0xE0)
            iLen = 3;
        else if ((c & 0xF8) == 0xF0)
            iLen = 4;
        if (i + iLen > szIn.size())
            iLen = 1;

        std::string szChar = szIn.substr(i, iLen);
        auto it = m_mGptToByte.find(szChar);
        if (it != m_mGptToByte.end()) {
            szOut += (char)it->second;
        } else {
            szOut += szChar;
        }
        i += iLen;
    }
    return szOut;
}

/*---------------------------------------------------------
 * FN: PreTokenize
 * DESC: splits text into chunks using gpt2/llama3 style
 *       pre tokenization regex patterns
 * PARMS: szText (input text)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::PreTokenize(const std::string &szText) const -> std::vector<std::string> {
    std::vector<std::string> vChunks;
    size_t iLen = szText.size();
    size_t i = 0;

    // 's|'t|'re|'ve|'m|'ll|'d
    // | ?\p{L}+
    // | ?\p{N}{1,3}
    // | ?[^\s\p{L}\p{N}]+
    // |\s+(?!\S)
    // |\s+
    auto bIsLetter = [](unsigned char c) -> bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
    auto bIsDigit = [](unsigned char c) -> bool { return c >= '0' && c <= '9'; };
    auto bIsWs = [](unsigned char c) -> bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };

    while (i < iLen) {
        unsigned char c = (unsigned char)szText[i];

        // 's 't 'm 'd 're 've 'll
        if (c == '\'' && i + 1 < iLen) {
            char cn = szText[i + 1];
            if (cn == 's' || cn == 'S' || cn == 't' || cn == 'T' || cn == 'm' || cn == 'M' || cn == 'd' || cn == 'D') {
                vChunks.push_back(szText.substr(i, 2));
                i += 2;
                continue;
            }
            if (i + 2 < iLen) {
                char c2 = szText[i + 2];
                if (((cn | 32) == 'r' && (c2 | 32) == 'e') || ((cn | 32) == 'v' && (c2 | 32) == 'e') ||
                    ((cn | 32) == 'l' && (c2 | 32) == 'l')) {
                    vChunks.push_back(szText.substr(i, 3));
                    i += 3;
                    continue;
                }
            }
        }

        // space+letters
        if (c == ' ' && i + 1 < iLen && bIsLetter((unsigned char)szText[i + 1])) {
            size_t iStart = i;
            i++;
            while (i < iLen && bIsLetter((unsigned char)szText[i]))
                i++;
            vChunks.push_back(szText.substr(iStart, i - iStart));
            continue;
        }

        // letters
        if (bIsLetter(c)) {
            size_t iStart = i;
            while (i < iLen && bIsLetter((unsigned char)szText[i]))
                i++;
            vChunks.push_back(szText.substr(iStart, i - iStart));
            continue;
        }

        // space+digits [3]
        if (c == ' ' && i + 1 < iLen && bIsDigit((unsigned char)szText[i + 1])) {
            size_t iStart = i;
            i++;
            int iCount = 0;
            while (i < iLen && bIsDigit((unsigned char)szText[i]) && iCount < 3) {
                i++;
                iCount++;
            }
            vChunks.push_back(szText.substr(iStart, i - iStart));
            continue;
        }

        // digits only [3]
        if (bIsDigit(c)) {
            size_t iStart = i;
            int iCount = 0;
            while (i < iLen && bIsDigit((unsigned char)szText[i]) && iCount < 3) {
                i++;
                iCount++;
            }
            vChunks.push_back(szText.substr(iStart, i - iStart));
            continue;
        }

        // nls
        if (c == '\n' || c == '\r') {
            size_t iStart = i;
            while (i < iLen && (szText[i] == '\n' || szText[i] == '\r'))
                i++;
            vChunks.push_back(szText.substr(iStart, i - iStart));
            continue;
        }

        // whitespace nnl
        if (bIsWs(c)) {
            size_t iStart = i;
            while (i < iLen && bIsWs((unsigned char)szText[i]) && szText[i] != '\n' && szText[i] != '\r')
                i++;
            vChunks.push_back(szText.substr(iStart, i - iStart));
            continue;
        }

        // punctuation/other
        {
            size_t iStart = i;
            if (c == ' ')
                i++;
            if (i < iLen) {
                unsigned char cc = (unsigned char)szText[i];
                while (i < iLen && !bIsLetter(cc) && !bIsDigit(cc) && !bIsWs(cc)) {
                    i++;
                    if (i < iLen)
                        cc = (unsigned char)szText[i];
                }
            }
            if (i > iStart) {
                vChunks.push_back(szText.substr(iStart, i - iStart));
            } else {
                vChunks.push_back(szText.substr(i, 1));
                i++;
            }
        }
    }

    return vChunks;
}

// <<<s_start(json)
// --- json helpers
/*---------------------------------------------------------
 * FN: SkipWs
 * DESC: advances position past whitespace in a string
 * PARMS: s (json string), p (current position, modified)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::SkipWs(const std::string &s, size_t &p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r'))
        p++;
}

/*---------------------------------------------------------
 * FN: ParseJsonString
 * DESC: parses a json quoted string at pos p
 * PARMS: s (json string), p (current position, modified)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::ParseJsonString(const std::string &s, size_t &p) -> std::string {
    const char *ptr = s.data() + p;
    const char *pEnd = s.data() + s.size();
    std::string szOut = ParseJsonStringPtr(ptr, pEnd);
    p = ptr - s.data();
    return szOut;
}

/*---------------------------------------------------------
 * FN: SkipJsonValue
 * DESC: skips over an arbitrary json value at position p
 * PARMS: s (json string), p (current position, modified)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::SkipJsonValue(const std::string &s, size_t &p) {
    SkipWs(s, p);
    if (p >= s.size())
        return;

    if (s[p] == '"') {
        ParseJsonString(s, p);
    } else if (s[p] == '{') {
        p++;
        int d = 1;
        while (p < s.size() && d > 0) {
            if (s[p] == '"')
                ParseJsonString(s, p);
            else {
                if (s[p] == '{')
                    d++;
                else if (s[p] == '}')
                    d--;
                p++;
            }
        }
    } else if (s[p] == '[') {
        p++;
        int d = 1;
        while (p < s.size() && d > 0) {
            if (s[p] == '"')
                ParseJsonString(s, p);
            else {
                if (s[p] == '[')
                    d++;
                else if (s[p] == ']')
                    d--;
                p++;
            }
        }
    } else {
        while (p < s.size() && s[p] != ',' && s[p] != '}' && s[p] != ']' && s[p] != ' ' && s[p] != '\n')
            p++;
    }
}

/*---------------------------------------------------------
 * FN: ParseJsonStringPtr
 * DESC: fast pointer based json string parser
 * PARMS: p (pointer, modified), pEnd (end of buffer)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::ParseJsonStringPtr(const char *&p, const char *pEnd) -> std::string {
    // hi its me unium, welcome back to unium rants on dumb shit bc istg my blood is boiling
    // why does every fucking """lightweight""" cpp project end up with a custom json parser
    // like look at this shit, im manually checking for fucking backslashes and switch and
    // fucking switch casing "n" and "r" and "t" like its fucking 1995
    //
    // oh yeah and the headache that is unicode, jsons INSISTENCE on utf16 surrogate pairs
    // feels like someone shot me in the nuts. you have to fucking bitshift 0x10000 and like
    // some masking bs, TO GET A FUCKING EMOJI TO WORK
    //
    // and remember one obo error in this pointer magic hell and the whole tokenizer goes
    // into orbit like im fucking nasa and this shit is apollo 11. its literally either:
    // a) import boost or nlohman smth smth (great now this shit is 900mb and takes 50 years
    //                                       to compile from scratch)
    // b) do whatever the fuck i am doing right now (great now WE have a headache because
    //                                               json was written when utf16 was the
    //                                               "final form of unicode"!!!!! they were
    //                                               wrong. ucs2/utf16 was not the final
    //                                               form of unicode.)
    if (p >= pEnd || *p != '"')
        return "";
    p++;

    const char *pStart = p;
    const char *pScan = p;
    bool bHasEscape = false;
    while (pScan < pEnd && *pScan != '"') {
        if (*pScan == '\\') {
            bHasEscape = true;
            pScan++;
            if (pScan < pEnd)
                pScan++;
        } else
            pScan++;
    }

    if (!bHasEscape) {
        std::string szOut(pStart, pScan - pStart);
        p = (pScan < pEnd) ? pScan + 1 : pScan;
        return szOut;
    }

    p = pStart;
    std::string szOut;
    szOut.reserve(pScan - pStart);
    while (p < pEnd && *p != '"') {
        if (*p == '\\' && p + 1 < pEnd) {
            p++;
            switch (*p) {
            case '"':
                szOut += '"';
                p++;
                break;
            case '\\':
                szOut += '\\';
                p++;
                break;
            case '/':
                szOut += '/';
                p++;
                break;
            case 'n':
                szOut += '\n';
                p++;
                break;
            case 'r':
                szOut += '\r';
                p++;
                break;
            case 't':
                szOut += '\t';
                p++;
                break;
            case 'b':
                szOut += '\b';
                p++;
                break;
            case 'f':
                szOut += '\f';
                p++;
                break;
            case 'u': {
                p++;
                if (p + 4 > pEnd)
                    break;
                uint32_t uCode = 0;
                for (int k = 0; k < 4; k++) {
                    uCode <<= 4;
                    char h = p[k];
                    if (h >= '0' && h <= '9')
                        uCode |= (h - '0');
                    else if (h >= 'a' && h <= 'f')
                        uCode |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F')
                        uCode |= (h - 'A' + 10);
                }
                p += 4;
                if (uCode >= 0xD800 && uCode <= 0xDBFF && p + 5 < pEnd && p[0] == '\\' && p[1] == 'u') {
                    p += 2;
                    uint32_t uLow = 0;
                    for (int k = 0; k < 4; k++) {
                        uLow <<= 4;
                        char h = p[k];
                        if (h >= '0' && h <= '9')
                            uLow |= (h - '0');
                        else if (h >= 'a' && h <= 'f')
                            uLow |= (h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F')
                            uLow |= (h - 'A' + 10);
                    }
                    p += 4;
                    uCode = 0x10000 + ((uCode - 0xD800) << 10) + (uLow - 0xDC00);
                }
                if (uCode < 0x80) {
                    szOut += (char)uCode;
                } else if (uCode < 0x800) {
                    szOut += (char)(0xC0 | (uCode >> 6));
                    szOut += (char)(0x80 | (uCode & 0x3F));
                } else if (uCode < 0x10000) {
                    szOut += (char)(0xE0 | (uCode >> 12));
                    szOut += (char)(0x80 | ((uCode >> 6) & 0x3F));
                    szOut += (char)(0x80 | (uCode & 0x3F));
                } else {
                    szOut += (char)(0xF0 | (uCode >> 18));
                    szOut += (char)(0x80 | ((uCode >> 12) & 0x3F));
                    szOut += (char)(0x80 | ((uCode >> 6) & 0x3F));
                    szOut += (char)(0x80 | (uCode & 0x3F));
                }
                break;
            }
            default:
                szOut += *p;
                p++;
                break;
            }
        } else {
            szOut += *p;
            p++;
        }
    }
    if (p < pEnd)
        p++;
    return szOut;
}
// >>>s_end(json)

// <<<s_start(shit)
/*---------------------------------------------------------
 * FN: bLoadFromJsonFile
 * DESC: loads a hf tokenizer.json file and populates vocab,
 *       merges and special tokens
 * PARMS: szPath (path to tokenizer.json)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::bLoadFromJsonFile(const std::string &szPath) -> bool {
    auto tLoadStart = std::chrono::high_resolution_clock::now();

    std::ifstream ifs(szPath, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "  error: cannot open " << szPath << std::endl;
        return false;
    }

    ifs.seekg(0, std::ios::end);
    size_t lFileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::cout << "  tokenizer: reading " << (lFileSize / (1024 * 1024)) << " MB..." << std::flush;

    std::string szJson;
    szJson.resize(lFileSize);
    ifs.read(&szJson[0], lFileSize);
    ifs.close();

    const char *pJson = szJson.data();
    const char *pEnd = pJson + szJson.size();

    std::cout << " done" << std::endl;

    m_vVocab.clear();
    m_mTextToId.clear();
    m_mMerges.clear();
    m_mMergeRank.clear();

    std::cout << "  tokenizer: parsing added_tokens..." << std::flush;
    struct SAddedToken {
        std::string szContent;
        int32_t iId;
        bool bSpecial;
    };
    std::vector<SAddedToken> vAddedTokens;

    {
        size_t lPos = szJson.find("\"added_tokens\"");
        if (lPos != std::string::npos) {
            lPos = szJson.find('[', lPos);
            if (lPos != std::string::npos) {
                lPos++;
                SkipWs(szJson, lPos);
                while (lPos < szJson.size() && szJson[lPos] != ']') {
                    SkipWs(szJson, lPos);
                    if (szJson[lPos] == ',') {
                        lPos++;
                        continue;
                    }
                    if (szJson[lPos] != '{')
                        break;
                    lPos++;

                    int32_t iId = -1;
                    bool bSpecial = false;
                    std::string szContent;

                    while (lPos < szJson.size() && szJson[lPos] != '}') {
                        SkipWs(szJson, lPos);
                        if (szJson[lPos] == ',' || szJson[lPos] == '}') {
                            if (szJson[lPos] == ',')
                                lPos++;
                            continue;
                        }
                        std::string szKey = ParseJsonString(szJson, lPos);
                        SkipWs(szJson, lPos);
                        if (lPos < szJson.size() && szJson[lPos] == ':')
                            lPos++;
                        SkipWs(szJson, lPos);

                        if (szKey == "id") {
                            std::string szNum;
                            while (lPos < szJson.size() &&
                                   ((szJson[lPos] >= '0' && szJson[lPos] <= '9') || szJson[lPos] == '-')) {
                                szNum += szJson[lPos++];
                            }
                            if (!szNum.empty())
                                iId = std::stoi(szNum);
                        } else if (szKey == "content") {
                            szContent = ParseJsonString(szJson, lPos);
                        } else if (szKey == "special") {
                            SkipWs(szJson, lPos);
                            if (lPos + 4 <= szJson.size() && szJson.compare(lPos, 4, "true") == 0) {
                                bSpecial = true;
                                lPos += 4;
                            } else if (lPos + 5 <= szJson.size() && szJson.compare(lPos, 5, "false") == 0) {
                                lPos += 5;
                            }
                        } else {
                            SkipJsonValue(szJson, lPos);
                        }
                    }
                    if (lPos < szJson.size() && szJson[lPos] == '}')
                        lPos++;

                    if (iId >= 0 && !szContent.empty()) {
                        vAddedTokens.push_back({szContent, iId, bSpecial});
                    }
                }
            }
        }
    }

    int iSpecialCount = 0;
    for (const auto &at : vAddedTokens) {
        if (at.bSpecial)
            iSpecialCount++;
    }
    std::cout << " " << vAddedTokens.size() << " added (" << iSpecialCount << " special)" << std::endl;

    std::cout << "  tokenizer: parsing vocab..." << std::flush;
    int32_t iMaxId = -1;
    int32_t iVocabCount = 0;

    {
        size_t lModel = szJson.find("\"model\"");
        if (lModel == std::string::npos) {
            std::cerr << "\n  error: no 'model' key" << std::endl;
            return false;
        }
        size_t lColon = szJson.find(':', lModel + 7);
        if (lColon == std::string::npos)
            return false;
        size_t lModelObj = szJson.find('{', lColon);
        if (lModelObj == std::string::npos)
            return false;

        size_t lVocab = szJson.find("\"vocab\"", lModelObj);
        if (lVocab == std::string::npos) {
            std::cerr << "\n  error: no 'vocab' in model" << std::endl;
            return false;
        }
        size_t lVColon = szJson.find(':', lVocab + 7);
        if (lVColon == std::string::npos)
            return false;
        size_t lVObj = szJson.find('{', lVColon);
        if (lVObj == std::string::npos)
            return false;

        const char *p = pJson + lVObj + 1;

        m_vVocab.reserve(130000);
        m_mTextToId.reserve(200000);

        while (p < pEnd) {
            while (p < pEnd && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ','))
                p++;
            if (p >= pEnd || *p == '}')
                break;

            std::string szToken = ParseJsonStringPtr(p, pEnd);

            while (p < pEnd && *p != ':')
                p++;
            if (p < pEnd)
                p++;
            while (p < pEnd && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
                p++;

            int32_t iId = 0;
            bool bNeg = false;
            if (p < pEnd && *p == '-') {
                bNeg = true;
                p++;
            }
            while (p < pEnd && *p >= '0' && *p <= '9') {
                iId = iId * 10 + (*p - '0');
                p++;
            }
            if (bNeg)
                iId = -iId;

            EnsureVocabSize(iId);
            m_vVocab[iId] = {szToken, 0.0f, false, false};
            m_mTextToId[szToken] = iId;

            if (iId > iMaxId)
                iMaxId = iId;
            iVocabCount++;

            if ((iVocabCount % 50000) == 0) {
                std::cout << " " << iVocabCount << "..." << std::flush;
            }
        }
    }
    std::cout << " " << iVocabCount << " tokens" << std::endl;

    for (const auto &at : vAddedTokens) {
        EnsureVocabSize(at.iId);

        const std::string &szOld = m_vVocab[at.iId].m_szText;
        if (!szOld.empty() && szOld != at.szContent) {
            m_mTextToId.erase(szOld);
        }

        m_vVocab[at.iId] = {at.szContent, 0.0f, at.bSpecial, true};
        m_mTextToId[at.szContent] = at.iId;

        if (at.iId > iMaxId)
            iMaxId = at.iId;
    }

    std::cout << "  tokenizer: parsing merges..." << std::flush;
    int32_t iMergeCount = 0;

    // i hate you
    {
        size_t lModel = szJson.find("\"model\"");
        size_t lMerges = szJson.find("\"merges\"", lModel);
        if (lMerges != std::string::npos) {
            size_t lMColon = szJson.find(':', lMerges);
            if (lMColon == std::string::npos)
                goto skip_merges;
            size_t lArr = szJson.find('[', lMColon);
            if (lArr == std::string::npos)
                goto skip_merges;

            const char *p = pJson + lArr + 1;

            m_mMerges.reserve(300000);
            m_mMergeRank.reserve(300000);

            while (p < pEnd && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
                p++;
            if (p >= pEnd)
                goto skip_merges;

            bool bArrayFormat = (*p == '[');
            int32_t iRank = 0;

            if (bArrayFormat) {
                while (p < pEnd) {
                    while (p < pEnd && *p != '[' && *p != ']')
                        p++;
                    if (p >= pEnd || *p == ']')
                        break;
                    p++;

                    while (p < pEnd && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
                        p++;

                    std::string szLeft = ParseJsonStringPtr(p, pEnd);

                    while (p < pEnd && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ','))
                        p++;

                    std::string szRight = ParseJsonStringPtr(p, pEnd);

                    while (p < pEnd && *p != ']')
                        p++;
                    if (p < pEnd)
                        p++;

                    std::string szMerged = szLeft + szRight;
                    auto itL = m_mTextToId.find(szLeft);
                    auto itR = m_mTextToId.find(szRight);
                    auto itM = m_mTextToId.find(szMerged);

                    if (itL != m_mTextToId.end() && itR != m_mTextToId.end() && itM != m_mTextToId.end()) {
                        int64_t lKey = lPackPair(itL->second, itR->second);
                        m_mMerges[lKey] = itM->second;
                        m_mMergeRank[lKey] = iRank;
                        iMergeCount++;
                    }
                    iRank++;

                    if ((iRank % 100000) == 0) {
                        std::cout << " " << iRank << "..." << std::flush;
                    }
                }
            } else if (*p == '"') {
                while (p < pEnd) {
                    while (p < pEnd && *p != '"' && *p != ']')
                        p++;
                    if (p >= pEnd || *p == ']')
                        break;

                    std::string szMerge = ParseJsonStringPtr(p, pEnd);

                    size_t iSp = szMerge.find(' ');
                    if (iSp == std::string::npos) {
                        iRank++;
                        continue;
                    }

                    std::string szLeft(szMerge, 0, iSp);
                    std::string szRight(szMerge, iSp + 1);
                    std::string szMerged = szLeft + szRight;

                    auto itL = m_mTextToId.find(szLeft);
                    auto itR = m_mTextToId.find(szRight);
                    auto itM = m_mTextToId.find(szMerged);

                    if (itL != m_mTextToId.end() && itR != m_mTextToId.end() && itM != m_mTextToId.end()) {
                        int64_t lKey = lPackPair(itL->second, itR->second);
                        m_mMerges[lKey] = itM->second;
                        m_mMergeRank[lKey] = iRank;
                        iMergeCount++;
                    }
                    iRank++;

                    if ((iRank % 100000) == 0) {
                        std::cout << " " << iRank << "..." << std::flush;
                    }
                }
            }
        }
    }
skip_merges:
    std::cout << " " << iMergeCount << " merges" << std::endl;

    auto lfnFind = [&](const std::string &szName) -> int32_t {
        auto it = m_mTextToId.find(szName);
        return (it != m_mTextToId.end()) ? it->second : -1;
    };

    m_iUnkId = std::max(0, lfnFind("<unk>"));
    m_iBosId = lfnFind("<|begin_of_text|>");
    if (m_iBosId < 0)
        m_iBosId = lfnFind("<s>");
    if (m_iBosId < 0)
        m_iBosId = 0;
    m_iEosId = lfnFind("<|eot_id|>");
    if (m_iEosId < 0)
        m_iEosId = lfnFind("<|end_of_text|>");
    if (m_iEosId < 0)
        m_iEosId = lfnFind("<|endoftext|>");
    if (m_iEosId < 0)
        m_iEosId = lfnFind("<|im_end|>");
    if (m_iEosId < 0)
        m_iEosId = lfnFind("</s>");
    if (m_iEosId < 0)
        m_iEosId = 0;

    int32_t iGptCount = 0;
    for (const auto &info : m_vVocab) {
        if (info.m_bSpecial || info.m_bAdded)
            continue;
        if (info.m_szText.size() >= 2 && (unsigned char)info.m_szText[0] == 0xc4 &&
            (unsigned char)info.m_szText[1] == 0xa0) {
            iGptCount++;
        }
    }
    m_bByteLevel = (iGptCount > 100);
    if (m_bByteLevel) {
        BuildGptByteMap();
    }

    m_vAddedTokens.clear();
    for (int32_t i = 0; i < (int32_t)m_vVocab.size(); i++) {
        if (m_vVocab[i].m_bAdded && !m_vVocab[i].m_szText.empty()) {
            m_vAddedTokens.push_back({m_vVocab[i].m_szText, i});
        }
    }
    std::sort(m_vAddedTokens.begin(), m_vAddedTokens.end(), [](const SAddedTokenEntry &a, const SAddedTokenEntry &b) {
        return a.m_szText.size() > b.m_szText.size();
    });

    auto tLoadEnd = std::chrono::high_resolution_clock::now();
    double dLoadMs = std::chrono::duration<double, std::milli>(tLoadEnd - tLoadStart).count();

    std::cout << "  tokenizer: " << (iMaxId + 1) << " tokens, " << iMergeCount << " merges, " << m_vAddedTokens.size()
              << " added"
              << " (byte-level=" << (m_bByteLevel ? "yes" : "no") << ", bos=" << m_iBosId << ", eos=" << m_iEosId
              << ") loaded in " << (int)dLoadMs << " ms" << std::endl;

    return true;
}

/*---------------------------------------------------------
 * FN: Encode
 * DESC: encodes text into a vector of token ids, and also
 *       handles added/special tokens and bpe stuff
 * PARMS: szText (input text)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::Encode(const std::string &szText) const -> std::vector<int32_t> {
    if (szText.empty())
        return {};

    struct SSeg {
        bool bAdded;
        int32_t iId;
        std::string szText;
    };
    std::vector<SSeg> vSegs;

    size_t iPos = 0;
    while (iPos < szText.size()) {
        bool bFound = false;
        for (const auto &at : m_vAddedTokens) {
            if (iPos + at.m_szText.size() <= szText.size() &&
                szText.compare(iPos, at.m_szText.size(), at.m_szText) == 0) {
                vSegs.push_back({true, at.m_iId, ""});
                iPos += at.m_szText.size();
                bFound = true;
                break;
            }
        }
        if (!bFound) {
            if (vSegs.empty() || vSegs.back().bAdded) {
                vSegs.push_back({false, 0, ""});
            }
            vSegs.back().szText += szText[iPos];
            iPos++;
        }
    }

    std::vector<int32_t> viResult;
    for (const auto &seg : vSegs) {
        if (seg.bAdded) {
            viResult.push_back(seg.iId);
        } else {
            auto vi = EncodeBpe(seg.szText);
            viResult.insert(viResult.end(), vi.begin(), vi.end());
        }
    }
    return viResult;
}

/*---------------------------------------------------------
 * FN: EncodeBpe
 * DESC: applies pre tokenization and bpe to a bit of text
 * PARMS: szText (input text segment)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::EncodeBpe(const std::string &szText) const -> std::vector<int32_t> {
    if (szText.empty())
        return {};

    if (m_bByteLevel) {
        auto vChunks = PreTokenize(szText);
        std::vector<int32_t> viResult;
        for (const auto &szChunk : vChunks) {
            std::string szGpt = szTextToGptBytes(szChunk);
            auto vi = EncodeBpeChunk(szGpt);
            viResult.insert(viResult.end(), vi.begin(), vi.end());
        }
        return viResult;
    } else {
        return EncodeBpeChunk(szText);
    }
}

/*---------------------------------------------------------
 * FN: EncodeBpeChunk
 * DESC: applies bpe merges to a single pre tokenized chunk
 * PARMS: szChunk (pre-tokenized chunk)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::EncodeBpeChunk(const std::string &szChunk) const -> std::vector<int32_t> {
    if (szChunk.empty())
        return {};

    std::vector<int32_t> viTokens;
    viTokens.reserve(szChunk.size());

    // split into individual chars, kys utf8
    size_t i = 0;
    while (i < szChunk.size()) {
        unsigned char c = (unsigned char)szChunk[i];
        int iLen = 1;
        if ((c & 0x80) == 0) // 0xxxxxxx 1b
            iLen = 1;
        else if ((c & 0xE0) == 0xC0) // 110xxxxx 2b
            iLen = 2;
        else if ((c & 0xF0) == 0xE0) // 1110xxxx 3b
            iLen = 3;
        else if ((c & 0xF8) == 0xF0) // 11110xxx 4b
            iLen = 4;
        if (i + iLen > szChunk.size())
            iLen = 1;

        std::string szChar = szChunk.substr(i, iLen);
        auto it = m_mTextToId.find(szChar);
        if (it != m_mTextToId.end()) {
            viTokens.push_back(it->second);
        } else {
            viTokens.push_back(m_iUnkId);
        }
        i += iLen;
    }

    if (viTokens.size() < 2)
        return viTokens;

    while (true) {
        int32_t iBestRank = INT32_MAX;
        int64_t lBestKey = -1;
        bool bFound = false;

        for (size_t j = 0; j < viTokens.size() - 1; j++) {
            int64_t lKey = lPackPair(viTokens[j], viTokens[j + 1]);
            auto itRank = m_mMergeRank.find(lKey);
            if (itRank != m_mMergeRank.end()) {
                if (itRank->second < iBestRank) {
                    iBestRank = itRank->second;
                    lBestKey = lKey;
                    bFound = true;
                }
            }
        }

        if (!bFound)
            break;

        auto itMerged = m_mMerges.find(lBestKey);
        int32_t iMergedId = itMerged->second;

        std::vector<int32_t> viNewTokens;
        viNewTokens.reserve(viTokens.size());

        size_t j = 0;
        while (j < viTokens.size()) {
            if (j + 1 < viTokens.size()) {
                int64_t lKey = lPackPair(viTokens[j], viTokens[j + 1]);
                if (lKey == lBestKey) {
                    viNewTokens.push_back(iMergedId);
                    j += 2;
                    continue;
                }
            }
            viNewTokens.push_back(viTokens[j]);
            j++;
        }
        viTokens = std::move(viNewTokens);
        if (viTokens.size() < 2)
            break;
    }

    return viTokens;
}

/*---------------------------------------------------------
 * FN: Decode
 * DESC: decodes a vector of token ids back into text,
 *       while also skippinf special tokens
 * PARMS: viTokens (token ID vector)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::Decode(const std::vector<int32_t> &viTokens) const -> std::string {
    std::string szOut;
    for (int32_t iToken : viTokens) {
        if (iToken < 0 || iToken >= (int32_t)m_vVocab.size())
            continue;
        const auto &info = m_vVocab[iToken];
        if (info.m_bSpecial)
            continue;
        szOut += info.m_szText;
    }
    if (m_bByteLevel) {
        szOut = szGptBytesToText(szOut);
    }
    return szOut;
}

/*---------------------------------------------------------
 * FN: DecodeToken
 * DESC: decodes a single token id into its text
 *       representation
 * PARMS: iToken (token ID)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::DecodeToken(int32_t iToken) const -> std::string {
    if (iToken < 0 || iToken >= (int32_t)m_vVocab.size())
        return "<INVALID>";
    const auto &info = m_vVocab[iToken];
    if (info.m_bSpecial)
        return info.m_szText;
    if (info.m_bAdded)
        return info.m_szText;

    std::string szOut = info.m_szText;
    if (m_bByteLevel) {
        szOut = szGptBytesToText(szOut);
    }
    return szOut;
}

/*---------------------------------------------------------
 * FN: iVocabSize
 * DESC: returns the total vocabulary size
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::iVocabSize() const -> int32_t { return (int32_t)m_vVocab.size(); }

/*---------------------------------------------------------
 * FN: iBosId
 * DESC: returns the bos token id
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::iBosId() const -> int32_t { return m_iBosId; }

/*---------------------------------------------------------
 * FN: iEosId
 * DESC: returns the eos token id
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::iEosId() const -> int32_t { return m_iEosId; }

/*---------------------------------------------------------
 * FN: iUnkId
 * DESC: returns the unknown token id
 * PARMS: none
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::iUnkId() const -> int32_t { return m_iUnkId; }

/*---------------------------------------------------------
 * FN: iLookup
 * DESC: looks up a token string and returns its id
 *       (or -1 if not found)
 * PARMS: szToken (token text to look up)
 * AUTH: unium (23.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::iLookup(const std::string &szToken) const -> int32_t {
    auto it = m_mTextToId.find(szToken);
    return (it != m_mTextToId.end()) ? it->second : -1;
}
// >>>s_end(shit)

// no touchy
/*---------------------------------------------------------
 * FN: AddToken
 * DESC: add token to vocab
 * PARMS: szText, iId, fScore
 * AUTH: unium (15.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::AddToken(const std::string &szText, int32_t iId, float fScore) {
    EnsureVocabSize(iId);
    m_vVocab[iId] = {szText, fScore, false, false};
    m_mTextToId[szText] = iId;
}

/*---------------------------------------------------------
 * FN: AddSpecialToken
 * DESC: add spectok to vocab
 * PARMS: szText, iId
 * AUTH: unium (15.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::AddSpecialToken(const std::string &szText, int32_t iId) {
    EnsureVocabSize(iId);
    m_vVocab[iId] = {szText, 0.0f, true, true};
    m_mTextToId[szText] = iId;
}

/*---------------------------------------------------------
 * FN: BuildFromVocab
 * DESC: make merge table from vocab
 * PARMS: nothing
 * AUTH: unium (15.02.26)
 *-------------------------------------------------------*/
void CBpeTokenizer::BuildFromVocab() {
    m_mMerges.clear();
    m_mMergeRank.clear();

    std::vector<std::pair<int32_t, float>> vSorted;
    for (int32_t i = 0; i < (int32_t)m_vVocab.size(); i++) {
        if (!m_vVocab[i].m_bSpecial && !m_vVocab[i].m_bAdded && m_vVocab[i].m_szText.size() > 1) {
            vSorted.push_back({i, m_vVocab[i].m_fScore});
        }
    }
    std::sort(vSorted.begin(), vSorted.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

    int32_t iRank = 0;
    for (auto &[iId, fScore] : vSorted) {
        const std::string &szText = m_vVocab[iId].m_szText;
        for (size_t iSplit = 1; iSplit < szText.size(); iSplit++) {
            if ((unsigned char)szText[iSplit] >= 0x80 && (unsigned char)szText[iSplit] < 0xC0)
                continue;
            std::string szL = szText.substr(0, iSplit);
            std::string szR = szText.substr(iSplit);
            auto itL = m_mTextToId.find(szL);
            auto itR = m_mTextToId.find(szR);
            if (itL != m_mTextToId.end() && itR != m_mTextToId.end()) {
                int64_t lKey = lPackPair(itL->second, itR->second);
                if (m_mMerges.find(lKey) == m_mMerges.end()) {
                    m_mMerges[lKey] = iId;
                    m_mMergeRank[lKey] = iRank++;
                }
                break;
            }
        }
    }

    m_vAddedTokens.clear();
    for (int32_t i = 0; i < (int32_t)m_vVocab.size(); i++) {
        if (m_vVocab[i].m_bAdded && !m_vVocab[i].m_szText.empty()) {
            m_vAddedTokens.push_back({m_vVocab[i].m_szText, i});
        }
    }
    std::sort(m_vAddedTokens.begin(), m_vAddedTokens.end(), [](const SAddedTokenEntry &a, const SAddedTokenEntry &b) {
        return a.m_szText.size() > b.m_szText.size();
    });
}

/*---------------------------------------------------------
 * FN: EncodeWithBos
 * DESC: encodes text with bostok
 * PARMS: szText
 * AUTH: unium (15.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::EncodeWithBos(const std::string &szText) const -> std::vector<int32_t> {
    auto vi = Encode(szText);
    vi.insert(vi.begin(), m_iBosId);
    return vi;
}

/*---------------------------------------------------------
 * FN: EncodeWithBosEos
 * DESC: encodes text + bos and eos toks
 * PARMS: szText (input text)
 * AUTH: unium (15.02.26)
 *-------------------------------------------------------*/
auto CBpeTokenizer::EncodeWithBosEos(const std::string &szText) const -> std::vector<int32_t> {
    auto vi = Encode(szText);
    vi.insert(vi.begin(), m_iBosId);
    vi.push_back(m_iEosId);
    return vi;
}

} // namespace TK
