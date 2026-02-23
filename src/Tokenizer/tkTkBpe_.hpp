// Created by Unium on 23.02.26

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace TK {

struct SMerge {
    int32_t m_iLeft;
    int32_t m_iRight;
    int32_t m_iResult;
    int32_t m_iRank;
};

struct STokenInfo {
    std::string m_szText;
    float m_fScore;
    bool m_bSpecial; // true | skip during decode (eg: <|begin_of_text|>)
    bool m_bAdded;   // true | match as whole string during encode and never bpe split
};

struct SAddedTokenEntry {
    std::string m_szText;
    int32_t m_iId;
};

class CBpeTokenizer {
public:
    /*---------------------------------------------------------
     * FN: CBpeTokenizer
     * DESC: default constructor
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    CBpeTokenizer();

    /*---------------------------------------------------------
     * FN: ~CBpeTokenizer
     * DESC: destructor
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    ~CBpeTokenizer();

    // <<<s_start(shit)
    // --- loading
    /*---------------------------------------------------------
     * FN: bLoadFromJsonFile
     * DESC: loads a hf tokenizer.json file and populates vocab,
     *       merges and special tokens
     * PARMS: szPath (path to tokenizer.json)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto bLoadFromJsonFile(const std::string &szPath) -> bool;

    // --- encoding
    /*---------------------------------------------------------
     * FN: Encode
     * DESC: encodes text into a vector of token ids, and also
     *       handles added/special tokens and bpe stuff
     * PARMS: szText (input text)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto Encode(const std::string &szText) const -> std::vector<int32_t>;

    // --- decoding
    /*---------------------------------------------------------
     * FN: Decode
     * DESC: decodes a vector of token ids back into text,
     *       while also skippinf special tokens
     * PARMS: viTokens (token ID vector)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto Decode(const std::vector<int32_t> &viTokens) const -> std::string;

    /*---------------------------------------------------------
     * FN: DecodeToken
     * DESC: decodes a single token id into its text
     *       representation
     * PARMS: iToken (token ID)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto DecodeToken(int32_t iToken) const -> std::string;

    // --- info
    /*---------------------------------------------------------
     * FN: iVocabSize
     * DESC: returns the total vocabulary size
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto iVocabSize() const -> int32_t;

    /*---------------------------------------------------------
     * FN: iBosId
     * DESC: returns the bos token id
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto iBosId() const -> int32_t;

    /*---------------------------------------------------------
     * FN: iEosId
     * DESC: returns the eos token id
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto iEosId() const -> int32_t;

    /*---------------------------------------------------------
     * FN: iUnkId
     * DESC: returns the unknown token id
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto iUnkId() const -> int32_t;

    /*---------------------------------------------------------
     * FN: iLookup
     * DESC: looks up a token string and returns its id
     *       (or -1 if not found)
     * PARMS: szToken (token text to look up)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto iLookup(const std::string &szToken) const -> int32_t;
    // >>>s_end(shit)

    // old shit dont touch!!
    /*---------------------------------------------------------
     * FN: AddToken
     * DESC: add token to vocab
     * PARMS: szText, iId, fScore
     * AUTH: unium (15.02.26)
     *-------------------------------------------------------*/
    void AddToken(const std::string &szText, int32_t iId, float fScore = 0.0f);

    /*---------------------------------------------------------
     * FN: AddSpecialToken
     * DESC: add spectok to vocab
     * PARMS: szText, iId
     * AUTH: unium (15.02.26)
     *-------------------------------------------------------*/
    void AddSpecialToken(const std::string &szText, int32_t iId);

    /*---------------------------------------------------------
     * FN: BuildFromVocab
     * DESC: make merge table from vocab
     * PARMS: nothing
     * AUTH: unium (15.02.26)
     *-------------------------------------------------------*/
    void BuildFromVocab();

    /*---------------------------------------------------------
     * FN: EncodeWithBos
     * DESC: encodes text with bostok
     * PARMS: szText
     * AUTH: unium (15.02.26)
     *-------------------------------------------------------*/
    auto EncodeWithBos(const std::string &szText) const -> std::vector<int32_t>;

    /*---------------------------------------------------------
     * FN: EncodeWithBosEos
     * DESC: encodes text + bos and eos toks
     * PARMS: szText (input text)
     * AUTH: unium (15.02.26)
     *-------------------------------------------------------*/
    auto EncodeWithBosEos(const std::string &szText) const -> std::vector<int32_t>;

private:
    std::vector<STokenInfo> m_vVocab;
    std::unordered_map<std::string, int32_t> m_mTextToId;
    std::unordered_map<int64_t, int32_t> m_mMerges;
    std::unordered_map<int64_t, int32_t> m_mMergeRank;
    std::vector<SAddedTokenEntry> m_vAddedTokens;

    int32_t m_iBosId = 0;
    int32_t m_iEosId = 0;
    int32_t m_iUnkId = 0;

    bool m_bByteLevel = false;
    std::string m_vszByteToGpt[256];
    std::unordered_map<std::string, uint8_t> m_mGptToByte;

    int32_t m_viByteCharToId[256];
    struct SGptByteEntry {
        char m_szData[4];
        uint8_t m_iLen;
    };
    SGptByteEntry m_vGptBytesFast[256];

    uint8_t m_viGptToByteDirectAscii[128];
    bool m_vbGptToByteDirectValid[128];

    /*---------------------------------------------------------
     * FN: EnsureVocabSize
     * DESC: grows the vocab vector if needed to fit iId
     * PARMS: iId (token ID to ensure space for)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    void EnsureVocabSize(int32_t iId);

    /*---------------------------------------------------------
     * FN: lPackPair
     * DESC: packs two 32b token ids into a single 64b key for
     *       merge lookup
     * PARMS: iLeft (left token ID), iRight (right token ID)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    static auto lPackPair(int32_t iLeft, int32_t iRight) -> int64_t;

    /*---------------------------------------------------------
     * FN: BuildGptByteMap
     * DESC: builds gpt2 styled byte to unicode mapping tables
     * PARMS: none
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    void BuildGptByteMap();

    /*---------------------------------------------------------
     * FN: szTextToGptBytes
     * DESC: converts raw text bytes to gpt2 unicode encoding
     * PARMS: szIn (raw input text)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto szTextToGptBytes(const std::string &szIn) const -> std::string;

    /*---------------------------------------------------------
     * FN: szGptBytesToText
     * DESC: converts gpt2 unicode encoding back to raw bytes
     * PARMS: szIn (GPT-encoded input)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto szGptBytesToText(const std::string &szIn) const -> std::string;

    /*---------------------------------------------------------
     * FN: PreTokenize
     * DESC: splits text into chunks using gpt2/llama3 style
     *       pre tokenization regex patterns
     * PARMS: szText (input text)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto PreTokenize(const std::string &szText) const -> std::vector<std::string>;

    /*---------------------------------------------------------
     * FN: EncodeBpe
     * DESC: applies pre tokenization and bpe to a bit of text
     * PARMS: szText (input text segment)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto EncodeBpe(const std::string &szText) const -> std::vector<int32_t>;

    /*---------------------------------------------------------
     * FN: EncodeBpeChunk
     * DESC: applies bpe merges to a single pre tokenized chunk
     * PARMS: szChunk (pre-tokenized chunk)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    auto EncodeBpeChunk(const std::string &szChunk) const -> std::vector<int32_t>;

    // <<<s_start(json)
    // --- json helpers
    /*---------------------------------------------------------
     * FN: SkipWs
     * DESC: advances position past whitespace in a string
     * PARMS: s (json string), p (current position, modified)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    static void SkipWs(const std::string &s, size_t &p);

    /*---------------------------------------------------------
     * FN: ParseJsonString
     * DESC: parses a json quoted string at pos p
     * PARMS: s (json string), p (current position, modified)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    static auto ParseJsonString(const std::string &s, size_t &p) -> std::string;

    /*---------------------------------------------------------
     * FN: ParseJsonStringPtr
     * DESC: fast pointer based json string parser
     * PARMS: p (pointer, modified), pEnd (end of buffer)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    static auto ParseJsonStringPtr(const char *&p, const char *pEnd) -> std::string;

    /*---------------------------------------------------------
     * FN: SkipJsonValue
     * DESC: skips over an arbitrary json value at position p
     * PARMS: s (json string), p (current position, modified)
     * AUTH: unium (23.02.26)
     *-------------------------------------------------------*/
    static void SkipJsonValue(const std::string &s, size_t &p);
    // >>>s_end(json)

    static inline int iUtf8CharLen(unsigned char c) {
        if ((c & 0x80) == 0)
            return 1;
        if ((c & 0xE0) == 0xC0)
            return 2;
        if ((c & 0xF0) == 0xE0)
            return 3;
        if ((c & 0xF8) == 0xF0)
            return 4;
        return 1;
    }
};
} // namespace TK
