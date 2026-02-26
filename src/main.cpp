// Created by Unium on 26.02.26

#include "Model/mdMdInfr.hpp"
#include "Model/mdMdLoad.hpp"
#include "Tokenizer/tkTkBpe_.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

// <<<s_start(helpers)
// --- cli helpers
/*---------------------------------------------------------
 * FN: PrintUsage
 * DESC: prints usage info to stdout
 * PARMS: szProgName (argv[0])
 * AUTH: unium (26.02.26)
 *-------------------------------------------------------*/
// --- from ../../llmcpp/src/main.cpp
static void PrintUsage(const char *szProgName) {
    std::cout << "usage:" << std::endl;
    std::cout << "  " << szProgName << " <model_dir> <tokenizer.json> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "options:" << std::endl;
    std::cout << "  --q8                  quantize weights to int8" << std::endl;
    std::cout << "  --max-tokens <int>    max generation tokens (default: 512)" << std::endl;
    std::cout << "  --temp <float>        temperature (default: 0.7)" << std::endl;
    std::cout << "  --top-k <int>         top-k (default: 40)" << std::endl;
    std::cout << "  --top-p <float>       top-p (default: 0.9)" << std::endl;
    std::cout << "  --rep-penalty <float> repetition penalty (default: 1.2)" << std::endl;
    std::cout << "  --system <string>     system prompt (default: built-in)" << std::endl;
}

/*---------------------------------------------------------
 * FN: szBuildChatPrompt
 * DESC: builds chat prompt with im_start/im_end tags from
 *       conversation history
 * PARMS: vszHistory (alternating user/assistant turns),
 *        szSystem (system prompt)
 * AUTH: unium (26.02.26)
 *-------------------------------------------------------*/
// --- from ../../llmcpp/src/main.cpp
static auto szBuildChatPrompt(const std::vector<std::pair<std::string, std::string>> &vHistory,
                              const std::string &szSystem) -> std::string {
    std::string sz;

    sz += "<|im_start|>system\n";
    sz += szSystem;
    sz += "\n<|im_end|>\n";

    for (const auto &sTurn : vHistory) {
        sz += "<|im_start|>user\n";
        sz += sTurn.first;
        sz += "<|im_end|>\n";

        if (!sTurn.second.empty()) {
            sz += "<|im_start|>assistant\n";
            sz += sTurn.second;
            sz += "<|im_end|>\n";
        }
    }

    sz += "<|im_start|>assistant\n";
    return sz;
}
// >>>s_end(helpers)

// <<<s_start(generate)
// --- generation
// --- from ../../llmcpp/src/main.cpp
struct SGenResult {
    std::string szText;
    int32_t iTokens = 0;
    double dTotalMs = 0.0;
    double dPrefillMs = 0.0;
    double dGenMs = 0.0;
    double dTokPerSec = 0.0;
};

/*---------------------------------------------------------
 * FN: sGenerate
 * DESC: runs full generation loop, prints tokens as they
 *       come (streaming to stdout)
 * PARMS: sInfer (inference state), tok (tokenizer),
 *        szPrompt (full chat prompt), sSampler (config),
 *        iMaxTokens (generation limit)
 * AUTH: unium (26.02.26)
 *-------------------------------------------------------*/
// --- from ../../llmcpp/src/main.cpp
static auto sGenerate(MD::CInferState &sInfer, const TK::CBpeTokenizer &tok, const std::string &szPrompt,
                      const MD::SSamplerConfig &sSampler, int32_t iMaxTokens) -> SGenResult {
    SGenResult sResult;

    sInfer.Reset();

    auto viPrompt = tok.Encode(szPrompt);
    if (viPrompt.empty())
        return sResult;

    auto tStart = std::chrono::high_resolution_clock::now();

    MT::CTensor tLogits;
    for (size_t i = 0; i < viPrompt.size(); i++)
        tLogits = sInfer.Forward(viPrompt[i]);

    auto tPrefillEnd = std::chrono::high_resolution_clock::now();
    sResult.dPrefillMs = std::chrono::duration<double, std::milli>(tPrefillEnd - tStart).count();

    int32_t iEosId = tok.iEosId();
    std::string szGenerated;

    for (int32_t iStep = 0; iStep < iMaxTokens; iStep++) {
        int32_t iNext = sInfer.iSample(tLogits, sSampler);

        if (iNext == iEosId)
            break;

        std::string szToken = tok.DecodeToken(iNext);

        if (szToken == "<|im_end|>" || szToken == "<|endoftext|>" || szToken == "<|im_start|>" ||
            szToken == "<|end_of_text|>")
            break;

        std::cout << szToken << std::flush;

        szGenerated += szToken;
        sResult.iTokens++;

        tLogits = sInfer.Forward(iNext);
    }

    std::cout << std::endl;

    auto tEnd = std::chrono::high_resolution_clock::now();
    sResult.dTotalMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    sResult.dGenMs = std::chrono::duration<double, std::milli>(tEnd - tPrefillEnd).count();
    sResult.dTokPerSec = sResult.iTokens > 0 ? sResult.iTokens / (sResult.dGenMs / 1000.0) : 0.0;
    sResult.szText = szGenerated;

    return sResult;
}
// >>>s_end(generate)

// <<<s_start(main)
// --- entry point
/*---------------------------------------------------------
 * FN: main
 * DESC: interactive chat loop. loads model + tokenizer,
 *       then reads user input in a loop and generates
 *       responses with streaming output.
 * PARMS: argc, argv
 * AUTH: unium (26.02.26)
 *-------------------------------------------------------*/
// --- from ../../llmcpp/src/main.cpp
int main(int argc, char *argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        PrintUsage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    std::string szModelDir = argv[1];

    if (argc < 3) {
        std::cerr << "  error: need tokenizer path as second argument" << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }

    std::string szTokPath = argv[2];

    MD::SSamplerConfig sSampler;
    sSampler.fTemperature = 0.7f;
    sSampler.fTopP = 0.9f;
    sSampler.iTopK = 40;
    sSampler.fRepPenalty = 1.2f;

    int32_t iMaxTokens = 512;
    bool bQuantize = false;
    std::string szSystem = "You are a helpful assistant.";

    for (int i = 3; i < argc; i++) {
        std::string szArg = argv[i];

        if (szArg == "--q8" || szArg == "--quantize") {
            bQuantize = true;
        } else if (szArg == "--max-tokens" && i + 1 < argc) {
            iMaxTokens = std::stoi(argv[++i]);
        } else if (szArg == "--temp" && i + 1 < argc) {
            sSampler.fTemperature = std::stof(argv[++i]);
        } else if (szArg == "--top-k" && i + 1 < argc) {
            sSampler.iTopK = std::stoi(argv[++i]);
        } else if (szArg == "--top-p" && i + 1 < argc) {
            sSampler.fTopP = std::stof(argv[++i]);
        } else if (szArg == "--rep-penalty" && i + 1 < argc) {
            sSampler.fRepPenalty = std::stof(argv[++i]);
        } else if (szArg == "--seed" && i + 1 < argc) {
            sSampler.iSeed = std::stoi(argv[++i]);
        } else if (szArg == "--system" && i + 1 < argc) {
            szSystem = argv[++i];
        }
    }

    MD::SModel sModel;
    if (!MD::bLoadModel(szModelDir, sModel)) {
        std::cerr << "  error: failed to load model" << std::endl;
        return 1;
    }
    std::cout << std::endl;

    if (bQuantize) {
        MD::QuantizeModel(sModel);
        std::cout << std::endl;
    }

    MD::PrintModelInfo(sModel);

    TK::CBpeTokenizer tok;
    std::string szTokFile = szTokPath;
    if (std::filesystem::is_directory(szTokFile))
        szTokFile += "/tokenizer.json";

    if (!tok.bLoadFromJsonFile(szTokFile)) {
        std::cerr << "  error: failed to load tokenizer from " << szTokFile << std::endl;
        return 1;
    }
    std::cout << std::endl;

    MD::CInferState sInfer(&sModel);

    std::cout << "  config: temp=" << sSampler.fTemperature << " top_k=" << sSampler.iTopK
              << " top_p=" << sSampler.fTopP << " rep=" << sSampler.fRepPenalty << " max=" << iMaxTokens << std::endl;
    std::cout << "  system: " << szSystem.substr(0, 80) << (szSystem.size() > 80 ? "..." : "") << std::endl;
    std::cout << std::endl;
    std::cout << "  type your message and press enter. /quit to exit, /clear to reset." << std::endl;
    std::cout << std::endl;

    std::vector<std::pair<std::string, std::string>> vHistory;

    while (true) {
        std::cout << "> " << std::flush;

        std::string szInput;
        if (!std::getline(std::cin, szInput))
            break;

        size_t iStart = szInput.find_first_not_of(" \t\n\r");
        if (iStart == std::string::npos)
            continue;
        size_t iEnd = szInput.find_last_not_of(" \t\n\r");
        szInput = szInput.substr(iStart, iEnd - iStart + 1);

        if (szInput.empty())
            continue;

        if (szInput == "/quit" || szInput == "/exit" || szInput == "/q")
            break;

        if (szInput == "/clear" || szInput == "/reset") {
            vHistory.clear();
            std::cout << "  conversation cleared." << std::endl;
            std::cout << std::endl;
            continue;
        }

        if (szInput == "/help") {
            std::cout << "  commands:" << std::endl;
            std::cout << "    /quit    exit" << std::endl;
            std::cout << "    /clear   reset conversation" << std::endl;
            std::cout << "    /help    show this" << std::endl;
            std::cout << std::endl;
            continue;
        }

        vHistory.push_back({szInput, ""});

        std::string szPrompt = szBuildChatPrompt(vHistory, szSystem);

        std::cout << std::endl;
        auto sResult = sGenerate(sInfer, tok, szPrompt, sSampler, iMaxTokens);

        vHistory.back().second = sResult.szText;

        std::cout << std::endl;
        std::cout << "  [" << sResult.iTokens << " tokens, " << std::fixed << std::setprecision(0) << sResult.dPrefillMs
                  << " ms prefill, " << std::setprecision(0) << sResult.dGenMs << " ms gen, " << std::setprecision(1)
                  << sResult.dTokPerSec << " tok/s]" << std::endl;
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "  bye bye!" << std::endl;
    return 0;
}
// >>>s_end(main)
