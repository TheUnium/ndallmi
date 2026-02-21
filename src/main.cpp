// Created by Unium on 11.02.26

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Tensor/mtTnTnsr.hpp"

using namespace MT;

// <<<s_start(testf)
// --- test framework
static int s_iTestsRun = 0;
static int s_iTestsPassed = 0;
static const char *s_szCurrentModule = nullptr;

struct STestEntry {
    const char *szName;
    const char *szModule;
    void (*pfnTest)();
};

static std::vector<STestEntry> &vtGetTests() {
    static std::vector<STestEntry> s_vtTests;
    return s_vtTests;
}

/*---------------------------------------------------------
 * FN: Check
 * DESC: asserts a condition or throws with message
 * PARMS: bCond (condition), szMsg (error message)
 * AUTH: unium (11.02.26)
 *-------------------------------------------------------*/
static void Check(bool bCond, const char *szMsg) {
    if (!bCond)
        throw std::runtime_error(szMsg);
}

/*---------------------------------------------------------
 * FN: CheckClose
 * DESC: asserts two floats are approximately equal
 * PARMS: fA (first), fB (second), fTol (tolerance)
 * AUTH: unium (11.02.26)
 *-------------------------------------------------------*/
static void CheckClose(float fA, float fB, float fTol = 1e-4f) {
    if (std::fabs(fA - fB) > fTol) {
        throw std::runtime_error("expected " + std::to_string(fA) + " ~ " + std::to_string(fB) +
                                 " (diff=" + std::to_string(std::fabs(fA - fB)) + ")");
    }
}

/*---------------------------------------------------------
 * FN: CheckTensorsClose
 * DESC: checks that two tensors have same shape and close vals
 * PARMS: tA (first), tB (second), fTol (tolerance)
 * AUTH: unium (11.02.26)
 *-------------------------------------------------------*/
static void CheckTensorsClose(const CTensor &tA, const CTensor &tB, float fTol = 1e-3f) {
    Check(tA.m_iNdim == tB.m_iNdim, "ndim mismatch");
    for (int i = 0; i < tA.m_iNdim; i++) {
        Check(tA.m_lShape[i] == tB.m_lShape[i], "shape mismatch");
    }
    for (int64_t i = 0; i < tA.lNumel(); i++) {
        float fDiff = std::fabs(tA.pfData()[i] - tB.pfData()[i]);
        if (fDiff > fTol) {
            throw std::runtime_error("tensor mismatch at flat[" + std::to_string(i) +
                                     "]: " + std::to_string(tA.pfData()[i]) + " vs " + std::to_string(tB.pfData()[i]) +
                                     " (diff=" + std::to_string(fDiff) + ")");
        }
    }
}

#define PP_CAT_I(a, b) a##b
#define PP_CAT(a, b) PP_CAT_I(a, b)

#define TEST_MODULE(mod)                                                                                               \
    namespace {                                                                                                        \
    static const char *PP_CAT(s_szMod_, __LINE__) = (s_szCurrentModule = mod);                                         \
    }

#define TEST(name)                                                                                                     \
    static void Test_##name();                                                                                         \
    struct SReg_##name {                                                                                               \
        SReg_##name() { vtGetTests().push_back({#name, s_szCurrentModule, Test_##name}); }                             \
    } s_reg_##name;                                                                                                    \
    static void Test_##name()

/*---------------------------------------------------------
 * FN: Bench
 * DESC: runs a function multiple times and prints avg time
 * PARMS: szName (label), iIters (iterations), lfnWork (func)
 * AUTH: unium (11.02.26)
 *-------------------------------------------------------*/
static void Bench(const char *szName, int iIters, std::function<void()> lfnWork) {
    for (int i = 0; i < 3; i++)
        lfnWork();
    auto tStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iIters; i++)
        lfnWork();
    auto tEnd = std::chrono::high_resolution_clock::now();
    double dMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "  [bench] " << szName << ": " << dMs / iIters << " ms avg (" << iIters << " iters)" << std::endl;
}

/*---------------------------------------------------------
 * FN: bMatchesFilter
 * DESC: checks if a test module matches the --tests filter
 * PARMS: szModule (module name), szFilter (filter string)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
static bool bMatchesFilter(const char *szModule, const char *szFilter) {
    if (szFilter == nullptr)
        return true;
    return std::strcmp(szModule, szFilter) == 0;
}

/*---------------------------------------------------------
 * FN: RunTests
 * DESC: runs all registered tests matching the filter
 * PARMS: szFilter (module filter, null for all)
 * AUTH: unium (14.02.26)
 *-------------------------------------------------------*/
static void RunTests(const char *szFilter) {
    const char *szLastModule = nullptr;
    for (auto &tEntry : vtGetTests()) {
        if (!bMatchesFilter(tEntry.szModule, szFilter))
            continue;

        if (szLastModule == nullptr || std::strcmp(szLastModule, tEntry.szModule) != 0) {
            std::cout << std::endl;
            std::cout << "--- " << tEntry.szModule << std::endl;
            szLastModule = tEntry.szModule;
        }

        s_iTestsRun++;
        std::cout << "  [test] " << tEntry.szName << "... ";
        try {
            tEntry.pfnTest();
            s_iTestsPassed++;
            std::cout << "passed!" << std::endl;
        } catch (const std::exception &e) {
            std::cout << "failed: " << e.what() << std::endl;
        } catch (...) {
            std::cout << "failed (?)" << std::endl;
        }
    }
}
// >>>s_end(testf)

// <<<s_start(includes)
// --- test modules
TEST_MODULE("tensor/tensor")
#include "Tests/tsTnTnsr.hpp"

TEST_MODULE("tensor/ops")
#include "Tests/tsTnOps_.hpp"
// >>>s_end(includes)

// <<<s_start(main)
// --- main
/*---------------------------------------------------------
 * FN: main
 * DESC: runs all tests and benchmarks, prints summary
 *       usage: ./llm --tests [module]
 *       modules: tensor/tensor, tensor/ops, or omit for all
 * PARMS: argc, argv
 * AUTH: unium (11.02.26)
 *-------------------------------------------------------*/
int main(int argc, char *argv[]) {
    const char *szFilter = nullptr;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--tests") == 0 && i + 1 < argc) {
            szFilter = argv[i + 1];
            i++;
        } else if (std::strcmp(argv[i], "--tests") == 0) {
        }
    }

    std::cout << "  tensorlib tests" << std::endl;
    if (szFilter) {
        std::cout << "  filter: " << szFilter << std::endl;
    }

    RunTests(szFilter);

    std::cout << std::endl;
    std::cout << "--- summary" << std::endl;
    std::cout << "  " << s_iTestsPassed << "/" << s_iTestsRun << " tests passed";
    if (s_iTestsPassed == s_iTestsRun) {
        std::cout << " ok";
    } else {
        std::cout << " (" << (s_iTestsRun - s_iTestsPassed) << " failed)";
    }
    std::cout << std::endl;
    std::cout << "--- end tests" << std::endl;

    return (s_iTestsPassed == s_iTestsRun) ? 0 : 1;
}
// >>>s_end(main)
