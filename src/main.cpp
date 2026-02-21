// Created by Unium on 11.02.26

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Tensor/mtTnTnsr.hpp"

using namespace MT;

// <<<s_start(testf)
// --- test framework
static int s_iTestsRun = 0;
static int s_iTestsPassed = 0;

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

#define TEST(name)                                                                                                     \
    static void Test_##name();                                                                                         \
    struct SReg_##name {                                                                                               \
        SReg_##name() {                                                                                                \
            s_iTestsRun++;                                                                                             \
            std::cout << "  [test] " << #name << "... ";                                                               \
            try {                                                                                                      \
                Test_##name();                                                                                         \
                s_iTestsPassed++;                                                                                      \
                std::cout << "passed!" << std::endl;                                                                   \
            } catch (const std::exception &e) {                                                                        \
                std::cout << "failed: " << e.what() << std::endl;                                                      \
            } catch (...) {                                                                                            \
                std::cout << "failed (?)" << std::endl;                                                                \
            }                                                                                                          \
        }                                                                                                              \
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
// >>>s_end(testf)

// <<<s_start(includes)
// --- test modules
#include "./Tests/tsTnTnsr.hpp"
// >>>s_end(includes)

// <<<s_start(main)
// --- main
/*---------------------------------------------------------
 * FN: main
 * DESC: runs all tests and benchmarks, prints summary
 * PARMS: none
 * AUTH: unium (11.02.26)
 *-------------------------------------------------------*/
int main() {
    std::cout << "  tensorlib tests" << std::endl;
    std::cout << std::endl;
    std::cout << "--- tests" << std::endl;
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
