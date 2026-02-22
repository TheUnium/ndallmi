// Created by Unium on 11.02.26

#include <cmath>
#include <cstring>
#include <iostream>

#include "Tests/tsTsTstf.hpp"

using namespace MT;

// <<<s_start(includes)
// --- test modules
TEST_MODULE("tensor/tensor")
#include "Tests/tsTnTnsr.hpp"

TEST_MODULE("tensor/ops")
#include "Tests/tsTnOps_.hpp"

TEST_MODULE("tensor/simd")
#include "Tests/tsTnSimd.hpp"

TEST_MODULE("thread/thread")
#include "Tests/tsThThrd.hpp"
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
