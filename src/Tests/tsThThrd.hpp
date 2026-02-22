// Created by Unium on 22.02.26

#pragma once

#include "../Thread/mtThPool.hpp"
#include "tsTsTstf.hpp"

#include <atomic>
#include <cmath>
#include <string>
#include <vector>

using namespace MT::TH;

// <<<s_start(tests)
// --- thread pool tests
// --- iGetNumCores
TEST(get_num_cores_positive) {
    int iCores = iGetNumCores();
    Check(iCores > 0, "iGetNumCores should return positive value");
}

TEST(get_num_cores_at_least_one) {
    int iCores = iGetNumCores();
    Check(iCores >= 1, "should have at least 1 core");
}

// --- constructors
TEST(pool_explicit_threads) {
    CThreadPool pool(4);
    Check(pool.iNumThreads() == 4, "should have 4 threads");
}

TEST(pool_single_thread) {
    CThreadPool pool(1);
    Check(pool.iNumThreads() == 1, "should have 1 thread");
}

TEST(pool_autodetect) {
    CThreadPool pool(0);
    Check(pool.iNumThreads() > 0, "autodetect should give positive thread count");
    Check(pool.iNumThreads() == iGetNumCores(), "autodetect should match iGetNumCores");
}

TEST(pool_negative_autodetect) {
    CThreadPool pool(-1);
    Check(pool.iNumThreads() > 0, "negative should autodetect");
    Check(pool.iNumThreads() == iGetNumCores(), "negative should match iGetNumCores");
}

TEST(pool_default_constructor) {
    CThreadPool pool;
    Check(pool.iNumThreads() > 0, "default should autodetect");
}

TEST(pool_many_threads) {
    CThreadPool pool(16);
    Check(pool.iNumThreads() == 16, "should have 16 threads");
}

// --- destructor / lifetime
TEST(pool_construct_destruct) {
    {
        CThreadPool pool(2);
    }
    Check(true, "pool constructed and destructed without issue");
}

TEST(pool_construct_destruct_no_work) {
    {
        CThreadPool pool(4);
    }
    Check(true, "pool destroyed with no work submitted");
}

// --- ParallelFor basic
TEST(parallel_for_zero_total) {
    CThreadPool pool(4);
    bool bCalled = false;
    pool.ParallelFor(0, [&](int64_t, int64_t) { bCalled = true; });
    Check(!bCalled, "should not call task for zero total");
}

TEST(parallel_for_negative_total) {
    CThreadPool pool(4);
    bool bCalled = false;
    pool.ParallelFor(-10, [&](int64_t, int64_t) { bCalled = true; });
    Check(!bCalled, "should not call task for negative total");
}

TEST(parallel_for_single_element) {
    CThreadPool pool(4);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(1, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(1, std::memory_order_relaxed);
    });
    Check(iSum.load() == 1, "single element should be processed once");
}

TEST(parallel_for_small_range) {
    CThreadPool pool(4);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(10, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(i, std::memory_order_relaxed);
    });
    Check(iSum.load() == 45, "sum of 0..9 should be 45");
}

TEST(parallel_for_covers_full_range) {
    CThreadPool pool(4);
    const int64_t lN = 1000;
    std::vector<std::atomic<int>> vHit(lN);
    for (int64_t i = 0; i < lN; i++)
        vHit[i].store(0, std::memory_order_relaxed);

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            vHit[i].fetch_add(1, std::memory_order_relaxed);
    });

    bool bAllHit = true;
    bool bNoDupes = true;
    for (int64_t i = 0; i < lN; i++) {
        if (vHit[i].load() == 0)
            bAllHit = false;
        if (vHit[i].load() > 1)
            bNoDupes = false;
    }
    Check(bAllHit, "every index should be hit exactly once");
    Check(bNoDupes, "no index should be processed more than once");
}

TEST(parallel_for_large_range) {
    CThreadPool pool(4);
    const int64_t lN = 100000;
    std::atomic<int64_t> iSum{0};

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSum.fetch_add(lLocal, std::memory_order_relaxed);
    });

    int64_t lExpected = (lN - 1) * lN / 2;
    Check(iSum.load() == lExpected, "sum should match arithmetic series");
}

TEST(parallel_for_exact_thread_count) {
    CThreadPool pool(4);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(4, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(i, std::memory_order_relaxed);
    });
    Check(iSum.load() == 6, "sum 0+1+2+3 = 6");
}

TEST(parallel_for_less_than_threads) {
    CThreadPool pool(8);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(3, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(i, std::memory_order_relaxed);
    });
    Check(iSum.load() == 3, "sum 0+1+2 = 3");
}

// --- ParallelFor with single thread pool (serial fallback path)
TEST(parallel_for_single_thread_pool) {
    CThreadPool pool(1);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(100, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(i, std::memory_order_relaxed);
    });
    int64_t lExpected = 99 * 100 / 2;
    Check(iSum.load() == lExpected, "single thread pool should still work");
}

// --- ParallelFor below threshold (< 64, runs inline)
TEST(parallel_for_below_threshold) {
    CThreadPool pool(4);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(63, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(i, std::memory_order_relaxed);
    });
    int64_t lExpected = 62 * 63 / 2;
    Check(iSum.load() == lExpected, "below threshold should run inline and produce correct sum");
}

TEST(parallel_for_at_threshold) {
    CThreadPool pool(4);
    std::atomic<int64_t> iSum{0};
    pool.ParallelFor(64, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iSum.fetch_add(i, std::memory_order_relaxed);
    });
    int64_t lExpected = 63 * 64 / 2;
    Check(iSum.load() == lExpected, "at threshold should use threads and produce correct sum");
}

// --- multiple calls on same pool
TEST(parallel_for_multiple_calls) {
    CThreadPool pool(4);

    for (int iter = 0; iter < 10; iter++) {
        std::atomic<int64_t> iSum{0};
        const int64_t lN = 1000;
        pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
            int64_t lLocal = 0;
            for (int64_t i = lStart; i < lEnd; i++)
                lLocal += i;
            iSum.fetch_add(lLocal, std::memory_order_relaxed);
        });
        int64_t lExpected = (lN - 1) * lN / 2;
        Check(iSum.load() == lExpected, ("iteration " + std::to_string(iter) + " should produce correct sum").c_str());
    }
}

TEST(parallel_for_many_sequential_calls) {
    CThreadPool pool(4);
    const int64_t lN = 500;

    for (int iter = 0; iter < 50; iter++) {
        std::atomic<int64_t> iSum{0};
        pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
            int64_t lLocal = 0;
            for (int64_t i = lStart; i < lEnd; i++)
                lLocal += i;
            iSum.fetch_add(lLocal, std::memory_order_relaxed);
        });
        int64_t lExpected = (lN - 1) * lN / 2;
        Check(iSum.load() == lExpected, ("call " + std::to_string(iter) + " correct").c_str());
    }
}

// --- correctness: write to array
TEST(parallel_for_write_array) {
    CThreadPool pool(4);
    const int64_t lN = 10000;
    std::vector<float> vf(lN, 0.0f);

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            vf[i] = (float)(i * i);
    });

    bool bCorrect = true;
    for (int64_t i = 0; i < lN; i++) {
        if (std::fabs(vf[i] - (float)(i * i)) > 1e-5f) {
            bCorrect = false;
            break;
        }
    }
    Check(bCorrect, "all elements should be i*i");
}

TEST(parallel_for_write_array_large) {
    CThreadPool pool(8);
    const int64_t lN = 1000000;
    std::vector<float> vf(lN, 0.0f);

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            vf[i] = (float)i + 0.5f;
    });

    bool bCorrect = true;
    for (int64_t i = 0; i < lN; i++) {
        if (std::fabs(vf[i] - ((float)i + 0.5f)) > 1e-5f) {
            bCorrect = false;
            break;
        }
    }
    Check(bCorrect, "large array should be correctly filled");
}

// --- non-overlapping ranges
TEST(parallel_for_no_overlap) {
    CThreadPool pool(4);
    const int64_t lN = 256;
    std::vector<std::atomic<int>> vCount(lN);
    for (int64_t i = 0; i < lN; i++)
        vCount[i].store(0, std::memory_order_relaxed);

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            vCount[i].fetch_add(1, std::memory_order_relaxed);
    });

    bool bAllOnce = true;
    for (int64_t i = 0; i < lN; i++) {
        if (vCount[i].load() != 1) {
            bAllOnce = false;
            break;
        }
    }
    Check(bAllOnce, "each index should be visited exactly once (no overlap)");
}

// --- global pool
TEST(global_pool_exists) {
    auto &pool = GetGlobalPool();
    Check(pool.iNumThreads() > 0, "global pool should have threads");
}

TEST(global_pool_same_instance) {
    auto &poolA = GetGlobalPool();
    auto &poolB = GetGlobalPool();
    Check(&poolA == &poolB, "global pool should return same instance");
}

TEST(global_pool_matches_cores) {
    auto &pool = GetGlobalPool();
    Check(pool.iNumThreads() == iGetNumCores(), "global pool should match core count");
}

// --- ParFor wrapper
TEST(parfor_basic) {
    std::atomic<int64_t> iSum{0};
    ParFor(1000, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSum.fetch_add(lLocal, std::memory_order_relaxed);
    });
    int64_t lExpected = 999 * 1000 / 2;
    Check(iSum.load() == lExpected, "ParFor should produce correct sum");
}

TEST(parfor_zero) {
    bool bCalled = false;
    ParFor(0, [&](int64_t, int64_t) { bCalled = true; });
    Check(!bCalled, "ParFor(0) should not call task");
}

TEST(parfor_negative) {
    bool bCalled = false;
    ParFor(-5, [&](int64_t, int64_t) { bCalled = true; });
    Check(!bCalled, "ParFor(-5) should not call task");
}

TEST(parfor_single) {
    std::atomic<int64_t> iCount{0};
    ParFor(1, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            iCount.fetch_add(1, std::memory_order_relaxed);
    });
    Check(iCount.load() == 1, "ParFor(1) should process one element");
}

TEST(parfor_write_array) {
    const int64_t lN = 5000;
    std::vector<double> vd(lN, 0.0);

    ParFor(lN, [&](int64_t lStart, int64_t lEnd) {
        for (int64_t i = lStart; i < lEnd; i++)
            vd[i] = std::sqrt((double)i);
    });

    bool bCorrect = true;
    for (int64_t i = 0; i < lN; i++) {
        if (std::fabs(vd[i] - std::sqrt((double)i)) > 1e-10) {
            bCorrect = false;
            break;
        }
    }
    Check(bCorrect, "ParFor should correctly compute sqrt for all elements");
}

// --- stress / edge cases
TEST(parallel_for_prime_total) {
    CThreadPool pool(4);
    const int64_t lN = 997;
    std::atomic<int64_t> iSum{0};

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSum.fetch_add(lLocal, std::memory_order_relaxed);
    });

    int64_t lExpected = (lN - 1) * lN / 2;
    Check(iSum.load() == lExpected, "prime-sized range should produce correct sum");
}

TEST(parallel_for_odd_thread_count) {
    CThreadPool pool(3);
    const int64_t lN = 10000;
    std::atomic<int64_t> iSum{0};

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSum.fetch_add(lLocal, std::memory_order_relaxed);
    });

    int64_t lExpected = (lN - 1) * lN / 2;
    Check(iSum.load() == lExpected, "odd thread count should work correctly");
}

TEST(parallel_for_two_threads) {
    CThreadPool pool(2);
    const int64_t lN = 10000;
    std::atomic<int64_t> iSum{0};

    pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSum.fetch_add(lLocal, std::memory_order_relaxed);
    });

    int64_t lExpected = (lN - 1) * lN / 2;
    Check(iSum.load() == lExpected, "two thread pool should produce correct sum");
}

TEST(parallel_for_varying_sizes) {
    CThreadPool pool(4);
    int64_t vlSizes[] = {1, 2, 3, 7, 15, 63, 64, 65, 100, 127, 128, 255, 256, 1000, 4096};

    for (int64_t lN : vlSizes) {
        std::atomic<int64_t> iSum{0};
        pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
            int64_t lLocal = 0;
            for (int64_t i = lStart; i < lEnd; i++)
                lLocal += i;
            iSum.fetch_add(lLocal, std::memory_order_relaxed);
        });
        int64_t lExpected = (lN - 1) * lN / 2;
        Check(iSum.load() == lExpected, ("size " + std::to_string(lN) + " should produce correct sum").c_str());
    }
}

// --- multiple pools simultaneously alive
TEST(multiple_pools) {
    CThreadPool poolA(2);
    CThreadPool poolB(4);

    std::atomic<int64_t> iSumA{0};
    std::atomic<int64_t> iSumB{0};

    poolA.ParallelFor(1000, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSumA.fetch_add(lLocal, std::memory_order_relaxed);
    });

    poolB.ParallelFor(1000, [&](int64_t lStart, int64_t lEnd) {
        int64_t lLocal = 0;
        for (int64_t i = lStart; i < lEnd; i++)
            lLocal += i;
        iSumB.fetch_add(lLocal, std::memory_order_relaxed);
    });

    int64_t lExpected = 999 * 1000 / 2;
    Check(iSumA.load() == lExpected, "pool A should produce correct sum");
    Check(iSumB.load() == lExpected, "pool B should produce correct sum");
}

// --- rapid create/destroy cycles
TEST(rapid_pool_lifecycle) {
    for (int i = 0; i < 20; i++) {
        CThreadPool pool(2);
        std::atomic<int64_t> iSum{0};
        pool.ParallelFor(200, [&](int64_t lStart, int64_t lEnd) {
            int64_t lLocal = 0;
            for (int64_t j = lStart; j < lEnd; j++)
                lLocal += j;
            iSum.fetch_add(lLocal, std::memory_order_relaxed);
        });
        int64_t lExpected = 199 * 200 / 2;
        Check(iSum.load() == lExpected, ("cycle " + std::to_string(i) + " should work").c_str());
    }
}
// >>>s_end(tests)

// <<<s_start(benches)
// --- benchmarks
TEST(bench_parfor_small) {
    CThreadPool pool(4);
    std::vector<float> vf(1024, 0.0f);

    Bench("ParFor 1024 elements", 10000, [&]() {
        pool.ParallelFor(1024, [&](int64_t lStart, int64_t lEnd) {
            for (int64_t i = lStart; i < lEnd; i++)
                vf[i] = (float)i * 0.5f;
        });
    });
}

TEST(bench_parfor_medium) {
    CThreadPool pool(4);
    std::vector<float> vf(65536, 0.0f);

    Bench("ParFor 64K elements", 5000, [&]() {
        pool.ParallelFor(65536, [&](int64_t lStart, int64_t lEnd) {
            for (int64_t i = lStart; i < lEnd; i++)
                vf[i] = (float)i * 0.5f;
        });
    });
}

TEST(bench_parfor_large) {
    CThreadPool pool(4);
    const int64_t lN = 1000000;
    std::vector<float> vf(lN, 0.0f);

    Bench("ParFor 1M elements", 500, [&]() {
        pool.ParallelFor(lN, [&](int64_t lStart, int64_t lEnd) {
            for (int64_t i = lStart; i < lEnd; i++)
                vf[i] = std::sqrt((float)i);
        });
    });
}

TEST(bench_parfor_global) {
    const int64_t lN = 1000000;
    std::vector<float> vf(lN, 0.0f);

    Bench("ParFor global 1M", 500, [&]() {
        ParFor(lN, [&](int64_t lStart, int64_t lEnd) {
            for (int64_t i = lStart; i < lEnd; i++)
                vf[i] = std::sqrt((float)i);
        });
    });
}

TEST(bench_parfor_overhead) {
    CThreadPool pool(4);
    volatile float kys;

    Bench("ParFor overhead (trivial work)", 50000, [&]() {
        pool.ParallelFor(256, [&](int64_t lStart, int64_t lEnd) {
            float fS = 0.0f;
            for (int64_t i = lStart; i < lEnd; i++)
                fS += (float)i;
            kys = fS;
        });
    });
}

TEST(bench_parfor_serial_baseline) {
    const int64_t lN = 1000000;
    std::vector<float> vf(lN, 0.0f);

    Bench("serial 1M sqrt", 500, [&]() {
        for (int64_t i = 0; i < lN; i++)
            vf[i] = std::sqrt((float)i);
    });
}

TEST(bench_pool_create_destroy) {
    Bench("pool create+destroy (4 threads)", 1000, []() {
        CThreadPool pool(4);
        (void)pool;
    });
}

TEST(bench_parfor_many_calls) {
    CThreadPool pool(4);
    std::vector<float> vf(1024, 0.0f);

    Bench("100x ParFor 1024", 100, [&]() {
        for (int iter = 0; iter < 100; iter++) {
            pool.ParallelFor(1024, [&](int64_t lStart, int64_t lEnd) {
                for (int64_t i = lStart; i < lEnd; i++)
                    vf[i] = (float)i;
            });
        }
    });
}
// >>>s_end(benches)
