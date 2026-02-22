// Created by Unium on 22.02.26

#include "mtThPool.hpp"

#include <algorithm>

// <<<s_start(spin)
// --- spin-wait macro: brief spin with pause
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#else
#define _mm_pause() ((void)0)
#endif
// >>>s_end(spin)

namespace MT {
namespace TH {
/*---------------------------------------------------------
 * FN: iGetNumCores
 * DESC: returns the number of hardware threads available
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto iGetNumCores() -> int {
    int iCores = (int)std::thread::hardware_concurrency();
    return (iCores > 0) ? iCores : 4;
}

/*---------------------------------------------------------
 * FN: iNumThreads
 * DESC: returns num of worker threads
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto CThreadPool::iNumThreads() const -> int { return m_iNumThreads; }

/*---------------------------------------------------------
 * FN: CThreadPool
 * DESC: constructs a thread pool with iNumThreads workers
 * PARMS: iNumThreads (0 = autodetect)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
CThreadPool::CThreadPool(int iNumThreads) {
    if (iNumThreads <= 0)
        iNumThreads = iGetNumCores();
    m_iNumThreads = iNumThreads;
    m_vWorkers = std::make_unique<SWorkerData[]>(iNumThreads);
    m_vThreads.reserve(iNumThreads);

    for (int i = 0; i < iNumThreads; i++) {
        m_vThreads.emplace_back(&CThreadPool::WorkerLoop, this, i);
    }
}

/*---------------------------------------------------------
 * FN: ~CThreadPool
 * DESC: shuts down all worker threads
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
CThreadPool::~CThreadPool() {
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        for (int i = 0; i < m_iNumThreads; i++) {
            m_vWorkers[i].m_iState.store(2, std::memory_order_release);
        }
    }
    m_cvWork.notify_all();
    for (auto &t : m_vThreads) {
        if (t.joinable())
            t.join();
    }
}

/*---------------------------------------------------------
 * FN: ParallelFor
 * DESC: splits range [0, lTotal) across threads
 *       each thread calls lfnTask(lStart, lEnd)
 *       with its assigned bits and blocks until all done
 * PARMS: lTotal (total iterations), lfnTask (work fn)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void CThreadPool::ParallelFor(int64_t lTotal, TaskFn lfnTask) {
    if (lTotal <= 0)
        return;

    if (lTotal < 64 || m_iNumThreads <= 1) {
        lfnTask(0, lTotal);
        return;
    }

    int iThreadsToUse = std::min(m_iNumThreads, (int)lTotal);
    int64_t lChunk = (lTotal + iThreadsToUse - 1) / iThreadsToUse;

    m_fnCurrentTask = lfnTask;
    m_iDoneCount.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);

    {
        for (int i = 0; i < iThreadsToUse; i++) {
            int64_t lStart = i * lChunk;
            int64_t lEnd = std::min(lStart + lChunk, lTotal);
            m_vWorkers[i].m_lStart.store(lStart, std::memory_order_relaxed);
            m_vWorkers[i].m_lEnd.store(lEnd, std::memory_order_relaxed);
            m_vWorkers[i].m_iState.store(1, std::memory_order_release);
        }

        m_cvWork.notify_all();
    }

    int iSpins = 0;
    while (m_iDoneCount.load(std::memory_order_acquire) < iThreadsToUse) {
        if (iSpins < 2000) {
            _mm_pause();
            iSpins++;
        } else {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cvDone.wait(lock, [&]() { return m_iDoneCount.load(std::memory_order_acquire) >= iThreadsToUse; });
        }
    }
}

/*---------------------------------------------------------
 * FN: WorkerLoop
 * DESC: main loop for each worker thread. uses hybrid
 *       polling (spin -> yield -> sleep) to minimize
 *       latency while preventing cpu starvation.
 * PARMS: iThreadId (this threads idx)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void CThreadPool::WorkerLoop(int iThreadId) {
    SWorkerData &w = m_vWorkers[iThreadId];

    while (true) {
        int s = w.m_iState.load(std::memory_order_acquire);
        if (s == 1)
            goto work;
        if (s == 2)
            return;

        for (int sp = 0; sp < 2000; sp++) {
            _mm_pause();
            s = w.m_iState.load(std::memory_order_acquire);
            if (s == 1)
                goto work;
            if (s == 2)
                return;
        }

        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cvWork.wait(lock, [&]() {
                int st = w.m_iState.load(std::memory_order_acquire);
                return st != 0;
            });
            s = w.m_iState.load(std::memory_order_acquire);
        }

        if (s == 2)
            return;
    work: {
        int64_t lStart = w.m_lStart.load(std::memory_order_relaxed);
        int64_t lEnd = w.m_lEnd.load(std::memory_order_relaxed);
        if (lStart < lEnd) {
            m_fnCurrentTask(lStart, lEnd);
        }
    }

        w.m_iState.store(0, std::memory_order_release);
        int iPrev = m_iDoneCount.fetch_add(1, std::memory_order_release);
        if (iPrev + 1 >= m_iNumThreads) {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_cvDone.notify_one();
        }
    }
}

/*---------------------------------------------------------
 * FN: GetGlobalPool
 * DESC: returns a reference to global pool
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto GetGlobalPool() -> CThreadPool & {
    static CThreadPool s_pool(0);
    return s_pool;
}

/*---------------------------------------------------------
 * FN: ParFor
 * DESC: wrapper around gp PF
 * PARMS: lTotal (total), lfnTask (work fn)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void ParFor(int64_t lTotal, CThreadPool::TaskFn lfnTask) { GetGlobalPool().ParallelFor(lTotal, lfnTask); }
} // namespace TH
} // namespace MT
