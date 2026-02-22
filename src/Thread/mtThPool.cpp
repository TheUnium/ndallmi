// Created by Unium on 22.02.26

#include "mtThPool.hpp"

#include <algorithm>

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
    if (iNumThreads <= 0) {
        iNumThreads = iGetNumCores();
    }
    m_iNumThreads = iNumThreads;
    m_vTasks.resize(iNumThreads);
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
        m_bShutdown = true;
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

    {
        std::lock_guard<std::mutex> lock(m_mtx);

        m_iActiveThreads = iThreadsToUse;

        for (int i = 0; i < iThreadsToUse; i++) {
            int64_t lStart = i * lChunk;
            int64_t lEnd = std::min(lStart + lChunk, lTotal);
            m_vTasks[i].m_lfnWork = lfnTask;
            m_vTasks[i].m_lStart = lStart;
            m_vTasks[i].m_lEnd = lEnd;
        }

        m_iTasksRemaining = iThreadsToUse;
        m_iGeneration++;
    }

    m_cvWork.notify_all();

    {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_cvDone.wait(lock, [&]() { return m_iTasksRemaining == 0; });
    }
}

/*---------------------------------------------------------
 * FN: WorkerLoop
 * DESC: main loop for each worker thread
 * PARMS: iThreadId (this threads idx)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void CThreadPool::WorkerLoop(int iThreadId) {
    int iLastGen = 0;

    while (true) {
        TaskFn lfnWork;
        int64_t lStart = 0;
        int64_t lEnd = 0;
        bool bParticipate = false;

        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cvWork.wait(lock, [&]() { return m_bShutdown || m_iGeneration != iLastGen; });

            if (m_bShutdown)
                return;

            iLastGen = m_iGeneration;

            if (iThreadId < m_iActiveThreads) {
                lfnWork = m_vTasks[iThreadId].m_lfnWork;
                lStart = m_vTasks[iThreadId].m_lStart;
                lEnd = m_vTasks[iThreadId].m_lEnd;
                bParticipate = true;
            }
        }

        if (!bParticipate) {
            continue;
        }

        if (lfnWork && lStart < lEnd) {
            lfnWork(lStart, lEnd);
        }

        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_iTasksRemaining--;
            if (m_iTasksRemaining == 0) {
                m_cvDone.notify_one();
            }
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
