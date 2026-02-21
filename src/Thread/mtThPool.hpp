// Created by Unium on 22.02.26

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace MT {
namespace TH {
/*---------------------------------------------------------
 * FN: iGetNumCores
 * DESC: returns the number of hardware threads available
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto iGetNumCores() -> int;

/*---------------------------------------------------------
 * FN: CThreadPool
 * DESC: thread pool stuff
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
class CThreadPool {
public:
    using TaskFn = std::function<void(int64_t, int64_t)>;

    /*---------------------------------------------------------
     * FN: CThreadPool
     * DESC: constructs a thread pool with iNumThreads workers
     * PARMS: iNumThreads (0 = autodetect)
     * AUTH: unium (22.02.26)
     *-------------------------------------------------------*/
    explicit CThreadPool(int iNumThreads = 0);

    /*---------------------------------------------------------
     * FN: ~CThreadPool
     * DESC: shuts down all worker threads
     * PARMS: none
     * AUTH: unium (22.02.26)
     *-------------------------------------------------------*/
    ~CThreadPool();

    // <<<ignore
    CThreadPool(const CThreadPool &) = delete;
    CThreadPool &operator=(const CThreadPool &) = delete;
    // >>>ignore

    /*---------------------------------------------------------
     * FN: ParallelFor
     * DESC: splits range [0, lTotal) across threads
     *       each thread calls lfnTask(lStart, lEnd)
     *       with its assigned bits and blocks until all done
     * PARMS: lTotal (total iterations), lfnTask (work fn)
     * AUTH: unium (22.02.26)
     *-------------------------------------------------------*/
    void ParallelFor(int64_t lTotal, TaskFn lfnTask);

    /*---------------------------------------------------------
     * FN: iNumThreads
     * DESC: returns num of worker threads
     * PARMS: none
     * AUTH: unium (22.02.26)
     *-------------------------------------------------------*/
    auto iNumThreads() const -> int;

private:
    struct STask {
        TaskFn m_lfnWork = nullptr;
        int64_t m_lStart = 0;
        int64_t m_lEnd = 0;
    };

    std::vector<std::thread> m_vThreads;
    int m_iNumThreads = 0;

    std::vector<STask> m_vTasks;
    std::mutex m_mtx;
    std::condition_variable m_cvWork;
    std::condition_variable m_cvDone;
    int m_iActiveThreads = 0;
    int m_iTasksRemaining = 0;
    int m_iGeneration = 0;
    bool m_bShutdown = false;

    /*---------------------------------------------------------
     * FN: WorkerLoop
     * DESC: main loop for each worker thread
     * PARMS: iThreadId (this threads idx)
     * AUTH: unium (22.02.26)
     *-------------------------------------------------------*/
    void WorkerLoop(int iThreadId);
};

/*---------------------------------------------------------
 * FN: GetGlobalPool
 * DESC: returns a reference to global pool
 * PARMS: none
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
auto GetGlobalPool() -> CThreadPool &;

/*---------------------------------------------------------
 * FN: ParFor
 * DESC: wrapper around gp PF
 * PARMS: lTotal (total), lfnTask (work fn)
 * AUTH: unium (22.02.26)
 *-------------------------------------------------------*/
void ParFor(int64_t lTotal, CThreadPool::TaskFn lfnTask);
} // namespace TH
} // namespace MT
