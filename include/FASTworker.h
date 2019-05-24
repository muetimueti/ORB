#ifndef ORBEXTRACTOR_FASTWORKER_H
#define ORBEXTRACTOR_FASTWORKER_H

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/core/core.hpp>
#include <future>

//TODO: remove include
#include <iostream>

const int CIRCLE_SIZE = 16;

const int CIRCLE_OFFSETS[16][2] =
        {{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
         {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}};

const int PIXELS_TO_CHECK[16] =
        {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15};

typedef struct LineResult
{
uchar* scores;
int* pos;
int ncandidates;
} LineResult;

class FASTworker
{
    struct FASTargs
    {
        uchar* ptr;
        int* offset;
        int threshold;
        int cols;
        LineResult* res;
    };


public:
    FASTworker(size_t nThreads, size_t th_ini, size_t th_min) : threads(nThreads),
        th_tab_init{}, th_tab_min{}, quit(false)
    {
        for (size_t i = 0; i < nThreads; ++i)
        {
            threads[i] = std::thread(&FASTworker::WorkWork, this);
        }

        for (size_t i = 0; i < 512; ++i)
        {
            int v = i - 255;
            if (v < -th_ini)
            {
                th_tab_init[i] = (uchar)1;
                th_tab_min[i] = (uchar)1;
            }
            else if (v > th_ini)
            {
                th_tab_init[i] = (uchar)2;
                th_tab_min[i] = (uchar)2;

            }
            else
            {
                th_tab_init[i] = (uchar)0;
                if (v < -th_min)
                {
                    th_tab_min[i] = (uchar)1;
                }
                else if (v > th_min)
                {
                    th_tab_min[i] = (uchar)2;
                }
                else
                    th_tab_min[i] = (uchar)0;
            }
        }
    }

    ~FASTworker()
    {
        quit = true;
        queuelock.unlock();
        cond.notify_all();
        for (auto &t : threads)
        {
            if (t.joinable())
                t.join();
        }
    }

    void PushLine(uchar* ptr, int offset[], int threshold, int cols, LineResult* res)
    {
        std::unique_lock<std::mutex> l(queuelock);
        FASTargs a{ptr, offset, threshold, };
        lineQueue.push(a);
        l.unlock();
        cond.notify_all();
    }

    void Reset()
    {
        //TODO: implement
        std::unique_lock<std::mutex> l(queuelock);
    }


private:

    void WorkWork()
    {
        std::unique_lock<std::mutex> lock(queuelock);

        while (!quit)
        {
            cond.wait(lock, [this]{return (!lineQueue.empty() || quit);});

            if (!lineQueue.empty())
            {
                auto line = lineQueue.front();
                lineQueue.pop();

                lock.unlock();
                processLine(line.ptr, line.offset, line.threshold, line.cols, line.res);
                lock.lock();
            }
        }
    }

    std::queue<FASTargs> lineQueue;

    std::vector<std::thread> threads;

    std::mutex queuelock;
    std::condition_variable cond;

    uchar th_tab_init[512];
    uchar th_tab_min[512];

    bool quit;

    void processLine(uchar* ptr, int offset[], int threshold, int cols, LineResult* res)
    {
        memset(&res->scores, 0, cols);
        memset(&res->pos, 0, cols);
        std::cout << "processing line...\n";
    }
};


#endif //ORBEXTRACTOR_FASTWORKER_H
