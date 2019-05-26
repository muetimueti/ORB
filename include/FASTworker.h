#ifndef ORBEXTRACTOR_FASTWORKER_H
#define ORBEXTRACTOR_FASTWORKER_H

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/core/core.hpp>
#include <future>
//#include "include/avx.h"

//TODO: remove include
#include <iostream>
#include "FAST.h"

const int CIRCLE_SIZE = 16;

const int CIRCLE_OFFSETS[16][2] =
        {{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
         {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}};

const int PIXELS_TO_CHECK[16] =
        {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15};

typedef struct LineResult
{
std::vector<uchar> scores;
//uchar* scores;
std::vector<int> pos;
//int* pos;
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
        std::promise<bool>* pr;
        LineResult* res;
    };

    struct FASTargs_img
    {
        cv::Mat* img;
        std::vector<cv::KeyPoint>* kpts;
        int* offset;
        int threshold;
        int lvl;
        std::promise<bool>* pr;

    };


public:
    FASTworker(size_t nThreads, size_t _th_ini, size_t _th_min) : threads(nThreads), th_tab_init{}, th_tab_min{},
        th_ini(_th_ini), th_min(_th_min), contP(CIRCLE_SIZE/2), onePointFiveCircles(contP + CIRCLE_SIZE + 1),
        quit(false), is_img_queue(true)
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

    void PushLine(uchar* ptr, int offset[], int threshold, int cols, std::promise<bool>* p, LineResult* res)
    {
        assert(!is_img_queue);

        std::unique_lock<std::mutex> l(queuelock);
        FASTargs a{ptr, offset, threshold, cols, p, res};
        //queue.push(a);
        l.unlock();
        cond.notify_all();
    }

    void PushImg(cv::Mat img, std::vector<cv::KeyPoint> kpts, int offset[], int threshold, int lvl,
            std::promise<bool>* p)
    {
        std::unique_lock<std::mutex> l(queuelock);
        FASTargs_img a{&img, &kpts, offset, threshold, lvl, };

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
            cond.wait(lock, [this]{return (!queue.empty() || quit);});

            if (!queue.empty())
            {
                //auto line = queue.front();
                auto args = queue.front();
                queue.pop();

                lock.unlock();
                //processLine(line.ptr, line.offset, line.threshold, line.cols, line.pr, line.res);
                processImg(*args.img, *args.kpts, args.offset, args.threshold, args.lvl, *args.pr);

                lock.lock();
            }
        }
    }

    std::queue<FASTargs_img> queue;

    std::vector<std::thread> threads;

    std::mutex queuelock;
    std::condition_variable cond;

    uchar th_tab_init[512];
    uchar th_tab_min[512];

    size_t th_ini;
    size_t th_min;

    size_t contP;
    size_t onePointFiveCircles;

    bool is_img_queue;

    bool quit;

    void processLine(uchar* ptr, int offset[], int threshold, int cols, std::promise<bool>* pr, LineResult* res)
    {
        /*
        uchar sc[cols];
        int ps[cols];
        memset(sc, 0, cols);
        memset(ps, 0, cols);
        */
        res->scores.resize(cols);
        res->pos.resize(cols);
        res->ncandidates = 0;


        uchar *threshold_tab;
        if (threshold == th_ini)
            threshold_tab = th_tab_init;
        else
            threshold_tab = th_tab_min;

        size_t x, k;
        for (x = 3; x < cols-3; ++x, ++ptr)
        {
            int val = ptr[0];                           //value of central pixel
            const uchar *tab = &threshold_tab[255] - val;       //shift threshold tab by val


            int discard = tab[ptr[offset[PIXELS_TO_CHECK[0]]]]
                          | tab[ptr[offset[PIXELS_TO_CHECK[1]]]];

            if (discard == 0)
                continue;

            bool gotoNextCol = false;
            for (k = 2; k < 16; k+=2)
            {
                discard &= tab[ptr[offset[PIXELS_TO_CHECK[k]]]]
                           | tab[ptr[offset[PIXELS_TO_CHECK[k+1]]]];
                if (k == 6 && discard == 0)
                {
                    gotoNextCol = true;
                    break;
                }
                if (k == 14 && discard == 0)
                {
                    gotoNextCol = true;
                }
            }
            if (gotoNextCol) // initial FAST-check failed
                continue;


            if (discard & 1) // check for continuous circle of pixels darker than threshold
            {
                size_t compare = val - threshold;
                size_t contPixels = 0;

                for (k = 0; k < onePointFiveCircles; ++k)
                {
                    int a = ptr[offset[k%CIRCLE_SIZE]];
                    if (a < compare)
                    {
                        ++contPixels;
                        if (contPixels > contP)
                        {
                            res->pos[res->ncandidates++] = x;
                            res->scores[x] = CornerScore(ptr, offset, threshold);
                            break;
                        }
                    } else
                        contPixels = 0;
                }
            }

            if (discard & 2) // check for continuous circle of pixels brighter than threshold
            {
                size_t compare = val + threshold;
                size_t contPixels = 0;

                for (k = 0; k < onePointFiveCircles; ++k)
                {
                    int a = ptr[offset[k%CIRCLE_SIZE]];
                    if (a < compare)
                    {
                        ++contPixels;
                        if (contPixels > contP)
                        {
                            res->pos[res->ncandidates++] = x;
                            res->scores[x] = CornerScore(ptr, offset, threshold);
                            break;
                        }
                    } else
                        contPixels = 0;
                }
            }
        }
        pr->set_value(true);
    }

    void processImg(cv::Mat& img, std::vector<cv::KeyPoint>& kpts, int* offset, int threshold, int lvl,
            std::promise<bool>& p)
    {
        kpts.clear();

        assert(threshold == th_ini || threshold == th_min);

        uchar *threshold_tab;
        if (threshold == th_ini)
            threshold_tab = th_tab_init;
        else
            threshold_tab = th_tab_min;


        uchar cornerScores[img.cols*3];
        int cornerPos[img.cols*3];

        memset(cornerScores, 0, img.cols*3);
        memset(cornerPos, 0, img.cols*3);

        uchar* currRowScores = &cornerScores[0];
        uchar* prevRowScores = &cornerScores[img.cols];
        uchar* pprevRowScores = &cornerScores[img.cols*2];

        int* currRowPos = &cornerPos[0];
        int* prevRowPos = &cornerPos[img.cols];
        int* pprevRowPos = &cornerPos[img.cols*2];


        int i, j, k, ncandidates = 0, ncandidatesprev = 0;

        for (i = 3; i < img.rows - 2; ++i)
        {
            const uchar* pointer = img.ptr<uchar>(i) + 3;

            ncandidatesprev = ncandidates;
            ncandidates = 0;

            int* tempPos = pprevRowPos;
            uchar* tempScores = pprevRowScores;

            pprevRowPos = prevRowPos;
            pprevRowScores = prevRowScores;
            prevRowPos = currRowPos;
            prevRowScores = currRowScores;

            currRowPos = tempPos;
            currRowScores = tempScores;

            memset(currRowPos, 0, img.cols);
            memset(currRowScores, 0, img.cols);

            if (i < img.rows - 3) // skip last row
            {
                for (j = 3; j < img.cols-3; ++j, ++pointer)
                {
                    int val = pointer[0];                           //value of central pixel
                    const uchar *tab = &threshold_tab[255] - val;       //shift threshold tab by val


                    int discard = tab[pointer[offset[PIXELS_TO_CHECK[0]]]]
                                  | tab[pointer[offset[PIXELS_TO_CHECK[1]]]];

                    if (discard == 0)
                        continue;

                    bool gotoNextCol = false;
                    for (k = 2; k < 16; k+=2)
                    {
                        discard &= tab[pointer[offset[PIXELS_TO_CHECK[k]]]]
                                   | tab[pointer[offset[PIXELS_TO_CHECK[k+1]]]];
                        if (k == 6 && discard == 0)
                        {
                            gotoNextCol = true;
                            break;
                        }
                        if (k == 14 && discard == 0)
                        {
                            gotoNextCol = true;
                        }
                    }
                    if (gotoNextCol) // initial FAST-check failed
                        continue;


                    if (discard & 1) // check for continuous circle of pixels darker than threshold
                    {
                        int compare = val - threshold;
                        int contPixels = 0;

                        for (k = 0; k < onePointFiveCircles; ++k)
                        {
                            int a = pointer[offset[k%CIRCLE_SIZE]];
                            if (a < compare)
                            {
                                ++contPixels;
                                if (contPixels > contP)
                                {
                                    currRowPos[ncandidates++] = j;
                                    currRowScores[j] = CornerScore(pointer, offset, threshold);

                                    break;
                                }
                            } else
                                contPixels = 0;
                        }
                    }

                    if (discard & 2) // check for continuous circle of pixels brighter than threshold
                    {
                        int compare = val + threshold;
                        int contPixels = 0;

                        for (k = 0; k < onePointFiveCircles; ++k)
                        {
                            int a = pointer[offset[k%CIRCLE_SIZE]];
                            if (a > compare)
                            {
                                ++contPixels;
                                if (contPixels > contP)
                                {
                                    currRowPos[ncandidates++] = j;
                                    currRowScores[j] = CornerScore(pointer, offset, threshold);

                                    break;
                                }
                            } else
                                contPixels = 0;
                        }
                    }
                }
            }


            if (i == 3)
                continue;

            for (k = 0; k < ncandidatesprev; ++k)
            {
                int pos = prevRowPos[k];
                float score = prevRowScores[pos];

                if (score > pprevRowScores[pos-1] && score > pprevRowScores[pos] && score > pprevRowScores[pos+1] &&
                    score > prevRowScores[pos+1] && score > prevRowScores[pos-1] &&
                    score > currRowScores[pos-1] && score > currRowScores[pos] && score > currRowScores[pos+1])
                {
                    kpts.emplace_back(cv::KeyPoint((float)pos, (float)(i-1), 7.f, -1, (float)score, lvl));
                }
            }
        }
    }

    int CornerScore(const uchar* ptr, const int offset[], int threshold)
    {
        int val = ptr[0];
        int i;
        int diff[onePointFiveCircles];
        for (i = 0; i < CIRCLE_SIZE; ++i)
        {
            diff[i] = (val - ptr[offset[i]]);
        }
        for ( ; i < onePointFiveCircles; ++i)
        {
            diff[i] = diff[i-CIRCLE_SIZE];
        }

        int a0 = threshold;
        for (i = 0; i < CIRCLE_SIZE; i += 2)
        {
            int a;
            if (diff[i+1] < diff[i+2])
                a = diff[i+1];
            else
                a = diff[i+2];

            if (diff[i+3] < a)
                a = diff[i+3];
            if (a0 > a)
                continue;

            if (diff[i+4] < a)
                a = diff[i+4];
            if (diff[i+5] < a)
                a = diff[i+5];
            if (diff[i+6] < a)
                a = diff[i+6];
            if (diff[i+7] < a)
                a = diff[i+7];
            if (diff[i+8] < a)
                a = diff[i+8];

            int c;
            if (a < diff[i])
                c = a;
            else
                c = diff[i];

            if (c > a0)
                a0 = c;
            if (diff[i+9] < a)
                a = diff[i+9];
            if (a > a0)
                a0 = a;
        }

        int b0 = -a0;
        for (i = 0; i < CIRCLE_SIZE; i += 2)
        {
            int b;
            if (diff[i+1] > diff[i+2])
                b = diff[i+1];
            else
                b = diff[i+2];

            if (diff[i+3] > b)
                b = diff[i+3];
            if (diff[i+4] > b)
                b = diff[i+4];
            if (diff[i+5] > b)
                b = diff[i+5];

            if (b0 < b)
                continue;

            if (diff[i+6] > b)
                b = diff[i+6];
            if (diff[i+7] > b)
                b = diff[i+7];
            if (diff[i+8] > b)
                b = diff[i+8];

            int c;
            if (diff[i] > b)
                c = diff[i];
            else
                c = b;

            if (c < b0)
                b0 = c;
            if (diff[i+9] > b)
                b = diff[i+9];
            if (b < b0)
                b0 = b;
        }
        return -b0 - 1;

        /*
        using namespace blorp;

        const int K = 8, N = K*3 + 1;
        int k, v = ptr[0];
        short d[N];
        for( k = 0; k < N; k++ )
            d[k] = (short)(v - ptr[offset[k]]);

        v_int16x8 q0 = v_setall_s16(-1000), q1 = v_setall_s16(1000);
        for (k = 0; k < 16; k += 8)
        {
            v_int16x8 v0 = v_load(d + k + 1);
            v_int16x8 v1 = v_load(d + k + 2);
            v_int16x8 a = v_min(v0, v1);
            v_int16x8 b = v_max(v0, v1);
            v0 = v_load(d + k + 3);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 4);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 5);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 6);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 7);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 8);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k);
            q0 = v_max(q0, v_min(a, v0));
            q1 = v_min(q1, v_max(b, v0));
            v0 = v_load(d + k + 9);
            q0 = v_max(q0, v_min(a, v0));
            q1 = v_min(q1, v_max(b, v0));
        }
        q0 = v_max(q0, v_setzero_s16() - q1);
        threshold = v_reduce_max(q0) - 1;

        return threshold;
         */
    }
};


#endif //ORBEXTRACTOR_FASTWORKER_H
