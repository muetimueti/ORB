#ifndef ORBEXTRACTOR_FAST_H
#define ORBEXTRACTOR_FAST_H

#include <opencv2/core/core.hpp>
#include "include/Types.h"
#define FASTWORKERS 0

const int CIRCLE_SIZE = 16;

const int CIRCLE_OFFSETS[16][2] =
        {{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
         {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}};

const int PIXELS_TO_CHECK[16] =
        {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15};

class FASTdetector
{
public:
    FASTdetector(int _iniThreshold, int _minThreshold, int _nlevels);

    ~FASTdetector() = default;

    void SetStepVector(std::vector<int> &_steps);

    void SetFASTThresholds(int ini, int min);

    void FAST(img_t img, std::vector<kvis::KeyPoint> &keypoints, int threshold, int lvl);

    enum ScoreType
    {
    OPENCV,
    HARRIS,
    SUM,
    EXPERIMENTAL
    };

    void inline SetScoreType(ScoreType t)
    {
        scoreType = t;
    }

    ScoreType inline GetScoreType()
    {
        return scoreType;
    }

    void inline SetLevels(int nlvls)
    {
        pixelOffset.resize(nlvls * CIRCLE_SIZE);
    }

    static float CornerScore_Experimental(const uchar* ptr, int lvl);

    //FASTworker workerPool;

protected:

    int iniThreshold;
    int minThreshold;

    int nlevels;

    int continuousPixelsRequired;
    int onePointFiveCircles;

    ScoreType scoreType;

    std::vector<int> pixelOffset;
    std::vector<int> steps;

    uchar threshold_tab_init[512];
    uchar threshold_tab_min[512];


    template <typename scoretype>
    void FAST_t(img_t &img, std::vector<kvis::KeyPoint> &keypoints, int threshold, int lvl);

    float CornerScore_Harris(const uchar* ptr, int lvl);

    float CornerScore_Sum(const uchar* ptr, const int offset[]);

    float CornerScore(const uchar* pointer, const int offset[], int threshold);

#if FASTWORKERS
public:
    void FAST_mt(cv::Mat &img, std::vector<knuff::KeyPoint> &keypoints, int threshold, int lvl);
#endif
};


#endif //ORBEXTRACTOR_FAST_H
