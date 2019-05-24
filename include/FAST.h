#ifndef ORBEXTRACTOR_FAST_H
#define ORBEXTRACTOR_FAST_H

#include <opencv2/core/core.hpp>
#include <thread>
#include "include/FASTworker.h"


class FASTdetector
{
public:
    FASTdetector(int _iniThreshold, int _minThreshold, int _nlevels);

    ~FASTdetector() = default;

    void SetStepVector(std::vector<int> &_steps);

    void SetFASTThresholds(int ini, int min);

    void FAST(cv::Mat img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl);

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

    static float CornerScore_Experimental(const uchar* ptr, int lvl);

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

    FASTworker workerPool;


    template <typename scoretype>
    void FAST_t(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl);

    float CornerScore_Harris(const uchar* ptr, int lvl);

    float CornerScore_Sum(const uchar* ptr, const int offset[]);

    float CornerScore(const uchar* pointer, const int offset[], int threshold);


public:
    void FAST_mt(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, int threshold, int lvl);
};


#endif //ORBEXTRACTOR_FAST_H
