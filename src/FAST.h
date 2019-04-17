#ifndef ORBEXTRACTOR_FAST_H
#define ORBEXTRACTOR_FAST_H

#include <opencv2/core/core.hpp>
#include <vector>


class FAST
{
public:
    FAST(int iniThreshold, int minThreshold);

    ~FAST() = default;

    void operator()(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, int threshold, int level);

protected:
    int CornerScore(const uchar* pointer, int threshold);

    void CheckDims(cv::Mat &img);

    int initialThreshold;
    int minimumThreshold;

    uchar threshold_tab_min[512];
    uchar threshold_tab_init[512];

    int continuousPixelsRequired;
    int onePointFiveCircles;

    int imageCols;
    int imageRows;

    int pixelOffset[16];
};


#endif //ORBEXTRACTOR_FAST_H
