#ifndef ORBEXTRACTOR_FAST_H
#define ORBEXTRACTOR_FAST_H

#include <opencv2/core/core.hpp>

class FAST
{
public:
    FAST(int iniThreshold, int minThreshold);

    ~FAST() = default;

    void operator()(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<int> &pixelOffset,
            int threshold, int level);

protected:
    int CornerScore(const uchar* pointer, const int offset[], int threshold);

    void CheckDims(cv::Mat &img);

    int initialThreshold;
    int minimumThreshold;

    uchar threshold_tab_min[512];
    uchar threshold_tab_init[512];

    //std::vector<int> pixelOffset;

    int continuousPixelsRequired;
    int onePointFiveCircles;

    int imageCols;
    int imageRows;


};


#endif //ORBEXTRACTOR_FAST_H
