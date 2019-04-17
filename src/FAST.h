#ifndef ORBEXTRACTOR_FAST_H
#define ORBEXTRACTOR_FAST_H

#include <opencv2/core/core.hpp>


class FAST
{
public:
    FAST(int iniThreshold, int minThreshold);

    ~FAST() = default;

    void operator()(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, int threshold, int level);

protected:
    int CornerScore(const uchar* pointer, int threshold);

    int initialThreshold;
    int minimumThreshold;

    uchar threshold_tab_min[512];
    uchar threshold_tab_init[512];
};


#endif //ORBEXTRACTOR_FAST_H
