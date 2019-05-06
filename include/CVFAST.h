#ifndef ORBEXTRACTOR_CVFAST_H
#define ORBEXTRACTOR_CVFAST_H

#include "include/main.h"

void FAST_cv(cv::InputArray _img, std::vector <cv::KeyPoint> &keypoints, int threshold, bool nonmax_suppression = true);

#endif //ORBEXTRACTOR_CVFAST_H
