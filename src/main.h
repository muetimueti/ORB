#ifndef ORBEXTRACTOR_MAIN_H
#define ORBEXTRACTOR_MAIN_H

#include <string>
#include <opencv2/core/core.hpp>
#include "ORBextractor.h"

#include <unistd.h>

void DisplayKeypoints(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar &color,
                      int thickness = 1, int radius = 8, int drawAngular = 0);

void AddRandomKeypoints(std::vector<cv::KeyPoint> &keypoints);

#endif //ORBEXTRACTOR_MAIN_H
