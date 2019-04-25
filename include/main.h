#ifndef ORBEXTRACTOR_MAIN_H
#define ORBEXTRACTOR_MAIN_H

#include <string>
#include <opencv2/core/core.hpp>
#include "include/ORBextractor.h"

#include <unistd.h>

#ifndef NDEBUG
#  define D(x) x
#else
# define D(x)
#endif

enum MODE {DESC_RUNTIME = 0, FAST_RUNTIME = 1};

void DisplayKeypoints(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar &color,
                      int thickness = 1, int radius = 8, int drawAngular = 0);

void SingleImageMode(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                     int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular);

void SequenceMode(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                    int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular);

void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
        std::vector<double> &vTimestamps);

void MeasureExecutionTime(int numIterations, ORB_SLAM2::ORBextractor &extractor, cv::Mat &imagem, MODE mode);
void AddRandomKeypoints(std::vector<cv::KeyPoint> &keypoints);
void LoadHugeImage(ORB_SLAM2::ORBextractor &extractor);

#endif //ORBEXTRACTOR_MAIN_H