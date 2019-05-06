#ifndef ORBEXTRACTOR_MAIN_H
#define ORBEXTRACTOR_MAIN_H

#include <string>
#include <opencv2/core/core.hpp>
#include "include/ORBextractor.h"
#include "include/referenceORB.h"

#include <unistd.h>

#ifndef NDEBUG
#  define D(x) x
#else
# define D(x)
#endif

struct Descriptor_Pair
{
    int byte1;
    int byte2;
    int index;
};

enum MODE {DESC_RUNTIME = 0, FAST_RUNTIME = 1};

void SingleImageMode(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                     int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular);

void SequenceMode(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                    int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular);

void SortKeypoints(std::vector<cv::KeyPoint> &kpts);

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> CompareKeypoints(std::vector<cv::KeyPoint> &kpts1, std::string name1,
        std::vector<cv::KeyPoint> &kpts2, std::string name2, int imgNr, bool print = false);

std::vector<Descriptor_Pair> CompareDescriptors (cv::Mat &desc1, std::string name1, cv::Mat &desc2, std::string name2,
                                                 int nkpts, int imgNr, bool print = false);

void LoadHugeImage(ORB_SLAM2::ORBextractor &extractor);
void LoadHugeImage(ORB_SLAM_REF::referenceORB &extractor);

void DisplayKeypoints(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar &color,
                      int thickness = 1, int radius = 8, int drawAngular = 0, std::string windowname = "test");

void DrawCellGrid(cv::Mat &image, int minX, int maxX, int minY, int maxY, int cellSize);

void MeasureExecutionTime(int numIterations, ORB_SLAM2::ORBextractor &extractor, cv::Mat &imagem, MODE mode);

void DistributionComparisonSuite(ORB_SLAM2::ORBextractor &extractor, cv::Mat &imgColor, cv::Scalar &color,
                                 int thickness, int radius, bool drawAngular, bool distributePerLevel);

void AddRandomKeypoints(std::vector<cv::KeyPoint> &keypoints);

void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
                std::vector<double> &vTimestamps);

#endif //ORBEXTRACTOR_MAIN_H

