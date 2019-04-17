#ifndef ORBEXTRACTOR_COMPARISON_DESCRIPTORS_H
#define ORBEXTRACTOR_COMPARISON_DESCRIPTORS_H

#include "ORBextractor.h"
#include <opencv2/core.hpp>


using namespace cv;
using namespace std;

class Comparison_Descriptors
{
public:
    void Compute(vector<Mat> &imagePyramid, vector < vector<KeyPoint> > &allKeypoints, vector<KeyPoint>& _keypoints,
                 OutputArray _descriptors, const std::vector<Point>& _pattern);
};


#endif //ORBEXTRACTOR_COMPARISON_DESCRIPTORS_H
