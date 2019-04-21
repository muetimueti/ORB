#include "Comparison_Descriptors.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;


static bool firstInLvl;

const float factorPI = (float)(CV_PI/180.f);
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    float angle = (float)kpt.angle*factorPI;
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    //std::cout << "refkpt=" << kpt.pt << ", angle=" << angle << ", a=" << a << ", b=" << b << "\n";
    //std::cout << "\nrefstep = " << step << "\n";



    //std::cout << "ref candidate I: " << (int)center[0] << "\n";

#define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]



    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0, t1, val;
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        //std::cout << "reference: t0 < t1 mit i=" << i << ": " <<  (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[0] << " = " << (int)t0 << ", t1 at " << pattern[1] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 1st point: " << (int)cvRound(pattern[0].x*b + pattern[0].y*a)*step + cvRound(pattern[0].x*a - pattern[0].y*b) <<
        //    ", ref Mat idx at 2nd point: " << (int)cvRound(pattern[1].x*b + pattern[1].y*a)*step + cvRound(pattern[1].x*a - pattern[1].y*b) << "\n";

        //std::cout << "idx1 = rd(" << pattern[i].x << "*" << a << "-" << pattern[i].y << "*" << b << ") + rd(" << pattern[i].x << "*" <<
        //          b << " + " << pattern[i].y << "*" << a << ")*" << step << "\n";

        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[2] << " = " << (int)t0 << ", t1 at " << pattern[3] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 3rd point: " << (int)cvRound(pattern[2].x*b + pattern[2].y*a)*step + cvRound(pattern[2].x*a - pattern[2].y*b) <<
        //          ", ref Mat idx at 4th point: " << (int)cvRound(pattern[3].x*b + pattern[3].y*a)*step + cvRound(pattern[3].x*a - pattern[3].y*b) << "\n";
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[4] << " = " << (int)t0 << ", t1 at " << pattern[5] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 5th point: " << (int)cvRound(pattern[4].x*b + pattern[4].y*a)*step + cvRound(pattern[4].x*a - pattern[4].y*b) <<
        //          ", ref Mat idx at 6th point: " << (int)cvRound(pattern[5].x*b + pattern[5].y*a)*step + cvRound(pattern[5].x*a - pattern[5].y*b) << "\n";
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[6] << " = " << (int)t0 << ", t1 at " << pattern[7] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 7th point: " << (int)cvRound(pattern[6].x*b + pattern[6].y*a)*step + cvRound(pattern[6].x*a - pattern[6].y*b) <<
        //          ", ref Mat idx at 8th point: " << (int)cvRound(pattern[7].x*b + pattern[7].y*a)*step + cvRound(pattern[7].x*a - pattern[7].y*b) << "\n";
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[8] << " = " << (int)t0 << ", t1 at " << pattern[9] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 9th point: " << (int)cvRound(pattern[8].x*b + pattern[8].y*a)*step + cvRound(pattern[8].x*a - pattern[8].y*b) <<
        //          ", ref Mat idx at 10th point: " << (int)cvRound(pattern[9].x*b + pattern[9].y*a)*step + cvRound(pattern[9].x*a - pattern[9].y*b) << "\n";
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[10] << " = " << (int)t0 << ", t1 at " << pattern[11] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 11th point: " << (int)cvRound(pattern[10].x*b + pattern[10].y*a)*step + cvRound(pattern[10].x*a - pattern[10].y*b) <<
        //          ", ref Mat idx at 12th point: " << (int)cvRound(pattern[11].x*b + pattern[11].y*a)*step + cvRound(pattern[11].x*a - pattern[11].y*b) << "\n";
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[12] << " = " << (int)t0 << ", t1 at " << pattern[13] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 13th point: " << (int)cvRound(pattern[12].x*b + pattern[12].y*a)*step + cvRound(pattern[12].x*a - pattern[12].y*b) <<
        //          ", ref Mat idx at 14th point: " << (int)cvRound(pattern[13].x*b + pattern[13].y*a)*step + cvRound(pattern[13].x*a - pattern[13].y*b) << "\n";
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        //std::cout << "reference: t0 < t1 mit i=" << i << ": " << (t0 < t1) << "\n";
        //std::cout << "t0 at " << pattern[14] << " = " << (int)t0 << ", t1 at " << pattern[15] << " = " << (int)t1 << "\n";
        //std::cout << "ref Mat idx at 15th point: " << (int)cvRound(pattern[14].x*b + pattern[14].y*a)*step + cvRound(pattern[14].x*a - pattern[14].y*b) <<
        //          ", ref Mat idx at 16th point: " << (int)cvRound(pattern[15].x*b + pattern[15].y*a)*step + cvRound(pattern[15].x*a - pattern[15].y*b) << "\n";

        desc[i] = (uchar)val;
        //std::cout << "compdesc[" << i << "]=" << (int)desc[i] << "\n";
    }

#undef GET_VALUE
}



static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)  //TODO: revert to "size_t i = 0; i < keypoints.size(); i++"
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}


void Comparison_Descriptors::Compute(vector<Mat> &imagePyramid, vector < vector<KeyPoint> > &allKeypoints,
        OutputArray _descriptors, const std::vector<Point>& pattern)
{
    Mat descriptors;

    int nlevels = 8;  //TODO: revert to 8

    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)          //TODO: revert
        nkeypoints += (int)allKeypoints[level].size();
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }



    int offset = 0;
    for (int level = 0; level < nlevels; ++level)  //TODO: revert to level = 0; level < nlevels
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image
        Mat workingMat = imagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        //std::cout << "working mat step: " << (int)workingMat.step << "\n";

        firstInLvl = true;
        // Compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        /*
        if (level != 0)
        {
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
         */
    }
}

