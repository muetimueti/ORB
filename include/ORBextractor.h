#ifndef ORBEXTRACTOR_ORBEXTRACTOR_H
#define ORBEXTRACTOR_ORBEXTRACTOR_H

#include <vector>
#include "include/Distribution.h"

#ifndef NDEBUG
#   define D(x) x
#else
#   define D(x)
#endif


namespace ORB_SLAM2
{

class ORBextractor
{
public:

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor() = default;


    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);

    void operator()(cv::InputArray inputImage, cv::InputArray mask,
                                  std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors,
                                  Distribution::DistributionMethod distributionMode, bool distributePerLevel = true);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return scaleFactorVec;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return invScaleFactorVec;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return levelSigma2Vec;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return invLevelSigma2Vec;
    }

    std::vector<cv::Mat> imagePyramid;


    //TODO: remove from master--------------------------------------------------------------------------

    long testingFAST(cv::Mat &img, std::vector<cv::KeyPoint> &kpts, bool myFAST, bool printTime);

    void testingDescriptors(cv::Mat myDescriptors, cv::Mat compDescriptors, int nkpts, bool printdif,
                            int start, int end, bool testTime, cv::OutputArray _descr);

    void Tests(cv::InputArray inputImage, std::vector<cv::KeyPoint> &resKeypoints,
               cv::OutputArray outputDescriptors, bool myFAST = true, bool myDesc = true);

    ///--------------------------------------------------------------------------------------------------

protected:

    static float IntensityCentroidAngle(const uchar* pointer, int step);


    void ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts);

    void ComputeDescriptors(std::vector<std::vector<cv::KeyPoint>> &allkpts, cv::Mat &descriptors);


    void DivideAndFAST(std::vector<std::vector<cv::KeyPoint> >& allKeypoints,
                       Distribution::DistributionMethod mode = Distribution::QUADTREE,
                       bool divideImage = true, int cellSize = 30, bool distributePerLevel = true);

    void FAST(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, int &threshold, int level = 0);

    int CornerScore(const uchar *pointer, const int offset[], int &threshold);
    int OptimizedCornerScore(const uchar *pointer, const int offset[], int &threshold);

    void ComputeScalePyramid(cv::Mat &image);

    std::vector<cv::Point> pattern;

    //inline float getScale(int lvl);


    uchar threshold_tab_min[512];
    uchar threshold_tab_init[512];


    int continuousPixelsRequired;
    int onePointFiveCircles;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;


    std::vector<int> pixelOffset;

    std::vector<int> nfeaturesPerLevelVec;


    std::vector<float> scaleFactorVec;
    std::vector<float> invScaleFactorVec;
    std::vector<float> levelSigma2Vec;
    std::vector<float> invLevelSigma2Vec;


    //TODO: remove from master

    template<class T>
    void PrintArray(T *array, const std::string &name, int start, int end);

    void printInternalValues();

    static void PrintKeypoints(std::vector<cv::KeyPoint> &kpts);

    static void PrintKeypoints(std::vector<cv::KeyPoint> &kpts, int start, int end);

    static void PrintKeypoints(std::vector<cv::KeyPoint> &kpts, int start, int end, bool printResponse);

    static void CompareKeypointVectors(std::vector<cv::KeyPoint> &vec1, std::vector<cv::KeyPoint> &vec2);

    /////////////////////

};


}

#endif //ORBEXTRACTOR_ORBEXTRACTOR_H