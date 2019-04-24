#ifndef ORBEXTRACTOR_ORBEXTRACTOR_H
#define ORBEXTRACTOR_ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv/cv.h>


#ifndef NDEBUG
#   define D(x) x
#else
#   define D(x)
#endif


namespace ORB_SLAM2
{

class ExtractorNode
{
public:
    ExtractorNode():leaf(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> nodeKpts;
    cv::Point2i UL, UR, LL, LR;
    //std::list<ExtractorNode>::iterator lit;
    bool leaf;
};

class ORBextractor
{
public:

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor() = default;


    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);

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

    void testingFAST(cv::Mat &img, std::vector<cv::KeyPoint> &kpts, bool myFAST, bool printTime);

    void testingDescriptors(cv::Mat myDescriptors, cv::Mat compDescriptors, int nkpts, bool printdif,
                            int start, int end);

    void Tests(cv::InputArray inputImage, std::vector<cv::KeyPoint> &resKeypoints,
               cv::OutputArray outputDescriptors, bool myFAST = true, bool myDesc = true);

    //---------------------------------------------------------------------------------------------------

protected:

    CV_INLINE  int  myRound( float value )
    {
    #if defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
        return (int)lrint(value);
    #else
        // while this is not IEEE754-compliant rounding, it's usually a good enough approximation
      return (int)(value + (value >= 0 ? 0.5f : -0.5f));
    #endif
    }

    static float IntensityCentroidAngle(const uchar* pointer, int step);
    static void RetainBestN(std::vector<cv::KeyPoint> &kpts, int N);
    static bool ResponseComparison(const cv::KeyPoint &k1, const cv::KeyPoint &k2);

    void ComputeScalePyramid(cv::Mat &image);

    void DivideAndFAST(std::vector<std::vector<cv::KeyPoint> >& allKeypoints,
                       bool distributeKpts = true, bool divideImage = true, int cellSize = 30);
    void DistributeKeypoints(std::vector<cv::KeyPoint>& kpts, const int &minX,
                             const int &maxX, const int &minY, const int &maxY, const int &level);

    void FAST(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, int threshold, int level = 0);

    int CornerScore(const uchar *pointer, const int offset[], int threshold);

    void ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts);

    void ComputeDescriptors(std::vector<std::vector<cv::KeyPoint>> &allkpts, cv::Mat &descriptors);


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
    //std::vector<int> stepVec;

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

    //

};


}

#endif //ORBEXTRACTOR_ORBEXTRACTOR_H