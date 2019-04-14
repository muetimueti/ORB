//
// Created by ralph on 3/31/19.
//

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
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor() = default;

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using a quadtree.
    // Mask is ignored in the current implementation
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

protected:

    void ComputeScalePyramid(cv::Mat &image);
    void DivideAndFAST(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
    std::vector<cv::KeyPoint> DistributeQuadTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void FAST(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, int threshold, int level = 0);


    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);



    std::vector<cv::Point> pattern;

    inline float getScale(int lvl);


    uchar threshold_tab_min[512];
    uchar threshold_tab_init[512];
    int pixelOffset[16];

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;



    std::vector<int> nfeaturesPerLevelVec;

    std::vector<int> umaxVec;

    std::vector<float> scaleFactorVec;
    std::vector<float> invScaleFactorVec;
    std::vector<float> levelSigma2Vec;
    std::vector<float> invLevelSigma2Vec;


    D(
            template<class T>
            void PrintArray(T *array, const std::string &name, int start, int end);

            void printInternalValues();
    )
};


}

#endif //ORBEXTRACTOR_ORBEXTRACTOR_H
