#ifndef ORBEXTRACTOR_DISTRIBUTION_H
#define ORBEXTRACTOR_DISTRIBUTION_H

#include <vector>
#include "include/Common.h"
#include <list>


class ExtractorNode
{
public:
    ExtractorNode():leaf(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> nodeKpts;
    cv::Point2i UL, UR, LL, LR;
    std::list<ExtractorNode>::iterator lit;
    bool leaf;
};

class Distribution
{
public:

    enum DistributionMethod
    {
    NAIVE,
    QUADTREE,
    QUADTREE_ORBSLAMSTYLE,
    GRID,
    ANMS_KDTREE,
    ANMS_RT,
    SSC,
    KEEP_ALL
    };

    static void DistributeKeypoints(std::vector<cv::KeyPoint> &kpts, int minX, int maxX, int minY,
                             int maxY, int N, DistributionMethod mode);

protected:

    static void DistributeKeypointsNaive(std::vector<cv::KeyPoint> &kpts, int N);

    static void DistributeKeypointsQuadTree(std::vector<cv::KeyPoint> &kpts, int minX,
                                     int maxX, int minY, int maxY, int N);

    static void DistributeKeypointsQuadTree_ORBSLAMSTYLE(std::vector<cv::KeyPoint> &kpts, int minX,
                                                  int maxX, int minY, int maxY, int N);

    static void DistributeKeypointsGrid(std::vector<cv::KeyPoint> &kpts, int minX,
                             int maxX, int minY, int maxY, int N);

    static void DistributeKeypointsKdT_ANMS(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon);

    static void DistributeKeypointsRT_ANMS(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon);

    static void DistributeKeypointsSSC(std::vector<cv::KeyPoint> &kpts, int rows, int cols, int N, float epsilon);

};

#endif