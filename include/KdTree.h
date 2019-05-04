#ifndef ORBEXTRACTOR_KDTREE_H
#define ORBEXTRACTOR_KDTREE_H

#include <opencv2/core/types.hpp>

class KdTree
{
public:
    explicit KdTree(int maxLeafSize = 25) : root{}, kptVec{}, leafMax(maxLeafSize) {};

    ~KdTree() = default;

    void BuildTree(std::vector<cv::KeyPoint> &kpts);

protected:
    struct BoundingBox
    {
    int minX, maxX, minY, maxY;
    };
    struct Node
    {
        cv::Point2f point;
        int axis;
        Node* left;
        Node* right;
        int size;
        BoundingBox bbox;
    };


    Node root;
    int leafMax;
    std::vector<cv::KeyPoint* > kptPtrs;

    void divideNode(std::vector<cv::KeyPoint> &kpts);
};

struct xGreater
{
    bool operator()(cv::KeyPoint *k1, cv::KeyPoint *k2) {return k1->pt.x > k2->pt.x;}
};
struct YGreater
{
bool operator()(cv::KeyPoint *k1, cv::KeyPoint *k2) {return k1->pt.y > k2->pt.y;}
};
/**
 * @param kpts must already be sorted by response(descending)!
 */
void KdTree::BuildTree(std::vector<cv::KeyPoint> &kpts)
{
    kptPtrs.resize(kpts.size());
    for (unsigned int i = 0; i < kpts.size(); ++i)
    {
        kptPtrs[i] = &kpts[i];
    }
    Node n;
    n.axis = 0;
    n.size = kpts.size();
    std::nth_element(kptPtrs.begin(), kptPtrs.begin() + kptPtrs.size(), kptPtrs.end(), xGreater());

}


#endif //ORBEXTRACTOR_KDTREE_H

