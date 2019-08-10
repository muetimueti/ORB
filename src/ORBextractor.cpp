#include <string>
#include <iostream>
#include <iterator>
#include <opencv2/imgproc/imgproc.hpp>
#include "include/ORBextractor.h"
#include "include/ORBconstants.h"
#include <unistd.h>
#include <chrono>

#include <saiga/core/util/Range.h>
#include "include/2Dimgeffects.h"
//#include <saiga/core/image/templatedImage.h>
#include "saigacpy/templatedImage.h"


#ifndef NDEBUG
#   define D(x) x


#else
#   define D(x)
#endif

#define MYFAST 1
#define TESTFAST 0

#if TESTFAST
#include "include/avx.h"
#endif

#if !MYFAST and !TESTFAST
#include <opencv2/features2d/features2d.hpp>
#endif

namespace ORB_SLAM2
{

float ORBextractor::IntensityCentroidAngle(const uchar* pointer, int step)
{
    //m10 ~ x^1y^0, m01 ~ x^0y^1
    int x, y, m01 = 0, m10 = 0;

    int half_patch = PATCH_SIZE / 2;

    for (x = -half_patch; x <= half_patch; ++x)
    {
        m10 += x * pointer[x];
    }

    for (y = 1; y <= half_patch; ++y)
    {
        int cols = CIRCULAR_ROWS[y];
        int sumY = 0;
        for (x = -cols; x <= cols; ++x)
        {
            int uptown = pointer[x + y*step];
            int downtown = pointer[x - y*step];
            sumY += uptown - downtown;
            m10 += x * (uptown + downtown);
        }
        m01 += y * sumY;
    }

    return std::atan2((float)m01, (float)m10);
}


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST):
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), iniThFAST(_iniThFAST),
        minThFAST(_minThFAST), stepsChanged(true), levelToDisplay(-1), softSSCThreshold(10), prevDims(-1, -1),
        kptDistribution(Distribution::DistributionMethod::SSC), pixelOffset{}, fast(_iniThFAST, _minThFAST, _nlevels)
        /*, fileInterface(), saveFeatures(false), usePrecomputedFeatures(false)*/
{
    SetnLevels(_nlevels);

    SetFASTThresholds(_iniThFAST, _minThFAST);

    SetnFeatures(nfeatures);

    const int nPoints = 512;
    const auto tempPattern = (const kvis::Point*) bit_pattern_31_;
    std::copy(tempPattern, tempPattern+nPoints, std::back_inserter(pattern));
}

void ORBextractor::SetnFeatures(int n)
{
    if (n < 1 || n > 10000)
        return;

    nfeatures = n;

    float fac = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float) pow((double) fac, (double) nlevels));

    int sumFeatures = 0;
    for (int i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = myRound(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
}

void ORBextractor::SetFASTThresholds(int ini, int min)
{
    if ((ini == iniThFAST && min == minThFAST))
        return;

    iniThFAST = std::min(255, std::max(1, ini));
    minThFAST = std::min(iniThFAST, std::max(1, min));

    fast.SetFASTThresholds(ini, min);
}


void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask,
                              std::vector<kvis::KeyPoint>& resultKeypoints, cv::OutputArray outputDescriptors)
{
    //this->operator()(inputImage, mask, resultKeypoints, outputDescriptors, true);

    cv::Mat image = inputImage.getMat();
    message_assert("Image must be single-channel!",image.type() == CV_8UC1);

    Saiga::ImageView<uchar> saigaImage = Saiga::MatToImageView<uchar>(image);
    Saiga::ImageView<uchar> saigaDescriptors;

    this->operator()(saigaImage, resultKeypoints, saigaDescriptors, true);

    cv::Mat cvRes = Saiga::ImageViewToMat<uchar>(saigaDescriptors);
}

/** @overload
 * @param inputImage single channel img-matrix
 * @param mask ignored
 * @param resultKeypoints keypoint vector in which results will be stored
 * @param outputDescriptors matrix in which descriptors will be stored
 * @param distributePerLevel true->distribute kpts per octave, false->distribute kpts per image
 */
void ORBextractor::operator()(img_t& image, std::vector<kvis::KeyPoint>& resultKeypoints,
        img_t& outputDescriptors, bool distributePerLevel)
{
    /////////////
    //resultKeypoints.emplace_back(kvis::KeyPoint(100, 100, 7, 10, 10, 0));
    //return;
    ////////////////
    //std::chrono::high_resolution_clock::time_point funcEntry = std::chrono::high_resolution_clock::now();



    message_assert(image.size() > 0, "image empty");

    if (prevDims.x != image.cols || prevDims.y != image.rows)
        stepsChanged = true;

    ComputeScalePyramid(image);

    SetSteps();

    //using namespace std::chrono;
    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

#if 0
    //resultKeypoints.reserve(100);


    resultKeypoints.emplace_back(kvis::KeyPoint(100, 100, 7, -1, 100, 0));

    //PrintKeyPoints(resultKeypoints);

    std::cout << "sz of reskpts in extractor: " << resultKeypoints.size() << "\n";
    std::cout << "address of reskpts[0]: " << &resultKeypoints[0] << "\n";
    std::cout << "address of reskpts inside extractor: " << &resultKeypoints << "\n";
    return;
#endif

    DivideAndFAST(resultKeypoints, kptDistribution, true, 30, distributePerLevel);

    if (!distributePerLevel)
    {
        ComputeAngles(resultKeypoints);
        for (auto& kpt : resultKeypoints)
        {
            float size = PATCH_SIZE * scaleFactorVec[kpt.octave];
            float scale = scaleFactorVec[kpt.octave];
            if (kpt.octave)
            {
                kpt.size = size;
                kpt.pt *= scale;
            }

        }

        Distribution::DistributeKeypoints(resultKeypoints, 0, imagePyramid[0].cols, 0, imagePyramid[0].rows,
                                          nfeatures, kptDistribution, softSSCThreshold);
    }

    if (distributePerLevel)
    {
        ComputeAngles(resultKeypoints);

    }

    int nkpts = resultKeypoints.size();


    Saiga::ImageBase d(nkpts, 32, 32);
    img_t data = Saiga::ImageView<uchar>(d);
    img_t BRIEFdescriptors(nkpts, 32, &data);
    //TODO: just make it a vector instead? Imageview easier to convert to mat tho



    //ComputeDescriptors(resultKeypoints, BRIEFdescriptors);

    if (distributePerLevel)
    {
        for (auto& kpt : resultKeypoints)
        {
            float size = PATCH_SIZE * scaleFactorVec[kpt.octave];
            float scale = scaleFactorVec[kpt.octave];
            if (kpt.octave)
            {
                kpt.size = size;
                kpt.pt *= scale;
            }
        }
    }

    std::cout << "kpts.sz: " << resultKeypoints.size() << "\n";
    //PrintKeyPoints(resultKeypoints);

    /*
    if (saveFeatures)
    {
        fileInterface.SaveFeatures(resultKeypoints);
        //TODO: port fileinterface to saiga
        //fileInterface.SaveDescriptors(BRIEFdescriptors);
    }
     */

    //ensure feature detection always takes 50ms
    /*
    unsigned long maxDuration = 50000;
    std::chrono::high_resolution_clock::time_point funcExit = std::chrono::high_resolution_clock::now();
    auto funcDuration = std::chrono::duration_cast<std::chrono::microseconds>(funcExit-funcEntry).count();
    assert(funcDuration <= maxDuration);
    if (funcDuration < maxDuration)
    {
        auto sleeptime = maxDuration - funcDuration;
        usleep(sleeptime);
    }
     */
}


void ORBextractor::ComputeAngles(std::vector<kvis::KeyPoint> &allkpts)
{
#pragma omp parallel for
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (int i = 0; i < featuresPerLevelActual[lvl]; ++i)
        {
            int idx = lvl > 0 ? featuresPerLevelActual[lvl-1] + i : i;
            allkpts[idx].angle = IntensityCentroidAngle(&imagePyramid[lvl](myRound(allkpts[idx].pt.y),
                    myRound(allkpts[idx].pt.x)), imagePyramid[lvl].pitchBytes);
        }
    }
}


void ORBextractor::ComputeDescriptors(std::vector<kvis::KeyPoint> &allkpts, img_t &descriptors)
{
    const auto degToRadFactor = (float)(CV_PI/180.f);
    const kvis::Point* p = &pattern[0];

    int current = 0;

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        img_t lvlClone(imagePyramid[lvl]);
        //imagePyramid[lvl].copyTo(lvlClone);
        kvis::GaussianBlur<uchar>(lvlClone, lvlClone, 7, 7, 2, 2);

        cv::Mat lvlCloneCV = Saiga::ImageViewToMat(lvlClone);

        const int step = (int)lvlClone.pitchBytes;

        int idx = lvl > 0 ? featuresPerLevelActual[lvl-1] : 0;
        int i = 0, nkpts = featuresPerLevelActual[lvl];
        for (int k = 0; k < nkpts; ++k, ++current)
        {
            const kvis::KeyPoint& kpt = allkpts[idx];
            auto descPointer = descriptors.rowPtr(current);        //ptr to beginning of current descriptor
            const uchar* pixelPointer = &lvlClone(myRound(kpt.pt.y), myRound(kpt.pt.x));  //ptr to kpt in img

            float angleRad = kpt.angle * degToRadFactor;
            auto a = (float)cos(angleRad), b = (float)sin(angleRad);

            int byte = 0, v0, v1, idx0, idx1;
            for (i = 0; i <= 512; i+=2)
            {
                if (i > 0 && i%16 == 0) //working byte full
                {
                    descPointer[i/16 - 1] = (uchar)byte;  //write current byte to descriptor-mat
                    byte = 0;      //reset working byte
                    if (i == 512)  //break out after writing very last byte, so oob indices aren't accessed
                        break;
                }

                idx0 = myRound(p[i].x*a - p[i].y*b) + myRound(p[i].x*b + p[i].y*a)*step;
                idx1 = myRound(p[i+1].x*a - p[i+1].y*b) + myRound(p[i+1].x*b + p[i+1].y*a)*step;

                v0 = pixelPointer[idx0];
                v1 = pixelPointer[idx1];

                byte |= (v0 < v1) << ((i%16)/2); //write comparison bit to current byte
            }
        }
    }
}


/**
 * @param allkpts KeyPoint vector in which the result will be stored
 * @param mode decides which method to call for keypoint distribution over image, see Distribution.h
 * @param divideImage  true-->divide image into cellSize x cellSize cells, run FAST per cell
 * @param cellSize must be greater than 16 and lesser than min(rows, cols) of smallest image in pyramid
 */
void ORBextractor::DivideAndFAST(std::vector<kvis::KeyPoint>& resultkpts,
        Distribution::DistributionMethod mode, bool divideImage, int cellSize, bool distributePerLevel)
{
    const int minimumX = EDGE_THRESHOLD - 3, minimumY = minimumX;
    {
        int c = std::min(imagePyramid[nlevels-1].rows, imagePyramid[nlevels-1].cols);
        assert(cellSize < c && cellSize > 16);

        int minLvl = 0, maxLvl = nlevels;
        if (levelToDisplay != -1)
        {
            minLvl = levelToDisplay;
            maxLvl = minLvl + 1;
        }

//#pragma omp parallel for
        for (int lvl = minLvl; lvl < maxLvl; ++lvl)
        {
            std::vector<kvis::KeyPoint> levelkpts;
            levelkpts.clear();
            levelkpts.reserve(nfeatures*10);

            const int maximumX = imagePyramid[lvl].cols - EDGE_THRESHOLD + 3;
            const int maximumY = imagePyramid[lvl].rows - EDGE_THRESHOLD + 3;
            const float width = maximumX - minimumX;
            const float height = maximumY - minimumY;

            const int npatchesInX = width / cellSize;
            const int npatchesInY = height / cellSize;
            const int patchWidth = ceil(width / npatchesInX);
            const int patchHeight = ceil(height / npatchesInY);

#if THREADEDPATCHES
            int nCells = npatchesInX * npatchesInY;
            int offset[CIRCLE_SIZE];
            for (int i = 0; i < CIRCLE_SIZE; ++i)
            {
                offset[i] = pixelOffset[lvl*CIRCLE_SIZE + i];
            }
            std::vector<std::promise<bool>> promises(nCells);
            int curCell = 0;

            std::vector<std::vector<kvis::KeyPoint>> cellkptvecs;
#endif

            for (int py = 0; py < npatchesInY; ++py)
            {
                float startY = minimumY + py * patchHeight;
                float endY = startY + patchHeight + 6;

                if (startY >= maximumY-3)
                {
                    continue;
                }

                if (endY > maximumY)
                {
                    endY = maximumY;
                }


                for (int px = 0; px < npatchesInX; ++px)
                {
                    float startX = minimumX + px * patchWidth;
                    float endX = startX + patchWidth + 6;

                    if (startX >= maximumX-6)
                    {
                        continue;
                    }


                    if (endX > maximumX)
                    {
                        endX = maximumX;
                    }


                    //std::chrono::high_resolution_clock::time_point FASTEntry =
                    //        std::chrono::high_resolution_clock::now();

#if MYFAST
#if THREADEDPATCHES
                    fast.workerPool.PushImg(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                            cellkptvecs[curCell], offset, iniThFAST, lvl, &promises[curCell]);
                    ++curCell;

#else
                    std::vector<kvis::KeyPoint> patchKpts;
                    img_t patch = imagePyramid[lvl].subImageView(startY, startX, endY-startY, endX-startX);

                    fast.FAST(patch, patchKpts, iniThFAST, lvl);
                    if (patchKpts.empty())
                    {
                        fast.FAST(patch, patchKpts, minThFAST, lvl);
                    }
                    //std::cout << "patchkpts.sz: " << patchKpts.size() << "\n";
#endif
#elif TESTFAST
                    std::vector<kvis::KeyPoint> patchKpts;
                    blorp::FAST_t<16>(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                      patchKpts, iniThFAST, true);
                    if (patchKpts.empty())
                        blorp::FAST_t<16>(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                          patchKpts, minThFAST, true);

#else
                    std::vector<kvis::KeyPoint> patchKpts;
                    cv::FAST(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                            patchKpts, iniThFAST, true, cv::FastFeatureDetector::TYPE_9_16);
                    if (patchKpts.empty())
                    {
                        cv::FAST(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                            patchKpts, minThFAST, true, cv::FastFeatureDetector::TYPE_9_16);
                    }
#endif
#if !THREADEDPATCHES
                    if(patchKpts.empty())
                        continue;

                    for (auto &kpt : patchKpts)
                    {
                        kpt.pt.y += py * patchHeight;
                        kpt.pt.x += px * patchWidth;
                        levelkpts.emplace_back(kpt);
                    }
#endif
                }
            }

            //resultkpts.reserve(nfeatures + levelkpts.size());

            if (distributePerLevel)
                Distribution::DistributeKeypoints(levelkpts, minimumX, maximumX, minimumY, maximumY,
                                                  nfeaturesPerLevelVec[lvl], mode, softSSCThreshold);

            for (auto &kpt : levelkpts)
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
            }
            featuresPerLevelActual[lvl] = levelkpts.size();
            resultkpts.insert(resultkpts.end(), levelkpts.begin(), levelkpts.end());
        }
    }
}

void ORBextractor::ComputeScalePyramid(img_t& image)
{
    imagePyramid[0] = image;

    for (int lvl = 1; lvl < nlevels; ++ lvl)
    {
        int width = (int)myRound(image.cols * invScaleFactorVec[lvl]);
        int height = (int)myRound(image.rows * invScaleFactorVec[lvl]);

        Saiga::TemplatedImage<uchar> t(height, width);
        image.copyScaleLinear(t.getImageView());

        imagePyramid[lvl] = t.getImageView();
        //ImageDisplay(imagePyramid[lvl]);
    }
}


void ORBextractor::SetSteps()
{
    if (stepsChanged)
    {
        std::vector<int> steps(nlevels);
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            steps[lvl] = (int)imagePyramid[lvl].pitchBytes;
        }
        fast.SetLevels(nlevels);
        fast.SetStepVector(steps);

        stepsChanged = false;
    }
}

void ORBextractor::SetnLevels(int n)
{
    nlevels = std::max(std::min(12, n), 2);
    scaleFactorVec.resize(nlevels);
    invScaleFactorVec.resize(nlevels);
    imagePyramid.resize(nlevels);
    nfeaturesPerLevelVec.resize(nlevels);
    featuresPerLevelActual.resize(nlevels);
    levelSigma2Vec.resize(nlevels);
    invLevelSigma2Vec.resize(nlevels);
    pixelOffset.resize(nlevels * CIRCLE_SIZE);

    SetScaleFactor(scaleFactor);

    float fac = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float) pow((double) fac, (double) nlevels));
    int sumFeatures = 0;
    for (int i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = myRound(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
}

void ORBextractor::SetScaleFactor(float s)
{
    stepsChanged = true;
    scaleFactor = std::max(std::min(1.5f, s), 1.001f);
    scaleFactorVec[0] = 1.f;
    invScaleFactorVec[0] = 1.f;

    SetSteps();

    for (int i = 1; i < nlevels; ++i) {
        scaleFactorVec[i] = scaleFactor * scaleFactorVec[i - 1];
        invScaleFactorVec[i] = 1 / scaleFactorVec[i];

        levelSigma2Vec[i] = scaleFactorVec[i] * scaleFactorVec[i];
        invLevelSigma2Vec[i] = 1.f / levelSigma2Vec[i];
    }
}

void ORBextractor::FilterTest(img_t& img)
{
    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    cv::Mat cvImg = Saiga::ImageViewToMat<uchar>(img);
    cv::imshow("test", cvImg);
    cv::waitKey(0);

    kvis::GaussianBlur<uchar>(img, img, 7, 7, 2, 2);

    cvImg = Saiga::ImageViewToMat<uchar>(img);
    cv::imshow("test", cvImg);
    cv::waitKey(0);
}

void ORBextractor::PrintKeyPoints(std::vector<kvis::KeyPoint>& kpts)
{
    std::cout << "Printing keypoints:\n";
    for (auto& kpt : kpts)
    {
        std::cout << kpt << "\n";
    }
    std::cout << "\n\n";
}

void ORBextractor::ImageDisplay(img_t& img)
{
    cv::Mat test = Saiga::ImageViewToMat<uchar>(img);
    cv::namedWindow("test", CV_WINDOW_AUTOSIZE);
    cv::imshow("test", test);
    cv::waitKey(0);
}
}