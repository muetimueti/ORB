#include <string>
#include <iostream>
#include <iterator>
#include <opencv2/imgproc/imgproc.hpp>
#include "include/ORBextractor.h"
#include "include/ORBconstants.h"
#include <unistd.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfor-loop-analysis"

#ifndef NDEBUG
#   define D(x) x
#   include <opencv2/highgui/highgui.hpp>
#   include <opencv2/features2d/features2d.hpp>
#   include <chrono>
#   include "include/referenceORB.h"
#   include "include/CVFAST.h"
#else
#   define D(x)
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

    return cv::fastAtan2((float)m01, (float)m10);
}



ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                           int _iniThFAST, int _minThFAST):
        nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
        iniThFAST(_iniThFAST), minThFAST(_minThFAST), threshold_tab_min{}, threshold_tab_init{}, pixelOffset{}

{

    scaleFactorVec.resize(nlevels);
    invScaleFactorVec.resize(nlevels);
    imagePyramid.resize(nlevels);
    nfeaturesPerLevelVec.resize(nlevels);
    pixelOffset.resize(nlevels * CIRCLE_SIZE);

    continuousPixelsRequired = CIRCLE_SIZE / 2;
    onePointFiveCircles = CIRCLE_SIZE + continuousPixelsRequired + 1;

    iniThFAST = std::min(255, std::max(0, iniThFAST));
    minThFAST = std::min(iniThFAST, std::max(0, minThFAST));

    //initialize threshold tabs for init and min threshold
    int i;
    for (i = 0; i < 512; ++i)
    {
        int v = i - 255;
        if (v < -iniThFAST)
        {
            threshold_tab_init[i] = (uchar)1;
            threshold_tab_min[i] = (uchar)1;
        } else if (v > iniThFAST)
        {
            threshold_tab_init[i] = (uchar)2;
            threshold_tab_min[i] = (uchar)2;

        } else
        {
            threshold_tab_init[i] = (uchar)0;
            if (v < -minThFAST)
            {
                threshold_tab_min[i] = (uchar)1;
            } else if (v > minThFAST)
            {
                threshold_tab_min[i] = (uchar)2;
            } else
                threshold_tab_min[i] = (uchar)0;
        }
    }

    scaleFactorVec[0] = 1.f;
    invScaleFactorVec[0] = 1.f;

    for (i = 1; i < nlevels; ++i) {
        scaleFactorVec[i] = scaleFactor * scaleFactorVec[i - 1];
        invScaleFactorVec[i] = 1 / scaleFactorVec[i];
    }


    float fac = 1.f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1.f - fac) / (1.f - (float) pow((double) fac, (double) nlevels));

    int sumFeatures = 0;
    for (i = 0; i < nlevels - 1; ++i)
    {
        nfeaturesPerLevelVec[i] = myRound(nDesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevelVec[i];
        nDesiredFeaturesPerScale *= fac;
    }
    nfeaturesPerLevelVec[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    const int nPoints = 512;
    const auto tempPattern = (const cv::Point*) bit_pattern_31_;
    std::copy(tempPattern, tempPattern+nPoints, std::back_inserter(pattern));
}


void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask,
        std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors)
{
    if (inputImage.empty())
        return;

    cv::Mat image = inputImage.getMat();
    assert(image.type() == CV_8UC1);

    ComputeScalePyramid(image);


    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (int i = 0; i < CIRCLE_SIZE; ++i)
        {
            pixelOffset[lvl*CIRCLE_SIZE + i] =
                    CIRCLE_OFFSETS[i][0] + CIRCLE_OFFSETS[i][1] * (int)imagePyramid[lvl].step1();
        }
    }


    std::vector<std::vector<cv::KeyPoint>> allkpts;
    DivideAndFAST(allkpts, Distribution::QUADTREE_ORBSLAMSTYLE, true);

    ComputeAngles(allkpts);

    cv::Mat BRIEFdescriptors;
    int nkpts = 0;
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        nkpts += (int)allkpts[lvl].size();
    }
    if (nkpts <= 0)
    {
        outputDescriptors.release();
    } else
    {
        outputDescriptors.create(nkpts, 32, CV_8U);
        BRIEFdescriptors = outputDescriptors.getMat();
    }

    resultKeypoints.clear();
    resultKeypoints.reserve(nkpts);

    ComputeDescriptors(allkpts, BRIEFdescriptors);

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        int size = PATCH_SIZE * scaleFactorVec[lvl];
        float scale = scaleFactorVec[lvl];
        for (auto &kpt : allkpts[lvl])
        {
            kpt.size = size;
            if (lvl)
                kpt.pt *= scale;
        }
    }

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        resultKeypoints.insert(resultKeypoints.end(), allkpts[lvl].begin(), allkpts[lvl].end());
    }
}

/** @overload
 * @param inputImage single channel img-matrix
 * @param mask ignored
 * @param resultKeypoints keypoint vector in which results will be stored
 * @param outputDescriptors matrix in which descriptors will be stored
 * @param distributionMode decides the method to call for kpt-distribution, see Distribution.h
 */

void ORBextractor::operator()(cv::InputArray inputImage, cv::InputArray mask,
        std::vector<cv::KeyPoint> &resultKeypoints, cv::OutputArray outputDescriptors,
        Distribution::DistributionMethod distributionMode, bool distributePerLevel)
{
    std::chrono::high_resolution_clock::time_point funcEntry = std::chrono::high_resolution_clock::now();

    if (inputImage.empty())
        return;

    cv::Mat image = inputImage.getMat();
    assert(image.type() == CV_8UC1);

    ComputeScalePyramid(image);


    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (int i = 0; i < CIRCLE_SIZE; ++i)
        {
            pixelOffset[lvl*CIRCLE_SIZE + i] =
                    CIRCLE_OFFSETS[i][0] + CIRCLE_OFFSETS[i][1] * (int)imagePyramid[lvl].step1();
        }
    }


    std::vector<std::vector<cv::KeyPoint>> allkpts;

    //using namespace std::chrono;
    //high_resolution_clock::time_point t1 = high_resolution_clock::now();

    DivideAndFAST(allkpts, distributionMode, true, 30, distributePerLevel);


    ComputeAngles(allkpts);

    //high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //auto d = duration_cast<microseconds>(t2-t1).count();
    //std::cout << "\nmy comp time for FAST + distr: " << d << "\n";

    cv::Mat BRIEFdescriptors;
    int nkpts = 0;
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        nkpts += (int)allkpts[lvl].size();
    }
    if (nkpts <= 0)
    {
        outputDescriptors.release();
    }
    else
    {
        outputDescriptors.create(nkpts, 32, CV_8U);
        BRIEFdescriptors = outputDescriptors.getMat();
    }

    resultKeypoints.clear();
    resultKeypoints.reserve(nkpts);

    ComputeDescriptors(allkpts, BRIEFdescriptors);

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        int size = PATCH_SIZE * scaleFactorVec[lvl];
        float scale = scaleFactorVec[lvl];
        for (auto &kpt : allkpts[lvl])
        {
            kpt.size = size;
            if (lvl)
                kpt.pt *= scale;
        }
    }

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        resultKeypoints.insert(resultKeypoints.end(), allkpts[lvl].begin(), allkpts[lvl].end());
    }

    //TODO: activate max duration
    //ensure feature detection always takes 50ms
    unsigned long maxDuration = 50000;
    std::chrono::high_resolution_clock::time_point funcExit = std::chrono::high_resolution_clock::now();
    auto funcDuration = std::chrono::duration_cast<std::chrono::microseconds>(funcExit-funcEntry).count();
    //assert(funcDuration <= maxDuration);
    //if (funcDuration < maxDuration)
    //{
    //    auto sleeptime = maxDuration - funcDuration;
    //    usleep(sleeptime);
    //}
}


void ORBextractor::ComputeAngles(std::vector<std::vector<cv::KeyPoint>> &allkpts)
{
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        for (auto &kpt : allkpts[lvl])
        {
            kpt.angle = IntensityCentroidAngle(&imagePyramid[lvl].at<uchar>(myRound(kpt.pt.y), myRound(kpt.pt.x)),
                                               imagePyramid[lvl].step1());
        }
    }
}


void ORBextractor::ComputeDescriptors(std::vector<std::vector<cv::KeyPoint>> &allkpts, cv::Mat &descriptors)
{
    const auto degToRadFactor = (float)(CV_PI/180.f);
    const cv::Point* p = &pattern[0];

    int current = 0;

    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        cv::Mat lvlClone = imagePyramid[lvl].clone();
        cv::GaussianBlur(lvlClone, lvlClone, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        const int step = (int)lvlClone.step;


        int i = 0, nkpts = allkpts[lvl].size();
        for (int k = 0; k < nkpts; ++k, ++current)
        {
            const cv::KeyPoint &kpt = allkpts[lvl][k];
            auto descPointer = descriptors.ptr<uchar>(current);        //ptr to beginning of current descriptor
            const uchar* pixelPointer = &lvlClone.at<uchar>(myRound(kpt.pt.y), myRound(kpt.pt.x));  //ptr to kpt in img

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
void ORBextractor::DivideAndFAST(std::vector<std::vector<cv::KeyPoint>> &allkpts,
        Distribution::DistributionMethod mode, bool divideImage, int cellSize, bool distributePerLevel)
{
    allkpts.resize(nlevels);

    const int minimumX = EDGE_THRESHOLD - 3, minimumY = minimumX;

    if (!divideImage)
    {
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            std::vector<cv::KeyPoint> levelKpts;
            levelKpts.clear();
            levelKpts.reserve(nfeatures*10);

            const int maximumX = imagePyramid[lvl].cols - EDGE_THRESHOLD + 3;
            const int maximumY = imagePyramid[lvl].rows - EDGE_THRESHOLD + 3;

            //std::cout << "lvl " << lvl << ": minX=" << minimumX << ", maxX=" << maximumX <<
            //   ", minY=" << minimumY << ", maxY=" << maximumY << "\n";

            //cv::Range colSelect(minimumX, maximumX);
            //cv::Range rowSelect(minimumY, maximumY);
            //cv::Mat levelMat = imagePyramid[lvl](rowSelect, colSelect);

            this->FAST(imagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                    levelKpts, iniThFAST, lvl);

            if (levelKpts.empty())
                this->FAST(imagePyramid[lvl].rowRange(minimumY, minimumY).colRange(minimumX, maximumX),
                        levelKpts, minThFAST, lvl);


            if(levelKpts.empty())
                continue;

            allkpts[lvl].reserve(nfeaturesPerLevelVec[lvl]);

            if (distributePerLevel)
                Distribution::DistributeKeypoints(levelKpts, minimumX, maximumX, minimumY, maximumY,
                        nfeaturesPerLevelVec[lvl], mode);


            allkpts[lvl] = levelKpts;

            for (auto &kpt : allkpts[lvl])
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
                //kpt.angle = IntensityCentroidAngle(&imagePyramid[lvl].at<uchar>(
                //        myRound(kpt.pt.x), myRound(kpt.pt.y)), imagePyramid[lvl].step1());
            }
        }
    }
    else
    {
        int c = std::min(imagePyramid[nlevels-1].rows, imagePyramid[nlevels-1].cols);
        assert(cellSize < c && cellSize > 16);
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            std::vector<cv::KeyPoint> levelKpts;
            levelKpts.clear();
            levelKpts.reserve(nfeatures*10);

            const int maximumX = imagePyramid[lvl].cols - EDGE_THRESHOLD + 3;
            const int maximumY = imagePyramid[lvl].rows - EDGE_THRESHOLD + 3;
            const float width = maximumX - minimumX;
            const float height = maximumY - minimumY;

            const int npatchesInX = width / cellSize;
            const int npatchesInY = height / cellSize;
            const int patchWidth = ceil(width / npatchesInX);
            const int patchHeight = ceil(height / npatchesInY);

            /*
            //Todo: remove
            std::cout << "pyramid[" << lvl << "]: cols = " << imagePyramid[lvl].cols <<
            ", rows = " << imagePyramid[lvl].rows << "\nmaximumX = " << maximumX << ", maximumY = " << maximumY <<
            ", minimumX = " << minimumX << ", minimumY = " << minimumY <<
            "\nwidth = " << width << ", height = " << height << ", npatchesinX = " << npatchesInX << ", npatchesinY = "
            << npatchesInY << ", patchWidth = " << patchWidth << ", patchHeight = " << patchHeight << "\n";
             */

            for (int py = 0; py < npatchesInY; ++py)
            {
                float startY = minimumY + py * patchHeight;
                float endY = startY + patchHeight + 6;

                if (startY >= maximumY-3)
                    continue;

                if (endY > maximumY)
                    endY = maximumY;


                for (int px = 0; px < npatchesInX; ++px)
                {
                    float startX = minimumX + px * patchWidth;
                    float endX = startX + patchWidth + 6;

                    if (startX >= maximumX-6)
                        continue;

                    if (endX > maximumX)
                        endX = maximumX;

                    //cv::Range colSelect(startX, endX);
                    //cv::Range rowSelect(startY, endY);
                    //cv::Mat patch = imagePyramid[lvl](rowSelect, colSelect);
                    std::vector<cv::KeyPoint> patchKpts;


                    std::chrono::high_resolution_clock::time_point FASTEntry = std::chrono::high_resolution_clock::now();
                    //this->FAST(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                    //        patchKpts, iniThFAST, lvl);
                    //FAST_cv(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                    //        patchKpts, iniThFAST, true);
                    cv::FAST(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                            patchKpts, iniThFAST, true);

                    if (patchKpts.empty())

                        //this->FAST(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                        //       patchKpts, minThFAST, lvl);
                        //FAST_cv(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                        //         patchKpts, minThFAST, true);
                        cv::FAST(imagePyramid[lvl].rowRange(startY, endY).colRange(startX, endX),
                                 patchKpts, minThFAST, true);



                    if(patchKpts.empty())
                        continue;

                    for (auto &kpt : patchKpts)
                    {
                        kpt.pt.y += py * patchHeight;
                        kpt.pt.x += px * patchWidth;
                        levelKpts.emplace_back(kpt);
                    }
                }
            }

            allkpts[lvl].reserve(nfeatures);

            if (distributePerLevel)
                Distribution::DistributeKeypoints(levelKpts, minimumX, maximumX, minimumY, maximumY,
                        nfeaturesPerLevelVec[lvl], mode);


            allkpts[lvl] = levelKpts;



            for (auto &kpt : allkpts[lvl])
            {
                kpt.pt.y += minimumY;
                kpt.pt.x += minimumX;
                kpt.octave = lvl;
                //kpt.angle = IntensityCentroidAngle(&imagePyramid[lvl].at<uchar>(
                //        myRound(kpt.pt.x), myRound(kpt.pt.y)), imagePyramid[lvl].step1());
            }
        }
    }
    if (!distributePerLevel)
    {
        int nkpts = 0;
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            nkpts += allkpts[lvl].size();
        }
        std::vector<cv::KeyPoint> temp(nkpts);
        for (int lvl = 0; lvl < nlevels; ++lvl)
        {
            temp.insert(temp.end(), allkpts[lvl].begin(), allkpts[lvl].end());
        }
        Distribution::DistributeKeypoints(temp, 0, imagePyramid[0].cols, 0, imagePyramid[0].rows, nfeatures, mode);

        for (int lvl = 0; lvl < nlevels; ++lvl)
            allkpts[lvl].clear();

        for (auto &kpt : temp)
        {
            allkpts[kpt.octave].emplace_back(kpt);
        }
    }
}


//move to separate file?
void ORBextractor::FAST(cv::Mat img, std::vector<cv::KeyPoint> &keypoints, int &threshold, int level)
{
   keypoints.clear();

    int offset[CIRCLE_SIZE];
    for (int i = 0; i < CIRCLE_SIZE; ++i)
    {
        offset[i] = pixelOffset[level*CIRCLE_SIZE + i];
    }

    assert(threshold == minThFAST || threshold == iniThFAST); //only initial or min threshold should be passed


    uchar *threshold_tab;
    if (threshold == iniThFAST)
        threshold_tab = threshold_tab_init;
    else
        threshold_tab = threshold_tab_min;


    uchar cornerScores[img.cols*3];
    int cornerPos[img.cols*3];

    memset(cornerScores, 0, img.cols*3);
    memset(cornerPos, 0, img.cols*3);

    uchar* currRowScores = &cornerScores[0];
    uchar* prevRowScores = &cornerScores[img.cols];
    uchar* pprevRowScores = &cornerScores[img.cols*2];

    int* currRowPos = &cornerPos[0];
    int* prevRowPos = &cornerPos[img.cols];
    int* pprevRowPos = &cornerPos[img.cols*2];



    int i, j, k, ncandidates = 0, ncandidatesprev = 0;

    for (i = 3; i < img.rows - 2; ++i)
    {
        const uchar* pointer = img.ptr<uchar>(i) + 3;

        ncandidatesprev = ncandidates;
        ncandidates = 0;

        int* tempPos = pprevRowPos;
        uchar* tempScores = pprevRowScores;

        pprevRowPos = prevRowPos;
        pprevRowScores = prevRowScores;
        prevRowPos = currRowPos;
        prevRowScores = currRowScores;

        currRowPos = tempPos;
        currRowScores = tempScores;

        memset(currRowPos, 0, img.cols);
        memset(currRowScores, 0, img.cols);

        if (i < img.rows - 3) // skip last row
        {
            for (j = 3; j < img.cols-3; ++j, ++pointer)
            {
                int val = pointer[0];                           //value of central pixel
                const uchar *tab = &threshold_tab[255] - val;       //shift threshold tab by val

                /*
                //TODO:remove
                std::cout << "\ncurrent i(=y): " << i << ", current j(=x): " << j;
                if(j == 52 && i == 178)
                {
                    std::cout << "\ntest\n";
                }
                ////////////
                */

                int discard = tab[pointer[offset[PIXELS_TO_CHECK[0]]]]
                              | tab[pointer[offset[PIXELS_TO_CHECK[1]]]];

                if (discard == 0)
                    continue;

                bool gotoNextCol = false;
                for (k = 2; k < 16; k+=2)
                {
                    discard &= tab[pointer[offset[PIXELS_TO_CHECK[k]]]]
                               | tab[pointer[offset[PIXELS_TO_CHECK[k+1]]]];
                    if (k == 6 && discard == 0)
                    {
                        gotoNextCol = true;
                        break;
                    }
                    if (k == 14 && discard == 0)
                    {
                        gotoNextCol = true;
                    }

                }
                if (gotoNextCol) // initial FAST-check failed
                    continue;


                if (discard & 1) // check for continuous circle of pixels darker than threshold
                {
                    int compare = val - threshold;
                    int contPixels = 0;

                    for (k = 0; k < onePointFiveCircles; ++k)
                    {
                        int a = pointer[offset[k%CIRCLE_SIZE]];
                        if (a < compare)
                        {
                            ++contPixels;
                            if (contPixels > continuousPixelsRequired)
                            {
                                currRowPos[ncandidates++] = j;

                                currRowScores[j] = OptimizedCornerScore(pointer, offset, threshold);
                                break;
                            }
                        } else
                            contPixels = 0;
                    }
                }

                if (discard & 2) // check for continuous circle of pixels brighter than threshold
                {
                    int compare = val + threshold;
                    int contPixels = 0;

                    for (k = 0; k < onePointFiveCircles; ++k)
                    {
                        int a = pointer[offset[k%CIRCLE_SIZE]];
                        if (a > compare)
                        {
                            ++contPixels;
                            if (contPixels > continuousPixelsRequired)
                            {
                                currRowPos[ncandidates++] = j;

                                currRowScores[j] = OptimizedCornerScore(pointer, offset, threshold);
                                break;
                            }
                        } else
                            contPixels = 0;
                    }
                }
            }
        }


        if (i == 3)   //skip first row
            continue;

        for (k = 0; k < ncandidatesprev; ++k)
        {
            int pos = prevRowPos[k];
            int score = prevRowScores[pos];

            //TODO: remove after debugging FAST
            /*
            if (i == 35 && score == 42)
            {
                std::cout <<"scores of row with y = 35 (any patch), x = " << pos << "\n"
                            "pprevrow[x-1]: "<<(int)pprevRowScores[pos-1]<<
                ", pprevrow[x]: " << (int)pprevRowScores[pos] << ", pprevrow[x+1]: " << (int)pprevRowScores[pos+1] <<
                "\nprevrow[x-1]: " << (int)prevRowScores[pos-1] << ", candodate: " << score << ", prevrow[x+1]: " <<
                (int)prevRowScores[pos+1] << "\ncurrow[x-1]: " << (int)currRowScores[pos-1] << ", currow[x]: " <<
                (int)currRowScores[pos] << ", currow[x+1]: " << (int)currRowScores[pos+1] << "\n\n";
            }
             */


                //////////////////////////////////////

            if (score > pprevRowScores[pos-1] && score > pprevRowScores[pos] && score > pprevRowScores[pos+1] &&
                score > prevRowScores[pos+1] && score > prevRowScores[pos-1] &&
                score > currRowScores[pos-1] && score > currRowScores[pos] && score > currRowScores[pos+1])
            {
                keypoints.emplace_back(cv::KeyPoint((float)pos, (float)(i-1),
                                                    7.f, -1, (float)score, level));
            }
        }
    }
}



int ORBextractor::CornerScore(const uchar* pointer, const int offset[], int &threshold)
{
    int val = pointer[0];
    int i;
    int diff[onePointFiveCircles];
    for (i = 0; i < CIRCLE_SIZE; ++i)
    {
        diff[i] = (val - pointer[offset[i]]);
    }
    for ( ; i < onePointFiveCircles; ++i)
    {
        diff[i] = diff[i-CIRCLE_SIZE];
    }

    int a0 = threshold;
    for (i = 0; i < CIRCLE_SIZE; i += 2)
    {
        int a = std::min(diff[i + 1], diff[i + 2]);
        a = std::min(a, diff[i + 3]);
        if (a <= a0)
            continue;
        a = std::min(a, diff[i + 4]);
        a = std::min(a, diff[i + 5]);
        a = std::min(a, diff[i + 6]);
        a = std::min(a, diff[i + 7]);
        a = std::min(a, diff[i + 8]);
        a0 = std::max(a0, std::min(a, diff[i]));
        a0 = std::max(a0, std::min(a, diff[i + 9]));
    }

    int b0 = -a0;
    for (i = 0; i < CIRCLE_SIZE; i += 2)
    {
        int b = std::max(diff[i + 1], diff[i + 2]);
        b = std::max(b, diff[i + 3]);
        b = std::max(b, diff[i + 4]);
        b = std::max(b, diff[i + 5]);
        if (b >= b0)
            continue;
        b = std::max(b, diff[i + 6]);
        b = std::max(b, diff[i + 7]);
        b = std::max(b, diff[i + 8]);
        b0 = std::min(b0, std::max(b, diff[i]));
        b0 = std::min(b0, std::max(b, diff[i + 9]));
    }

    return -b0 - 1;
}

int ORBextractor::OptimizedCornerScore(const uchar* pointer, const int offset[], int &threshold)
{
    int val = pointer[0];
    int i;
    int diff[onePointFiveCircles];
    for (i = 0; i < CIRCLE_SIZE; ++i)
    {
        diff[i] = (val - pointer[offset[i]]);
    }
    for ( ; i < onePointFiveCircles; ++i)
    {
        diff[i] = diff[i-CIRCLE_SIZE];
    }

    int a0 = threshold;
    for (i = 0; i < CIRCLE_SIZE; i += 2)
    {
        int a;
        if (diff[i+1] < diff[i+2])
            a = diff[i+1];
        else
            a = diff[i+2];

        if (diff[i+3] < a)
            a = diff[i+3];
        if (a0 > a)
            continue;

        if (diff[i+4] < a)
            a = diff[i+4];
        if (diff[i+5] < a)
            a = diff[i+5];
        if (diff[i+6] < a)
            a = diff[i+6];
        if (diff[i+7] < a)
            a = diff[i+7];
        if (diff[i+8] < a)
            a = diff[i+8];

        int c;
        if (a < diff[i])
            c = a;
        else
            c = diff[i];

        if (c > a0)
            a0 = c;
        if (diff[i+9] < a)
            a = diff[i+9];
        if (a > a0)
            a0 = a;
    }

    int b0 = -a0;
    for (i = 0; i < CIRCLE_SIZE; i += 2)
    {
        int b;
        if (diff[i+1] > diff[i+2])
            b = diff[i+1];
        else
            b = diff[i+2];

        if (diff[i+3] > b)
            b = diff[i+3];
        if (diff[i+4] > b)
            b = diff[i+4];
        if (diff[i+5] > b)
            b = diff[i+5];

        if (b0 < b)
            continue;

        if (diff[i+6] > b)
            b = diff[i+6];
        if (diff[i+7] > b)
            b = diff[i+7];
        if (diff[i+8] > b)
            b = diff[i+8];

        int c;
        if (diff[i] > b)
            c = diff[i];
        else
            c = b;

        if (c < b0)
            b0 = c;
        if (diff[i+9] > b)
            b = diff[i+9];
        if (b < b0)
            b0 = b;
    }
    return -b0 - 1;
}


void ORBextractor::ComputeScalePyramid(cv::Mat &image)
{
    for (int lvl = 0; lvl < nlevels; ++ lvl)
    {
        int width = (int)myRound(image.cols * invScaleFactorVec[lvl]); // 1.f / getScale(lvl));
        int height = (int)myRound(image.rows * invScaleFactorVec[lvl]); // 1.f / getScale(lvl));
        int doubleEdge = EDGE_THRESHOLD * 2;
        int borderedWidth = width + doubleEdge;
        int borderedHeight = height + doubleEdge;

        //Size sz(width, height);
        //Size borderedSize(borderedWidth, borderedHeight);

        cv::Mat borderedImg(borderedHeight, borderedWidth, image.type());
        cv::Range rowRange(EDGE_THRESHOLD, height + EDGE_THRESHOLD);
        cv::Range colRange(EDGE_THRESHOLD, width + EDGE_THRESHOLD);

        //imagePyramid[lvl] = borderedImg(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, width, height));
        imagePyramid[lvl] = borderedImg(rowRange, colRange);


        if (lvl)
        {
            cv::resize(imagePyramid[lvl-1], imagePyramid[lvl], cv::Size(width, height), 0, 0, CV_INTER_LINEAR);

            cv::copyMakeBorder(imagePyramid[lvl], borderedImg, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               EDGE_THRESHOLD, cv::BORDER_REFLECT_101+cv::BORDER_ISOLATED);
        }
        else
        {
            cv::copyMakeBorder(image, borderedImg, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               cv::BORDER_REFLECT_101);
        }
    }
}

/*
float ORBextractor::getScale(int lvl)
{
    return std::pow(scaleFactor, (double)lvl);
}
*/

//TODO:remove from master
//debug

void ORBextractor::Tests(cv::InputArray inputImage, std::vector<cv::KeyPoint> &resKeypoints,
        cv::OutputArray outputDescriptors, bool myFAST, bool myDesc)
{
    if (inputImage.empty())
        return;
    cv::Mat image = inputImage.getMat();
    assert(image.type() == CV_8UC1);

    ComputeScalePyramid(image);


    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        int step = (int)imagePyramid[lvl].step1();
        for (int i = 0; i < CIRCLE_SIZE; ++i)
        {
            pixelOffset[lvl*CIRCLE_SIZE + i] = CIRCLE_OFFSETS[i][0] + CIRCLE_OFFSETS[i][1] * step;
        }
    }

    resKeypoints.clear();
    resKeypoints.reserve(nfeatures);

    std::vector<std::vector<cv::KeyPoint>> allMyKeypoints;


    //TODO: remove ////////////////////////////////////////////////////////////////////////////////////////////////////
    int offset[CIRCLE_SIZE];
    for (int i = 0; i < CIRCLE_SIZE; ++i)
    {
        offset[i] = pixelOffset[i];
    }


    std::vector<cv::KeyPoint> testkpts;
    int N = 1000;
    testkpts.reserve(N);
    this->FAST(image, testkpts, iniThFAST, 0);

    std::vector<uchar*> ptrvec;
    for (auto &kpt : testkpts)
    {
        uchar* ptr = &image.at<uchar>(myRound(kpt.pt.y), myRound(kpt.pt.x));
        ptrvec.emplace_back(ptr);

    }

    std::chrono::high_resolution_clock::time_point stdminstart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
    {
        this->CornerScore(ptrvec[i], offset, iniThFAST);
    }


    std::chrono::high_resolution_clock::time_point stdminend = std::chrono::high_resolution_clock::now();

    auto stdminduration = std::chrono::duration_cast<std::chrono::microseconds>(stdminend-stdminstart).count();
    std::cout << "\nnormal duration = " << stdminduration << " microseconds.\n";


    std::chrono::high_resolution_clock::time_point ifstart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
    {
        this->OptimizedCornerScore(ptrvec[i], offset, iniThFAST);
    }
    std::chrono::high_resolution_clock::time_point ifend = std::chrono::high_resolution_clock::now();


    auto ifduration = std::chrono::duration_cast<std::chrono::microseconds>(ifend-ifstart).count();
    std::cout << "\noptimized(?) duration = " << ifduration << " microseconds.\n";
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //TODO: remove
    return;

    //first bool: quad-tree distribution, 2nd bool: divide
    DivideAndFAST(allMyKeypoints, Distribution::QUADTREE, false, 30);

    ComputeAngles(allMyKeypoints);

    cv::Mat BRIEFdescriptors;
    int nkpts = 0;
    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        nkpts += (int)allMyKeypoints[lvl].size();
    }
    if (nkpts <= 0)
    {
        outputDescriptors.release();
    }
    else
    {
        outputDescriptors.create(nkpts, 32, CV_8U);
        BRIEFdescriptors = outputDescriptors.getMat();
    }



    ComputeDescriptors(allMyKeypoints, BRIEFdescriptors);

    for (int lvl = 1; lvl < nlevels; ++lvl)
    {
        float scale = scaleFactorVec[lvl];
        for (auto &kpt : allMyKeypoints[lvl])
        {
            kpt.pt *= scale;
        }
    }


    for (int lvl = 0; lvl < nlevels; ++lvl)
    {
        resKeypoints.insert(resKeypoints.end(), allMyKeypoints[lvl].begin(), allMyKeypoints[lvl].end());
    }

}

long ORBextractor::testingFAST(cv::Mat &img, std::vector<cv::KeyPoint> &kpts, bool myFAST, bool printTime)
{
    using namespace std::chrono;
    kpts.reserve(nfeatures*10);

    const int minimumX = EDGE_THRESHOLD - 3, minimumY = minimumX;
    const int maximumX = img.cols - EDGE_THRESHOLD + 3;
    const int maximumY = img.rows - EDGE_THRESHOLD + 3;

    cv::Range colSelect(minimumX, maximumX);
    cv::Range rowSelect(minimumY, maximumY);
    cv::Mat tempMat = img(rowSelect, colSelect);

    high_resolution_clock::time_point start = high_resolution_clock::now();


    if (myFAST)
    {
        FAST(tempMat, kpts, iniThFAST, 0);
    }
    else
    {
        FAST_cv(img, kpts, iniThFAST, true);
    }

    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    if (printTime)
    {
        std::cout << "\nExecution time of " << (myFAST? "my " : "openCV ") << "FAST was " << duration << " ms.\n";
    }
    return duration;
}

void ORBextractor::testingDescriptors(cv::Mat myDescriptors, cv::Mat compDescriptors, int nkpts, bool printdif,
        int start, int end, bool testTime, cv::OutputArray _descr)
{

    std::cout << "\ndesc empty?:" << myDescriptors.empty() << "\n";

    auto descPointer = myDescriptors.ptr<uchar>((int) 0);
    auto compPointer = compDescriptors.ptr<uchar>((int) 0);

    if (testTime)
    {
        using namespace std::chrono;



        cv::Mat ref_descriptors;
        cv::Mat my_descriptors;


        std::vector<std::vector<cv::KeyPoint>> allkpts;
        assert(!imagePyramid[0].empty());
        this->DivideAndFAST(allkpts, Distribution::QUADTREE, false);


        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int) allkpts[level].size();
        if (nkeypoints == 0)
            _descr.release();
        else
        {
            _descr.create(nkeypoints, 32, CV_8U);
            my_descriptors = _descr.getMat();
        }

        my_descriptors = _descr.getMat();


        /*
        for (int i = start; i < end; ++i)
        {
            std::cout << "myDescriptors[" << i << "] = " << (int)descPointer[i] << "\n";

        }

        for (int i = start; i < end; ++i)
        {
            std::cout << "compDescriptors[" << i << "] = " << (int)compPointer[i] << "\n";
        }

        std::cout << "desc size: " << myDescriptors.total() << "\n";
        std::cout << "comp des size: " << compDescriptors.total() << "\n";
         */

        bool eq = true;
        int count = 0;
        if (myDescriptors.total() == compDescriptors.total())
        {
            int N = nkpts * 32;
            for (int i = 0; i < N; ++i)
            {
                if (descPointer[i] != compPointer[i])
                {
                    ++count;
                    if (printdif)
                    {
                        std::cout << "dif: my[" << i << "]=" << (int) descPointer[i] <<
                                  ", comp[" << i << "]=" << (int) compPointer[i] << "\n";
                    }

                    eq = false;
                }
            }
        } else
        {
            std::cout << "\ndescs not of eq sz\n";
        }


        std::cout << "my and compDesc equal? -->" << (eq ? " yes" : " no") << "\n";
        std::cout << "differences: " << count << "\n";
    }
}



        void ORBextractor::printInternalValues()
        {
            std::cout << "\nInitialization-values of ORBextractor:" << std::endl <<
                      "nFeatures: " << nfeatures << std::endl <<
                      "scaleFactor: " << scaleFactor << std::endl <<
                      "nLevels: " << nlevels << std::endl <<
                      "initial FAST Threshold: " << iniThFAST << std::endl <<
                      "minimum FAST Threshold: " << minThFAST << std::endl;
        }

        template <class T>
        void ORBextractor::PrintArray(T *array, const std::string &name, int start, int end)
        {
            std::cout << std::endl;
            for (int i = start; i < end; ++i) {
                std::cout << name << "[" << i << "] = " << (int) array[i] << std::endl;
            }

        }

        void ORBextractor::PrintKeypoints(std::vector<cv::KeyPoint> &kpts)
        {
            std::cout << "\nKeypoints found: " << std::endl;
            for (auto &kpt : kpts)
            {
                std::cout << "kpt1: (" << kpt.pt.x << ", " << kpt.pt.y << ")\n";
            }
        }

        void ORBextractor::PrintKeypoints(std::vector<cv::KeyPoint> &kpts, int start, int end)
        {
            if (end <= start)
                return;
            std::cout << "\nKeypoints from " << start << " to " <<  end-1 << ":\n";
            for (int i = start; i < end; ++i)
            {
                std::cout << "kpt[" << i << "]: (" << kpts[i].pt.x << "," << kpts[i].pt.y << ")\n";
            }
        }

        void ORBextractor::PrintKeypoints(std::vector<cv::KeyPoint> &kpts, int start, int end, bool printResponse)
        {
            if (end <= start)
                return;
            std::cout << "\nKeypoints from " << start << " to " <<  end-1 << ":\n";
            for (int i = start; i < end; ++i)
            {
                std::cout << "kpt[" << i << "]: (" << kpts[i].pt.x << "," << kpts[i].pt.y << ")";
                if (printResponse)
                {
                    std::cout << ", with cornerscore of " << kpts[i].response;
                }
                std::cout << "\n";
            }
        }


        void ORBextractor::CompareKeypointVectors(std::vector<cv::KeyPoint> &vec1, std::vector<cv::KeyPoint> &vec2)
        {
            if (vec1.size() != vec2.size())
            {
                std::cout << "\nVectors are not the same size:\n";
                std::cout << "Size of vector 1: " << vec1.size() << std::endl;
                std::cout << "Size of vector 2: " << vec2.size() << std::endl;
            }

            int dif = 0;
            int same = 0;

            bool first = true;
            int i;

            for (i = 0; i < vec1.size(); ++i)
            {
                if (vec1[i].pt.x == vec2[i].pt.x &&
                    vec1[i].pt.y == vec2[i].pt.y &&
                    vec1[i].angle == vec2[i].angle)
                {
                    ++same;
                    //std::cout << "\nkeypoints at " << i << " are identical\n";
                    continue;
                }
                if (first)
                {
                    std::cout << "\nFirst differing kpt is at index " << i << std::endl;
                    first = false;
                }
                ++dif;
            }

            if (dif == 0)
            {
                std::cout << "\nVectors are identical!\n";
            } else
            {
                std::cout << "\n" << dif << " keypoints were different, " << same <<
                          " keypoints were identical (and at the same index)\n";
            }
        }
}

#pragma clang diagnostic pop


