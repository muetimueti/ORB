#include "include/FAST.h"


const int CIRCLE_SIZE = 16;
const int CIRCLE_OFFSETS[16][2] =
        {{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
         {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}};
const int PIXELS_TO_CHECK[16] =
        {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15};


FAST::FAST(int iniThreshold, int minThreshold) : initialThreshold(iniThreshold), minimumThreshold(minThreshold),
    imageCols(0), imageRows(0), threshold_tab_init{}, threshold_tab_min{}
{
    continuousPixelsRequired = CIRCLE_SIZE / 2;
    onePointFiveCircles = CIRCLE_SIZE + continuousPixelsRequired + 1;


    initialThreshold = std::min(255, std::max(0, initialThreshold));
    minimumThreshold = std::min(initialThreshold, std::max(0, minimumThreshold));

    //initialize threshold tabs for init and min threshold
    int i;
    for (i = 0; i < 512; ++i)
    {
        int v = i - 255;
        if (v < -initialThreshold)
        {
            threshold_tab_init[i] = (uchar)1;
            threshold_tab_min[i] = (uchar)1;
        } else if (v > initialThreshold)
        {
            threshold_tab_init[i] = (uchar)2;
            threshold_tab_min[i] = (uchar)2;

        } else
        {
            threshold_tab_init[i] = (uchar)0;
            if (v < -minimumThreshold)
            {
                threshold_tab_min[i] = (uchar)1;
            } else if (v > minimumThreshold)
            {
                threshold_tab_min[i] = (uchar)2;
            } else
                threshold_tab_min[i] = (uchar)0;
        }
    }
}

/**
 *
 * @param img single channel image matrix
 * @param keypoints results
 * @param pixelOffset mat indices for hitting 16px circle
 * @param threshold FAST threshold
 * @param level octave of img pyramid
 */

void FAST::operator()(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, std::vector<int> &pixelOffset,
        int threshold, int level)
{
    keypoints.clear();

    int offset[CIRCLE_SIZE];
    for (int i = 0; i < CIRCLE_SIZE; ++i)
    {
        offset[i] = pixelOffset[level*CIRCLE_SIZE + i];
    }

    if (!(threshold == minimumThreshold || threshold == initialThreshold))
        //only initial or min threshold should be passed
        return;

    uchar *threshold_tab;
    if (threshold == initialThreshold)
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

    for (i = 3; i < img.rows-2; ++i)
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


        for (j = 3; j < img.cols-3; ++j, ++pointer)
        {
            int val = pointer[0];                           //value of central pixel
            const uchar *tab = &threshold_tab[255] - val;       //shift threshold tab by val


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

                            currRowScores[j] = CornerScore(pointer, offset, threshold);
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

                            currRowScores[j] = CornerScore(pointer, offset, threshold);
                            break;
                        }
                    } else
                        contPixels = 0;
                }
            }
        }

        if (i == 3)   //skip first row
            continue;

        for (k = 0; k < ncandidatesprev; ++k)
        {
            int pos = prevRowPos[k];
            int score = prevRowScores[pos];

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


int FAST::CornerScore(const uchar* pointer, const int offset[], int threshold)
{
    int val = pointer[0];
    int i;
    int diff[onePointFiveCircles];
    for (i = 0; i < onePointFiveCircles; ++i)
    {
        diff[i] = (val - pointer[offset[i%CIRCLE_SIZE]]);
    }

    int a0 = threshold;
    for (i = 0; i < CIRCLE_SIZE; i+=2)
    {
        int a = std::min(diff[i+1], diff[i+2]);
        a = std::min(a, diff[i+3]);
        if (a <= a0)
            continue;
        a = std::min(a, diff[i+4]);
        a = std::min(a, diff[i+5]);
        a = std::min(a, diff[i+6]);
        a = std::min(a, diff[i+7]);
        a = std::min(a, diff[i+8]);
        a0 = std::max(a0, std::min(a, diff[i]));
        a0 = std::max(a0, std::min(a, diff[i+9]));
    }

    int b0 = -a0;
    for (i = 0; i < CIRCLE_SIZE; i+=2)
    {
        int b = std::max(diff[i+1], diff[i+2]);
        b = std::max(b, diff[i+3]);
        b = std::max(b, diff[i+4]);
        b = std::max(b, diff[i+5]);
        if (b >= b0)
            continue;
        b = std::max(b, diff[i+6]);
        b = std::max(b, diff[i+7]);
        b = std::max(b, diff[i+8]);
        b0 = std::min(b0, std::max(b, diff[i]));
        b0 = std::min(b0, std::max(b, diff[i+9]));
    }

    return -b0 - 1;
}
