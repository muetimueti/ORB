#include "include/CVFAST.h"
#include <iostream>
#include <chrono>
using namespace cv;

void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
    static const int offsets16[][2] =
            {
                    {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
                    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
            };

    static const int offsets12[][2] =
            {
                    {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
                    {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
            };

    static const int offsets8[][2] =
            {
                    {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
                    {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
            };

    const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                              patternSize == 12 ? offsets12 :
                              patternSize == 8  ? offsets8  : 0;

    CV_Assert(pixel && offsets);

    int k = 0;
    for( ; k < patternSize; k++ )
        pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
    for( ; k < 25; k++ )
        pixel[k] = pixel[k - patternSize];
}

#if VERIFY_CORNERS
static void testCorner(const uchar* ptr, const int pixel[], int K, int N, int threshold) {
    // check that with the computed "threshold" the pixel is still a corner
    // and that with the increased-by-1 "threshold" the pixel is not a corner anymore
    for( int delta = 0; delta <= 1; delta++ )
    {
        int v0 = std::min(ptr[0] + threshold + delta, 255);
        int v1 = std::max(ptr[0] - threshold - delta, 0);
        int c0 = 0, c1 = 0;

        for( int k = 0; k < N; k++ )
        {
            int x = ptr[pixel[k]];
            if(x > v0)
            {
                if( ++c0 > K )
                    break;
                c1 = 0;
            }
            else if( x < v1 )
            {
                if( ++c1 > K )
                    break;
                c0 = 0;
            }
            else
            {
                c0 = c1 = 0;
            }
        }
        CV_Assert( (delta == 0 && std::max(c0, c1) > K) ||
                   (delta == 1 && std::max(c0, c1) <= K) );
    }
}
#endif


int cornerScore_cv(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

#if CV_SIMD128
    if (hasSIMD128())
    {
        v_int16x8 q0 = v_setall_s16(-1000), q1 = v_setall_s16(1000);
        for (k = 0; k < 16; k += 8)
        {
            v_int16x8 v0 = v_load(d + k + 1);
            v_int16x8 v1 = v_load(d + k + 2);
            v_int16x8 a = v_min(v0, v1);
            v_int16x8 b = v_max(v0, v1);
            v0 = v_load(d + k + 3);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 4);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 5);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 6);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 7);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k + 8);
            a = v_min(a, v0);
            b = v_max(b, v0);
            v0 = v_load(d + k);
            q0 = v_max(q0, v_min(a, v0));
            q1 = v_min(q1, v_max(b, v0));
            v0 = v_load(d + k + 9);
            q0 = v_max(q0, v_min(a, v0));
            q1 = v_min(q1, v_max(b, v0));
        }
        q0 = v_max(q0, v_setzero_s16() - q1);
        threshold = v_reduce_max(q0) - 1;
    }
    else
#endif
    {

        int a0 = threshold;
        for( k = 0; k < 16; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            a = std::min(a, (int)d[k+3]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+4]);
            a = std::min(a, (int)d[k+5]);
            a = std::min(a, (int)d[k+6]);
            a = std::min(a, (int)d[k+7]);
            a = std::min(a, (int)d[k+8]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+9]));
        }

        int b0 = -a0;
        for( k = 0; k < 16; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            b = std::max(b, (int)d[k+4]);
            b = std::max(b, (int)d[k+5]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+6]);
            b = std::max(b, (int)d[k+7]);
            b = std::max(b, (int)d[k+8]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+9]));
        }

        threshold = -b0 - 1;
    }

#if VERIFY_CORNERS
    testCorner(ptr, pixel, K, N, threshold);
#endif
    return threshold;
}

void FAST_cv(InputArray _img, std::vector <KeyPoint> &keypoints, int threshold, bool nonmax_suppression)
{
    Mat img = _img.getMat();
    const int K = 16 / 2, N = 16 + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int) img.step, 16);

#if CV_SIMD128
    const int quarterPatternSize = patternSize/4;
    v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char)threshold), K16 = v_setall_u8((char)K);
    bool hasSimd = hasSIMD128();
#if CV_TRY_AVX2
    Ptr<opt_AVX2::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
    if(CV_CPU_HAS_SUPPORT_AVX2)
        fast_t_impl_avx2 = opt_AVX2::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for (i = -255; i <= 255; i++)
        threshold_tab[i + 255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer <uchar> _buf((img.cols + 16) * 3 * (sizeof(int) + sizeof(uchar)) + 128);
    uchar *buf[3];
    //buf[0] = _buf.data();
    buf[1] = buf[0] + img.cols;
    buf[2] = buf[1] + img.cols;
    int *cpbuf[3];
    cpbuf[0] = (int *) alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols * 3);

    for (i = 3; i < img.rows - 2; i++)
    {
        const uchar *ptr = img.ptr<uchar>(i) + 3;
        uchar *curr = buf[(i - 3) % 3];
        int *cornerpos = cpbuf[(i - 3) % 3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if (i < img.rows - 3)
        {
            j = 3;
#if CV_SIMD128
            if( hasSimd )
            {
                if( patternSize == 16 )
                {
#if CV_TRY_AVX2
                    if (fast_t_impl_avx2)
                        fast_t_impl_avx2->process(j, ptr, curr, cornerpos, ncorners);
#endif
                    //vz if (j <= (img.cols - 27)) //it doesn't make sense using vectors for less than 8 elements
                    {
                        for (; j < img.cols - 16 - 3; j += 16, ptr += 16)
                        {
                            v_uint8x16 v = v_load(ptr);
                            v_int8x16 v0 = v_reinterpret_as_s8((v + t) ^ delta);
                            v_int8x16 v1 = v_reinterpret_as_s8((v - t) ^ delta);

                            v_int8x16 x0 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[0]), delta));
                            v_int8x16 x1 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[quarterPatternSize]), delta));
                            v_int8x16 x2 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[2*quarterPatternSize]), delta));
                            v_int8x16 x3 = v_reinterpret_as_s8(v_sub_wrap(v_load(ptr + pixel[3*quarterPatternSize]), delta));

                            v_int8x16 m0, m1;
                            m0 = (v0 < x0) & (v0 < x1);
                            m1 = (x0 < v1) & (x1 < v1);
                            m0 = m0 | ((v0 < x1) & (v0 < x2));
                            m1 = m1 | ((x1 < v1) & (x2 < v1));
                            m0 = m0 | ((v0 < x2) & (v0 < x3));
                            m1 = m1 | ((x2 < v1) & (x3 < v1));
                            m0 = m0 | ((v0 < x3) & (v0 < x0));
                            m1 = m1 | ((x3 < v1) & (x0 < v1));
                            m0 = m0 | m1;

                            int mask = v_signmask(m0);
                            if( mask == 0 )
                                continue;
                            if( (mask & 255) == 0 )
                            {
                                j -= 8;
                                ptr -= 8;
                                continue;
                            }

                            v_int8x16 c0 = v_setzero_s8();
                            v_int8x16 c1 = v_setzero_s8();
                            v_uint8x16 max0 = v_setzero_u8();
                            v_uint8x16 max1 = v_setzero_u8();
                            for( k = 0; k < N; k++ )
                            {
                                v_int8x16 x = v_reinterpret_as_s8(v_load((ptr + pixel[k])) ^ delta);
                                m0 = v0 < x;
                                m1 = x < v1;

                                c0 = v_sub_wrap(c0, m0) & m0;
                                c1 = v_sub_wrap(c1, m1) & m1;

                                max0 = v_max(max0, v_reinterpret_as_u8(c0));
                                max1 = v_max(max1, v_reinterpret_as_u8(c1));
                            }

                            max0 = v_max(max0, max1);
                            int m = v_signmask(K16 < max0);

                            for( k = 0; m > 0 && k < 16; k++, m >>= 1 )
                            {
                                if(m & 1)
                                {
                                    cornerpos[ncorners++] = j+k;
                                    if(nonmax_suppression)
                                        curr[j+k] = (uchar)cornerScore<patternSize>(ptr+k, pixel, threshold);
                                }
                            }
                        }
                    }
                }
            }
#endif
            for (; j < img.cols - 3; j++, ptr++)
            {
                int v = ptr[0];
                const uchar *tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                if (d == 0)
                    continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if (d == 0)
                    continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if (d & 1)
                {
                    int vt = v - threshold, count = 0;

                    for (k = 0; k < N; k++)
                    {
                        int x = ptr[pixel[k]];
                        if (x < vt)
                        {
                            if (++count > K)
                            {
                                cornerpos[ncorners++] = j;
                                if (nonmax_suppression)
                                    curr[j] = (uchar) cornerScore_cv(ptr, pixel, threshold);
                                break;
                            }
                        } else
                            count = 0;
                    }
                }

                if (d & 2)
                {
                    int vt = v + threshold, count = 0;

                    for (k = 0; k < N; k++)
                    {
                        int x = ptr[pixel[k]];
                        if (x > vt)
                        {
                            if (++count > K)
                            {
                                cornerpos[ncorners++] = j;
                                if (nonmax_suppression)
                                    curr[j] = (uchar) cornerScore_cv(ptr, pixel, threshold);
                                break;
                            }
                        } else
                            count = 0;
                    }
                }
            }
        }

        cornerpos[-1] = ncorners;

        if (i == 3)
            continue;

        const uchar *prev = buf[(i - 4 + 3) % 3];
        const uchar *pprev = buf[(i - 5 + 3) % 3];
        cornerpos = cpbuf[(i - 4 + 3) % 3];
        ncorners = cornerpos[-1];

        for (k = 0; k < ncorners; k++)
        {
            j = cornerpos[k];
            int score = prev[j];
            /*
            if (i == 35)
            {
                std::cout <<"\nreference fast!!!:\npatch containing (52,178) and (68,178), scores of row with y = 35, x = " << j << "\n"
                           "pprevrow[x-1]: "<<(int)pprev[j-1]<<
                          ", pprevrow[x]: " << (int)pprev[j] << ", pprevrow[x+1]: " << (int)pprev[j+1] <<
                          "\nprevrow[x-1]: " << (int)prev[j-1] << ", candidate: " << score << ", prevrow[x+1]: " <<
                          (int)prev[j+1] << "\ncurrow[x-1]: " << (int)curr[j-1] << ", currow[x]: " <<
                          (int)curr[j] << ", currow[x+1]: " << (int)curr[j+1] << "\n\n";
            }
             */

            if (!nonmax_suppression ||
                (score > prev[j + 1] && score > prev[j - 1] &&
                 score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1] &&
                 score > curr[j - 1] && score > curr[j] && score > curr[j + 1]))
            {
                keypoints.push_back(KeyPoint((float) j, (float) (i - 1), 7.f, -1, (float) score));
            }
        }
    }
}


