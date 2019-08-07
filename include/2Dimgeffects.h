#ifndef ORBEXTRACTOR_2DIMGEFFECTS_H
#define ORBEXTRACTOR_2DIMGEFFECTS_H

#include "include/Types.h"
//#include <saiga/core/util/statistics.h>

namespace kvis
{
void MakeKernel1D(std::vector<double>& k, double sigma, int sz)
{
    //k = Saiga::gaussianBlurKernel(sz/2, sigma);
    //return;
    double sigmaX = sigma > 0 ? sigma : ((sz - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum = 0;
    int i = 0;
    for (i = 0; i < sz; ++i)
    {
        double x = i - (sz - 1) * 0.5;
        double t = std::exp(scale2X * x * x);
        k[i] = t;
        sum += t;
    }
    sum = 1./sum;
    for (i = 0; i < sz; ++i)
    {
        k[i]*=sum;
    }
}

template <typename T>
void ApplySeparableFilter(bool dim, std::vector<double>& kernel, img_t& img)
{
    //int max = dim == 0 ? img.cols : dim == 1 ? img.rows : -1;
    int hk = kernel.size()/2;
    double val;
    if (!dim)
    {
        for (int y = 0; y < img.rows; ++y)
        {
            for (int x = 0; x < img.cols; ++x)
            {
                val = 0;
                for (int k = -hk; k <= hk; ++k)
                {
                    if (x < kernel.size() && k < 0)
                    {
                        //k = -k;
                    }
                    val += ((double)img(y, x+k))*kernel[k];
                }
                img(y, x) = (T)val;
            }
        }
    }
    else
    {
        for (int x = 0; x < img.cols; ++x)
        {
            for (int y = 0; y < img.rows; ++y)
            {
                val = 0;
                for (int k = -hk; k <= hk; ++k)
                {
                    if (y < kernel.size() && k < 0)
                    {
                        //k = -k;
                    }
                    val += ((double)img(y+k, x))*kernel[k];
                }
                img(y, x) = (T)val;
            }
        }
    }

}

template<typename T>
void GaussianBlur(img_t &src, img_t &dst, int xK, int yK, double sigmaX, double sigmaY)
{
    message_assert(xK%2==1 && yK%2==1, "wrong kernel size");
    int hx = xK / 2;
    int hy = yK / 2;
    std::vector<double> kernelX(xK);
    std::vector<double> kernelY(yK);
    MakeKernel1D(kernelX, sigmaX, xK);
    MakeKernel1D(kernelY, sigmaY, yK);

    dst = Saiga::ImageView<T>(src.rows, src.cols, src.pitchBytes, dst.data);
    ApplySeparableFilter<uchar>(0, kernelX, dst);
    ApplySeparableFilter<uchar>(1, kernelY, dst);
    /*
    for (int x = 0; x < src.cols; ++x)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            double val = 0;
            for (int kx = -hx; kx < hx; ++kx)
            {
                for (int ky = -hy; ky < hy; ++ky)
                {
                    if (x<xK && kx<0)
                    {
                        kx = -kx;
                    }
                    if (y<yK && ky<0)
                    {
                        ky = -ky;
                    }
                    val += ((double)src(y+ky, x+kx)*kernelX[kx]*kernelY[ky]);
                    //std::cout << "Val+=" << ((double)src(y+ky, x+kx)*kernelX[kx]*kernelY[ky]) << "\n";
                }
            }
            //std::cout << "Final Val=" << val << "\n\n";
            dst(y, x) = (T)val;
        }
    }
     */
}

} //namespace kvis



#endif //ORBEXTRACTOR_2DIMGEFFECTS_H
