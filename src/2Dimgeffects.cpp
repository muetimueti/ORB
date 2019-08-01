#include "include/2Dimgeffects.h"


template<typename T>
void kvis::GaussianBlur(img_t &src, img_t &dst, int xK, int yK, double sigmaX, double sigmaY)
{
    message_assert(xK%2==1 && yK%2==1, "wrong kernel size");
    int hx = xK / 2;
    int hy = yK / 2;
    double kernelX[xK];
    double kernelY[yK];
    MakeKernel1D(kernelX, sigmaX, xK);
    MakeKernel1D(kernelY, sigmaY, yK);

    dst = Saiga::ImageView<T>(src.rows, src.cols, src.pitchBytes, dst.data);
    for (int x = 0; x < src.cols; ++x)
    {
        for (int y = 0; y < src.rows; ++y)
        {
            dst(y, x) = 0;
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
                    dst(y, x) += (T)((double)src(y+ky, x+kx)*kernelX[kx]*kernelY[ky]);
                }
            }
        }
    }
}

void kvis::MakeKernel1D(double k[], double sigma, int sz)
{
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