#ifndef ORBEXTRACTOR_2DIMGEFFECTS_H
#define ORBEXTRACTOR_2DIMGEFFECTS_H

#include "include/Types.h"

namespace kvis
{

    template<typename T>
    void GaussianBlur(img_t &src, img_t &dst, int xK, int yK, double sigmaX, double sigmaY);

    void MakeKernel1D(double k[], double sigma, int sz);


} //namespace kvis



#endif //ORBEXTRACTOR_2DIMGEFFECTS_H
