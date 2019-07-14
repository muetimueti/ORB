#ifndef ORBEXTRACTOR_2DIMGEFFECTS_H
#define ORBEXTRACTOR_2DIMGEFFECTS_H

#include "include/Types.h"

namespace kvis
{
    void GaussianBlur(img_t &src, img_t &dst, int xK, int yK, double sigmaX, double sigmaY);

} //namespace kvis

#endif //ORBEXTRACTOR_2DIMGEFFECTS_H
