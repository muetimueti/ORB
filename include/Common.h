#ifndef ORBEXTRACTOR_COMMON_H
#define ORBEXTRACTOR_COMMON_H

#include <opencv2/core/core.hpp>

CV_INLINE  int myRound( float value )
{
#if defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
    return (int)lrint(value);
#else
    // not IEEE754-compliant rounding
      return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

#endif //ORBEXTRACTOR_COMMON_H
