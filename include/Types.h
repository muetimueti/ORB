#ifndef ORBEXTRACTOR_TYPES_H
#define ORBEXTRACTOR_TYPES_H

#include <iostream>
#include "saigacpy/opencv.h"
#include "saigacpy/assert.h"

#include <opencv2/core/types.hpp>


#define message_assert(expr, msg) assert(( (void)(msg), (expr) ))

typedef unsigned char uchar;

typedef Saiga::ImageView<uchar> img_t;

inline  int myRound( float value )
{
#if defined HAVE_LRINT || defined CV_ICC || defined __GNUC__
    return (int)lrint(value);
#else
    // not IEEE754-compliant rounding
      return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

namespace kvis
{
struct Point
{
    int x;
    int y;

    Point() : x(0), y(0) {}

    Point(int _x, int _y) : x(_x), y(_y) {}

    bool inline operator==(const Point &other) const
    {
    return x == other.x && y == other.y;
    }

    template <typename T> inline
    friend Point operator*(const T s, const Point& pt)
    {
        return Point(pt.x*s, pt.y*s);
    }

    template <typename T> inline
    friend void operator*=(Point& pt, const T s)
    {
        pt.x*=s;
        pt.y*=s;
    }

    friend std::ostream& operator<<(std::ostream& os, const Point& pt)
    {
        os << "[" << pt.x << "," << pt.y << "]";
        return os;
    }
};
class KeyPoint
{
public:
    Point pt;  // Points coordinates (x,y) in image space
    float size;
    float angle;
    float response;
    int octave;

    KeyPoint() : pt(), size(0), angle(-1), response(0), octave(0) {}

    explicit KeyPoint(Point _pt, float _size=0, float _angle=-1, float _response=0, int _octave=0) :
        pt(_pt), size(_size), angle(_angle), response(_response), octave(_octave) {}

    KeyPoint(int _x, int _y, float _size=0, float _angle=-1, float _response=0, int _octave=0) :
            pt(_x, _y), size(_size), angle(_angle), response(_response), octave(_octave) {}

    bool operator==(const KeyPoint& other) const
    {
        return pt == other.pt && size == other.size && angle == other.angle && response == other.response &&
           octave == other.octave;
    }

    cv::KeyPoint kvisToCv()
    {
        cv::KeyPoint p(this->pt.x, this->pt.y, this->size, this->angle, this->response, this->octave);
        return p;
    }

    static kvis::KeyPoint cvToKvis(cv::KeyPoint &p)
    {
        kvis::KeyPoint k(myRound(p.pt.x), myRound(p.pt.y), p.size, p.angle, p.response, p.octave);
    }

    friend std::ostream& operator<<(std::ostream& os, const KeyPoint& kpt)
    {
        os << kpt.pt << ": size=" << kpt.size << ", angle=" << kpt.angle << ", response=" << kpt.response <<
            ", octave=" << kpt.octave;
        return os;
    }
};
} //namespace kvis

#endif //ORBEXTRACTOR_TYPES_H
