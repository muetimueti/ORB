#ifndef ORBEXTRACTOR_AVX_H
#define ORBEXTRACTOR_AVX_H

#include "opencv2/core/utility.hpp"
#include <algorithm>
#include <immintrin.h>
#include "include/FAST.h"
#define CV_SIMD128 1
#define CV_TRY_AVX2 1

#define OPENCV_HAL_ADD(a, b) ((a) + (b))
#define OPENCV_HAL_AND(a, b) ((a) & (b))
#define OPENCV_HAL_NOP(a) (a)
#define OPENCV_HAL_1ST(a, b) (a)

namespace blorp
{

template<typename _Tp, size_t fixed_size = 1024/sizeof(_Tp)+8> class AutoBuffer2
{
public:
    typedef _Tp value_type;

    //! the default constructor
    AutoBuffer2();
    //! constructor taking the real buffer size
    explicit AutoBuffer2(size_t _size);

    //! the copy constructor
    AutoBuffer2(const AutoBuffer2<_Tp, fixed_size>& buf);
    //! the assignment operator
    AutoBuffer2<_Tp, fixed_size>& operator = (const AutoBuffer2<_Tp, fixed_size>& buf);

    //! destructor. calls deallocate()
    ~AutoBuffer2();

    //! allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
    void allocate(size_t _size);
    //! deallocates the buffer if it was dynamically allocated
    void deallocate();
    //! resizes the buffer and preserves the content
    void resize(size_t _size);
    //! returns the current buffer size
    size_t size() const;
    //! returns pointer to the real buffer, stack-allocated or heap-allocated
    inline _Tp* data() { return ptr; }
    //! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    inline const _Tp* data() const { return ptr; }

#if !defined(OPENCV_DISABLE_DEPRECATED_COMPATIBILITY) // use to .data() calls instead
    //! returns pointer to the real buffer, stack-allocated or heap-allocated
    operator _Tp* () { return ptr; }
    //! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    operator const _Tp* () const { return ptr; }
#else
    //! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
    inline _Tp& operator[] (size_t i) { CV_DbgCheckLT(i, sz, "out of range"); return ptr[i]; }
    //! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
    inline const _Tp& operator[] (size_t i) const { CV_DbgCheckLT(i, sz, "out of range"); return ptr[i]; }
#endif

protected:
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    //! size of the real buffer
    size_t sz;
    //! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
    _Tp buf[(fixed_size > 0) ? fixed_size : 1];
};

template<typename _Tp, size_t fixed_size> inline
AutoBuffer2<_Tp, fixed_size>::AutoBuffer2()
{
    ptr = buf;
    sz = fixed_size;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer2<_Tp, fixed_size>::AutoBuffer2(size_t _size)
{
    ptr = buf;
    sz = fixed_size;
    allocate(_size);
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer2<_Tp, fixed_size>::AutoBuffer2(const AutoBuffer2<_Tp, fixed_size>& abuf )
{
    ptr = buf;
    sz = fixed_size;
    allocate(abuf.size());
    for( size_t i = 0; i < sz; i++ )
        ptr[i] = abuf.ptr[i];
}

template<typename _Tp, size_t fixed_size> inline AutoBuffer2<_Tp, fixed_size>&
AutoBuffer2<_Tp, fixed_size>::operator = (const AutoBuffer2<_Tp, fixed_size>& abuf)
{
    if( this != &abuf )
    {
        deallocate();
        allocate(abuf.size());
        for( size_t i = 0; i < sz; i++ )
            ptr[i] = abuf.ptr[i];
    }
    return *this;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer2<_Tp, fixed_size>::~AutoBuffer2()
{ deallocate(); }

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer2<_Tp, fixed_size>::allocate(size_t _size)
{
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    deallocate();
    sz = _size;
    if(_size > fixed_size)
    {
        ptr = new _Tp[_size];
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer2<_Tp, fixed_size>::deallocate()
{
    if( ptr != buf )
    {
        delete[] ptr;
        ptr = buf;
        sz = fixed_size;
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer2<_Tp, fixed_size>::resize(size_t _size)
{
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    size_t i, prevsize = sz, minsize = MIN(prevsize, _size);
    _Tp* prevptr = ptr;

    ptr = _size > fixed_size ? new _Tp[_size] : buf;
    sz = _size;

    if( ptr != prevptr )
        for( i = 0; i < minsize; i++ )
            ptr[i] = prevptr[i];
    for( i = prevsize; i < _size; i++ )
        ptr[i] = _Tp();

    if( prevptr != buf )
        delete[] prevptr;
}

template<typename _Tp, size_t fixed_size> inline size_t
AutoBuffer2<_Tp, fixed_size>::size() const
{ return sz; }

namespace hal {

enum StoreMode
{
STORE_UNALIGNED = 0,
STORE_ALIGNED = 1,
STORE_ALIGNED_NOCACHE = 2
};

}

struct v_uint8x16
{
typedef uchar lane_type;
typedef __m128i vector_type;
enum { nlanes = 16 };

v_uint8x16() : val(_mm_setzero_si128()) {}
explicit v_uint8x16(__m128i v) : val(v) {}
v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
           uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
{
    val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                        (char)v4, (char)v5, (char)v6, (char)v7,
                        (char)v8, (char)v9, (char)v10, (char)v11,
                        (char)v12, (char)v13, (char)v14, (char)v15);
}
uchar get0() const
{
    return (uchar)_mm_cvtsi128_si32(val);
}

__m128i val;
};

struct v_int8x16
{
typedef schar lane_type;
typedef __m128i vector_type;
enum { nlanes = 16 };

v_int8x16() : val(_mm_setzero_si128()) {}
explicit v_int8x16(__m128i v) : val(v) {}
v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
          schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
{
    val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                        (char)v4, (char)v5, (char)v6, (char)v7,
                        (char)v8, (char)v9, (char)v10, (char)v11,
                        (char)v12, (char)v13, (char)v14, (char)v15);
}
schar get0() const
{
    return (schar)_mm_cvtsi128_si32(val);
}

__m128i val;
};

struct v_int16x8
{
typedef short lane_type;
typedef __m128i vector_type;
enum { nlanes = 8 };

v_int16x8() : val(_mm_setzero_si128()) {}
explicit v_int16x8(__m128i v) : val(v) {}
v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
{
    val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                         (short)v4, (short)v5, (short)v6, (short)v7);
}
short get0() const
{
    return (short)_mm_cvtsi128_si32(val);
}

__m128i val;
};

struct v_uint16x8
{
typedef ushort lane_type;
typedef __m128i vector_type;
enum { nlanes = 8 };

v_uint16x8() : val(_mm_setzero_si128()) {}
explicit v_uint16x8(__m128i v) : val(v) {}
v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
{
    val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                         (short)v4, (short)v5, (short)v6, (short)v7);
}
ushort get0() const
{
    return (ushort)_mm_cvtsi128_si32(val);
}

__m128i val;
};

#define OPENCV_HAL_IMPL_SSE_INITVEC(_Tpvec, _Tp, suffix, zsuffix, ssuffix, _Tps, cast) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(_mm_setzero_##zsuffix()); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(_mm_set1_##ssuffix((_Tps)v)); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(cast(a.val)); }

OPENCV_HAL_IMPL_SSE_INITVEC(v_int16x8, short, s16, si128, epi16, short, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_uint8x16, uchar, u8, si128, epi8, char, OPENCV_HAL_NOP)

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(_Tpvec, _Tp) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(_mm_loadu_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(_mm_load_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ return _Tpvec(_mm_loadl_epi64((const __m128i*)ptr)); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return _Tpvec(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)ptr0), \
                                     _mm_loadl_epi64((const __m128i*)ptr1))); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ _mm_storeu_si128((__m128i*)ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ _mm_store_si128((__m128i*)ptr, a.val); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ _mm_stream_si128((__m128i*)ptr, a.val); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
{ \
    if( mode == hal::STORE_UNALIGNED ) \
        _mm_storeu_si128((__m128i*)ptr, a.val); \
    else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
        _mm_stream_si128((__m128i*)ptr, a.val); \
    else \
        _mm_store_si128((__m128i*)ptr, a.val); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, a.val); } \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a.val, a.val)); }

OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint8x16, uchar)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int16x8, short)

#define OPENCV_HAL_IMPL_SSE_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_min, _mm_min_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_max, _mm_max_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_min, _mm_min_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_max, _mm_max_epi16)

#define OPENCV_HAL_IMPL_SSE_BIN_OP(bin_op, _Tpvec, intrin) \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
    { \
        return _Tpvec(intrin(a.val, b.val)); \
    } \
    inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
    { \
        a.val = intrin(a.val, b.val); \
        return a; \
    }
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint8x16, _mm_adds_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint8x16, _mm_subs_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int8x16, _mm_adds_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int8x16, _mm_subs_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint16x8, _mm_adds_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint16x8, _mm_subs_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int16x8, _mm_adds_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int16x8, _mm_subs_epi16)

#define OPENCV_HAL_IMPL_SSE_LOGIC_OP(_Tpvec, suffix, not_const) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(&, _Tpvec, _mm_and_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(|, _Tpvec, _mm_or_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(^, _Tpvec, _mm_xor_##suffix) \
    inline _Tpvec operator ~ (const _Tpvec& a) \
    { \
        return _Tpvec(_mm_xor_##suffix(a.val, not_const)); \
    }
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint8x16, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int8x16, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint16x8, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int16x8, si128, _mm_set1_epi32(-1))

#define OPENCV_HAL_IMPL_SSE_64BIT_CMP_OP(_Tpvec, cast) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return cast(v_reinterpret_as_f64(a) == v_reinterpret_as_f64(b)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return cast(v_reinterpret_as_f64(a) != v_reinterpret_as_f64(b)); }


OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_sub_wrap, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_sub_wrap, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_mul_wrap, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_mul_wrap, _mm_mullo_epi16)

#define OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(_Tpvec, suffix, pack_op, and_op, signmask, allmask) \
inline int v_signmask(const _Tpvec& a) \
{ \
    return and_op(_mm_movemask_##suffix(pack_op(a.val)), signmask); \
} \
inline bool v_check_all(const _Tpvec& a) \
{ return and_op(_mm_movemask_##suffix(a.val), allmask) == allmask; } \
inline bool v_check_any(const _Tpvec& a) \
{ return and_op(_mm_movemask_##suffix(a.val), allmask) != 0; }

#define OPENCV_HAL_PACKS(a) _mm_packs_epi16(a, a)
inline __m128i v_packq_epi32(__m128i a)
{
    __m128i b = _mm_packs_epi32(a, a);
    return _mm_packs_epi16(b, b);
}

OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(_Tpvec, scalartype, func, suffix, sbit) \
inline scalartype v_reduce_##func(const v_##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
    return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype v_reduce_##func(const v_u##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    __m128i smask = _mm_set1_epi16(sbit); \
    val = _mm_xor_si128(val, smask); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
    return (unsigned scalartype)(_mm_cvtsi128_si32(val) ^  sbit); \
}
#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8_SUM(_Tpvec, scalartype, suffix) \
inline scalartype v_reduce_sum(const v_##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 8)); \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 4)); \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 2)); \
    return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype v_reduce_sum(const v_u##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 8)); \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 4)); \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 2)); \
    return (unsigned scalartype)_mm_cvtsi128_si32(val); \
}
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, max, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, min, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8_SUM(int16x8, short, 16)

#define OPENCV_HAL_IMPL_SSE_INITVEC(_Tpvec, _Tp, suffix, zsuffix, ssuffix, _Tps, cast) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(_mm_setzero_##zsuffix()); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(_mm_set1_##ssuffix((_Tps)v)); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(cast(a.val)); }

OPENCV_HAL_IMPL_SSE_INITVEC(v_int8x16, schar, s8, si128, epi8, char, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_uint16x8, ushort, u16, si128, epi16, short, OPENCV_HAL_NOP)



#define OPENCV_HAL_IMPL_SSE_INT_CMP_OP(_Tpuvec, _Tpsvec, suffix, sbit) \
inline _Tpuvec operator == (const _Tpuvec& a, const _Tpuvec& b) \
{ return _Tpuvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpuvec operator != (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpuvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec operator == (const _Tpsvec& a, const _Tpsvec& b) \
{ return _Tpsvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpsvec operator != (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpuvec operator < (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask))); \
} \
inline _Tpuvec operator > (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask))); \
} \
inline _Tpuvec operator <= (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpuvec operator >= (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpsvec operator < (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(b.val, a.val)); \
} \
inline _Tpsvec operator > (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(a.val, b.val)); \
} \
inline _Tpsvec operator <= (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec operator >= (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(b.val, a.val), not_mask)); \
}

OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint8x16, v_int8x16, epi8, (char)-128)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint16x8, v_int16x8, epi16, (short)-32768)

namespace blorp
{

class FAST_t_patternSize16_AVX2
{
public:
    static cv::Ptr<FAST_t_patternSize16_AVX2> getImpl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel);
    virtual void process(int &j, const uchar* &ptr, uchar* curr, int* cornerpos, int &ncorners) = 0;
    virtual ~FAST_t_patternSize16_AVX2() {};
};

class FAST_t_patternSize16_AVX2_Impl final: public FAST_t_patternSize16_AVX2
{
public:
FAST_t_patternSize16_AVX2_Impl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel):
        cols(_cols), nonmax_suppression(_nonmax_suppression), pixel(_pixel)
{
    //patternSize = 16
    t256c = (char)_threshold;
    threshold = std::min(std::max(_threshold, 0), 255);
}

virtual void process(int &j, const uchar* &ptr, uchar* curr, int* cornerpos, int &ncorners) override
{
static const __m256i delta256 = _mm256_broadcastsi128_si256(_mm_set1_epi8((char)(-128))), K16_256 = _mm256_broadcastsi128_si256(_mm_set1_epi8((char)8));
const __m256i t256 = _mm256_broadcastsi128_si256(_mm_set1_epi8(t256c));
for (; j < cols - 32 - 3; j += 32, ptr += 32)
{
__m256i m0, m1;
__m256i v0 = _mm256_loadu_si256((const __m256i*)ptr);

__m256i v1 = _mm256_xor_si256(_mm256_subs_epu8(v0, t256), delta256);
v0 = _mm256_xor_si256(_mm256_adds_epu8(v0, t256), delta256);

__m256i x0 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[0])), delta256);
__m256i x1 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[4])), delta256);
__m256i x2 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[8])), delta256);
__m256i x3 = _mm256_sub_epi8(_mm256_loadu_si256((const __m256i*)(ptr + pixel[12])), delta256);

m0 = _mm256_and_si256(_mm256_cmpgt_epi8(x0, v0), _mm256_cmpgt_epi8(x1, v0));
m1 = _mm256_and_si256(_mm256_cmpgt_epi8(v1, x0), _mm256_cmpgt_epi8(v1, x1));
m0 = _mm256_or_si256(m0, _mm256_and_si256(_mm256_cmpgt_epi8(x1, v0), _mm256_cmpgt_epi8(x2, v0)));
m1 = _mm256_or_si256(m1, _mm256_and_si256(_mm256_cmpgt_epi8(v1, x1), _mm256_cmpgt_epi8(v1, x2)));
m0 = _mm256_or_si256(m0, _mm256_and_si256(_mm256_cmpgt_epi8(x2, v0), _mm256_cmpgt_epi8(x3, v0)));
m1 = _mm256_or_si256(m1, _mm256_and_si256(_mm256_cmpgt_epi8(v1, x2), _mm256_cmpgt_epi8(v1, x3)));
m0 = _mm256_or_si256(m0, _mm256_and_si256(_mm256_cmpgt_epi8(x3, v0), _mm256_cmpgt_epi8(x0, v0)));
m1 = _mm256_or_si256(m1, _mm256_and_si256(_mm256_cmpgt_epi8(v1, x3), _mm256_cmpgt_epi8(v1, x0)));
m0 = _mm256_or_si256(m0, m1);

unsigned int mask = _mm256_movemask_epi8(m0); //unsigned is important!
if (mask == 0){
continue;
}
if ((mask & 0xffff) == 0)
{
j -= 16;
ptr -= 16;
continue;
}

__m256i c0 = _mm256_setzero_si256(), c1 = c0, max0 = c0, max1 = c0;
for (int k = 0; k < 25; k++)
{
__m256i x = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(ptr + pixel[k])), delta256);
m0 = _mm256_cmpgt_epi8(x, v0);
m1 = _mm256_cmpgt_epi8(v1, x);

c0 = _mm256_and_si256(_mm256_sub_epi8(c0, m0), m0);
c1 = _mm256_and_si256(_mm256_sub_epi8(c1, m1), m1);

max0 = _mm256_max_epu8(max0, c0);
max1 = _mm256_max_epu8(max1, c1);
}

max0 = _mm256_max_epu8(max0, max1);
unsigned int m = _mm256_movemask_epi8(_mm256_cmpgt_epi8(max0, K16_256));

for (int k = 0; m > 0 && k < 32; k++, m >>= 1)
if (m & 1)
{
cornerpos[ncorners++] = j + k;
if (nonmax_suppression)
{

short d[25];
for (int q = 0; q < 25; q++)
d[q] = (short)(ptr[k] - ptr[k + pixel[q]]);
v_int16x8 q0 = v_setall_s16(-1000), q1 = v_setall_s16(1000);
for (int q = 0; q < 16; q += 8)
{
v_int16x8 v0_ = v_load(d + q + 1);
v_int16x8 v1_ = v_load(d + q + 2);
v_int16x8 a = v_min(v0_, v1_);
v_int16x8 b = v_max(v0_, v1_);
v0_ = v_load(d + q + 3);
a = v_min(a, v0_);
b = v_max(b, v0_);
v0_ = v_load(d + q + 4);
a = v_min(a, v0_);
b = v_max(b, v0_);
v0_ = v_load(d + q + 5);
a = v_min(a, v0_);
b = v_max(b, v0_);
v0_ = v_load(d + q + 6);
a = v_min(a, v0_);
b = v_max(b, v0_);
v0_ = v_load(d + q + 7);
a = v_min(a, v0_);
b = v_max(b, v0_);
v0_ = v_load(d + q + 8);
a = v_min(a, v0_);
b = v_max(b, v0_);
v0_ = v_load(d + q);
q0 = v_max(q0, v_min(a, v0_));
q1 = v_min(q1, v_max(b, v0_));
v0_ = v_load(d + q + 9);
q0 = v_max(q0, v_min(a, v0_));
q1 = v_min(q1, v_max(b, v0_));
}
q0 = v_max(q0, v_setzero_s16() - q1);

curr[j + k] = (uchar)(v_reduce_max(q0) - 1);//(uchar)FASTdetector::CornerScore_Experimental(ptr, 0);
}
}
}
_mm256_zeroupper();
}

virtual ~FAST_t_patternSize16_AVX2_Impl() override {};

private:
int cols;
char t256c;
int threshold;
bool nonmax_suppression;
const int* pixel;
};

cv::Ptr<FAST_t_patternSize16_AVX2> FAST_t_patternSize16_AVX2::getImpl(int _cols, int _threshold, bool _nonmax_suppression, const int* _pixel)
{
    return cv::Ptr<FAST_t_patternSize16_AVX2>(new FAST_t_patternSize16_AVX2_Impl(_cols, _threshold, _nonmax_suppression, _pixel));
}

}


int cornerScore(const uchar* ptr, const int pixel[], int threshold)
{
    const int K = 8, N = K*3 + 1;
    int k, v = ptr[0];
    short d[N];
    for( k = 0; k < N; k++ )
        d[k] = (short)(v - ptr[pixel[k]]);

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
    return threshold;
}

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

template<int patternSize>
void FAST_t(cv::InputArray _img, std::vector<cv::KeyPoint>& keypoints, int threshold, bool nonmax_suppression)
{
    cv::Mat img = _img.getMat();
    const int K = patternSize / 2, N = patternSize + K + 1;
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

#if CV_SIMD128
    const int quarterPatternSize = patternSize / 4;
    v_uint8x16 delta = v_setall_u8(0x80), t = v_setall_u8((char) threshold), K16 = v_setall_u8((char) K);
    bool hasSimd = true;
#if CV_TRY_AVX2
    cv::Ptr<blorp::FAST_t_patternSize16_AVX2> fast_t_impl_avx2;
    fast_t_impl_avx2 = blorp::FAST_t_patternSize16_AVX2::getImpl(img.cols, threshold, nonmax_suppression, pixel);
#endif

#endif

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for (i = -255; i <= 255; i++)
        threshold_tab[i + 255] = (uchar) (i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer2<uchar> _buf((img.cols + 16) * 3 * (sizeof(int) + sizeof(uchar)) + 128);
    uchar *buf[3];
    buf[0] = _buf.data();
    buf[1] = buf[0] + img.cols;
    buf[2] = buf[1] + img.cols;
    int *cpbuf[3];
    cpbuf[0] = (int *) cv::alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
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
            if (fast_t_impl_avx2)
                fast_t_impl_avx2->process(j, ptr, curr, cornerpos, ncorners);

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
                            curr[j+k] = (uchar)cornerScore(ptr+k, pixel, threshold);
                    }
                }
            }
        }
        cornerpos[-1] = ncorners;

        if( i == 3 )
            continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
        {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
            {
                keypoints.emplace_back(cv::KeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
            }
        }
    }
}

}

#endif