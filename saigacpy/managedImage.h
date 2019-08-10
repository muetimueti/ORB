/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/imageFormat.h"
#include "saiga/core/image/imageView.h"
#include "saiga/core/util/DataStructures/ArrayView.h"

#include <vector>

namespace Saiga
{
#define DEFAULT_ALIGNMENT 4
/**
 * Note: The first scanline is at position data[0].
 */

class SAIGA_CORE_API Image : public ImageBase
{
public:
    using byte_t   = unsigned char;
    ImageType type = TYPE_UNKNOWN;

protected:
    std::vector<byte_t> vdata;

public:
    Image() {}
    Image(ImageType type) : type(type) {}
    Image(int h, int w, ImageType type)
            : ImageBase(h, w, iAlignUp(elementSize(type) * w, DEFAULT_ALIGNMENT)), type(type)
    {
        create();
    }
    Image(const std::string& file)
    {
        auto res = load(file);
        SAIGA_ASSERT(res);
    }

    // Note: This creates a copy of img
    template <typename T>
    Image(ImageView<T> img)
    {
        setFormatFromImageView(img);
        create();
        img.copyTo(getImageView<T>());
    }

    void create()
    {
        SAIGA_ASSERT(width > 0 && height > 0 && type != TYPE_UNKNOWN);

        if (pitchBytes == 0)
        {
            pitchBytes = iAlignUp(elementSize(type) * width, DEFAULT_ALIGNMENT);
        }

        vdata.resize(size());

        SAIGA_ASSERT(valid());
    }

    void create(int h, int w)
    {
        pitchBytes = 0;
        height     = h;
        width      = w;
        create();
    }

    void create(int h, int w, ImageType t)
    {
        pitchBytes = 0;
        height     = h;
        width      = w;
        type       = t;
        create();
    }

    void create(int h, int w, int p, ImageType t)
    {
        pitchBytes = p;
        create(h, w, t);
    }

    void clear()
    {
        (*this) = Image();
    }

    void free()
    {
        pitchBytes = 0;
        vdata.clear();
        vdata.shrink_to_fit();
    }
    /**
     * @brief makeZero
     * Sets all data to 0.
     */
    void makeZero()
    {
        std::fill(vdata.begin(), vdata.end(), 0);
    }

    /**
     * @brief valid
     * Checks if this image has at least 1 pixel and a valid type.
     */
    bool valid() const
    {
        return width > 0 && height > 0 && pitchBytes > 0 && type != TYPE_UNKNOWN && size() == vdata.size();
    }

    void* data() { return vdata.data(); }
    const void* data() const { return vdata.data(); }

    uint8_t* data8() { return vdata.data(); }
    const uint8_t* data8() const { return vdata.data(); }


    template <typename T>
    inline T& at(int y, int x)
    {
        return reinterpret_cast<T*>(rowPtr(y))[x];
    }

    inline void* rowPtr(int y)
    {
        auto ptr = data8() + y * pitchBytes;
        return ptr;
    }

    inline const void* rowPtr(int y) const
    {
        auto ptr = data8() + y * pitchBytes;
        return ptr;
    }


    template <typename T>
    ImageView<T> getImageView()
    {
        SAIGA_ASSERT(ImageTypeTemplate<T>::type == type);
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    template <typename T>
    ImageView<const T> getConstImageView() const
    {
        SAIGA_ASSERT(ImageTypeTemplate<T>::type == type);
        ImageView<const T> res(*this);
        res.data = data();
        return res;
    }

    template <typename T>
    void setFormatFromImageView(ImageView<T> v)
    {
        ImageBase::operator=(v);
        type               = ImageTypeTemplate<T>::type;
        pitchBytes         = 0;
    }

    template <typename T>
    void createEmptyFromImageView(ImageView<T> v)
    {
        setFormatFromImageView<T>(v);
        create();
    }


    bool load(const std::string& path);
    bool loadFromMemory(ArrayView<const char> data);

    bool save(const std::string& path) const;

    // save in a custom saiga format
    // this can handle all image types
    bool loadRaw(const std::string& path);
    bool saveRaw(const std::string& path) const;

    /**
     * Tries to convert the given image to a storable format.
     * For example:
     * Floating point images are converted to 8-bit grayscale images.
     */
    bool saveConvert(const std::string& path, float minValue = 0, float maxValue = 1);


    std::vector<uint8_t> compress();
    void decompress(std::vector<uint8_t> data);

    SAIGA_CORE_API friend std::ostream& operator<<(std::ostream& os, const Image& f);
};



}  // namespace Saiga
