#ifndef ORBEXTRACTOR_FEATUREFILEINTERFACE_H
#define ORBEXTRACTOR_FEATUREFILEINTERFACE_H

#include <string>
#include <opencv2/core/mat.hpp>
#include "include/Types.h"
#include <sys/stat.h>

class FeatureFileInterface
{
public:
    explicit FeatureFileInterface(std::string &_path) : path(_path), counter(0) {}
    FeatureFileInterface() : path(), counter(0) {}

    bool SaveFeatures(std::vector<knuff::KeyPoint> &kpts);

    std::vector<knuff::KeyPoint> LoadFeatures(std::string &path);

    bool SaveDescriptors(cv::Mat &descriptors);

    cv::Mat LoadDescriptors(std::string &path);

    std::string GetFilenameFromPath(std::string &path);

    bool CheckExistence(std::string &path);

    void SetPath(std::string &_path)
    {
        path = _path;
        struct stat buf{};
        bool dex = (stat(path.c_str(), &buf) == 0);
        if (!dex)
        {
            mkdir(path.c_str(), S_IRWXU);
            std::string fpath = path + "features/";
            std::string dpath = path + "descriptors/";
            mkdir(fpath.c_str(), S_IRWXU);
            mkdir(dpath.c_str(), S_IRWXU);
        }
    }

private:
    std::string path;
    int counter;
};

#endif //ORBEXTRACTOR_FEATUREFILEINTERFACE_H
