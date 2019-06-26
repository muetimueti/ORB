#include "include/FeatureFileInterface.h"
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <assert.h>
#include <sstream>

#define message_assert(expr, msg) assert(( (void)(msg), (expr) ))

using namespace std;

/** fileformat:
 * x y size angle response octave
 * 1 kpt per line
 */

bool FeatureFileInterface::SaveFeatures(vector<knuff::KeyPoint> &kpts)
{
    message_assert("Save path must be set", !path.empty());
    cout << "Saving Features for image to " << path << "...\n";

    string filename = path + "features/" + to_string(counter++) + ".orbf";

    fstream file;
    file.open(filename, ios::out);
    if (!file.is_open())
    {
        cerr << "Failed to open " << filename << "...\n";
        return false;
    }


    for (auto &kpt : kpts)
    {
        file << kpt.pt.x << " " << kpt.pt.y << " " << kpt.size << " " << kpt.angle << " " << kpt.response << " " <<
                kpt.octave << "\n";
    }
    file.close();
    return true;
}


vector<knuff::KeyPoint> FeatureFileInterface::LoadFeatures(std::string &path)
{
    vector<knuff::KeyPoint> kpts;

    fstream file;
    file.open(path, ios::in);
    if (!file.is_open())
    {
        cerr << "Failed to open " << path << "...\n";
        return kpts;
    }

    string line;
    stringstream ss;
    while (getline(file, line))
    {
        if (!line.empty())
        {

        }
    }
}


bool FeatureFileInterface::SaveDescriptors(cv::Mat &descriptors)
{
    message_assert("Save path must be set", !path.empty());
    cout << "\nSaving descriptors for image to " << path << "...\n";

    string filename = "descriptors/" + to_string(counter) + ".orb";

    fstream file;
    file.open(path, ios::out);
    if (!file.is_open())
    {
        cerr << "Failed to open " << path << "...\n";
        return false;
    }

    //TODO

    return true;
}


cv::Mat FeatureFileInterface::LoadDescriptors(string &path)
{

}


string FeatureFileInterface::GetFilenameFromPath(string &path)
{
    string name;
    string delim = "/";
    name = string(path);
    int n = name.rfind(delim, name.length()-2);
    return name.substr(n+1, name.length()-n-2);
}

bool FeatureFileInterface::CheckExistence(string &path)
{
    struct stat buf{};
    return (stat(path.c_str(), &buf) == 0);
}

#undef message_assert