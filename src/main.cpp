#include <iostream>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/main.h"

#ifndef NDEBUG
#  define D(x) x
#  include <random>
#  include <chrono>
#else
# define D(x)
#endif


using namespace std;


int main(int argc, char **argv)
{
    std::chrono::high_resolution_clock::time_point program_start = std::chrono::high_resolution_clock::now();
    if (argc != 3)
    {
        cerr << "required arguments: <path to settings> <path to image>" << endl;
    }
    string imgPath = string(argv[2]);
    cv::Mat image;

    image = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        cerr << "Failed to load image at" << imgPath << "!" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "\nImage loaded successfully!\n" << endl;

    string settingsPath = string(argv[1]);
    cv::FileStorage settingsFile(settingsPath, cv::FileStorage::READ);
    if (!settingsFile.isOpened())
    {
        cerr << "Failed to load ORB settings at" << settingsPath << "!" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "ORB Settings loaded successfully!\n" << endl;


    int nFeatures = settingsFile["ORBextractor.nFeatures"];
    float scaleFactor = settingsFile["ORBextractor.scaleFactor"];
    int nLevels = settingsFile["ORBextractor.nLevels"];
    int FASTThresholdInit = settingsFile["ORBextractor.iniThFAST"];
    int FASTThresholdMin = settingsFile["ORBextractor.minThFAST"];

    cv::Scalar color = cv::Scalar(settingsFile["Color.r"], settingsFile["Color.g"], settingsFile["Color.b"]);
    int thickness = settingsFile["Line.thickness"];
    int radius = settingsFile["Circle.radius"];
    int drawAngular = settingsFile["drawAngular"];


    ORB_SLAM2::ORBextractor extractor (nFeatures, scaleFactor, nLevels,
                                       FASTThresholdInit, FASTThresholdMin);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cv::Mat imgColor;
    image.copyTo(imgColor);

    cv::cvtColor(imgColor, image, CV_BGR2GRAY);

    extractor(image, cv::Mat(), keypoints, descriptors);

    //MeasureExecutionTime(1, extractor, image);

    //extractor.testingFAST(image, keypoints, false, true);
    //extractor.testingFAST(image, keypoints, true, true);



    //DisplayKeypoints(imgColor, keypoints, color, thickness, radius, drawAngular);



    //LoadHugeImage(extractor);

    std::chrono::high_resolution_clock::time_point program_end = std::chrono::high_resolution_clock::now();
    auto program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();

    std::cout << "\nProgram duration: " << program_duration << " microseconds.\n";

    return 0;
}

void LoadHugeImage(ORB_SLAM2::ORBextractor &extractor)
{
    string path = string("/home/ralph/Downloads/world.topo.bathy.200407.3x21600x10800.png");
    cv::Mat img;
    img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty())
    {
        cerr << "Failed to load image at" << path << "!" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "\nImage loaded successfully!\n" << endl;
    cout << "Running cv::FAST repeatedly with " << img.cols << "x" << img.rows << " image...\n";
    vector<cv::KeyPoint> kpts;

    extractor.testingFAST(img, kpts, false, false);
}

void DisplayKeypoints(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar &color,
                      int thickness, int radius, int drawAngular)
{
    cv::namedWindow("Keypoints", cv::WINDOW_AUTOSIZE);
    cv::imshow("Keypoints", image);
    cv::waitKey(0);

    for (const cv::KeyPoint &k : keypoints)
    {
        cv::Point2f point = k.pt;
        cv::circle(image, point, radius, color, 1, CV_AA);
        //cv::rectangle(image, cv::Point2f(point.x-1, point.y-1),
        //              cv::Point2f(point.x+1, point.y+1), color, thickness, CV_AA);
        cv::circle(image, point, 2, color, 1, CV_AA);
        if (drawAngular)
        {
            int len = radius;
            float angleRad =  k.angle * CV_PI / 180.f;
            float cos = std::cos(angleRad);
            float sin = std::sin(angleRad);
            int x = (int)round(point.x + len * cos);
            int y = (int)round(point.y + len * sin);
            cv::Point2f target = cv::Point2f(x, y);
            cv::line(image, point, target, color, thickness, CV_AA);
        }
    }
    cv::imshow("Keypoints", image);
    cv::waitKey(0);
}

//TODO: remove from master

void MeasureExecutionTime(int numIterations, ORB_SLAM2::ORBextractor &extractor, cv::Mat &img)
{
    using namespace std::chrono;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    high_resolution_clock::time_point cvStart = high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i)
    {
        //tests args: cv::InputArray inputImage, std::vector<cv::KeyPoint> &resKeypoints,
        //        cv::OutputArray outputDescriptors, bool myFAST, bool myDesc
        //extractor.Tests(img, kpts, desc, false, true);

        extractor.testingFAST(img, kpts, false, false);
    }
    high_resolution_clock::time_point cvEnd = high_resolution_clock::now();

    high_resolution_clock::time_point myStart = high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i)
    {
        //extractor.Tests(img, kpts, desc, true, true);

        extractor.testingFAST(img, kpts, true, false);
    }
    high_resolution_clock::time_point myEnd = high_resolution_clock::now();

    auto cvDuration = duration_cast<microseconds>(cvEnd - cvStart).count();
    auto myDuration = duration_cast<microseconds>(myEnd - myStart).count();

    std::cout << "\nExecution time of " << numIterations << " iterations with openCV: " << cvDuration << " microseconds.\n";
    std::cout << "\nExecution time of " << numIterations << " iterations with my impl: " << myDuration << " microseconds.\n";
}


void AddRandomKeypoints(std::vector<cv::KeyPoint> &keypoints)
{
    int nKeypoints = 150;
    keypoints.clear();
    keypoints.reserve(nKeypoints);
    for (int i =0; i < nKeypoints; ++i)
    {
        auto x = static_cast<float>(20 + (rand() % static_cast<int>(620 - 20 + 1)));
        auto y = static_cast<float>(20 + (rand() % static_cast<int>(460 - 20 + 1)));
        auto angle = static_cast<float>(0 + (rand() % static_cast<int>(359 - 0 + 1)));
        keypoints.emplace_back(cv::KeyPoint(x, y, 7.f, angle, 0));
    }
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{

    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}