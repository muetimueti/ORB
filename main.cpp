#include <iostream>
#include "main.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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


    ORB_SLAM2::ORBextractor* extractor = new ORB_SLAM2::ORBextractor(nFeatures, scaleFactor, nLevels,
        FASTThresholdInit, FASTThresholdMin);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;


    (*extractor)(image, cv::Mat(), keypoints, descriptors);



    D(AddRandomKeypoints(keypoints));

    DisplayKeypoints(image, keypoints, color, thickness, radius, drawAngular);


    return 0;
}

void DisplayKeypoints(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar &color,
        int thickness, int radius, int drawAngular)
{
    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    cv::imshow("test", image);
    cv::waitKey(0);

    for (int i = 0; i < keypoints.size(); ++i)
    {
        cv::Point2f point = keypoints[i].pt;
        cv::circle(image, point, radius, color, 1, CV_AA);
        //cv::rectangle(image, cv::Point2f(point.x-1, point.y-1),
        //              cv::Point2f(point.x+1, point.y+1), color, thickness, CV_AA);
        cv::circle(image, point, 2, color, 1, CV_AA);
        if (drawAngular)
        {
            int len = radius;
            float angleRad =  keypoints[i].angle * 3.141592653589793238 / 180;
            float cos = std::cos(angleRad);
            float sin = std::sin(angleRad);
            int x = (int)round(point.x + len * cos);
            int y = (int)round(point.y + len * sin);
            cv::Point2f target = cv::Point2f(x, y);
            cv::line(image, point, target, color, thickness, CV_AA);
        }
    }
    cv::imshow("test", image);
    cv::waitKey(0);
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