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

    if (argc != 4)
    {
        cerr << "required arguments: <path to settings> <path to image / sequence> "
                "<mode: 0-> single image / 1-> image sequence>" << endl;
    }

    string settingsPath = string(argv[1]);
    cv::FileStorage settingsFile(settingsPath, cv::FileStorage::READ);
    if (!settingsFile.isOpened())
    {
        cerr << "Failed to load ORB settings at" << settingsPath << "!" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "\nORB Settings loaded successfully!\n" << endl;


    int nFeatures = settingsFile["ORBextractor.nFeatures"];
    float scaleFactor = settingsFile["ORBextractor.scaleFactor"];
    int nLevels = settingsFile["ORBextractor.nLevels"];
    int FASTThresholdInit = settingsFile["ORBextractor.iniThFAST"];
    int FASTThresholdMin = settingsFile["ORBextractor.minThFAST"];

    cv::Scalar color = cv::Scalar(settingsFile["Color.r"], settingsFile["Color.g"], settingsFile["Color.b"]);
    int thickness = settingsFile["Line.thickness"];
    int radius = settingsFile["Circle.radius"];
    int drawAngular = settingsFile["drawAngular"];

    string imgPath = string(argv[2]);

    string m = argv[3];
    int mode = std::stoi(m);

    if (mode == 0)
    {
        SingleImageMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin,
                        color, thickness, radius, drawAngular);
    }
    else if (mode == 1)
    {
        SequenceMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin,
                     color, thickness, radius, drawAngular);
    }


   std::chrono::high_resolution_clock::time_point program_end = std::chrono::high_resolution_clock::now();
   auto program_duration = std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();

   /*
   ofstream log;
   log.open("/home/ralph/Documents/ComparisonLog.txt", ios::app);
   if (!log.is_open())
       cerr << "\nFailed to open log file...\n";
   log << "Program duration (mine): " << program_duration << " microseconds.\n";
    */
   std::cout << "\nProgram duration: " << program_duration << " microseconds.\n" <<
   "(equals ~" <<  (float)program_duration / 1000000.f << " seconds)\n";

   return 0;
}


void SingleImageMode(string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                        int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular)
{
    cout << "\nStarting in single image mode...\n";
    cv::Mat image;

    image = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        cerr << "Failed to load image at" << imgPath << "!" << endl;
        exit(EXIT_FAILURE);
    }
    cout << "\nImage loaded successfully!\n" << endl;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::vector<cv::KeyPoint> refkeypoints;
    cv::Mat refdescriptors;

    std::vector<cv::KeyPoint> keypointsAll;

    ORB_SLAM2::ORBextractor extractor (nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    DistributionComparisonSuite(extractor, image, color, thickness, radius, drawAngular, false);
    return;

    ORB_SLAM_REF::referenceORB refExtractor (nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    cv::Mat imgColor;
    cv::Mat imgColor2;
    image.copyTo(imgColor);
    image.copyTo(imgColor2);

    if (image.channels() == 3)
        cv::cvtColor(imgColor, image, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(imgColor, image, CV_BGRA2GRAY);

    ///CHANGE FUNCTION CALLS FOR TESTS HERE ///////////////////////////////////////////////////////////////////////


    //LoadHugeImage(refExtractor);
    extractor(image, cv::Mat(), keypoints, descriptors, Distribution::SSC, false);
    extractor(image, cv::Mat(), keypointsAll, descriptors, Distribution::KEEP_ALL, true);

    std::cout << "\nnkpts: " << keypoints.size() << "\n";

    //refExtractor(image, cv::Mat(), refkeypoints, refdescriptors);

    /*
    vector<int> mysizes(2000, 0);
    vector<int> refsizes(2000, 0);
    for (int i = 0; i < 1000; ++i)
    {
        keypoints.clear();
        refkeypoints.clear();
        extractor(image, cv::Mat(), keypoints, descriptors, DISTRIBUTION_QUADTREE_ORBSLAMSTYLE);
        refExtractor(image, cv::Mat(), refkeypoints, refdescriptors);
        ++mysizes[keypoints.size()];
        ++refsizes[refkeypoints.size()];
    }
    for (int i = 900; i < 1100; ++i)
    {
        std::cout << "Number of times my keypoint vec was size " << i << ": " << mysizes[i] << "\n";
        std::cout << "Number of times ref keypoint vec was size " << i << ": " << refsizes[i] << "\n";
    }
     */

    //DistributionComparisonSuite(extractor, imgColor, color, thickness, radius, drawAngular);


    //extractor.testingDescriptors(cv::Mat(), cv::Mat(), keypoints.size(), false, 0, 0, true, cv::Mat());

    //extractor.Tests(image, keypoints, descriptors, true, true);

    //extractor.testingFAST(image, keypoints, false, true);
    //extractor.testingFAST(image, keypoints, true, true);

    //MeasureExecutionTime(1, extractor, image, FAST_RUNTIME);


    //DisplayKeypoints(imgColor2, refkeypoints, color, thickness, radius, drawAngular, "reference");

    //DrawCellGrid(imgColor, 16, imgColor.cols-16, 16, imgColor.rows-16, 80);
    DisplayKeypoints(imgColor2, keypointsAll, color, thickness, radius, drawAngular, "all");
    DisplayKeypoints(imgColor, keypoints, color, thickness, radius, drawAngular, "kdtreeANMS");

    //CompareKeypoints(keypoints, "mine", refkeypoints, "reference", -1, true);
    //CompareDescriptors(descriptors, "mine", refdescriptors, "reference", keypoints.size(), -1, true);


    //LoadHugeImage(extractor);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}


void SequenceMode(string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                    int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular)
{
    cout << "\nStarting in sequence mode...\n";

    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(imgPath)+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    ORB_SLAM2::ORBextractor myExtractor(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    ORB_SLAM_REF::referenceORB refExtractor(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    cout << "\n/////////////////////////\n"
           << "Images in sequence: " << nImages << "\n";

    bool eqkpts = true;
    bool eqdescriptors = true;

    long myTotalDuration = 0;
    long refTotalDuration = 0;

    cv::Mat img;
    for(int ni=0; ni<nImages; ni++)
    {
        //cout << "\nNow processing image nr. " << ni << "...\n";
        // Read image from file
        img = cv::imread(string(imgPath) + "/" + vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (img.empty())
        {
            cerr << endl << "Failed to load image at: "
            << string(imgPath) << "/" << vstrImageFilenames[ni] << endl;
            exit(EXIT_FAILURE);
        }

        cv::cvtColor(img, img, CV_BGR2GRAY);

        vector<cv::KeyPoint> mykpts;
        cv::Mat mydescriptors;

        vector<cv::KeyPoint> refkpts;
        cv::Mat refdescriptors;



        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        refExtractor(img, cv::Mat(), refkpts, refdescriptors);


        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();


        myExtractor(img, cv::Mat(), mykpts, mydescriptors, Distribution::QUADTREE);
        chrono::high_resolution_clock ::time_point t3 = chrono::high_resolution_clock::now();

        auto refduration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        auto myduration = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();

        myTotalDuration += myduration;
        refTotalDuration += refduration;

        /** compare kpts and descriptors per image:
        vector<std::pair<cv::KeyPoint, cv::KeyPoint>> kptDiffs;
        kptDiffs = CompareKeypoints(mykpts, string("my kpts"), refkpts, string("reference kpts"), ni, true);

        if (!kptDiffs.empty())
            eqkpts = false;

        int nkpts = mykpts.size();

        vector<Descriptor_Pair> descriptorDiffs;
        descriptorDiffs = CompareDescriptors(mydescriptors, "my descriptors", refdescriptors,
                                                 "reference descriptors", nkpts, ni, true);
        if (!descriptorDiffs.empty())
            eqdescriptors = false;
        */


        /** time measurement per image:
         *
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

        refExtractor(img, cv::Mat(), kpts, descriptors);

        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();

        myExtractor(img, cv::Mat(), kpts, descriptors);

        chrono::high_resolution_clock ::time_point t3 = chrono::high_resolution_clock::now();

        auto refduration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        auto myduration = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();


        std::cout << "Computation time with my ORB for image nr. " << ni << ": " << myduration << " microseconds.\n";
        std::cout << "Computation time with ref ORB for image nr. " << ni << ": " << refduration << " microseconds.\n";
         */

    }
    cout << "\n" << (eqkpts ? "All keypoints across all images were equal!\n" : "Not all keypoints are equal...:(\n");
    cout << "\n" << (eqdescriptors ? "All descriptors across all images and keypoints were equal!\n" :
                        "Not all descriptors were equal... :(\n");

    cout << "\nTotal computation time using my orb: " << myTotalDuration/1000 <<
        " milliseconds, which averages to ~" << myTotalDuration/nImages << " microseconds.\n";
    cout << "\nTotal computation time using ref orb: " << refTotalDuration/1000 <<
        " milliseconds, which averages to ~" << refTotalDuration/nImages << " microseconds.\n";
}


struct CompareXThenY
{
    bool operator()(cv::KeyPoint &k1, cv::KeyPoint &k2) const
    {
        return (k1.pt.x < k2.pt.x || (k1.pt.x == k2.pt.x && k1.pt.y < k2.pt.y));
    }

};
void SortKeypoints(vector<cv::KeyPoint> &kpts)
{
    std::sort(kpts.begin(), kpts.end(), CompareXThenY());
}


vector<std::pair<cv::KeyPoint, cv::KeyPoint>> CompareKeypoints(vector<cv::KeyPoint> &kpts1, string name1,
        vector<cv::KeyPoint> &kpts2, string name2, int imgNr, bool print)
{
    //SortKeypoints(kpts1);
    //SortKeypoints(kpts2);
    int sz1 = kpts1.size();
    int sz2 = kpts2.size();
    if (print)
        cout << "\nSize of vector " << name1 << ": " << sz1 << ", size of vector " << name2 << ": "  << sz2 << "\n";

    int N;
    if (sz1 > sz2)
        N = sz2;
    else
        N = sz1;

    vector<std::pair<cv::KeyPoint, cv::KeyPoint>> differences;
    differences.reserve(N);

    bool eq = true;
    int count = 0;
    for (int i = 0; i < N; ++i) //TODO: revert to i = 0; i < N
    {
        if (!(kpts1[i].pt.x == kpts2[i].pt.x && kpts1[i].pt.y == kpts2[i].pt.y &&
            kpts1[i].octave == kpts2[i].octave && kpts1[i].response == kpts2[i].response &&
            kpts1[i].size == kpts2[i].size && kpts1[i].angle == kpts2[i].angle))
        {
            eq = false;
            ++count;
            differences.emplace_back(make_pair(kpts1[i], kpts2[i]));
            if (print)
            {
                cout << "\ndiffering keypoint at idx " << i << " in image " <<
                        (imgNr == -1? "" : std::to_string(imgNr)) << "\n";
                cout << "kpt1: " << kpts1[i].pt << ", kpt2: " << kpts2[i].pt << "\n";
                cout << "kpt1.angle=" << kpts1[i].angle << ", kpt2.angle=" << kpts2[i].angle << "\n" <<
                    "kpt1.octave=" << kpts1[i].octave << ", kpt2.octave=" << kpts2[i].octave << "\n" <<
                    "kpt1.response=" << kpts1[i].response << ", kpt2.response=" << kpts2[i].response << "\n" <<
                    "kpt1.size=" << kpts1[i].size << ", kpt2.size=" << kpts2[i].size << "\n";
            }
        }
    }
    if (print && eq)
        cout << "\nKeypoints from image " << (imgNr == -1? "" : std::to_string(imgNr)) << " are equal.\n";

    return differences;
}


vector<Descriptor_Pair> CompareDescriptors (cv::Mat &desc1, string name1, cv::Mat &desc2, string name2,
                                                int nkpts, int imgNr, bool print)
{
    vector<Descriptor_Pair> differences;

    uchar* ptr1 = &desc1.at<uchar>(0);
    uchar* ptr2 = &desc2.at<uchar>(0);

    assert(desc1.size == desc2.size);

    bool eq = true;

    int N = nkpts * 32;
    for (int i = 0; i < N; ++i)
    {
        if ((int)ptr1[i] != (int)ptr2[i])
        {
            eq = false;
            Descriptor_Pair d;
            d.byte1 = (int)ptr1[i];
            d.byte2 = (int)ptr2[i];
            d.index = i;
            differences.emplace_back(d);
        }
    }

    if (print)
    {
        cout << "\nDescriptors of kpts of image " << (imgNr == -1? "" : std::to_string(imgNr)) <<
        (eq? " are equal.\n" : " are not equal\n");
    }

    return differences;
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

    if (img.channels() == 3)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, img, CV_BGRA2GRAY);

   extractor(img, cv::Mat(), kpts, cv::Mat());
}

/**
 * @overload
 */
void LoadHugeImage(ORB_SLAM_REF::referenceORB &extractor)
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
    vector<cv::KeyPoint> kpts;

    if (img.channels() == 3)
        cv::cvtColor(img, img, CV_BGR2GRAY);
    else if (img.channels() == 4)
        cv::cvtColor(img, img, CV_BGRA2GRAY);


    extractor(img, cv::Mat(), kpts, cv::Mat());
}

void DisplayKeypoints(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Scalar &color,
                     int thickness, int radius, int drawAngular, string windowname)
{
   cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
   cv::imshow(windowname, image);
   //cv::waitKey(0);

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
   cv::imshow(windowname, image);
   cv::waitKey(0);
}

void DrawCellGrid(cv::Mat &image, int minX, int maxX, int minY, int maxY, int cellSize)
{
    const float width = maxX - minX;
    const float height = maxY - minY;

    int c = std::min(myRound(width), myRound(height));
    assert(cellSize < c && cellSize > 16);

    const int cellCols = width / cellSize;
    const int cellRows = height / cellSize;
    const int cellWidth = std::ceil(width / cellCols);
    const int cellHeight = std::ceil(height / cellRows);

    for (int y = 0; y <= cellRows; ++y)
    {
        cv::Point2f start(minX, minY + y*cellHeight);
        cv::Point2f end(maxX, minY + y*cellHeight);
        cv::line(image, start, end, cv::Scalar(100, 0, 255), 1, CV_AA);
    }
    for (int x = 0; x <= cellCols; ++x)
    {
        cv::Point2f start(minX + x*cellWidth, minY);
        cv::Point2f end(minX + x*cellWidth, maxY);
        cv::line(image, start, end, cv::Scalar(100, 0, 255), 1, CV_AA);
    }
}

void MeasureExecutionTime(int numIterations, ORB_SLAM2::ORBextractor &extractor, cv::Mat &img, MODE mode)
{
   using namespace std::chrono;
    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;

   if (mode == DESC_RUNTIME)
   {

   }
   else if (mode == FAST_RUNTIME)
   {
       int N = numIterations;

       std::chrono::high_resolution_clock::time_point tp_myfast = std::chrono::high_resolution_clock::now();
       for (int i = 0; i < N; ++i)
       {
           extractor.testingFAST(img, kpts, true, false);
       }

       std::chrono::high_resolution_clock::time_point tp_midpoint = std::chrono::high_resolution_clock::now();

       for (int i = 0; i < N; ++i)
       {
           extractor.testingFAST(img, kpts, false, false);
       }
       std::chrono::high_resolution_clock::time_point tp_cvfast = std::chrono::high_resolution_clock::now();

       auto myduration = std::chrono::duration_cast<std::chrono::microseconds>(tp_midpoint - tp_myfast).count();
       auto cvduration = std::chrono::duration_cast<std::chrono::microseconds>(tp_cvfast - tp_midpoint).count();

       std::cout << "\nduration of " << N << " iteration of myfast: " << myduration << " microseconds\n";
       std::cout << "\nduration of " << N << " iterations of cvfast: " << cvduration << " microseconds\n";
   }

}

void DistributionComparisonSuite(ORB_SLAM2::ORBextractor &extractor, cv::Mat &imgColor, cv::Scalar &color,
        int thickness, int radius, bool drawAngular, bool distributePerLevel)
{
    typedef std::chrono::high_resolution_clock clk;
    cv::Mat imgGray;
    imgColor.copyTo(imgGray);


    std::vector<cv::KeyPoint> kptsAll;
    std::vector<cv::KeyPoint> kptsNaive;
    std::vector<cv::KeyPoint> kptsQuadtree;
    std::vector<cv::KeyPoint> kptsQuadtreeORBSLAMSTYLE;
    std::vector<cv::KeyPoint> kptsGrid;
    std::vector<cv::KeyPoint> kptsANMS_KDTree;
    std::vector<cv::KeyPoint> kptsANMS_RT;
    std::vector<cv::KeyPoint> kptsSSC;

    cv::Mat descriptors;

    if (imgGray.channels() == 3)
        cv::cvtColor(imgGray, imgGray, CV_BGR2GRAY);
    else if (imgGray.channels() == 4)
        cv::cvtColor(imgGray, imgGray, CV_BGRA2GRAY);

    cv::Mat imgAll;
    cv::Mat imgNaive;
    cv::Mat imgQuadtree;
    cv::Mat imgQuadtreeORBSLAMSTYLE;
    cv::Mat imgGrid;
    cv::Mat imgANMS_KDTree;
    cv::Mat imgANMS_RT;
    cv::Mat imgSSC;

    imgColor.copyTo(imgAll);
    imgColor.copyTo(imgNaive);
    imgColor.copyTo(imgQuadtree);
    imgColor.copyTo(imgQuadtreeORBSLAMSTYLE);
    imgColor.copyTo(imgGrid);
    imgColor.copyTo(imgANMS_KDTree);
    imgColor.copyTo(imgANMS_RT);
    imgColor.copyTo(imgSSC);

    clk::time_point tStart = clk::now();
    clk::time_point t1;
    clk::time_point t2;
    clk::time_point t3;
    clk::time_point t4;
    clk::time_point t5;
    clk::time_point t6;
    clk::time_point t7;
    clk::time_point tEnd;

    if (distributePerLevel)
    {
        extractor(imgGray, cv::Mat(), kptsAll, descriptors, Distribution::KEEP_ALL);
        t1 = clk::now();
        extractor(imgGray, cv::Mat(), kptsNaive, descriptors, Distribution::NAIVE);
        t2 = clk::now();
        extractor(imgGray, cv::Mat(), kptsQuadtree, descriptors, Distribution::QUADTREE);
        t3 = clk::now();
        extractor(imgGray, cv::Mat(), kptsQuadtreeORBSLAMSTYLE, descriptors, Distribution::QUADTREE_ORBSLAMSTYLE);
        t4 = clk::now();
        extractor(imgGray, cv::Mat(), kptsGrid, descriptors, Distribution::GRID);
        t5 = clk::now();
        extractor(imgGray, cv::Mat(), kptsANMS_KDTree, descriptors, Distribution::ANMS_KDTREE);
        t6 = clk::now();
        extractor(imgGray, cv::Mat(), kptsANMS_RT, descriptors, Distribution::ANMS_RT);
        t7 = clk::now();
        extractor(imgGray, cv::Mat(), kptsSSC, descriptors, Distribution::SSC);
    }
    else
    {
        extractor(imgGray, cv::Mat(), kptsAll, descriptors, Distribution::KEEP_ALL, false);
        t1 = clk::now();
        extractor(imgGray, cv::Mat(), kptsNaive, descriptors, Distribution::NAIVE, false);
        t2 = clk::now();
        extractor(imgGray, cv::Mat(), kptsQuadtree, descriptors, Distribution::QUADTREE, false);
        t3 = clk::now();
        extractor(imgGray, cv::Mat(), kptsQuadtreeORBSLAMSTYLE, descriptors,
                Distribution::QUADTREE_ORBSLAMSTYLE,false);
        t4 = clk::now();
        extractor(imgGray, cv::Mat(), kptsGrid, descriptors, Distribution::GRID, false);
        t5 = clk::now();
        extractor(imgGray, cv::Mat(), kptsANMS_KDTree, descriptors, Distribution::ANMS_KDTREE, false);
        t6 = clk::now();
        extractor(imgGray, cv::Mat(), kptsANMS_RT, descriptors, Distribution::ANMS_RT, false);
        t7 = clk::now();
        extractor(imgGray, cv::Mat(), kptsSSC, descriptors, Distribution::SSC, false);
    }
    tEnd = clk::now();

    auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t1-tStart).count();
    auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    auto d3 = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();
    auto d4 = std::chrono::duration_cast<std::chrono::microseconds>(t4-t3).count();
    auto d5 = std::chrono::duration_cast<std::chrono::microseconds>(t5-t4).count();
    auto d6 = std::chrono::duration_cast<std::chrono::microseconds>(t6-t5).count();
    auto d7 = std::chrono::duration_cast<std::chrono::microseconds>(t7-t6).count();
    auto d8 = std::chrono::duration_cast<std::chrono::microseconds>(tEnd-t7).count();

    cout << "\nComplete computation time for each distribution:"
            "\nAll Keypoints kept: " << d1 << " microseconds" <<
            "\nTopN: " << d2 << " microseconds" <<
            "\nQuadtree: " << d3 << " microseconds" <<
            "\nQuadtre ORBSLAMSTYLE: " << d4 << " microseconds" <<
            "\nBucketing: " << d5 << " microseconds" <<
            "\nANMS (KDTree): " << d6 << " microseconds" <<
            "\nANMS (Range Tree): " << d7 << " microseconds" <<
            "\nSuppression via Square Covering: " << d8 << " microseconds\n";

    DisplayKeypoints(imgAll, kptsAll, color, thickness, radius, drawAngular, "all");
    DisplayKeypoints(imgNaive, kptsNaive, color, thickness, radius, drawAngular, "naive");
    DisplayKeypoints(imgQuadtree, kptsQuadtree, color, thickness, radius, drawAngular, "quadtree");
    DisplayKeypoints(imgQuadtreeORBSLAMSTYLE, kptsQuadtreeORBSLAMSTYLE, color, thickness, radius, drawAngular,
            "quadtree ORBSLAM");
    DisplayKeypoints(imgGrid, kptsGrid, color, thickness, radius, drawAngular, "Grid");
    DisplayKeypoints(imgANMS_KDTree, kptsANMS_KDTree, color, thickness, radius, drawAngular, "KDTree ANMS");
    DisplayKeypoints(imgANMS_RT, kptsANMS_RT, color, thickness, radius, drawAngular, "Range Tree ANMS");
    DisplayKeypoints(imgSSC, kptsSSC, color, thickness, radius, drawAngular, "SSC");
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