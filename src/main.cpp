#include <iostream>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/main.h"
#include <pangolin/pangolin.h>

#ifndef NDEBUG
#  define D(x) x
#  include <random>
#  include <chrono>
#include <opencv2/features2d.hpp>

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
    else
    {
        PerformanceMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);
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
   pangolin::QuitAll();
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

    //DistributionComparisonSuite(extractor, image, color, thickness, radius, drawAngular, false);
    //return;

    //ORB_SLAM_REF::referenceORB refExtractor (nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    cv::Mat imgColor;
    cv::Mat imgColor2;
    image.copyTo(imgColor);
    image.copyTo(imgColor2);

    if (image.channels() == 3)
        cv::cvtColor(imgColor, image, CV_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(imgColor, image, CV_BGRA2GRAY);


    bool distributePerLevel = false;

    pangolin::CreateWindowAndBind("Menu",210,520);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(210));

    pangolin::Var<bool> menuAll("menu.All Keypoints",false,false);
    pangolin::Var<bool> menuTopN("menu.TopN",false,false);
    pangolin::Var<bool> menuBucketing("menu.Bucketing",true,false);
    pangolin::Var<bool> menuQuadtreeORBSLAMSTYLE("menu.Quadtree",false,false);
    pangolin::Var<bool> menuANMS_KDT("menu.KDTree-ANMS",false,false);
    pangolin::Var<bool> menuANMS_RT("menu.Range-Tree-ANMS",false,false);
    pangolin::Var<bool> menuSSC("menu.SSC",false,false);
    pangolin::Var<bool> menuRANMS("menu.RANMS", false, false);
    pangolin::Var<bool> menuDistrPerLvl("menu.Distribute Per Level", false, true);
    pangolin::Var<int> menuNFeatures("menu.Desired Features", 1000, 1, 2000);
    pangolin::Var<int> menuActualkpts("menu.Features Actual", false, 0);
    pangolin::Var<int> menuSetInitThreshold("menu.Init FAST Threshold", FASTThresholdInit, 1, 40);
    pangolin::Var<int> menuSetMinThreshold("menu.Min FAST Threshold", FASTThresholdMin, 1, 40);
    pangolin::Var<std::string> menuText("menu.----- FAST-SCORE: -----");
    pangolin::Var<bool> menuScoreOpenCV("menu.OpenCV", true, false);
    pangolin::Var<bool> menuScoreHarris("menu.Harris", false, false);
    pangolin::Var<bool> menuScoreSum("menu.Sum", false, false);
    pangolin::Var<bool> menuScoreExp("menu.Experimental", false, false);
    pangolin::Var<bool> menuExit("menu.EXIT", false, false);


    pangolin::FinishFrame();

    cv::namedWindow(string(imgPath));
    cv::moveWindow(string(imgPath), 80, 260);


    while (true)
    {
        cv::Mat imgGray;
        imgColor.copyTo(imgGray);

        if (imgGray.channels() == 3)
            cv::cvtColor(imgGray, imgGray, CV_BGR2GRAY);
        else if (imgGray.channels() == 4)
            cv::cvtColor(imgGray, imgGray, CV_BGRA2GRAY);

        cv::Mat displayImg;
        imgColor.copyTo(displayImg);

        extractor(imgGray, cv::Mat(), keypoints, descriptors, distributePerLevel);

        if (extractor.GetDistribution() == Distribution::GRID && !distributePerLevel)
        {
            DrawCellGrid(displayImg, 0, displayImg.cols, 0, displayImg.rows, BUCKETING_GRID_SIZE);
        }
        DisplayKeypoints(displayImg, keypoints, color, thickness, radius, drawAngular, string(imgPath));

        cv::waitKey(33);

        int n = menuNFeatures;
        if (n != nFeatures)
        {
            nFeatures = n;
            extractor.SetnFeatures(n);
        }


        menuActualkpts = keypoints.size();
        keypoints.clear();
        if (menuAll)
        {
            extractor.SetDistribution(Distribution::KEEP_ALL);
            menuAll = false;
        }
        if (menuTopN)
        {
            extractor.SetDistribution(Distribution::NAIVE);
            menuTopN = false;
        }
        if (menuBucketing)
        {
            extractor.SetDistribution(Distribution::GRID);
            menuBucketing = false;
        }
        if (menuQuadtreeORBSLAMSTYLE)
        {
            extractor.SetDistribution(Distribution::QUADTREE_ORBSLAMSTYLE);
            menuQuadtreeORBSLAMSTYLE = false;
        }
        if (menuANMS_KDT)
        {
            extractor.SetDistribution(Distribution::ANMS_KDTREE);
            menuANMS_KDT = false;
        }
        if (menuANMS_RT)
        {
            extractor.SetDistribution(Distribution::ANMS_RT);
            menuANMS_RT = false;
        }
        if (menuSSC)
        {
            extractor.SetDistribution(Distribution::SSC);
            menuSSC = false;
        }
        if (menuRANMS)
        {
            extractor.SetDistribution(Distribution::RANMS);
            menuRANMS = false;
        }

        if (menuDistrPerLvl && !distributePerLevel)
            distributePerLevel = true;

        else if (!menuDistrPerLvl && distributePerLevel)
            distributePerLevel = false;

        if (menuSetInitThreshold != FASTThresholdInit || menuSetMinThreshold != FASTThresholdMin)
        {
            FASTThresholdInit = menuSetInitThreshold;
            if (menuSetMinThreshold > menuSetInitThreshold)
                menuSetMinThreshold = menuSetInitThreshold;
            FASTThresholdMin = menuSetMinThreshold;
            extractor.SetFASTThresholds(FASTThresholdInit, FASTThresholdMin);
        }

        if (menuScoreExp)
        {
            extractor.SetScoreType(FASTdetector::EXPERIMENTAL);
            menuScoreExp = false;
        }
        if (menuScoreHarris)
        {
            extractor.SetScoreType(FASTdetector::HARRIS);
            menuScoreHarris = false;
        }
        if (menuScoreOpenCV)
        {
            extractor.SetScoreType(FASTdetector::OPENCV);
            menuScoreOpenCV = false;
        }
        if (menuScoreSum)
        {
            extractor.SetScoreType(FASTdetector::SUM);
            menuScoreSum = false;
        }


        if (menuExit)
            return;

        pangolin::FinishFrame();
    }
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

    cout << "\n-------------------------\n"
           << "Images in sequence: " << nImages << "\n";

    bool eqkpts = true;
    bool eqdescriptors = true;

    long myTotalDuration = 0;
    long refTotalDuration = 0;

    cv::Mat img;

    pangolin::CreateWindowAndBind("Menu",210,680);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(210));

    pangolin::Var<bool> menuPause("menu. ~PAUSE~", false, true);
    pangolin::Var<bool> menuAll("menu.All Keypoints",false,false);
    pangolin::Var<bool> menuTopN("menu.TopN",false,false);
    pangolin::Var<bool> menuBucketing("menu.Bucketing",true,false);
    pangolin::Var<bool> menuQuadtreeORBSLAMSTYLE("menu.Quadtree",false,false);
    pangolin::Var<bool> menuANMS_KDT("menu.KDTree-ANMS",false,false);
    pangolin::Var<bool> menuANMS_RT("menu.Range-Tree-ANMS",false,false);
    pangolin::Var<bool> menuSSC("menu.SSC",false,false);
    pangolin::Var<bool> menuRANMS("menu.RANMS", false, false);
    pangolin::Var<bool> menuDistrPerLvl("menu.Distribute Per Level", false, true);
    pangolin::Var<int> menuNFeatures("menu.Desired Features", 1000, 1, 2000);
    pangolin::Var<int> menuActualkpts("menu.Features Actual", false, 0);
    pangolin::Var<int> menuSetInitThreshold("menu.Init FAST Threshold", FASTThresholdInit, 5, 40);
    pangolin::Var<int> menuSetMinThreshold("menu.Min FAST Threshold", FASTThresholdMin, 1, 39);
    pangolin::Var<std::string> menuText("menu.----- FAST-SCORE: -----");
    pangolin::Var<bool> menuScoreOpenCV("menu.OpenCV", true, false);
    pangolin::Var<bool> menuScoreHarris("menu.Harris", false, false);
    pangolin::Var<bool> menuScoreSum("menu.Sum", false, false);
    pangolin::Var<bool> menuScoreExp("menu.Experimental", false, false);
    pangolin::Var<float> menuScaleFactor("menu.Scale Factor", scaleFactor, 1.001, 1.2);
    pangolin::Var<int> menuNLevels("menu.nLevels", nLevels, 2, 8);
    pangolin::Var<bool> menuSingleLvlOnly("menu.Dispay Keypoints of single level:", false, true);
    pangolin::Var<int> menuChosenLvl("menu.Limit to Level", 0, 0, myExtractor.GetLevels()-1);
    pangolin::Var<int> menuMeanProcessingTime("menu.Mean Processing Time", 0);
    pangolin::Var<int> menuLastFrametime("menu.Last Frame", 0);

    pangolin::FinishFrame();

    cv::namedWindow(string(imgPath));
    cv::moveWindow(string(imgPath), 240, 260);
    string imgTrackbar = string("image nr");

    int nn = 0;
    cv::createTrackbar(imgTrackbar, string(imgPath), &nn, nImages);
    /** Trackbar call if opencv was compiled without Qt support:
    //cv::createTrackbar(imgTrackbar, string(imgPath), nullptr, nImages);
     */

    //cv::createButton("btn", nullptr, nullptr, cv::QT_PUSH_BUTTON, false);

    int count = 0;
    bool distributePerLevel = false;
    int soloLvl = -1;

    for(int ni=0; ni<nImages; ni++)
    {
        cv::setTrackbarPos("image nr", string(imgPath), ni);
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

        cv::Mat imgGray;
        cv::cvtColor(img, imgGray, CV_BGR2GRAY);

        vector<cv::KeyPoint> mykpts;
        cv::Mat mydescriptors;

        vector<cv::KeyPoint> refkpts;
        cv::Mat refdescriptors;



        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        //refExtractor(img, cv::Mat(), refkpts, refdescriptors);

        //std::cout << "\ncurrent img: " << string(imgPath) + vstrImageFilenames[ni];

        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();


        myExtractor(imgGray, cv::Mat(), mykpts, mydescriptors, distributePerLevel);
        chrono::high_resolution_clock ::time_point t3 = chrono::high_resolution_clock::now();

        auto refduration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        auto myduration = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();

        ++count;

        myTotalDuration += myduration;
        refTotalDuration += refduration;

        pangolin::FinishFrame();

        if (myExtractor.GetDistribution() == Distribution::GRID && !distributePerLevel)
        {
            DrawCellGrid(img, 0, img.cols, 0, img.rows, BUCKETING_GRID_SIZE);
        }
        DisplayKeypoints(img, mykpts, color, thickness, radius, drawAngular, string(imgPath));
        cv::waitKey(1000/30);

        //gui stuff
        if (cv::getTrackbarPos(imgTrackbar, string(imgPath)) != ni)
            ni = cv::getTrackbarPos(imgTrackbar, string(imgPath));

        int n = menuNFeatures;
        if (n != nFeatures)
        {
            nFeatures = n;
            myExtractor.SetnFeatures(n);
        }

        float scaleF = menuScaleFactor;
        if (scaleF != myExtractor.GetScaleFactor())
        {
            myExtractor.SetScaleFactor(scaleF);
        }

        int nlvl = menuNLevels;
        if (nlvl != myExtractor.GetLevels())
        {
            myExtractor.SetnLevels(nlvl);
        }

        menuLastFrametime = myduration/1000;
        menuMeanProcessingTime = myTotalDuration/1000 / count;

        menuActualkpts = mykpts.size();

        if (menuAll)
        {
            myExtractor.SetDistribution(Distribution::KEEP_ALL);
            myTotalDuration = 0;
            count = 0;
            menuAll = false;
        }
        if (menuTopN)
        {
            myExtractor.SetDistribution(Distribution::NAIVE);
            myTotalDuration = 0;
            count = 0;
            menuTopN = false;
        }
        if (menuBucketing)
        {
            myExtractor.SetDistribution(Distribution::GRID);
            myTotalDuration = 0;
            count = 0;
            menuBucketing = false;
        }
        if (menuQuadtreeORBSLAMSTYLE)
        {
            myExtractor.SetDistribution(Distribution::QUADTREE_ORBSLAMSTYLE);
            myTotalDuration = 0;
            count = 0;
            menuQuadtreeORBSLAMSTYLE = false;
        }
        if (menuANMS_KDT)
        {
            myExtractor.SetDistribution(Distribution::ANMS_KDTREE);
            myTotalDuration = 0;
            count = 0;
            menuANMS_KDT = false;
        }
        if (menuANMS_RT)
        {
            myExtractor.SetDistribution(Distribution::ANMS_RT);
            myTotalDuration = 0;
            count = 0;
            menuANMS_RT = false;
        }
        if (menuSSC)
        {
            myExtractor.SetDistribution(Distribution::SSC);
            myTotalDuration = 0;
            count = 0;
            menuSSC = false;
        }
        if (menuRANMS)
        {
            myExtractor.SetDistribution(Distribution::RANMS);
            myTotalDuration = 0;
            count = 0;
            menuRANMS = false;
        }

        if (menuSingleLvlOnly && (soloLvl != menuChosenLvl))
        {
            soloLvl = menuChosenLvl;
            menuDistrPerLvl = true;
            myExtractor.SetLevelToDisplay(soloLvl);
        }
        if (!menuSingleLvlOnly)
        {
            soloLvl = -1;
            myExtractor.SetLevelToDisplay(-1);
        }

        if (menuDistrPerLvl && !distributePerLevel)
            distributePerLevel = true;

        else if (!menuDistrPerLvl && distributePerLevel)
            distributePerLevel = false;

        if (menuPause)
            --ni;

        if (menuSetInitThreshold != FASTThresholdInit || menuSetMinThreshold != FASTThresholdMin)
        {
            FASTThresholdInit = menuSetInitThreshold;
            FASTThresholdMin = menuSetMinThreshold;
            myExtractor.SetFASTThresholds(FASTThresholdInit, FASTThresholdMin);
        }

        if (menuScoreExp)
        {
            myExtractor.SetScoreType(FASTdetector::EXPERIMENTAL);
            menuScoreExp = false;
        }
        if (menuScoreHarris)
        {
            myExtractor.SetScoreType(FASTdetector::HARRIS);
            menuScoreHarris = false;
        }
        if (menuScoreOpenCV)
        {
            myExtractor.SetScoreType(FASTdetector::OPENCV);
            menuScoreOpenCV = false;
        }
        if (menuScoreSum)
        {
            myExtractor.SetScoreType(FASTdetector::SUM);
            menuScoreSum = false;
        }
        /*
        if (menuMultithreading && !multithreading)
        {
            myExtractor.SetMultithreading(true);
            multithreading = true;
        }
        else if (!menuMultithreading && multithreading)
        {
            myExtractor.SetMultithreading(false);
            multithreading = false;
        }
         */





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
    //cout << "\n" << (eqkpts ? "All keypoints across all images were equal!\n" : "Not all keypoints are equal...:(\n");
    //cout << "\n" << (eqdescriptors ? "All descriptors across all images and keypoints were equal!\n" :
    //                    "Not all descriptors were equal... :(\n");

    cout << "\nTotal computation time: " << myTotalDuration/1000 << " milliseconds\n";
    //cout << "\nTotal computation time using ref orb: " << refTotalDuration/1000 <<
    //    " milliseconds, which averages to ~" << refTotalDuration/nImages << " microseconds.\n";
}


void PerformanceMode(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                     int FASTThresholdMin)
{
    cout << "\nStarting in Performance Mode...\n";
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(imgPath)+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    vector<long> vTimesTrack;
    vTimesTrack.resize(nImages);

    ORB_SLAM2::ORBextractor extractor(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    cout << "\n-------------------------\n"
         << "Images in sequence: " << nImages << "\n";

    cout << "\nSettings:\nnFeatures: " << nFeatures << "\nscaleFactor: " << scaleFactor << "\nnLevels: " << nLevels <<
        "\nFAST Thresholds: " << FASTThresholdInit << ", " << FASTThresholdMin << "\n";

    long totalDuration = 0;

    cv::Mat img;

    using clk = std::chrono::high_resolution_clock;

    clk::time_point t0 = clk::now();
    for (size_t d = 0; d < 7; ++d)
    {
        if (d == 1)
            continue;
        extractor.SetDistribution(static_cast<Distribution::DistributionMethod>(d));
        cout << "\nTesting performance with distribution " << static_cast<Distribution::DistributionMethod>(d)
            << "...";
        totalDuration = 0;
        vTimesTrack = vector<long>(nImages);
        for (int ni = 0; ni < nImages; ++ni)
        {
            img = cv::imread(string(imgPath) + "/" + vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);

            cv::Mat imgGray;
            cv::cvtColor(img, imgGray, CV_BGR2GRAY);

            vector<cv::KeyPoint> kpts;
            cv::Mat descriptors;

            clk::time_point t1 = clk::now();
            extractor(imgGray, cv::Mat(), kpts, descriptors, true);
            clk::time_point t2 = clk::now();
            long d = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
            totalDuration += d;
            vTimesTrack[ni] = d;
            //if (ni%100 == 0)
            //    cout << "\nNow at image " << ni;
        }
        std::sort(vTimesTrack.begin(), vTimesTrack.end());
        cout << "\nTotal time: " << (double)totalDuration/1000000.0 << " seconds"
                 "\nmean: " << (double)totalDuration/nImages/1000.0 << " milliseconds"
                 "\nmedian: " << (double)vTimesTrack[nImages/2]/1000.0 << " milliseconds";
        cout << "\n-------------------------\n";
    }

}

void SortKeypoints(vector<cv::KeyPoint> &kpts)
{
    std::sort(kpts.begin(), kpts.end(), [](const cv::KeyPoint &k1, const cv::KeyPoint &k2)
        {return (k1.pt.x < k2.pt.x || (k1.pt.x == k2.pt.x && k1.pt.y < k2.pt.y));});
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
    for (int i = 0; i < N; ++i)
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
       //cv::circle(image, point, 2, color, 1, CV_AA);
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
        cv::line(image, start, end, cv::Scalar(0, 0, 255), 1, CV_AA);
    }
    for (int x = 0; x <= cellCols; ++x)
    {
        cv::Point2f start(minX + x*cellWidth, minY);
        cv::Point2f end(minX + x*cellWidth, maxY);
        cv::line(image, start, end, cv::Scalar(0, 0, 255), 1, CV_AA);
    }
}

//TODO: update execution time function to make usable again
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
           //extractor.testingFAST(img, kpts, true, false);
       }

       std::chrono::high_resolution_clock::time_point tp_midpoint = std::chrono::high_resolution_clock::now();

       for (int i = 0; i < N; ++i)
       {
           //extractor.testingFAST(img, kpts, false, false);
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
        extractor.SetDistribution(Distribution::KEEP_ALL);
        extractor(imgGray, cv::Mat(), kptsAll, descriptors, true);
        t1 = clk::now();
        extractor.SetDistribution(Distribution::NAIVE);
        extractor(imgGray, cv::Mat(), kptsNaive, descriptors, true);
        t2 = clk::now();
        extractor.SetDistribution(Distribution::RANMS);
        extractor(imgGray, cv::Mat(), kptsQuadtree, descriptors, true);
        t3 = clk::now();
        extractor.SetDistribution(Distribution::QUADTREE_ORBSLAMSTYLE);
        extractor(imgGray, cv::Mat(), kptsQuadtreeORBSLAMSTYLE, descriptors, true);
        t4 = clk::now();
        extractor.SetDistribution(Distribution::GRID);
        extractor(imgGray, cv::Mat(), kptsGrid, descriptors, true);
        t5 = clk::now();
        extractor.SetDistribution(Distribution::ANMS_KDTREE);
        extractor(imgGray, cv::Mat(), kptsANMS_KDTree, descriptors, true);
        t6 = clk::now();
        extractor.SetDistribution(Distribution::ANMS_RT);
        extractor(imgGray, cv::Mat(), kptsANMS_RT, descriptors, true);
        t7 = clk::now();
        extractor.SetDistribution(Distribution::SSC);
        extractor(imgGray, cv::Mat(), kptsSSC, descriptors, true);
    }
    else
    {
        extractor.SetDistribution(Distribution::KEEP_ALL);
        extractor(imgGray, cv::Mat(), kptsAll, descriptors, false);
        t1 = clk::now();
        extractor.SetDistribution(Distribution::NAIVE);
        extractor(imgGray, cv::Mat(), kptsNaive, descriptors, false);
        t2 = clk::now();
        extractor.SetDistribution(Distribution::RANMS);
        extractor(imgGray, cv::Mat(), kptsQuadtree, descriptors, false);
        t3 = clk::now();
        extractor.SetDistribution(Distribution::QUADTREE_ORBSLAMSTYLE);
        extractor(imgGray, cv::Mat(), kptsQuadtreeORBSLAMSTYLE, descriptors, false);
        t4 = clk::now();
        extractor.SetDistribution(Distribution::GRID);
        extractor(imgGray, cv::Mat(), kptsGrid, descriptors, false);
        t5 = clk::now();
        extractor.SetDistribution(Distribution::ANMS_KDTREE);
        extractor(imgGray, cv::Mat(), kptsANMS_KDTree, descriptors, false);
        t6 = clk::now();
        extractor.SetDistribution(Distribution::ANMS_RT);
        extractor(imgGray, cv::Mat(), kptsANMS_RT, descriptors, false);
        t7 = clk::now();
        extractor.SetDistribution(Distribution::SSC);
        extractor(imgGray, cv::Mat(), kptsSSC, descriptors, false);
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
            "\nRANMS: " << d3 << " microseconds" <<
            "\nQuadtre: " << d4 << " microseconds" <<
            "\nBucketing: " << d5 << " microseconds" <<
            "\nANMS (KDTree): " << d6 << " microseconds" <<
            "\nANMS (Range Tree): " << d7 << " microseconds" <<
            "\nSuppression via Square Covering: " << d8 << " microseconds\n";

    DisplayKeypoints(imgAll, kptsAll, color, thickness, radius, drawAngular, "all");
    cv::waitKey(0);
    DisplayKeypoints(imgNaive, kptsNaive, color, thickness, radius, drawAngular, "naive");
    cv::waitKey(0);
    DisplayKeypoints(imgQuadtree, kptsQuadtree, color, thickness, radius, drawAngular, "quadtree");
    cv::waitKey(0);
    DisplayKeypoints(imgQuadtreeORBSLAMSTYLE, kptsQuadtreeORBSLAMSTYLE, color, thickness, radius, drawAngular,
            "quadtree ORBSLAM");
    cv::waitKey(0);
    DisplayKeypoints(imgGrid, kptsGrid, color, thickness, radius, drawAngular, "Grid");
    cv::waitKey(0);
    DisplayKeypoints(imgANMS_KDTree, kptsANMS_KDTree, color, thickness, radius, drawAngular, "KDTree ANMS");
    cv::waitKey(0);
    DisplayKeypoints(imgANMS_RT, kptsANMS_RT, color, thickness, radius, drawAngular, "Range Tree ANMS");
    cv::waitKey(0);
    DisplayKeypoints(imgSSC, kptsSSC, color, thickness, radius, drawAngular, "SSC");
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