#include <iostream>
#include <fstream>
#include <iomanip>
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

#define PRECOMPUTEDFEATURES 0
#define FEATUREPATH "/home/ralph/SLAM/SavedFeatures/kitti_seq_07/1500f_1.100000s_3d/"


using namespace std;


int main(int argc, char **argv)
{
    std::chrono::high_resolution_clock::time_point program_start = std::chrono::high_resolution_clock::now();

    if (argc != 6)
    {
        cerr << "required arguments: <path to settings> <path to image / sequence> "
                "<mode: 0/1> <save path> <kitti/tum/euroc>" << endl;
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

    string strdataset = argv[5];
    Dataset dataset = strdataset == "kitti" ? kitti : strdataset == "euroc"? euroc : tum;

    if (mode == 0)
    {
        //SingleImageMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin,
        //                color, thickness, radius, drawAngular);
    }
    else if (mode == 1)
    {
        SequenceMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin,
                     color, thickness, radius, drawAngular, dataset);
    }
    else if (mode == 2)
    {
        PerformanceMode(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);
    }
    else if (mode == 3)
    {
        FilterTest(imgPath, nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin, dataset);
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

/* TODO: port to saiga
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


    std::vector<kvis::KeyPoint> keypoints;
    cv::Mat descriptors;

    std::vector<kvis::KeyPoint> refkeypoints;
    cv::Mat refdescriptors;

    std::vector<kvis::KeyPoint> keypointsAll;

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
    pangolin::Var<bool> menuDistrPerLvl("menu.Distribute Per Level", false, distributePerLevel);
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

 */
void SequenceMode(string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                    int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular,
                    Dataset dataset)
{
    bool stereo;
    stereo = dataset == kitti || dataset == euroc;
    cout << "\nStarting in sequence mode...\n";

    vector<string> vstrImageFilenamesLeft;
    vector<double> vTimestamps;
    vector<string> vstrImageFilenamesRight;

    if (dataset == tum)
    {
        string strFile = string(imgPath)+"/rgb.txt";
        LoadImagesTUM(strFile, vstrImageFilenamesLeft, vTimestamps);
    }

    else if (dataset == kitti)
    {
        LoadImagesKITTI(imgPath, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    }

    else if (dataset == euroc)
    {
        string pathLeft = imgPath, pathRight = imgPath, pathTimes = imgPath;
        pathLeft += "cam0/data/";
        pathRight += "cam1/data/";
        pathTimes += "MH03.txt";
        LoadImagesEUROC(pathLeft, pathRight, pathTimes,vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    }

    int nImages = vstrImageFilenamesLeft.size();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    ORB_SLAM2::ORBextractor myExtractor(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);
    ORB_SLAM2::ORBextractor myExtractorRight(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);


    cout << "\n-------------------------\n"
           << "Images in sequence: " << nImages << "\n";

    bool eqkpts = true;
    bool eqdescriptors = true;

    long myTotalDuration = 0;

    int softTh = 0;

    cv::Mat img;
    cv::Mat imgRight;

    pangolin::CreateWindowAndBind("Menu",210,800);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(210));

    pangolin::Var<bool> menuPause("menu. ~PAUSE~", true, true);
    pangolin::Var<bool> menuAll("menu.All Keypoints",false,false);
    pangolin::Var<bool> menuTopN("menu.TopN",false,false);
    pangolin::Var<bool> menuBucketing("menu.Bucketing",true,false);
    pangolin::Var<bool> menuQuadtreeORBSLAMSTYLE("menu.Quadtree",false,false);
    pangolin::Var<bool> menuANMS_KDT("menu.KDTree-ANMS",false,false);
    pangolin::Var<bool> menuANMS_RT("menu.Range-Tree-ANMS",false,false);
    pangolin::Var<bool> menuSSC("menu.SSC",false,false);
    pangolin::Var<bool> menuRANMS("menu.RANMS", false, false);
    pangolin::Var<bool> menuSoftSSC("menu.Soft SSC", false, false);
    pangolin::Var<int> menuSoftSSCThreshold("menu.Soft SSC Threshold", softTh, 0, 100);
    pangolin::Var<bool> menuVSSC("menu.VSSC", false, false);
    pangolin::Var<bool> menuDistrPerLvl("menu.Distribute Per Level", true, true);
    pangolin::Var<int> menuNFeatures("menu.Desired Features", nFeatures, 500, 2000);
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
    pangolin::Var<bool> menuSingleLvlOnly("menu.Dispay single level:", false, true);
    pangolin::Var<int> menuChosenLvl("menu.Limit to Level", 0, 0, myExtractor.GetLevels()-1);
    pangolin::Var<int> menuMeanProcessingTime("menu.Mean Processing Time", 0);
    pangolin::Var<int> menuLastFrametime("menu.Last Frame", 0);
    pangolin::Var<bool> menuSaveFeatures("menu.SAVE FEATURES", false, false);

    pangolin::FinishFrame();

    cv::namedWindow(string(imgPath));
    cv::moveWindow(string(imgPath), 240, 260);
    string imgTrackbar = string("image nr");

    int nn = 0;
    cv::createTrackbar(imgTrackbar, string(imgPath), &nn, nImages);
    /** Trackbar call if opencv was compiled without Qt support:
    //cv::createTrackbar(imgTrackbar, string(imgPath), nullptr, nImages);
     */

    cv::displayStatusBar(string(imgPath), "Current Distribution: Bucketing");

    int count = 0;
    bool distributePerLevel = menuDistrPerLvl;
    int soloLvl = -1;

#if PRECOMPUTEDFEATURES
    string loadPath = FEATUREPATH;
    myExtractor.EnablePrecomputedFeatures(true);
    myExtractor.SetLoadPath(loadPath);
    cv::displayStatusBar(string(imgPath), "Current Distribution: Loaded from File");
#endif

    for(int ni=0; ni<nImages; ni++)
    {
        cv::setTrackbarPos("image nr", string(imgPath), ni);


        /// images loaded from TUM dataset:
        //img = cv::imread(string(imgPath) + "/" + vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);

        ///images loaded from KITTI dataset:
        img = cv::imread(vstrImageFilenamesLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        if (stereo)
            imgRight = cv::imread(vstrImageFilenamesRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (img.empty())
        {
            cerr << endl << "Failed to load image at: "
            << string(imgPath) << "/" << vstrImageFilenamesLeft[ni] << endl;
            exit(EXIT_FAILURE);
        }

        cv::Mat imgGray;
        cv::Mat imgGrayRight;
        if (img.channels() > 1)
        {
            cv::cvtColor(img, imgGray, CV_BGR2GRAY);
            if (stereo)
                cv::cvtColor(imgRight, imgGrayRight, CV_BGR2GRAY);
        }
        else
        {
            imgGray = img;
            if (stereo)
                imgGrayRight = imgRight;
        }


        img_t saigaImg = Saiga::MatToImageView<uchar>(imgGray);
        img_t saigaImgRight;
        if (stereo)
            saigaImgRight = Saiga::MatToImageView<uchar>(imgRight);

        vector<kvis::KeyPoint> mykpts;
        std::cout << "address of mykpts before: " << &mykpts << "\n";
        img_t mydescriptors;
        cv::Mat cvDescriptors;

        vector<kvis::KeyPoint> mykptsRight;
        img_t mydescriptorsRight;



        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();


        myExtractor(saigaImg, mykpts, mydescriptors, distributePerLevel);
        //myExtractor(imgGray, cv::Mat(), mykpts, cvDescriptors);
        std::cout << "kpts.sz in main: " << mykpts.size() << "\n";
        std::cout << "address of mykpts in main: " << &mykpts << "\n\n\n";

        if (stereo)
            myExtractorRight(saigaImgRight, mykptsRight, mydescriptorsRight, distributePerLevel);

        chrono::high_resolution_clock::time_point t3 = chrono::high_resolution_clock::now();

        auto myduration = chrono::duration_cast<chrono::microseconds>(t3 - t2).count();

        ++count;

        myTotalDuration += myduration;

        pangolin::FinishFrame();

        if (myExtractor.GetDistribution() == Distribution::GRID && !distributePerLevel)
        {
            DrawCellGrid(img, 0, img.cols, 0, img.rows, BUCKETING_GRID_SIZE);
        }
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);


        DisplayKeypoints(img, mykpts, color, thickness, radius, drawAngular, string(imgPath));
        cv::waitKey(1);

        //gui stuff
        if (cv::getTrackbarPos(imgTrackbar, string(imgPath)) != ni)
        {
            ni = cv::getTrackbarPos(imgTrackbar, string(imgPath));
            //myExtractor.GetFileInterface()->SetCurrentImage(ni); TODO
        }


        int n = menuNFeatures;
        if (n != nFeatures)
        {
            nFeatures = n;
            myExtractor.SetnFeatures(n);
            if (stereo)
                myExtractorRight.SetnFeatures(n);
        }

        float scaleF = menuScaleFactor;
        if (scaleF != myExtractor.GetScaleFactor())
        {
            myExtractor.SetScaleFactor(scaleF);
            if (stereo)
                myExtractorRight.SetScaleFactor(scaleF);
        }

        int nlvl = menuNLevels;
        if (nlvl != myExtractor.GetLevels())
        {
            myExtractor.SetnLevels(nlvl);
            if (stereo)
                myExtractorRight.SetnLevels(nlvl);
        }

        menuLastFrametime = myduration/1000;
        menuMeanProcessingTime = myTotalDuration/1000 / count;

        menuActualkpts = mykpts.size();

        if (menuAll)
        {
            myExtractor.SetDistribution(Distribution::KEEP_ALL);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::KEEP_ALL);
            cv::displayStatusBar(string(imgPath), "Current Distribution: None (All Keypoints kept)");
            myTotalDuration = 0;
            count = 0;
            menuAll = false;
        }
        if (menuTopN)
        {
            myExtractor.SetDistribution(Distribution::NAIVE);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::NAIVE);
            cv::displayStatusBar(string(imgPath), "Current Distribution: Top N");
            myTotalDuration = 0;
            count = 0;
            menuTopN = false;
        }
        if (menuBucketing)
        {
            myExtractor.SetDistribution(Distribution::GRID);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::GRID);
            cv::displayStatusBar(string(imgPath), "Current Distribution: Bucketing");
            myTotalDuration = 0;
            count = 0;
            menuBucketing = false;
        }
        if (menuQuadtreeORBSLAMSTYLE)
        {
            myExtractor.SetDistribution(Distribution::QUADTREE_ORBSLAMSTYLE);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::QUADTREE_ORBSLAMSTYLE);
            cv::displayStatusBar(string(imgPath), "Current Distribution: Quadtree");
            myTotalDuration = 0;
            count = 0;
            menuQuadtreeORBSLAMSTYLE = false;
        }
        if (menuANMS_KDT)
        {
            myExtractor.SetDistribution(Distribution::ANMS_KDTREE);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::ANMS_KDTREE);
            cv::displayStatusBar(string(imgPath), "Current Distribution: KD-Tree");
            myTotalDuration = 0;
            count = 0;
            menuANMS_KDT = false;
        }
        if (menuANMS_RT)
        {
            myExtractor.SetDistribution(Distribution::ANMS_RT);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::ANMS_RT);
            cv::displayStatusBar(string(imgPath), "Current Distribution: Range Tree");
            myTotalDuration = 0;
            count = 0;
            menuANMS_RT = false;
        }
        if (menuSSC)
        {
            myExtractor.SetDistribution(Distribution::SSC);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::SSC);
            cv::displayStatusBar(string(imgPath), "Current Distribution: SSC");
            myTotalDuration = 0;
            count = 0;
            menuSSC = false;
        }
        if (menuRANMS)
        {
            myExtractor.SetDistribution(Distribution::RANMS);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::RANMS);
            cv::displayStatusBar(string(imgPath), "Current Distribution: Bucketed Soft SSC");
            myTotalDuration = 0;
            count = 0;
            menuRANMS = false;
        }
        if (menuSoftSSC)
        {
            myExtractor.SetDistribution(Distribution::SOFT_SSC);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::SOFT_SSC);
            cv::displayStatusBar(string(imgPath), "Current Distribution: Soft SSC");
            myTotalDuration = 0;
            count = 0;
            menuSoftSSC = false;
        }

        if (menuSoftSSCThreshold != softTh)
        {
            myExtractor.SetSoftSSCThreshold(menuSoftSSCThreshold);
            if(stereo)
                myExtractorRight.SetSoftSSCThreshold(menuSoftSSCThreshold);
            softTh = menuSoftSSCThreshold;
        }

        if (menuVSSC)
        {
            myExtractor.SetDistribution(Distribution::VSSC);
            if (stereo)
                myExtractorRight.SetDistribution(Distribution::VSSC);
            cv::displayStatusBar(string(imgPath), "Current Distribution: VSSC");
            myTotalDuration = 0;
            count = 0;
            menuVSSC = false;
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
        {
            --ni;
            //myExtractor.GetFileInterface()->SetCurrentImage(ni);
            //if (stereo)
            //    myExtractorRight.GetFileInterface()->SetCurrentImage(ni); TODO
        }


        if (menuSetInitThreshold != FASTThresholdInit || menuSetMinThreshold != FASTThresholdMin)
        {
            FASTThresholdInit = menuSetInitThreshold;
            FASTThresholdMin = menuSetMinThreshold;
            myExtractor.SetFASTThresholds(FASTThresholdInit, FASTThresholdMin);
            if (stereo)
                myExtractorRight.SetFASTThresholds(FASTThresholdInit, FASTThresholdMin);
        }

        if (menuScoreExp)
        {
            myExtractor.SetScoreType(FASTdetector::EXPERIMENTAL);
            if (stereo)
                myExtractorRight.SetScoreType(FASTdetector::EXPERIMENTAL);
            menuScoreExp = false;
        }
        if (menuScoreHarris)
        {
            myExtractor.SetScoreType(FASTdetector::HARRIS);
            if (stereo)
                myExtractorRight.SetScoreType(FASTdetector::HARRIS);
            menuScoreHarris = false;
        }
        if (menuScoreOpenCV)
        {
            myExtractor.SetScoreType(FASTdetector::OPENCV);
            if (stereo)
                myExtractorRight.SetScoreType(FASTdetector::OPENCV);
            menuScoreOpenCV = false;
        }
        if (menuScoreSum)
        {
            myExtractor.SetScoreType(FASTdetector::SUM);
            if (stereo)
                myExtractorRight.SetScoreType(FASTdetector::SUM);
            menuScoreSum = false;
        }
        /*
        if (menuSaveFeatures)
        {
#if PRECOMPUTEDFEATURES
            cerr << "Saving Features while using loaded features is not supported.\n";
            menuSaveFeatures = false;
#else
            ni = -1;
            myExtractor.SetFeatureSaving(true);
            menuSaveFeatures = false;
            string featureSavePath = "/home/ralph/SLAM/SavedFeatures/kitti_seq_07/left";

            myExtractor.SetFeatureSavePath(featureSavePath);
            if (menuPause)
            {
                menuPause = false;
            }
            FeatureFileInterface::fileInfo info;
            info.nLevels = myExtractor.GetLevels();
            info.nFeatures = menuNFeatures;
            info.scaleFactor = myExtractor.GetScaleFactor();
            info.SSCThreshold = softTh;
            info.kptDistribution = myExtractor.GetDistribution();
            myExtractor.GetFileInterface()->SaveInfo(info);

            if (stereo)
            {
                myExtractorRight.SetFeatureSaving(true);
                string featureSavePathRight = "/home/ralph/SLAM/SavedFeatures/kitti_seq_07/right";
                myExtractorRight.SetFeatureSavePath(featureSavePathRight);
                myExtractorRight.GetFileInterface()->SaveInfo(info);
            }
#endif

        }
         */




        /** compare kpts and descriptors per image:
        vector<std::pair<kvis::KeyPoint, kvis::KeyPoint>> kptDiffs;
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
    LoadImagesTUM(strFile, vstrImageFilenames, vTimestamps);

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

            img_t saigaImg = Saiga::MatToImageView<uchar>(imgGray);

            vector<kvis::KeyPoint> kpts;
            img_t descriptors;

            clk::time_point t1 = clk::now();
            extractor(saigaImg, kpts, descriptors, true);
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

void SortKeypoints(vector<kvis::KeyPoint> &kpts)
{
    std::sort(kpts.begin(), kpts.end(), [](const kvis::KeyPoint &k1, const kvis::KeyPoint &k2)
        {return (k1.pt.x < k2.pt.x || (k1.pt.x == k2.pt.x && k1.pt.y < k2.pt.y));});
}


vector<std::pair<kvis::KeyPoint, kvis::KeyPoint>> CompareKeypoints(vector<kvis::KeyPoint> &kpts1, string name1,
        vector<kvis::KeyPoint> &kpts2, string name2, int imgNr, bool print)
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

    vector<std::pair<kvis::KeyPoint, kvis::KeyPoint>> differences;
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

void DisplayKeypoints(cv::Mat &image, std::vector<kvis::KeyPoint> &keypoints, cv::Scalar &color,
                     int thickness, int radius, int drawAngular, string windowname)
{
   cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
   cv::imshow(windowname, image);
   //cv::waitKey(0);

   for (const kvis::KeyPoint &k : keypoints)
   {
       cv::Point2f point = cv::Point2f(k.pt.x, k.pt.y);
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
    std::vector<kvis::KeyPoint> kpts;
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

/*
void DistributionComparisonSuite(ORB_SLAM2::ORBextractor &extractor, cv::Mat &imgColor, cv::Scalar &color,
        int thickness, int radius, bool drawAngular, bool distributePerLevel)
{
    typedef std::chrono::high_resolution_clock clk;
    cv::Mat imgGray;
    imgColor.copyTo(imgGray);


    std::vector<kvis::KeyPoint> kptsAll;
    std::vector<kvis::KeyPoint> kptsNaive;
    std::vector<kvis::KeyPoint> kptsQuadtree;
    std::vector<kvis::KeyPoint> kptsQuadtreeORBSLAMSTYLE;
    std::vector<kvis::KeyPoint> kptsGrid;
    std::vector<kvis::KeyPoint> kptsANMS_KDTree;
    std::vector<kvis::KeyPoint> kptsANMS_RT;
    std::vector<kvis::KeyPoint> kptsSSC;

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
 */

void AddRandomKeypoints(std::vector<kvis::KeyPoint> &keypoints)
{
   int nKeypoints = 150;
   keypoints.clear();
   keypoints.reserve(nKeypoints);
   for (int i =0; i < nKeypoints; ++i)
   {
       auto x = static_cast<float>(20 + (rand() % static_cast<int>(620 - 20 + 1)));
       auto y = static_cast<float>(20 + (rand() % static_cast<int>(460 - 20 + 1)));
       auto angle = static_cast<float>(0 + (rand() % static_cast<int>(359 - 0 + 1)));
       keypoints.emplace_back(kvis::KeyPoint(x, y, 7.f, angle, 0));
   }
}

void FilterTest(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                int FASTThresholdMin, Dataset dataset)
{
    vector<string> vstrImageFilenamesLeft;
    vector<double> vTimestamps;
    vector<string> vstrImageFilenamesRight;

    ORB_SLAM2::ORBextractor extractor(nFeatures, scaleFactor, nLevels, FASTThresholdInit, FASTThresholdMin);

    if (dataset == tum)
    {
        string strFile = string(imgPath)+"/rgb.txt";
        LoadImagesTUM(strFile, vstrImageFilenamesLeft, vTimestamps);
    }

    else if (dataset == kitti)
    {
        LoadImagesKITTI(imgPath, vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    }

    else if (dataset == euroc)
    {
        string pathLeft = imgPath, pathRight = imgPath, pathTimes = imgPath;
        pathLeft += "cam0/data/";
        pathRight += "cam1/data/";
        pathTimes += "MH03.txt";
        LoadImagesEUROC(pathLeft, pathRight, pathTimes,vstrImageFilenamesLeft, vstrImageFilenamesRight, vTimestamps);
    }

    int nImages = vstrImageFilenamesLeft.size();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cv::Mat img = cv::imread(vstrImageFilenamesLeft[1], CV_LOAD_IMAGE_UNCHANGED);

    cv::Mat imgGray;
    if (img.channels() > 1)
    {
        cv::cvtColor(img, imgGray, CV_BGR2GRAY);
    }
    else
    {
        imgGray = img;
    }

    img_t saigaImg = Saiga::MatToImageView<uchar>(imgGray);

    extractor.FilterTest(saigaImg);

    cv::Mat img2 = cv::imread(vstrImageFilenamesLeft[1], CV_LOAD_IMAGE_UNCHANGED);
    cv::GaussianBlur(img2, img2, cv::Size(7, 7), 2, 2);
    cv::imshow("test", img2);
    cv::waitKey(0);
}


void LoadImagesTUM(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
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

void LoadImagesKITTI(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

void LoadImagesEUROC(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}

