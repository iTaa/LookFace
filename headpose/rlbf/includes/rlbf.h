#ifndef __RLBF_H_
#define __RLBF_H_

#include "../../../general_def.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
//#include <Windows.h>
#include "headers.h"
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <omp.h>
#endif

//#include <sys/time.h>
//#include "facedetect-dll.h"
//#pragma comment(lib,"libfacedetect.lib")

using namespace cv;
using namespace std;

namespace face_rlbf{
    class Rlbf{
    public:
        int     minsize;
        int     maxsize;

        float   angle;

        float   scalefactor;
        float   stridefactor;

        float   qthreshold;

        int     draw;

        int     usepyr;
        int     noclustering;
        int     verbose;

        bool    video_en;
        std::string pico_cascades_path;
        std::string video_path;

        std::string model_name;
        std::string image_name;

    public:
        Rlbf(){
            minsize = 128;
            maxsize = 1024;

            scalefactor = 1.1f;
            stridefactor = 0.1f;

            qthreshold = 5.0f;

            angle = 0.0f;

            usepyr = 1;
            noclustering = 0;
            verbose = 1;

            pico_cascades_path = "../../../pico/cascades/facefinder";
            video_path = "/home/jxgu/dataset/save/pose_reco.avi";

            model_name = "../../Model_helen/Model";

        };
        ~Rlbf(){};
    public:
        void    DrawPredictedImage(cv::Mat_<uchar> & image, cv::Mat_<double>& shape);
        void    Test(const char* ModelName);
        void    TestImage(const char* name, CascadeRegressor& rg);
        void    TestImagePico(const char* name, CascadeRegressor& rg);
        void    TestVideoPico(CascadeRegressor& rg);
        void    TestImage(const char* ModelName, const char* name);
        void    TestVideo(void);
        void    Train(const char* ModelName, std::string train_list);
        int     train_test(int argc, char* argv[]);
    };
}


#endif