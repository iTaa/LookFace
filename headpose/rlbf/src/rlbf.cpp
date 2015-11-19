
#include "rlbf.h"
#include "pico.h"

using namespace face_rlbf;
using namespace FacePico;

void Rlbf::DrawPredictedImage(cv::Mat_<uchar> & image, cv::Mat_<double>& shape)
{
	for (int i = 0; i < shape.rows; i++){
		cv::circle(image, cv::Point2f(shape(i, 0), shape(i, 1)), 2, (255));
	}
}

void Rlbf::Test(const char* ModelName)
{
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(ModelName);
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
	std::string file_names = "/home/jxgu/dataset/ibug/helen/trainset/train_jpgs.txt";
	LoadImages(images, ground_truth_shapes, bboxes, file_names);
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
	for (int i = 0; i < images.size(); i++){
		cv::Mat_<double> current_shape = ReProjection(cas_load.params_.mean_shape_, bboxes[i]);
        //struct timeval t1, t2;
        //gettimeofday(&t1, NULL);
        cv::Mat_<double> res = cas_load.Predict(images[i], current_shape, bboxes[i]);//, ground_truth_shapes[i]);

        //cout << res << std::endl;
        //cout << res - ground_truth_shapes[i] << std::endl;
        //double err = CalculateError(grodund_truth_shapes[i], res);
        //cout << "error: " << err << std::endl;

        DrawPredictedImage(images[i], res);
		//if (i == 10) break;
	}
    gettimeofday(&t2, NULL);
    double time_full = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0;
    cout << "time full: " << time_full << " : " << time_full/images.size() << endl;
	return;
}


void Rlbf::TestImage(const char* name, CascadeRegressor& rg)
{
	std::string fn_haar = "../../haarcascade_frontalface_alt2.xml";
	cv::CascadeClassifier haar_cascade;
	bool yes = haar_cascade.load(fn_haar);
	std::cout << "detector: " << yes << std::endl;
	cv::Mat_<uchar> image = cv::imread(name, 0);
		if (image.cols > 2000){
			cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
			//ground_truth_shape /= 3.0;
		}
		else if (image.cols > 1400 && image.cols <= 2000){
			cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
			//ground_truth_shape /= 2.0;
		}
    std::vector<cv::Rect> faces;

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);
    haar_cascade.detectMultiScale(image, faces, 1.1, 2, 0, cv::Size(30, 30));
    gettimeofday(&t2, NULL);
    cout << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    for (int i = 0; i < faces.size(); i++){
        cv::Rect faceRec = faces[i];
        BoundingBox bbox;
        bbox.start_x = faceRec.x;
        bbox.start_y = faceRec.y;
        bbox.width = faceRec.width;
        bbox.height = faceRec.height;
        bbox.center_x = bbox.start_x + bbox.width / 2.0;
        bbox.center_y = bbox.start_y + bbox.height / 2.0;
        cv::Mat_<double> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
        //cv::Mat_<double> tmp = image.clone();
        //DrawPredictedImage(tmp, current_shape);
        gettimeofday(&t1, NULL);
        cv::Mat_<double> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
        gettimeofday(&t2, NULL);
        cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
        cv::rectangle(image, faceRec, (255), 1);
        //cv::imshow("show image", image);
        //cv::waitKey(0);
        DrawPredictedImage(image, res);
        break;
    }
	return;
}

void Rlbf::TestVideoPico(CascadeRegressor& rg)
{
   
    std::vector<cv::Rect> faces;
    cv::Rect face_det;

    int size;
    FILE* file;

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    // set default parameters
    void* cascade = 0;

    file = fopen(pico_cascades_path.c_str(), "rb");       //"../../../pico/cascades/facefinder"

    fseek(file, 0L, SEEK_END);
    size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    cascade = malloc(size);

    fread(cascade, 1, size, file);

    printf("Finished load cascade\n");

    //
    fclose(file);

    FacePico::pico * pico = new FacePico::pico();

    cv::VideoCapture capture(video_path);

    if (!capture.isOpened())
    {
        printf("Can't open input video file\n");
        return;
    }

    std::string windowname = "--------------------";

    // framerate
    int64 int_t1,int_t0 = cv::getTickCount();
    double fps = 10;

    unsigned int fc = 0;

    while(1)
    {
        fc++;

        int key = cv::waitKey(1);

        cv::Mat_<uchar> image;  // channel = 1
        cv::Mat image_cap;      // channel = 3

        capture.read(image_cap);

        cvtColor(image_cap,image,CV_RGB2GRAY);

        #if 1
        if (image.cols > 2000){
            cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
            cv::resize(image_cap, image_cap, cv::Size(image_cap.cols / 3, image_cap.rows / 3), 0, 0, cv::INTER_LINEAR);
            //ground_truth_shape /= 3.0;
        }
        else if (image.cols > 1400 && image.cols <= 2000){
            cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
            cv::resize(image_cap, image_cap, cv::Size(image_cap.cols / 2, image_cap.rows / 2), 0, 0, cv::INTER_LINEAR);
            //ground_truth_shape /= 2.0;
        }
        #else
        float resize_scale = 480.0/image_cap.rows;

        cv::resize(image, image, cv::Size(image.cols * resize_scale, image.rows * resize_scale), 0, 0, cv::INTER_LINEAR);
        cv::resize(image_cap, image_cap, cv::Size(image.cols * resize_scale, image.rows * resize_scale), 0, 0, cv::INTER_LINEAR);
        #endif

        face_det = pico->process_image(image_cap, draw, usepyr, cascade, 10.0, minsize, maxsize, scalefactor, stridefactor, qthreshold, noclustering, verbose);
        
        #if 0
            face_det.x = face_det.x - face_det.width/10;
            face_det.y = face_det.y - face_det.height/10;
            face_det.width = face_det.width*(1+1/5);
            face_det.height = face_det.height*(1+1/5);
        #endif

        if(face_det.width >0 && face_det.height>0)
        {
            gettimeofday(&t2, NULL);
            cout << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;

            cv::Rect faceRec = face_det;
            BoundingBox bbox;
            bbox.start_x = faceRec.x;
            bbox.start_y = faceRec.y;
            bbox.width = faceRec.width;
            bbox.height = faceRec.height;
            bbox.center_x = bbox.start_x + bbox.width / 2.0;
            bbox.center_y = bbox.start_y + bbox.height / 2.0;
            cv::Mat_<double> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
            //cv::Mat_<double> tmp = image.clone();
            //DrawPredictedImage(tmp, current_shape);
            gettimeofday(&t1, NULL);
            cv::Mat_<double> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
            gettimeofday(&t2, NULL);
            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
            cv::rectangle(image, faceRec, (255), 1);
            //cv::imshow("show image", image);
            //cv::waitKey(0);
            DrawPredictedImage(image, res);

            // Work out the framerate
            if (fc % 10 == 0 && fc > 0)
            {
                int_t1 = cv::getTickCount();
                fps = 10.0 / (double(cv::getTickCount()-int_t0)/cv::getTickFrequency());
                int_t0 = int_t1;
            }
            if (fc >= 10)
                std::cout << "FPS: " << fps << std::endl;
        }

        cv::imshow(windowname, image);

        /*      
        for (int i = 0; i < faces.size(); i++){
            cv::Rect faceRec = faces[i];
            BoundingBox bbox;
            bbox.start_x = faceRec.x;
            bbox.start_y = faceRec.y;
            bbox.width = faceRec.width;
            bbox.height = faceRec.height;
            bbox.center_x = bbox.start_x + bbox.width / 2.0;
            bbox.center_y = bbox.start_y + bbox.height / 2.0;
            cv::Mat_<double> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
            //cv::Mat_<double> tmp = image.clone();
            //DrawPredictedImage(tmp, current_shape);
            gettimeofday(&t1, NULL);
            cv::Mat_<double> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
            gettimeofday(&t2, NULL);
            cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
            cv::rectangle(image, faceRec, (255), 1);
            //cv::imshow("show image", image);
            //cv::waitKey(0);
            DrawPredictedImage(image, res);
            break;
        }
        */
    }

    return;    
}

void Rlbf::TestImagePico(const char* name, CascadeRegressor& rg)
{
    cv::Mat_<uchar> image = cv::imread(name, 0);
    cv::Mat read_image = cv::imread(name, 0);

    if (image.cols > 2000){
        cv::resize(image, image, cv::Size(image.cols / 3, image.rows / 3), 0, 0, cv::INTER_LINEAR);
        cv::resize(read_image, read_image, cv::Size(read_image.cols / 3, read_image.rows / 3), 0, 0, cv::INTER_LINEAR);
        //ground_truth_shape /= 3.0;
    }
    else if (image.cols > 1400 && image.cols <= 2000){
        cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_LINEAR);
        cv::resize(read_image, read_image, cv::Size(read_image.cols / 2, read_image.rows / 2), 0, 0, cv::INTER_LINEAR);
        //ground_truth_shape /= 2.0;
    }
    
    std::vector<cv::Rect> faces;
    cv::Rect face_det;

    struct timeval t1, t2;
    double timeuse;
    gettimeofday(&t1, NULL);

    // set default parameters
    void* cascade = 0;
    cout<<"Start detect "<<endl;
    int size;
    FILE* file;

    file = fopen("../../../pico/cascades/facefinder", "rb");

    fseek(file, 0L, SEEK_END);
    size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    cascade = malloc(size);

    fread(cascade, 1, size, file);

    //
    fclose(file);

    FacePico::pico * pico = new FacePico::pico();

    int minsize;
    int maxsize;

    float angle;

    float scalefactor;
    float stridefactor;

    float qthreshold;

    int usepyr;
    int noclustering;
    int verbose;

    minsize = 128;
    maxsize = 1024;

    scalefactor = 1.1f;
    stridefactor = 0.1f;

    qthreshold = 5.0f;

    usepyr = 1;
    noclustering = 0;
    verbose = 1;

    cout<<"Start detect "<<endl;
    face_det = pico->process_image(read_image, draw, usepyr, cascade, angle, minsize, maxsize, scalefactor, stridefactor, qthreshold, noclustering, verbose);
    cout<<"detect x = "<<face_det.x << "y "<<face_det.y <<endl;

    faces.push_back(face_det);
    
    gettimeofday(&t2, NULL);
    cout << "face detected " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
    for (int i = 0; i < faces.size(); i++){
        cv::Rect faceRec = faces[i];
        BoundingBox bbox;
        bbox.start_x = faceRec.x;
        bbox.start_y = faceRec.y;
        bbox.width = faceRec.width;
        bbox.height = faceRec.height;
        bbox.center_x = bbox.start_x + bbox.width / 2.0;
        bbox.center_y = bbox.start_y + bbox.height / 2.0;
        cv::Mat_<double> current_shape = ReProjection(rg.params_.mean_shape_, bbox);
        //cv::Mat_<double> tmp = image.clone();
        //DrawPredictedImage(tmp, current_shape);
        gettimeofday(&t1, NULL);
        cv::Mat_<double> res = rg.Predict(image, current_shape, bbox);//, ground_truth_shapes[i]);
        gettimeofday(&t2, NULL);
        cout << "time predict: " << t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec)/1000000.0 << endl;
        cv::rectangle(image, faceRec, (255), 1);
        //cv::imshow("show image", image);
        //cv::waitKey(0);
        DrawPredictedImage(image, res);
        break;
    }
    return;
}

void Rlbf::TestVideo(void)
{
	CascadeRegressor cas_load;
	cas_load.LoadCascadeRegressor(model_name);
	//TestImage(name, cas_load);
    cout<<"Start"<<endl;
    TestVideoPico(cas_load);
    return;
}

void Rlbf::TestImage(const char* ModelName, const char* name)
{
    CascadeRegressor cas_load;
    cas_load.LoadCascadeRegressor(ModelName);
    //TestImage(name, cas_load);
    cout<<"Start"<<endl;
    TestImagePico(name, cas_load);
    return;
}


// training list example : "/home/jxgu/dataset/ibug/helen/trainset/train_jpgs.txt"

void Rlbf::Train(const char* ModelName, std::string train_list)
{
	std::vector<cv::Mat_<uchar> > images;
	std::vector<cv::Mat_<double> > ground_truth_shapes;
	std::vector<BoundingBox> bboxes;
    std::string file_names = train_list;
	LoadImages(images, ground_truth_shapes, bboxes, file_names);

	Parameters params;
    params.local_features_num_ = 300;
	params.landmarks_num_per_face_ = 68;
    params.regressor_stages_ = 6;
	params.local_radius_by_stage_.push_back(0.4);
    params.local_radius_by_stage_.push_back(0.3);
    params.local_radius_by_stage_.push_back(0.2);
	params.local_radius_by_stage_.push_back(0.1);
    params.local_radius_by_stage_.push_back(0.08);
    params.local_radius_by_stage_.push_back(0.05);
    params.tree_depth_ = 5;
    params.trees_num_per_forest_ = 8;
    params.initial_guess_ = 5;

	params.mean_shape_ = GetMeanShape(ground_truth_shapes, bboxes);
	CascadeRegressor cas_reg;
	cas_reg.Train(images, ground_truth_shapes, bboxes, params);

	cas_reg.SaveCascadeRegressor(ModelName);
	return;
}

int Rlbf::train_test(int argc, char* argv[])
{
    /*
    if (argc >= 3)
    {
        if (strcmp(argv[1], "train") == 0)
        {
            std::cout << "enter train\n";
            TestVideo(argv[2], "/home/jxgu/dataset/ibug/helen/trainset/train_jpgs.txt");

            return 0;
        }
        if (strcmp(argv[1], "test") == 0)
        {
            std::cout << "enter test\n";
            if (argc == 3){
                Test(argv[2]);
            }
            else{
                TestVideo(argv[2], argv[3]);
            }
            return 0;
        }
    }
    */
    std::cout << "use [./application train ModelName] or [./application test ModelName [image_name]] \n";
    return 0;
}

/*
int main(int argc, char* argv[])
{
//    int a[10];
//    #pragma omp parallel for
//    for(int i=0; i < 10; i++){
//        int index = 0;
//        for(int j = i; j<10; j++){
//            index += j;
//        }
//        a[i] = index;
//        std::cout << i << " :curent: " << index << std::endl;
//    }

//    for(int i=0; i < 10; i++){
//        std::cout << i << " :final: " << a[i] << std::endl;
//    }
	if (argc >= 3)
	{
		if (strcmp(argv[1], "train") == 0)
		{
			std::cout << "enter train\n";
			Train(argv[2]);

            return 0;
		}
		if (strcmp(argv[1], "test") == 0)
		{
			std::cout << "enter test\n";
            if (argc == 3){
                Test(argv[2]);
            }
            else{
                Test(argv[2], argv[3]);
            }
            return 0;
		}
	}

    std::cout << "use [./application train ModelName] or [./application test ModelName [image_name]] \n";
	return 0;
}
*/

/*
string fn_haar = "D:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";
cv::CascadeClassifier haar_cascade;
bool yes = haar_cascade.load(fn_haar);
cv::Mat img = cv::imread("helen/trainset/103770709_1.jpg");// "helen/trainset/232194_1.jpg");
cv::Mat gray;
double scale = 1.3f;
cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
Mat smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
equalizeHist(smallImg, smallImg);

std::vector<cv::Rect_<int> > faces;
haar_cascade.detectMultiScale(gray, faces,
1.1, 2, 0, Size(30,30));
printf_s("number of faces: %d\n", faces.size());
for (int i = 0; i < faces.size(); i++)
{
cv::Rect face_i = faces[i];
cv::Rect ret;
ret.x = face_i.x*scale;
ret.y = face_i.y*scale;
ret.width = (face_i.width - 1)*scale;
ret.height = (face_i.height - 1)*scale;
cv::Mat face = gray(face_i);
rectangle(img, face_i, (255, 255, 255), 1);
//rectangle(img, ret, (0, 0, 255), 1);
}
//imshow("ppµÄö¦ÕÕ", img);
//waitKey();

//return 0;

int * pResults = NULL;
pResults = facedetect_frontal((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
1.2f, 2, 24);
printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));
//print the detection results
for (int i = 0; i < (pResults ? *pResults : 0); i++)
{
short * p = ((short*)(pResults + 1)) + 6 * i;
int x = p[0];
int y = p[1];
int w = p[2];
int h = p[3];
int neighbors = p[4];

printf("face_rect=[%d, %d, %d, %d], neighbors=%d\n", x, y, w, h, neighbors);
cv::Rect faceRec(x,y,w,h);
cv::rectangle(img, faceRec, (0, 255, 0), 1);
}


cv::imshow("ppµÄö¦ÕÕ", img);
cv::waitKey();

*/
