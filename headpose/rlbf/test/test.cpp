
#include "rlbf.h"
#include "pico.h"

using namespace face_rlbf;
using namespace FacePico;

void rlbf_train(face_rlbf::Rlbf * Rlbf, std::string Train_model)
{
    Rlbf->Train(Train_model.c_str(),"/home/jxgu/dataset/ibug/helen/trainset/train_jpgs.txt");
}

void rlbf_test_video(face_rlbf::Rlbf * Rlbf)
{
    Rlbf->TestVideo();
}

int main(int argc, char* argv[])
{
    face_rlbf::Rlbf * Rlbf = new face_rlbf::Rlbf();

    Rlbf->video_path = "/home/jxgu/dataset/EssentialTremor.mp4";

    rlbf_test_video(Rlbf);

    std::cout << "use [./application train ModelName] or [./application test ModelName [image_name]] \n";
    return 0;
}

