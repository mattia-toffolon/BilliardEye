#include "utils/VideoReader.hpp"
#include "opencv2/videoio.hpp"

#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <string>

using namespace cv;
VideoReader::VideoReader(std::string filename){
    this->vid = VideoCapture(filename);
    if(!vid.isOpened()){
        throw vid;
    }
}
VideoReader::~VideoReader(){
    vid.release();
}
Mat VideoReader::nextFrame(){
    Mat frame;
    bool res = vid.read(frame);
    if(!res){
        return Mat();
    }
    return frame;
}
Mat VideoReader::lastFrame(){
    Mat frame;
    int curframe = static_cast<int>(vid.get(CAP_PROP_POS_FRAMES));
    int num = static_cast<int>(vid.get(CAP_PROP_FRAME_COUNT)-1);
    vid.set(CAP_PROP_POS_FRAMES, num);
    vid.read(frame);
    vid.set(CAP_PROP_POS_FRAMES, curframe);
    return frame;
}
int VideoReader::fps(){
    return this->vid.get(cv::CAP_PROP_FPS);
}
