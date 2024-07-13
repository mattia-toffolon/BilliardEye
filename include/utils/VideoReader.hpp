#ifndef VIDEOREADER
#define VIDEOREADER
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class VideoReader{
    cv::VideoCapture vid;
    public:
    VideoReader(std::string filename);
    ~VideoReader();
    cv::Mat nextFrame();
    cv::Mat lastFrame();
    int fps();
};
 
#endif
