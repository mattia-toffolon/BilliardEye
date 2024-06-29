#ifndef RENDER_TABLE
#define RENDER_TABLE
#include "tracking/TrackBalls.hpp"
#include "utils/VideoReader.hpp"
class TableRenderer{
    VideoReader vid;
    TrackBalls tracker;
    cv::Mat curimg;
    std::vector<Ball> bbs;
    cv::Mat transform;
    public:
    TableRenderer(VideoReader v, TrackBalls t, std::vector<Ball> balls, cv::Mat transform, int width, int cols);
    cv::Mat nextFrame();
};
#endif
