#ifndef TRACK_BALLS
#define TRACK_BALLS
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include "utils/balls.hpp"
class TrackBalls{
    std::vector<cv::Ptr<cv::TrackerCSRT>> multi;
    std::vector<Ball> bbs;
    public:
    TrackBalls(cv::Mat frame, std::vector<Ball> bb);
    std::vector<Ball> update(cv::Mat frame, std::vector<int> &removed);
};
#endif
