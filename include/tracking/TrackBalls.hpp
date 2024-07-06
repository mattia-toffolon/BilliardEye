#ifndef TRACK_BALLS
#define TRACK_BALLS
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
    /**
     * @param removed index of balls that are no longer tracked
     */
    std::vector<Ball> update(cv::Mat frame);
};
#endif
