#ifndef TRACK_BALLS
#define TRACK_BALLS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include "utils/balls.hpp"

class TrackBalls{
    std::vector<cv::Ptr<cv::TrackerCSRT>> multi_tracker;
    std::vector<Ball> balls;

    public:
    TrackBalls(cv::Mat frame, std::vector<Ball> bb);
    /**
     * @param removed index of balls that are no longer tracked
     */
    std::vector<Ball> update(cv::Mat frame, std::vector<int>& renderer_remove_idxs);

    void removeBalls(std::vector<int> indexKeepList, cv::Mat frame);
    private:


    float sqEuclideanDist(cv::Rect r1, cv::Rect r2);

    int getClosestBBoxIndex(cv::Rect tracked, std::vector<cv::Rect> found);

    void adjustBalls(std::vector<int> found_indexes, std::vector<int> lost_indexes, std::vector<cv::Rect> found_bboxes, cv::Mat frame, std::vector<int>& renderer_remove_idxs);
};

#endif
