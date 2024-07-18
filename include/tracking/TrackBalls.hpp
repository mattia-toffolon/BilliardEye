// AUTHOR: Toffolon Mattia

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

    const int DELTA = 5;

    public:

    /**
     * Tracker constructor
     * 
     * @param frame first video frame
     * @param bb vector of Ball struct containing informations on ball location and type
     */
    TrackBalls(cv::Mat frame, std::vector<Ball> bb);

    /**
     * Method that updates the vector of trackers and Ball structs using the given video frame
     * 
     * @param frame next video frame
     * @param renderer_remove_idxs vector of ball indexes shared with method caller
     * @return vector of updated Ball structs
     */
    std::vector<Ball> update(cv::Mat frame, std::vector<int>& renderer_remove_idxs);

    /**
     * Method that erases the selected balls and respective trackers
     * 
     * @param indexEraseList vector of ball indexes to be erased
     */
    void removeBalls(std::vector<int> indexEraseList);

    /**
     * Method that returns the Balls with their original bounding box
     * 
     * @return vector of Ball structs with real-size bounding boxes
     */
    std::vector<Ball> getRealBalls();

    private:

    /**
     * Method that handles the Ball tracking update failures
     * 
     * @param found_indexs vector of indexes of correctly updated Balls
     * @param lost_indexes vector of indexes of Balls which tracking update has failed
     * @param found_bboxes vector of newly found bounding boxes
     * @param frame next video frame
     * @param renderer_remove_idxs vector of ball indexes shared with method caller
     */
    void adjustBalls(std::vector<int> found_indexes, std::vector<int> lost_indexes, std::vector<cv::Rect> found_bboxes, cv::Mat frame, std::vector<int>& renderer_remove_idxs);

    /**
     * Method that return the index of the closest bounding box in a given vector from a given reference box
     * 
     * @param tracked reference bounding box
     * @param found vector of newly found bounding boxes in the image
     * @return index of the closest box in "found" from the reference one
     */
    int getClosestBBoxIndex(cv::Rect tracked, std::vector<cv::Rect> found);

    /**
     * Method that computes the squared Euclidean distance between two bounding boxes top-left corner
     * 
     * @param r1 first bounding box
     * @param r2 second bounding box
     * @return squared Euclidean distance between the two given boxes top-left corner
     */
    float sqEuclideanDist(cv::Rect r1, cv::Rect r2);
};

#endif