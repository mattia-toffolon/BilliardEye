// AUTHOR: Toffolon Mattia

#ifndef SEG_BALLS
#define SEG_BALLS

#include <opencv2/highgui.hpp>

/**
 * Function that performs ball localization and segmentation on the given image
 * 
 * @param img image on which ball localization shall be executed
 * @param mask table segmentation mask
 * @param transf perspective transformation matrix
 * @return vector of bounding boxes expressed as Rect
 */
std::vector<cv::Rect> getBBoxes(cv::Mat img, cv::Mat mask, cv::Mat transf);

/**
 * @brief Function that sets each non-zero value pixel to the absolute value of the difference 
 * between the mean table pixel intensity and the pixel one 
 * 
 * @param img image that has to be processed
 * @return processed image
 */
cv::Mat subtractTable(cv::Mat img);

/**
 * @brief Function that detects and returns circles in the image using HoughCircles OpenCV 
 * function with HOUGH_GRADIENT methos and the given parameters
 * 
 * @param img image on which circles shall be found
 * @param param1 first HOUGH_GRADIENT specific parameter
 * @param param2 second HOUGH_GRADIENT specific parameter
 * @param minRadius minimum radius of detectable circles
 * @param maxRadius maximum radius of detectable circles
 * @param draw boolean that controls the image printing
 * @return vector of circles expressed as Vec3f
 */
std::vector<cv::Vec3f> circlesFinder(cv::Mat img, double param1, double param2, int minRadius, int maxRadius, bool draw=false);

/**
 * Function that performs a custom merge of the two given sets of circles
 * 
 * @param first first vector of circles
 * @param second seconod vector of circles
 * @return final vector of circle
 */
std::vector<cv::Vec3f> smartCircleMerge(std::vector<cv::Vec3f> first, std::vector<cv::Vec3f> second);

/**
 * Function that converts the given circles into bounding boxes
 * 
 * @param circles vector of circles expressed as Vec3f
 * @return vector of bounding boxes expressed as Rect
 */
std::vector<cv::Rect> bboxConverter(std::vector<cv::Vec3f> circles);

/**
 * Function that filters the false positive bounding boxes from the given vector
 * 
 * @param img image on which ball localization is performed
 * @param persp_trans perspective transformation matrix
 * @param bboxes vector of bounding boxes 
 * @return vector of filtered bounding boxes
 */
std::vector<cv::Rect> purgeFP(cv::Mat img, cv::Mat persp_trans, std::vector<cv::Rect> bboxes);

/**
 * Function that filters the false positive bounding boxes from the given vector using the given altered image
 * 
 * @param img modified version of Canny output on original image
 * @param persp_trans perspective transformation matrix
 * @param bboxes vector of bounding boxes 
 * @param prev_vals vector of previously computed white filling percentages of the bounding boxes
 * @return vector of filtered bounding boxes using the given altered image
 */
std::vector<cv::Rect> purgeByCanny(cv::Mat img, cv::Mat persp_trans, std::vector<cv::Rect> bboxes, std::vector<float>& prev_vals);

/**
 * Function that refines the given bounding boxes
 * 
 * @param img original versionn of the image on which the circles have been previously found
 * @param bboxes vector of bounding boxes indicating the approximate ball locations
 * @param draw boolean that controls the image printing
 * @return vector of refined bounding boxes expressed as Rect
 */
std::vector<cv::Rect> refineBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes, bool draw=false);

#endif