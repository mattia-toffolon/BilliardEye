// AUTHOR: Toffolon Mattia

#ifndef BBOXES_UTILS
#define BBOXES_UTILS

#include <opencv2/highgui.hpp>

/**
 * Function that converts the given circle into a box
 * 
 * @param circle Vec3f representing a circle
 * @return circle converted into a box (Rect)
 */
cv::Rect toRect(cv::Vec3f circle);

/**
 * Function that converts the givenbox into a circle
 * 
 * @param box Rect representing a box
 * @return box converted into a circle (Vec3f)
 */
cv::Vec3f toCircle(cv::Rect box);

/**
 * Function that draws the given set of bounding boxes into the given image
 * 
 * @param img image on which the boxes shall be drawn
 * @param bboxes vector of boxes (Rect)
 */
void drawBBoxes(cv::Mat img, std::vector<cv::Rect> bboxes);

/**
 * @brief Function that prints two copies of the given image side by side,
 * one on which the first set of bounding boxes is drawn, the other in which the second set is used
 * 
 * @param img image on which the boxes shall be drawn
 * @param bboxes1 first vector of boxes (Rect)
 * @param bboxes2 second vector of boxes (Rect)
 */
void drawBBoxesCanvas(cv::Mat img, std::vector<cv::Rect> bboxes1, std::vector<cv::Rect> bboxes2);

/**
 * @brief Function that modifies the given boxes by increasing their width/height by 
 * the given factor while keeping the box center fixed 
 * 
 * @param bboxes vector of boxes (Rect)
 * @param MULT box width expansion factor
 * @return enhanced vector of boxes (Rect)
 */
std::vector<cv::Rect> expandBBoxes(std::vector<cv::Rect> bboxes, const float MULT);

#endif