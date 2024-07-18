// AUTHOR: Artico Giovanni

#ifndef SIDE_RECOGNITION
#define SIDE_RECOGNITION
#include <opencv2/imgproc.hpp>

/**
 * @brief returns the rectangle along the sides of the table
 *  
 * @param points vertices of the table, the points should be ordered (see segment/segTable.h->orderPoints)
 * @param thick the width of the rectangle orthogonal to the side
 */
std::vector<cv::Mat> getRotatedborders(const std::vector<cv::Point2f> points, const cv::Mat img, int thick = 20);
/**
 * @brief returns true if the (ordered) sides have a short side as first
 *  
 * @param sides the rotated rectangles of the sides of the table, better if canny has been applied to original image
 */
//
bool isShortFirst(std::vector<cv::Mat> sides);

cv::Mat getTransformation(cv::Mat img, std::vector<cv::Point2f> points, int width=0, int height=0);

#endif
