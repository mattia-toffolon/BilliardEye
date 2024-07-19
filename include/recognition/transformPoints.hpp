// AUTHOR: Artico Giovanni

#ifndef TRANSFORMPOINTS
#define TRANSFORMPOINTS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
/**
 * @brief utility to get the perspective transform given the vertices of the table 
 * @param width width of the destination image
 * @param height height of the destination image
 * @param points vertices of the table, the points should be ordered (see segment/segTable.h->orderPoints)
 * @param rotated true if the first side is not the short one
 */
cv::Mat transPoints(std::vector<cv::Point2f> points, int width, int height, bool rotated = false);
#endif
