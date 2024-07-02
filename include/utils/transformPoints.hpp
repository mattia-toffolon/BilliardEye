#ifndef TRANSFORMPOINTS
#define TRANSFORMPOINTS
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//utility to get the perspective transform given the vertices of
//the table, the width and height refer to the image itself
//rotated indicates if to rotate the table, as it expects the first
//side to be the short one of the table, if this is not the case 
//rotated should be set to true
cv::Mat transPoints(std::vector<cv::Point2f> points, int width, int height, bool rotated = false);
#endif
