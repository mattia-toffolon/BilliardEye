#ifndef BALLTYPE_H
#define BALLTYPE_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "utils/balls.hpp"

// BALL TYPE IDENTIFICATION
// The purpose of this library is to identify the type of a ball that
// has been located in a window of an image

BallType getBallType(cv::Mat image, cv::Rect2d position);

#endif /* BALLTYPE_H */
