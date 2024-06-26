#ifndef BALLTYPE_H
#define BALLTYPE_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/balls.hpp"

// BALL TYPE IDENTIFICATION
// The purpose of this library is to identify the type of a ball that
// has been located in a window of an image

// Possibly include table color among arguments?

/**
 * @brief Get an estimated BallType from a ball inside an image
 * 
 * @param image Image that contains the ball
 * @param window Precise location of ball
 * @return BallType 
 */
BallType getBallType(cv::Mat image, cv::Rect2d window);

/**
 * @brief Estimated percentage of ball which is colored
 * 
 * @param image Cropped grayscale portion of image containing only the ball
 * @return float 
 */
float ballFullness(cv::Mat ballCrop);

#endif /* BALLTYPE_H */
