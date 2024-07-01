#ifndef BALLTYPE_H
#define BALLTYPE_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/balls.hpp"

// BALL TYPE IDENTIFICATION
// The purpose of this library is to identify the type of a ball that
// has been located in a window of an image

// THRESHOLDS FOR BALL FULLNESS
const float cueballThreshold = 0.3;
const float stripedThreshold = 0.06;

// Possibly include table color among arguments?

/**
 * @brief Get an estimated BallType from a ball inside an image
 * 
 * @param ballCrop cropped image containing only the ball
 * @return BallType 
 */
BallType getBallType(cv::Mat ballCrop);

/**
 * @brief Estimated percentage of ball in view which is white
 * 
 * @param image Cropped grayscale portion of image containing only the ball
 * @param thresh Percentile threshold
 * @return float 
 */
float ballWhiteness(cv::Mat ballCrop, float thresh=0.85);

/**
 * @brief Equalize only part of image corresponding to mask
 * 
 * @param img grayscale image to stretch partial contrast of 
 * @param mask mask to restrict stretching - white pixels only (ball ellipse if empty)
 * @return cv::Mat partially stretched image
 */
cv::Mat equalizedMasked(cv::Mat img, cv::InputArray mask=cv::noArray());

#endif /* BALLTYPE_H */
