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
 * @brief Perform k-means clustering and return associated labels
 * 
 * @param points points to cluster
 * @param k number of clusters
 * @return std::vector<int> labels (in same input order)
 */
std::vector<int> clusterIndexes(std::vector<cv::Point3f> points, int k);

/**
 * @brief Return relative share of found colors in image
 * 
 * @param img image to partition 
 * @param k number of clusters
 * @return vector<float> percentages of each cluster
 */
std::vector<float> clusterPercentage(cv::Mat img, int k);

/**
 * @brief Return an image that shows the labeled clusters
 * 
 * @param img image to label
 * @param labels row-col list of labels
 * @param k number of clusters
 * @return cv::Mat labeled image
 */
cv::Mat drawClusters(cv::Mat img, int k);

/**
 * @brief Equalize only part of image corresponding to mask
 * 
 * @param img grayscale image to stretch partial contrast of 
 * @param mask mask to restrict stretching - white pixels only (ball ellipse if empty)
 * @return cv::Mat partially stretched image
 */
cv::Mat equalizedMasked(cv::Mat img, cv::InputArray mask=cv::noArray());

#endif /* BALLTYPE_H */
