#ifndef PERFTESTING_H
#define PERFTESTING_H

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <map>

#include "utils/balls.hpp"

/**
 * @brief Calculates IoU (percentage) of two `Rect`s
 * 
 * @param region1 first rectangle to compare
 * @param region2 second rectangle to compare
 * @return float IoU
 */
float intersectionOverUnion(cv::Rect region1, cv::Rect region2);

/**
 * @brief Calculates IoU (percentage) of two binary masks
 * 
 * @param mask1 1 channel `uchar` image
 * @param mask2 1 channel `uchar` image of same size as `mask1`
 * @return float IoU
 */
float intersectionOverUnion(cv::Mat mask1, cv::Mat mask2);

/**
 * @brief Return best IoU among candidates over given region
 * 
 * @param region `Rect` to calculate best IoU over
 * @param candidates candidate regions for IoU
 * @return float best IoU found
 */
float oneToManyIoU(cv::Rect region, std::vector<cv::Rect> candidates);

/**
 * @brief Return array of best IoUs found between pairs of one vector and the other
 * 
 * @param predictions vector of rects
 * @param ground_truth second vector of rects
 * @return std::vector<float> best IoU found for each member of `regions1`
 */
std::vector<float> manyToManyIoU(std::vector<cv::Rect> predictions, std::vector<cv::Rect> ground_truth);

/**
 * @brief Calculates a (discrete) precision-recall curve over given BBs
 * 
 * @param predictions predicted areas
 * @param truths ground truth areas
 * @param threshold how much IoU should be considered a positive match
 */
std::map<float,float> precisionRecallCurve(
    std::vector<cv::Rect> predictions, 
    std::vector<cv::Rect> truths,
    float threshold=0.5
);

/**
 * @brief Given a discrete curve, find the greates value with key >= of `key`
 * 
 * @param curve discrete curve (as output by `precisionRecallCurve`)
 * @param key minimum key value to consider
 * @return float highest value found
 */
float highestValueToTheRight(
    std::map<float,float> curve,
    float key
);

/**
 * @brief Calclate average precision of discrete precision-recall curve 
 * using the Pascal VOC method
 * 
 * @param prCurve the discrete P-R curve as outputted by `precisionRecallCurve`
 * @param steps how many discrete steps to divide the graph into while integrating
 * @return float the final average precision
 */
float averagePrecision(
    std::map<float,float> prCurve,
    int steps=10
);

#endif /* PERFTESTING_H */
