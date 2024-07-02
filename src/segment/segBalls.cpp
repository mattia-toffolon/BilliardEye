#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "segment/segBalls.h"
#include "utils/getTableColor.hpp"
#include "utils/drawBBoxes.hpp"

using namespace cv;
using namespace std;

// UNUSABLE
// Mat meanShiftClustering(Mat& inputImage, int spatialRadius, int colorRadius, int maxLevel)
// {
//     Mat labImage;
//     cvtColor(inputImage, labImage, COLOR_BGR2Lab);

//     TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxLevel, 1.0);
//     Size windowSize(4 * spatialRadius + 1, 4 * spatialRadius + 1);

//     Mat outputImage;
//     pyrMeanShiftFiltering(labImage, outputImage, spatialRadius, colorRadius, maxLevel, criteria);

//     cvtColor(outputImage, outputImage, COLOR_Lab2BGR);

//     return outputImage;
// }

vector<Vec3f> circlesFinder(Mat img, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius, bool draw) {
    std::vector<cv::Vec3f> out;
    HoughCircles(img, out, method, dp, minDist, param1, param2, minRadius, maxRadius);

    if(out.empty()) cout<<"No circles found"<<endl;

    if(draw) {
        for(Vec3i c : out) {
            Point center = Point(c[0], c[1]);
            int radius = c[2];
            circle(img, center, radius, Scalar(0,0,255), 2, LINE_AA);
        }
        imshow("window", img);
        waitKey(0);
    }

    return out;
}

vector<Rect> bboxConverter(vector<Vec3f> circles) {
    vector<Rect> bboxes;
    for(Vec3i c : circles) bboxes.push_back(Rect(c[0]-c[2], c[1]-c[2], 2*c[2], 2*c[2]));
    
    return bboxes;
}

void circlesFilter(Mat img, vector<cv::Vec3f>& circles, vector<Vec3b> tableColors) {

    vector<cv::Vec3f> balls;
    const int THR = 600;

    for(Vec3f c : circles) {
        Mat mask1 = Mat::zeros(img.size(), CV_8U);
        Mat mask2 = Mat::zeros(img.size(), CV_8U);
        Mat mask3 = Mat::zeros(img.size(), CV_8U);
        Mat roi = Mat::zeros(img.size(), CV_8U);
        circle(mask1, Point(c[0], c[1]), 1.5*c[2], Scalar(255), -1);
        bitwise_not(mask1, mask1);
        circle(mask2, Point(c[0], c[1]), 2.5*c[2], Scalar(255), -1);
        bitwise_and(mask1, mask2, mask3);

        Scalar avg = mean(img, mask3);
        Vec3b mean = Vec3b(round(avg[0]), round(avg[1]), round(avg[2]));

        float min_dist = squaredEuclideanDist(mean, tableColors[0]);
        for(int i=0; i<tableColors.size(); i++){
            float dist = squaredEuclideanDist(mean, tableColors[i]);
            min_dist = (dist<min_dist ? dist : min_dist);
        }
        // cout<<min_dist<<endl;

        if(min_dist < THR) balls.push_back(c);

        // img.copyTo(roi, mask3);
        // imshow("window", roi);
        // waitKey(0);
    }

    circles.swap(balls);
}

float squaredEuclideanDist(Vec3b pixel, Vec3b center) {
    float sum = 0;
    for(int i=0; i<3; i++)
        sum += (pixel[i]-center[i]) * (pixel[i]-center[i]);
    return sum;
}


Vec3b getClusterCentroid(Vec3b pixel, vector<Vec3b> centers) {
    Vec3b ret = centers[0];
    float minDist = squaredEuclideanDist(pixel, centers[0]);
    for(int i=1; i<centers.size(); i++) {
        float dist = squaredEuclideanDist(pixel, centers[i]);
        if(dist < minDist) {
            minDist = dist;
            ret = centers[i];
        }
    }
    return ret;
}

Mat subtractTable(Mat img) {
    // cvtColor(img, img, COLOR_BGR2HSV);
    // imshow("window", img);
    // waitKey(0);
    Vec3b tableColor = getTableColor(img);
    Mat out = Mat(img);

    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            Vec3b pixel = out.at<Vec3b>(i, j);
            if(pixel == Vec3b(0, 0, 0)) continue;
            out.at<Vec3b>(i, j) = Vec3b(abs(pixel[0]-tableColor[0]), abs(pixel[1]-tableColor[1]), abs(pixel[2]-tableColor[2]));
        }
    }

    return out;
}

// To be perfected...
// Mat quantizeColors(Mat img, int delta) {

    // Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();
    // Mat mask;
    // pBackSub->apply(img, mask);
    // imshow("window", mask);
    // waitKey(0);

    // Mat img_hsv;
    // cvtColor(img, img_hsv, COLOR_BGR2HSV);
    // imshow("window", img_hsv);
    // waitKey(0);

    // Vec3b tableColor = getTableColor(img);

    // // int delta = 50;
    // Scalar lowTableColor( tableColor[0]-delta, tableColor[1]-delta, tableColor[2]-(1.9*delta));   
    // Scalar highTableColor(tableColor[0]+delta, tableColor[1]+delta, tableColor[2]+(1.9*delta));
    // // Scalar lowTableColor(30, 180, 80);
    // // Scalar highTableColor(130, 255, 255);
    // Mat mask;
    // inRange(img_hsv, lowTableColor, highTableColor, mask);
    // imshow("window", mask);
    // waitKey(0);

    // bitwise_not(mask, mask); // focus on the balls
    // imshow("window", mask);
    // waitKey(0);

    // Mat crop;
    // bitwise_and(img, img, crop, mask);
    // imshow("window", crop);
    // waitKey(0);


    // return crop;


    // vector<Vec3b> true_colors;
    // true_colors.push_back(Vec3b(255, 255, 255)); // White
    // true_colors.push_back(Vec3b(255, 207, 0));   // Yellow
    // true_colors.push_back(Vec3b(0, 0, 255));     // Blue
    // true_colors.push_back(Vec3b(238, 28, 36));   // Red
    // true_colors.push_back(Vec3b(163, 73, 164));  // Purple
    // true_colors.push_back(Vec3b(255, 127, 39));  // Orange
    // true_colors.push_back(Vec3b(0, 128, 0));     // Green
    // true_colors.push_back(Vec3b(128, 0, 0));     // Burgundy/Maroon
    // true_colors.push_back(Vec3b(0, 0, 0));       // Black

    // vector<Vec3b> colors = true_colors;
    // int levels = 40+1;
    // for(Vec3b c : true_colors) {
    //     int diff1 = c[0];
    //     int diff2 = c[1];
    //     int diff3 = c[2];
    //     for(int i=levels-1; i>levels-10; i--) {
    //         colors.push_back(Vec3b((diff1/levels)*i,
    //                                (diff2/levels)*i,
    //                                (diff3/levels)*i));
    //     }
    // }

    // Vec3b tableColor = getTableColor(img);

    // vector<Vec3b> colors;
    // int levels = 50;
    // for(int i=0; i<=255; i+=255/levels) {
    //     colors.push_back(Vec3b(i, i, i));
    // }
    // colors.push_back(Vec3b(255, 255, 255));

    // colors.push_back(tableColor);


//     Mat out = Mat(img);

//     for(int i=0; i<out.rows; i++) {
//         for(int j=0; j<out.cols; j++) {
//             Vec3b pixel = out.at<Vec3b>(i, j);
//             if(pixel == Vec3b(0, 0, 0)) continue;
//             out.at<Vec3b>(i, j) = Vec3b(abs(pixel[0]-tableColor[0]), abs(pixel[1]-tableColor[1]), abs(pixel[2]-tableColor[2]));
//             // if(pixel == Vec3b(0, 0, 0)) continue;
//             // if(squaredEuclideanDist(pixel, tableColor) < 5000) out.at<Vec3b>(i, j) = tableColor;
//             // else out.at<Vec3b>(i, j) = Vec3b(0,0,0);
//             // Vec3b centroid = getClusterCentroid(pixel, colors);
//             // if(squaredEuclideanDist(centroid, pixel) < squaredEuclideanDist(tableColor, pixel)) {
//             //     if(squaredEuclideanDist(centroid, pixel) > 0.7 * squaredEuclideanDist(tableColor, pixel))out.at<Vec3b>(i, j) = Vec3b(0,0,0);
//             //     else out.at<Vec3b>(i, j) = centroid;
//             // }
//             // else out.at<Vec3b>(i, j) = tableColor;
//         }
//     }

//     return out;
// }