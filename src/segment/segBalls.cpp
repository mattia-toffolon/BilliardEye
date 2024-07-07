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

        Mat old = img.clone();

        for(Vec3i c : out) {
            Point center = Point(c[0], c[1]);
            int radius = c[2];
            circle(img, center, radius, Scalar(255,255,255), 1, LINE_AA);
        }
        imshow("window", img);
        waitKey(0);

        // Mat canvas = Mat::zeros(img.rows, 2*img.cols, img.type());
        // Mat roi1 = canvas(Rect(0, 0, old.cols, old.rows));
        // old.copyTo(roi1);
        // Mat roi2 = canvas(Rect(img.cols, 0, img.cols, img.rows));
        // img.copyTo(roi2);
        // imshow("window", canvas);
        // waitKey(0);

    }

    return out;
}

vector<Rect> bboxConverter(vector<Vec3f> circles) {
    vector<Rect> bboxes;
    for(Vec3i c : circles) bboxes.push_back(Rect(c[0]-c[2], c[1]-c[2], 2*c[2], 2*c[2]));
    
    return bboxes;
}

// vector<Vec3f> circlesFilter(Mat img, vector<Vec3f> circles, vector<Vec3b> tableColors, bool draw) {

//     vector<cv::Vec3f> balls;
//     const int THR1 = 200;
//     const int THR2 = 1600;
//     Mat roi;

//     for(Vec3f c : circles) {

//         Mat mask1 = Mat::zeros(img.size(), CV_8U);
//         circle(mask1, Point(c[0], c[1]), c[2], Scalar(255), -1);
//         Scalar avg = mean(img, mask1);
//         Vec3b mean1 = Vec3b(round(avg[0]), round(avg[1]), round(avg[2]));

//         float min_dist = squaredEuclideanDist(mean1, tableColors[0]);
//         for(int i=0; i<tableColors.size(); i++){
//             float dist = squaredEuclideanDist(mean1, tableColors[i]);
//             min_dist = (dist<min_dist ? dist : min_dist);
//         }
//         if(draw) {
//             cout<<min_dist<<endl;
//             roi = Mat::zeros(img.size(), CV_8U);
//             img.copyTo(roi, mask1);
//             imshow("window", roi);
//             waitKey(0);
//         }
//         if(min_dist < THR1) continue;

//         Mat mask2 = Mat::zeros(img.size(), CV_8U);
//         Mat mask3 = Mat::zeros(img.size(), CV_8U);
//         Mat mask4 = Mat::zeros(img.size(), CV_8U);
//         circle(mask2, Point(c[0], c[1]), 1.2*c[2], Scalar(255), -1);
//         bitwise_not(mask2, mask2);
//         circle(mask3, Point(c[0], c[1]), 2.5*c[2], Scalar(255), -1);
//         bitwise_and(mask2, mask3, mask4);

//         avg = mean(img, mask4);
//         Vec3b mean2 = Vec3b(round(avg[0]), round(avg[1]), round(avg[2]));

//         min_dist = squaredEuclideanDist(mean2, tableColors[0]);
//         for(int i=0; i<tableColors.size(); i++){
//             float dist = squaredEuclideanDist(mean2, tableColors[i]);
//             min_dist = (dist<min_dist ? dist : min_dist);
//         }
//         if(draw) {
//             cout<<min_dist<<endl;
//             roi = Mat::zeros(img.size(), CV_8U);
//             img.copyTo(roi, mask4);
//             imshow("window", roi);
//             waitKey(0);
//         }
//         if(min_dist < THR2) balls.push_back(c);
//     }

//     return balls;
// }

vector<Vec3f> circlesFilter(Mat img, vector<Vec3f> circles, vector<Vec3b> tableColors, int levels, bool draw) {

    vector<cv::Vec3f> balls;
    const int THR1 = 200;
    const int THR2 = 1600;
    Mat roi;

    for(Vec3f c : circles) {

        Mat mask1 = Mat::zeros(img.size(), CV_8U);
        circle(mask1, Point(c[0], c[1]), c[2], Scalar(255), -1);
        Scalar avg = mean(img, mask1);
        Vec3b mean1 = Vec3b(round(avg[0]), round(avg[1]), round(avg[2]));

        float min_dist = squaredEuclideanDist(mean1, tableColors[(int)c[1]%levels]);
        if(draw) {
            cout<<min_dist<<endl;
            roi = Mat::zeros(img.size(), CV_8U);
            img.copyTo(roi, mask1);
            imshow("window", roi);
            waitKey(0);
        }
        if(min_dist < THR1) continue;

        Mat mask2 = Mat::zeros(img.size(), CV_8U);
        Mat mask3 = Mat::zeros(img.size(), CV_8U);
        Mat mask4 = Mat::zeros(img.size(), CV_8U);
        circle(mask2, Point(c[0], c[1]), 1.2*c[2], Scalar(255), -1);
        bitwise_not(mask2, mask2);
        circle(mask3, Point(c[0], c[1]), 2.5*c[2], Scalar(255), -1);
        bitwise_and(mask2, mask3, mask4);

        avg = mean(img, mask4);
        Vec3b mean2 = Vec3b(round(avg[0]), round(avg[1]), round(avg[2]));

        min_dist = squaredEuclideanDist(mean2, tableColors[(int)c[1]%levels]);
        if(draw) {
            cout<<min_dist<<endl;
            roi = Mat::zeros(img.size(), CV_8U);
            img.copyTo(roi, mask4);
            imshow("window", roi);
            waitKey(0);
        }
        if(min_dist < THR2) balls.push_back(c);
    }

    return balls;
}

vector<Vec3f> circlesFilter2(Mat img, vector<Vec3f> circles, bool draw) {
    vector<Vec3f> out;
    Mat roi;
    for(Vec3f c : circles) {
        Mat mask = Mat::zeros(img.size(), CV_8U);
        Point2d p1 = Point(c[0]-c[2]*1.5, c[1]-c[2]*1.5);
        Point2d p2 = Point(c[0]+c[2]*1.5, c[1]+c[2]*1.5);
        rectangle(mask, p2, p1, Scalar(255), -1);
        roi = Mat::zeros(img.size(), CV_8U);
        img.copyTo(roi, mask);
        if(draw) {
            imshow("window", roi);
            waitKey(0);
        }

        vector<Vec3f> circle = circlesFinder(roi, HOUGH_GRADIENT, 1, img.rows/32, 90, 15, 5, 15, draw);
        if(!circle.empty()) out.push_back(circle[0]);
    }
    return out;
}

// Returns a vector of circles made of all elements from "first" and all elements (circles) from
// "second" which center does not lie inside no circle from "first"
vector<Vec3f> smartCircleMerge(vector<Vec3f> first, vector<Vec3f> second) {
    vector<Vec3f> out;
    for(Vec3f c : first) out.push_back(c);

    for(Vec3f c2 : second) {
        bool duplicate = false;
        for(Vec3f c1 : first) {
            float dist = (c1[0]-c2[0])*(c1[0]-c2[0]) + (c1[1]-c2[1])*(c1[1]-c2[1]);
            if(dist <= c1[2]*c1[2]) {
                duplicate = true;
                break;
            }
        }
        if(!duplicate) out.push_back(c2);
    }
    return out;
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
    Mat out = img.clone();

    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            Vec3b pixel = out.at<Vec3b>(i, j);
            if(pixel == Vec3b(0, 0, 0)) continue;
            out.at<Vec3b>(i, j) = Vec3b(abs(pixel[0]-tableColor[0]), abs(pixel[1]-tableColor[1]), abs(pixel[2]-tableColor[2]));
        }
    }

    return out;
}

vector<Rect> getBBoxes(Mat img, Mat tableMask) {

    Mat img_BGR = img.clone();
    Mat img_HSV;
    cvtColor(img_BGR, img_HSV, COLOR_BGR2HSV);
    // imshow("window", img_BGR);
    // waitKey(0);
    // imshow("window", img_HSV);
    // waitKey(0);

    Mat crop_BGR = Mat::zeros(img_BGR.size(), img_BGR.type());
    Mat crop_HSV = Mat::zeros(img_HSV.size(), img_HSV.type());
    img_BGR.copyTo(crop_BGR, tableMask);
    img_HSV.copyTo(crop_HSV, tableMask);
    // imshow("window", crop_BGR);
    // waitKey(0);
    // imshow("window", crop_HSV);
    // waitKey(0);

    Mat sub_BGR = subtractTable(crop_BGR);
    Mat sub_HSV = subtractTable(crop_HSV);
    // imshow("window", sub_BGR);
    // waitKey(0);
    // imshow("window", sub_HSV);
    // waitKey(0);

    Mat gray_BGR, gray_HSV;
    cvtColor(sub_BGR, gray_BGR, COLOR_BGR2GRAY);
    Mat HSV_levels[3];
    split(sub_HSV, HSV_levels);
    gray_HSV = HSV_levels[2];
    // imshow("window", gray_BGR);
    // waitKey(0);
    // imshow("window", gray_HSV);
    // waitKey(0);

    // Mat local_hist_BGR, local_hist_HSV;
    // Ptr<CLAHE> clahe = createCLAHE(20, Size(2,2));
    // clahe->apply(gray_BGR, local_hist_BGR);
    // clahe->apply(gray_HSV, local_hist_HSV);
    // imshow("window", local_hist_BGR);
    // waitKey(0);
    // imshow("window", local_hist_HSV);
    // waitKey(0);

    // GaussianBlur(gray_BGR, gray_BGR, Size(3,3), 0);
    // GaussianBlur(gray_HSV, gray_HSV, Size(3,3), 0);

    vector<Vec3f> circles_BGR = circlesFinder(gray_BGR, HOUGH_GRADIENT, 1, gray_BGR.rows/32, 90, 12, 5, 15, false);
    vector<Vec3f> circles_HSV = circlesFinder(gray_HSV, HOUGH_GRADIENT, 1, gray_HSV.rows/32, 90, 12, 5, 15, false);
    // vector<Rect> bboxes_BGR = bboxConverter(circles_BGR);
    // drawBBoxes(img, bboxes_BGR);
    // vector<Rect> bboxes_HSV = bboxConverter(circles_HSV);
    // drawBBoxes(img, bboxes_HSV);

    // vector<Vec3f> circles( circles_BGR.size() + circles_HSV.size() );
    // copy(circles_BGR.begin(), circles_BGR.end(), circles.begin());
    // copy(circles_HSV.begin(), circles_HSV.end(), circles.begin() + circles_BGR.size());
    vector<Vec3f> circles = smartCircleMerge(circles_BGR, circles_HSV);
    vector<Rect> bboxes_tmp = bboxConverter(circles);
    // drawBBoxes(img, bboxes_tmp);

    // const int levels = 3;
    // vector<Vec3b> tableColors_BGR = getTableColors(crop_BGR, levels);
    // vector<Vec3b> tableColors_HSV = getTableColors(crop_HSV, levels);

    // // vector<Vec3b> tableColors_BGR = getTableColorVariations(crop_BGR, false);
    // // vector<Vec3b> tableColors_HSV = getTableColorVariations(crop_HSV, true);
    // // for(Vec3b c : tableColors_BGR) cout<<c<<" ";
    // // cout<<endl<<endl;
    // // vector<Vec3b> tableColors_HSV = getTableColorVariations(crop_HSV, true);
    // // for(Vec3b c : tableColors_HSV) cout<<c<<" ";

    // // vector<Vec3f> filtered_circles = circlesFilter2(gray_HSV, circles, false);

    // vector<Vec3f> filtered_circles = circlesFilter(img_BGR, circles, tableColors_BGR, levels, false);
    // // vector<Vec3f> filtered_circles = circlesFilter(img_HSV, circles, tableColors_HSV, levels, false);

    // // vector<Vec3f> filtered_circles( filtered_circles1.size() + filtered_circles2.size() );
    // // copy(filtered_circles1.begin(), filtered_circles1.end(), filtered_circles1.begin());
    // // copy(filtered_circles2.begin(), filtered_circles2.end(), filtered_circles.begin() + filtered_circles1.size());

    // vector<Rect> bboxes = bboxConverter(filtered_circles);
    return bboxes_tmp;
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