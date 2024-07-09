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
#include "utils/PurgeFP.hpp"
#include "recognition/transformPoints.hpp"
#include "recognition/side_recognition.hpp"

using namespace cv;
using namespace std;

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

vector<Vec3f> refineCircles(Mat img, vector<Rect> bboxes, bool draw) {
    vector<Vec3f> out;
    const int delta = 8;
    Mat roi;
    for(Rect r : bboxes) {
        Mat mask = Mat::zeros(img.size(), CV_8U);
        Point2d p1 = Point(r.x-delta, r.y-delta);
        Point2d p2 = Point(r.x+r.width+delta, r.y+r.height+delta);
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
    vector<Vec3f> tmp;
    for(Vec3f c : first) tmp.push_back(c);

    for(Vec3f c2 : second) {
        bool duplicate = false;
        for(Vec3f c1 : first) {
            float dist = (c1[0]-c2[0])*(c1[0]-c2[0]) + (c1[1]-c2[1])*(c1[1]-c2[1]);
            if(dist <= c1[2]*c1[2]) {
                duplicate = true;
                break;
            }
        }
        if(!duplicate) tmp.push_back(c2);
    }

    vector<Vec3f> out;
    for(int i=0; i<tmp.size(); i++) {
        bool duplicate = false;
        Vec3f c1 = tmp[i];
        for(int j=i+1; j<tmp.size(); j++) {
            Vec3f c2 = tmp[j];
            float dist = (c1[0]-c2[0])*(c1[0]-c2[0]) + (c1[1]-c2[1])*(c1[1]-c2[1]);
            if(dist <= c1[2]*c1[2]) {
                duplicate = true;
                break;
            }
        }
        if(!duplicate) out.push_back(c1);
    }

    return out;
}

float squaredEuclideanDist(Vec3b pixel, Vec3b center) {
    float sum = 0;
    for(int i=0; i<3; i++)
        sum += (pixel[i]-center[i]) * (pixel[i]-center[i]);
    return sum;
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

vector<Rect> getBBoxes(Mat img, Mat mask, Mat transf) {

    Mat img_BGR = img.clone();
    Mat img_HSV;
    cvtColor(img_BGR, img_HSV, COLOR_BGR2HSV);
    // imshow("window", img_BGR);
    // waitKey(0);
    // imshow("window", img_HSV);
    // waitKey(0);

    Mat crop_BGR = Mat::zeros(img_BGR.size(), img_BGR.type());
    Mat crop_HSV = Mat::zeros(img_HSV.size(), img_HSV.type());
    img_BGR.copyTo(crop_BGR, mask);
    img_HSV.copyTo(crop_HSV, mask);
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
    vector<Rect> bboxes = bboxConverter(circles);
    drawBBoxes(img, bboxes);

    // vector<Vec3f> filtered_circles = circlesFilter(img_BGR, circles, tableColors_BGR, levels, false);
    // vector<Vec3f> filtered_circles = circlesFilter(img_HSV, circles, tableColors_HSV, levels, false);

    vector<Rect> filtered_bboxes = purgeFP(img, transf, bboxes);

    // const int delta = 5;
    // vector<Rect> ret;
    // for(Rect bbox : filtered_bboxes) {
    //     ret.push_back(Rect(bbox.x-delta, bbox.y-delta, bbox.width+2*delta, bbox.height+2*delta));
    // }

    // drawBBoxes(img, filtered_bboxes);

    // vector<Vec3f> ref_circles = refineCircles(gray_HSV, filtered_bboxes, true);
    // vector<Rect> new_bboxes = bboxConverter(ref_circles);

    return filtered_bboxes;
}