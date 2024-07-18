// AUTHOR: Toffolon Mattia

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "segment/segBalls.h"
#include "utils/getTableColor.hpp"
#include "utils/bboxesUtils.hpp"
#include "utils/perfTesting.h"
#include "recognition/transformPoints.hpp"
#include "recognition/side_recognition.hpp"

using namespace cv;
using namespace std;

vector<Vec3f> circlesFinder(Mat img, double param1, double param2, int minRadius, int maxRadius, bool draw) {
    std::vector<cv::Vec3f> out;
    HoughCircles(img, out, HOUGH_GRADIENT, 1, img.rows/32, param1, param2, minRadius, maxRadius);

    if(out.empty()) return out;

    if(draw) {
        Mat old = img.clone();
        for(Vec3i c : out) {
            Point center = Point(c[0], c[1]);
            int radius = c[2];
            circle(old, center, radius, Scalar(255,255,255), 1, LINE_AA);
        }
        imshow("window", old);
        waitKey(0);
    }

    return out;
}

vector<Rect> bboxConverter(vector<Vec3f> circles) {
    vector<Rect> bboxes;
    for(Vec3i c : circles) bboxes.push_back(Rect(c[0]-c[2], c[1]-c[2], 2*c[2], 2*c[2]));
    
    return bboxes;
}

vector<Rect> refineBBoxes(Mat img, vector<Rect> bboxes, bool draw) {
    assert(bboxes.size() > 0);

    int mean_width = 0;
    for(Rect r : bboxes) mean_width += r.width;
    mean_width /= bboxes.size();
    
    const float MULT = 2;
    const int THR = 4;

    vector<Vec3f> out;
    Mat roi, mask, bgdModel, fgdModel;
    for(Rect r : bboxes) {
        Mat mask;
        Rect new_r(r.x-(mean_width*MULT-r.width)/2, r.y-(mean_width*MULT-r.width)/2, mean_width*MULT, mean_width*MULT);

        grabCut(img, mask, new_r, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);
        Mat final = mask==3;

        if(draw) {
            imshow("window", img(new_r));
            waitKey(0);
            imshow("window", final);
            waitKey(0);
        }

        vector<Vec3f> circle = circlesFinder(final, 60, 10, mean_width/3, mean_width*0.71, draw);
        if(!circle.empty() && abs(circle[0][2]-r.width/2) > THR) {
            out.push_back(circle[0]);
        }
        else {
            out.push_back(toCircle(r));
        }
    }

    return bboxConverter(out);
}

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

Mat subtractTable(Mat img) {
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
    
    Mat crop_BGR = Mat::zeros(img_BGR.size(), img_BGR.type());
    Mat crop_HSV = Mat::zeros(img_HSV.size(), img_HSV.type());
    img_BGR.copyTo(crop_BGR, mask);
    img_HSV.copyTo(crop_HSV, mask);

    Mat sub_BGR = subtractTable(crop_BGR);
    Mat sub_HSV = subtractTable(crop_HSV);

    Mat gray_BGR, gray_HSV;
    cvtColor(sub_BGR, gray_BGR, COLOR_BGR2GRAY);
    Mat HSV_levels[3];
    split(sub_HSV, HSV_levels);
    gray_HSV = HSV_levels[2];

    vector<Vec3f> circles_BGR = circlesFinder(gray_BGR, 90, 12, 5, 15);
    vector<Vec3f> circles_HSV = circlesFinder(gray_HSV, 90, 12, 5, 15);

    vector<Vec3f> circles = smartCircleMerge(circles_BGR, circles_HSV);
    
    vector<Rect> bboxes = bboxConverter(circles);

    vector<Rect> filtered_bboxes = purgeFP(img, transf, bboxes);

    vector<Rect> ref_bboxes = refineBBoxes(crop_HSV, filtered_bboxes);

    return ref_bboxes;
}

vector<Rect> purgeFP(Mat img,  Mat transform, vector<Rect> bboxes){
    Mat canny_trns, canny, img_hsv;
    cvtColor(img, img_hsv,  COLOR_BGR2HSV);
    Canny(img_hsv, canny, 270, 300);

    warpPerspective(canny, canny_trns, transform, canny_trns.size());

    vector<float> fillings;
    vector<Rect> filtered_bboxes = purgeByCanny(canny_trns, transform, bboxes, fillings);

    Mat structuringElem1 = getStructuringElement(MORPH_ELLIPSE, Size(3,2));
    morphologyEx(canny_trns, canny_trns, MORPH_OPEN, structuringElem1, Point(-1,-1), 2);

    vector<Rect> filtered_bboxes_opened = purgeByCanny(canny_trns, transform, bboxes, fillings);

    vector<Rect> ret(filtered_bboxes);
    for(const Rect& b2 : filtered_bboxes_opened) {
        if(find(filtered_bboxes.begin(), filtered_bboxes.end(), b2) == filtered_bboxes.end()) {
            ret.push_back(b2);
        }
    }

    return ret;
}

vector<Rect> purgeByCanny(Mat canny_trns,  Mat transform, vector<Rect> bboxes, vector<float>& prev_vals) {
    set<int> eliminate;
    Mat elaborated, labels, stats, centroids;
    int label_count = connectedComponentsWithStats(canny_trns, labels, stats, centroids);

    vector<int> areas;
    for(int i = 1; i < label_count; i++){
        areas.push_back(stats.at<int>(i,CC_STAT_AREA));
    }
    sort(areas.begin(), areas.end());

    for(int i = 1; i < label_count; i++){
        if(stats.at<int>(i,CC_STAT_TOP) == 0 || stats.at<int>(i,CC_STAT_TOP) + stats.at<int>(i,CC_STAT_HEIGHT) == canny_trns.rows){
            eliminate.insert(i);
        }
        else if(stats.at<int>(i,CC_STAT_LEFT) == 0 || stats.at<int>(i,CC_STAT_LEFT) + stats.at<int>(i,CC_STAT_WIDTH) == canny_trns.cols){
            eliminate.insert(i);
        } 
    }

    Mat masked(canny_trns.size(), CV_8UC1);
    for(int r = 0; r < canny_trns.rows; r++){
        for(int c = 0; c < canny_trns.cols; c++){
            if(labels.at<int>(r,c) == 0) continue;
            if(eliminate.find(labels.at<int>(r,c)) == eliminate.end()){
                masked.at<char>(r,c) = static_cast<char>(255);
            }
            else{
                masked.at<char>(r,c) = static_cast<char>(0);
            }
        }
    }

    Mat final;
    warpPerspective(masked, final, transform.inv(), masked.size());

    const float THR = 0.24;
    const float MULT = 1.1;
    const float UPD = 0.3;

    vector<Rect> exp_bboxes = expandBBoxes(bboxes, MULT);

    bool previously_comp = !prev_vals.empty();

    const float SCALE_X = 0.98;
    const float SCALE_Y = 0.85;
    Rect safe_area(canny_trns.cols*(1-SCALE_X)/2, canny_trns.rows*(1-SCALE_Y)/2, canny_trns.cols*SCALE_X, canny_trns.rows*SCALE_Y);

    vector<Rect> exp_bboxes2 = expandBBoxes(bboxes, 1.5);
    vector<bool> overlapping(exp_bboxes2.size(), false);

    for(int i=0; i<exp_bboxes2.size(); i++) {
        for(int j=i+1; j<exp_bboxes2.size(); j++) {
            if(intersectionOverUnion(exp_bboxes2[i], exp_bboxes2[j]) > 0) {
                overlapping[i] = true;
                overlapping[j] = true;
            }
        }
    }

    vector<Rect> filtered_bboxes;
    for(int i=0; i<bboxes.size(); i++) {

        Mat cur = final(exp_bboxes[i]);
        float fill = sum(cur)[0]/(cur.rows*cur.cols*255.0);
        if(!previously_comp) prev_vals.push_back(fill);

        Point2f r_center = Point2f(bboxes[i].x+bboxes[i].width/2, bboxes[i].y+bboxes[i].height/2);
        vector<Point2f> r_center_vect{r_center};
        perspectiveTransform(r_center_vect, r_center_vect, transform);
        Point2f new_r_center = r_center_vect[0];

        if(safe_area.x <= new_r_center.x && new_r_center.x <= safe_area.x+safe_area.width &&
           safe_area.y <= new_r_center.y && new_r_center.y <= safe_area.y+safe_area.height && !overlapping[i]) {

            filtered_bboxes.push_back(bboxes[i]);
            continue;
        }

        if(fill > THR){
            if(previously_comp && abs(fill-prev_vals[i]) < UPD) continue;
            filtered_bboxes.push_back(bboxes[i]);
        }
    }

    return filtered_bboxes;
}