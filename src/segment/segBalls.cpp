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

const string WINDOW_NAME = "WINDOW";

vector<Rect> getBBoxes(Mat img, Mat mask, Mat transf) {

    // BGR and HSV versions of the image are obtained
    Mat img_BGR = img.clone();
    Mat img_HSV;
    cvtColor(img_BGR, img_HSV, COLOR_BGR2HSV);
    
    // Regions outside the table area are masked
    Mat crop_BGR = Mat::zeros(img_BGR.size(), img_BGR.type());
    Mat crop_HSV = Mat::zeros(img_HSV.size(), img_HSV.type());
    img_BGR.copyTo(crop_BGR, mask);
    img_HSV.copyTo(crop_HSV, mask);

    // Custom local operation is applied to image pixels
    Mat sub_BGR = subtractTable(crop_BGR);
    Mat sub_HSV = subtractTable(crop_HSV);

    // Grayscale version of the resulting images are obtained
    Mat gray_BGR, gray_HSV;
    cvtColor(sub_BGR, gray_BGR, COLOR_BGR2GRAY);
    Mat HSV_levels[3];
    split(sub_HSV, HSV_levels);
    gray_HSV = HSV_levels[2];

    // Circles in the images are found
    vector<Vec3f> circles_BGR = circlesFinder(gray_BGR, 90, 12, 5, 15);
    vector<Vec3f> circles_HSV = circlesFinder(gray_HSV, 90, 12, 5, 15);

    // The found circles are merged into a single set with a custom algorithm
    vector<Vec3f> circles = smartCircleMerge(circles_BGR, circles_HSV);
    
    // Circles are converted into bounding boxes
    vector<Rect> bboxes = bboxConverter(circles);

    // False positive boxes are removed
    vector<Rect> filtered_bboxes = purgeFP(img, transf, bboxes);

    // Bounding boxes are refined
    vector<Rect> ref_bboxes = refineBBoxes(crop_HSV, filtered_bboxes);

    return expandBBoxes2(ref_bboxes, 4);
}

Mat subtractTable(Mat img) {
    // Table cloth color is estimated
    Vec3b tableColor = getTableColor(img);
    Mat out = img.clone();
    // Pixels are set at the absolute value of the difference between pixel intensity and cloth intensity 
    for(int i=0; i<out.rows; i++) {
        for(int j=0; j<out.cols; j++) {
            Vec3b pixel = out.at<Vec3b>(i, j);
            if(pixel == Vec3b(0, 0, 0)) continue;
            out.at<Vec3b>(i, j) = Vec3b(abs(pixel[0]-tableColor[0]), abs(pixel[1]-tableColor[1]), abs(pixel[2]-tableColor[2]));
        }
    }

    return out;
}

vector<Vec3f> circlesFinder(Mat img, double param1, double param2, int minRadius, int maxRadius, bool draw) {
    // Circles in the image are found using HoughCircles
    std::vector<cv::Vec3f> out;
    HoughCircles(img, out, HOUGH_GRADIENT, 1, img.rows/32, param1, param2, minRadius, maxRadius);

    if(out.empty()) return out;

    // Printing of image with found circles
    if(draw) {
        Mat old = img.clone();
        for(Vec3i c : out) {
            Point center = Point(c[0], c[1]);
            int radius = c[2];
            circle(old, center, radius, Scalar(255,255,255), 1, LINE_AA);
        }
        imshow(WINDOW_NAME, old);
        waitKey(0);
    }

    return out;
}

vector<Vec3f> smartCircleMerge(vector<Vec3f> first, vector<Vec3f> second) {
    // All circles from the first vector are added
    vector<Vec3f> tmp;
    for(Vec3f c : first) tmp.push_back(c);
    // Circles from the second vector are added only if they do not overlap with circle from the first one
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
    // Additional check to remove overlapping circles 
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

vector<Rect> bboxConverter(vector<Vec3f> circles) {
    vector<Rect> bboxes;
    for(Vec3i c : circles) bboxes.push_back(Rect(c[0]-c[2], c[1]-c[2], 2*c[2], 2*c[2]));
    
    return bboxes;
}

vector<Rect> purgeFP(Mat img,  Mat transform, vector<Rect> bboxes){
    Mat canny_trns, canny, img_hsv;
    // Image in HSV format is obtained
    cvtColor(img, img_hsv,  COLOR_BGR2HSV);
    // Canny edge detector is used
    Canny(img_hsv, canny, 270, 300);
    // The transformed version of the Canny image is obtained
    warpPerspective(canny, canny_trns, transform, canny_trns.size());

    vector<float> fillings;
    // First set of filtered bounding boxes is obtained
    vector<Rect> filtered_bboxes = purgeByCanny(canny_trns, transform, bboxes, fillings);

    // Opening is performed with a custom structuring element
    Mat structuringElem1 = getStructuringElement(MORPH_ELLIPSE, Size(3,2));
    morphologyEx(canny_trns, canny_trns, MORPH_OPEN, structuringElem1, Point(-1,-1), 2);

    // Second set of filtered bounding boxes is obtained
    vector<Rect> filtered_bboxes_opened = purgeByCanny(canny_trns, transform, bboxes, fillings);

    // Boxes not previously found are added to the first set
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
    // Connected components with stats are computed on the given Canny transformed image
    int label_count = connectedComponentsWithStats(canny_trns, labels, stats, centroids);

    // Regions touching the table borders are set to be eliminated
    for(int i = 1; i < label_count; i++){
        if(stats.at<int>(i,CC_STAT_TOP) == 0 || stats.at<int>(i,CC_STAT_TOP) + stats.at<int>(i,CC_STAT_HEIGHT) == canny_trns.rows){
            eliminate.insert(i);
        }
        else if(stats.at<int>(i,CC_STAT_LEFT) == 0 || stats.at<int>(i,CC_STAT_LEFT) + stats.at<int>(i,CC_STAT_WIDTH) == canny_trns.cols){
            eliminate.insert(i);
        } 
    }

    // Pixels are set to 0 if touching the respective regions touch the table borders, 255 otherwise
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

    // Canny image in its original shape is obtained
    Mat final;
    warpPerspective(masked, final, transform.inv(), masked.size());

    // Constants used in the following sections that proved to be efficient for the task
    const float THR = 0.24;
    const float MULT1 = 1.1;
    const float MULT2 = 1.5;
    const float UPD = 0.3;

    // Bounding boxes width are multiplied by factor MULT1
    vector<Rect> exp_bboxes = expandBBoxes(bboxes, MULT1);

    // Check to verify whereas this step was performed previously
    bool previously_comp = !prev_vals.empty();

    // "Safe area" definition with respect to table transformed image width and height
    const float SCALE_X = 0.98;
    const float SCALE_Y = 0.85;
    Rect safe_area(canny_trns.cols*(1-SCALE_X)/2, canny_trns.rows*(1-SCALE_Y)/2, canny_trns.cols*SCALE_X, canny_trns.rows*SCALE_Y);

    // Bounding boxes are width are multiplied by factor MULT2
    vector<Rect> exp_bboxes2 = expandBBoxes(bboxes, MULT2);
    vector<bool> overlapping(exp_bboxes2.size(), false);

    // Bounding boxes are labeled as overlapping or non-overlapping
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

        // Percentage of white filling of the bounding boxes is computed and added to a vector is not previously done
        Mat cur = final(exp_bboxes[i]);
        float fill = sum(cur)[0]/(cur.rows*cur.cols*255.0);
        if(!previously_comp) prev_vals.push_back(fill);

        // Box center is projected in the trasfomed table image
        Point2f r_center = Point2f(bboxes[i].x+bboxes[i].width/2, bboxes[i].y+bboxes[i].height/2);
        vector<Point2f> r_center_vect{r_center};
        perspectiveTransform(r_center_vect, r_center_vect, transform);
        Point2f new_r_center = r_center_vect[0];

        // If the projected center lies in the "safe area" Rect, the box is considered True Positive and returned
        if(safe_area.x <= new_r_center.x && new_r_center.x <= safe_area.x+safe_area.width &&
           safe_area.y <= new_r_center.y && new_r_center.y <= safe_area.y+safe_area.height && !overlapping[i]) {

            filtered_bboxes.push_back(bboxes[i]);
            continue;
        }

        // If the box filling is above the THR and the difference with the prev computed one is below UPD, the box is considered True Positive and returned
        if(fill > THR){
            if(previously_comp && abs(fill-prev_vals[i]) < UPD) continue;
            filtered_bboxes.push_back(bboxes[i]);
        }
    }

    return filtered_bboxes;
}

vector<Rect> refineBBoxes(Mat img, vector<Rect> bboxes, bool draw) {
    assert(bboxes.size() > 0);

    // Mean box width is computed
    int mean_width = 0;
    for(Rect r : bboxes) mean_width += r.width;
    mean_width /= bboxes.size();
    
    // Constants used in the following sections that proved to be efficient for the task
    const float MULT = 2;
    const int THR = 4;

    vector<Vec3f> out;
    Mat roi, mask, bgdModel, fgdModel;
    for(Rect r : bboxes) {
        Mat mask;
        // Expanded box is used to define the region of interest
        Rect new_r(r.x-(mean_width*MULT-r.width)/2, r.y-(mean_width*MULT-r.width)/2, mean_width*MULT, mean_width*MULT);
        // GrabCut is used to perform foreground extraction
        grabCut(img, mask, new_r, bgdModel, fgdModel, 3, GC_INIT_WITH_RECT);
        Mat final = mask==3;

        if(draw) {
            imshow(WINDOW_NAME, img(new_r));
            waitKey(0);
            imshow(WINDOW_NAME, final);
            waitKey(0);
        }

        // Circles are found on the GrabCut output
        vector<Vec3f> circle = circlesFinder(final, 60, 10, mean_width/3, mean_width*0.71, draw);
        // If a circle is found and its radius is much different from half of the original box width, the new circle is keep, rejected otherwise
        if(!circle.empty() && abs(circle[0][2]-r.width/2) > THR) {
            out.push_back(circle[0]);
        }
        else {
            out.push_back(toCircle(r));
        }
    }

    // Updated circles are converted into boxes and returned
    return bboxConverter(out);
}