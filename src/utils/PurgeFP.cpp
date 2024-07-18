#include <opencv2/highgui.hpp>
#include "utils/PurgeFP.hpp"
#include "utils/bboxesUtils.hpp"
#include "utils/perfTesting.h"
#include <vector>
#include <set>
#include <iostream>

using namespace cv;
using namespace std;

vector<Rect> purgeFP(Mat img,  Mat transform, vector<Rect> bboxes){
    Mat canny_trns, canny, img_hsv;
    cvtColor(img, img_hsv,  COLOR_BGR2HSV);
    Canny(img_hsv, canny, 270, 300);
    // imshow("window", canny);
    // waitKey(0);

    warpPerspective(canny, canny_trns, transform, canny_trns.size());
    // imshow("window", canny_trns);
    // waitKey(0);

    vector<float> fillings;

    vector<Rect> filtered_bboxes = purgeByCanny(canny_trns, transform, bboxes, fillings);

    Mat structuringElem1 = getStructuringElement(MORPH_ELLIPSE, Size(3,2));
    morphologyEx(canny_trns, canny_trns, MORPH_OPEN, structuringElem1, Point(-1,-1), 2);
    // imshow("window", canny_trns);
    // waitKey(0);

    vector<Rect> filtered_bboxes_opened = purgeByCanny(canny_trns, transform, bboxes, fillings);

    // set<Rect> bboxes_set;
    // bboxes_set.insert(filtered_bboxes.begin(), filtered_bboxes.end());
    // bboxes_set.insert(filtered_bboxes_opened.begin(), filtered_bboxes_opened.end());
    // vector<Rect> ret(bboxes_set.begin(), bboxes_set.end());

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
    // imshow("window", masked);
    // waitKey(0);

    Mat final;
    warpPerspective(masked, final, transform.inv(), masked.size());
    // imshow("window", final);
    // waitKey(0);
    // drawBBoxes(final, bboxes);

    const float THR = 0.24;
    const float MULT = 1.1;
    const float UPD = 0.3;

    vector<Rect> exp_bboxes = expandBBoxes(bboxes, MULT);
    // drawBBoxes(final, exp_bboxes);

    bool previously_comp = !prev_vals.empty();

    const float SCALE_X = 0.98;
    const float SCALE_Y = 0.85;
    Rect safe_area(canny_trns.cols*(1-SCALE_X)/2, canny_trns.rows*(1-SCALE_Y)/2, canny_trns.cols*SCALE_X, canny_trns.rows*SCALE_Y);

    vector<Rect> exp_bboxes2 = expandBBoxes(bboxes, 1.5);
    // drawBBoxes(final, exp_bboxes2);
    vector<bool> overlapping(exp_bboxes2.size(), false);
    for(int i=0; i<exp_bboxes2.size(); i++) {
        for(int j=i+1; j<exp_bboxes2.size(); j++) {
            if(intersectionOverUnion(exp_bboxes2[i], exp_bboxes2[j]) > 0) {
                overlapping[i] = true;
                overlapping[j] = true;
            }
        }
    }

    // vector<Rect> tmp;
    // tmp.push_back(safe_area);
    // drawBBoxes(canny_trns, tmp);

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
        // else{
        //     cout<<bboxes[i]<<"  -  "<<sum(cur)[0]/(cur.rows*cur.cols*255.0)<<endl;
        // }
    }
    return filtered_bboxes;
}