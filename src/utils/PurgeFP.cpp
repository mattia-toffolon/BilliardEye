#include <opencv2/highgui.hpp>
#include "utils/PurgeFP.hpp"
#include "utils/drawBBoxes.hpp"
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

    // int median = areas[areas.size()/2];
    for(int i = 1; i < label_count; i++){
        //if(stats.at<int>(i,CC_STAT_AREA) > 3*median){
        //    eliminate.insert(i);
        //}
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

    // 0 FP, 3 FN
    const float THR = 0.24;
    const float MULT = 1.1;
    const float UPD = 0.3;

    vector<Rect> exp_bboxes = expandBBoxes(bboxes, MULT);
    drawBBoxes(final, exp_bboxes);

    bool previously_comp = !prev_vals.empty();

    vector<Rect> filtered_bboxes;
    for(int i=0; i<bboxes.size(); i++) {
        Mat cur = final(exp_bboxes[i]);
        float fill = sum(cur)[0]/(cur.rows*cur.cols*255.0);
        if(!previously_comp) prev_vals.push_back(fill);
        if(fill > THR){
            if(previously_comp && abs(fill-prev_vals[i] < UPD)) continue;
            filtered_bboxes.push_back(bboxes[i]);
        }
        else{
            cout<<bboxes[i]<<"  -  "<<sum(cur)[0]/(cur.rows*cur.cols*255.0)<<endl;
        }
    }
    return filtered_bboxes;
}