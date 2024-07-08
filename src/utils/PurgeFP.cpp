#include <opencv2/highgui.hpp>
#include "utils/PurgeFP.hpp"
#include "utils/drawBBoxes.hpp"
#include <vector>
#include <set>

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

    // vector<vector<Point>> contours;
    // findContours(canny, contours, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
    // Mat contours_img(canny.size(), CV_8UC3);
    // drawContours(contours_img, contours, -1, Scalar(0,255,0), 3);

    // imshow("window", contours_img);
    // waitKey(0);

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
    const float THR = 0.26;

    vector<Rect> filtered_bboxes;
    for(auto b : bboxes){
        Mat cur = final(b);
        if(sum(cur)[0]/(cur.rows*cur.cols*255.0) > THR){
            filtered_bboxes.push_back(b);
        }
    }
    return filtered_bboxes;
}
