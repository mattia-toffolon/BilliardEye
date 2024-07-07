#include "utils/PurgeFP.hpp"
#include "utils/PurgeFP.hpp"
#include <vector>
#include <set>
using namespace cv;
std::vector<Rect> purgeFP(Mat img,  Mat trnas, std::vector<Rect> bboxes){
    Mat can, cannie,img2;
    cvtColor(img, img2,  COLOR_BGR2HSV);
    Canny(img2,cannie, 250, 300);
    warpPerspective(cannie, can, trnas, can.size());
    std::vector<std::vector<Point>> contours;
    findContours(cannie, contours, RETR_TREE, CHAIN_APPROX_TC89_KCOS);
    Mat cunt(img.size(), CV_8UC3);
    Mat conto(cannie.size(), CV_8UC3);
    drawContours(conto, contours, -1, Scalar(0,255,0),3);
    std::set<int> eliminate;
    Mat elaborated,labels,stats,centroids;
    int label_count = connectedComponentsWithStats(can,labels,stats,centroids);
    std::vector<int> areas;
    for(int i = 1; i < label_count; i++){
        areas.push_back(stats.at<int>(i,CC_STAT_AREA));
    }
    std::sort(areas.begin(), areas.end());
    int median = areas[areas.size()/2];
    for(int i = 1; i < label_count; i++){
        //if(stats.at<int>(i,CC_STAT_AREA) > 3*median){
        //    eliminate.insert(i);
        //}
        if(stats.at<int>(i,CC_STAT_TOP) == 0 || stats.at<int>(i,CC_STAT_TOP) + stats.at<int>(i,CC_STAT_HEIGHT) == can.rows){
            eliminate.insert(i);
        }
        else if(stats.at<int>(i,CC_STAT_LEFT) == 0 || stats.at<int>(i,CC_STAT_LEFT) + stats.at<int>(i,CC_STAT_WIDTH) == can.cols){
            eliminate.insert(i);
        }
        
    }
    Mat masked(can.size(), CV_8UC1);
    for(int r = 0; r < can.rows; r++){
        for(int c = 0; c < can.cols; c++){
            if(labels.at<int>(r,c) == 0){
                continue;
            }
            if(eliminate.find(labels.at<int>(r,c)) == eliminate.end()){
                masked.at<char>(r,c) = static_cast<char>(255);

            }
            else{
                masked.at<char>(r,c) = static_cast<char>(0);
            }
        }
    }
    Mat finalo;
    warpPerspective(masked, finalo, trnas.inv(), masked.size());
    std::vector<Rect> bboxes2;
    for(auto b : bboxes){
        Mat cur = finalo(b);
        if(sum(cur)[0]/(cur.rows*cur.cols*255.0) > 0.1){
            bboxes2.push_back(b);
        }
    }
    return bboxes2;
}
