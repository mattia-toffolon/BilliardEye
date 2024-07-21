// AUTHOR: Toffolon Mattia

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/bboxesUtils.hpp"

using namespace cv;
using namespace std;

const string WINDOW_NAME = "WINDOW";

Rect toRect(Vec3f circle) {    
    // Given circle is converted to box (Rect) and returned
    return Rect(circle[0]-circle[2], circle[1]-circle[2], 2*circle[2], 2*circle[2]);
}

Vec3f toCircle(Rect box) {
    // Given box is converted to circle (Vec3f) and returned
    return Vec3f(box.x + box.width/2.0, box.y + box.height/2.0, box.width/2.0);
}

void drawBBoxes(Mat img, vector<Rect> bboxes) {
    if(img.data==NULL) {
        cout<<"Empty Image given";
        return;
    }

    Mat tmp = img.clone();

    // Given boxes are drawn on a copy of the given image which is then shown
    for(Rect r : bboxes) {
        Point p1 = Point(r.x, r.y);
        Point p2 = Point(r.x+r.width, r.y+r.height);
        rectangle(tmp, p1, p2, Scalar(0,255,0), 1, LINE_AA);
    }   

    imshow(WINDOW_NAME, tmp);
    waitKey(0);
}

void drawBBoxesCanvas(Mat img, vector<Rect> bboxes1, vector<Rect> bboxes2) {
    if(img.data==NULL) {
        cout<<"Empty Image given";
        return;
    }

    // First set of boxes are drawn on a copy of the given image
    Mat tmp1 = img.clone();
    for(Rect r : bboxes1) {
        Point p1 = Point(r.x, r.y);
        Point p2 = Point(r.x+r.width, r.y+r.height);
        rectangle(tmp1, p1, p2, Scalar(0,0,255), 1, LINE_AA);
    } 

    // Second set of boxes are drawn on another copy of the given image
    Mat tmp2 = img.clone();
    for(Rect r : bboxes2) {
        Point p1 = Point(r.x, r.y);
        Point p2 = Point(r.x+r.width, r.y+r.height);
        rectangle(tmp2, p1, p2, Scalar(0,0,255), 1, LINE_AA);
    }   

    // Produced images are put side by side and shown
    Mat canvas = Mat::zeros(img.rows, 2*img.cols, img.type());
    Mat roi1 = canvas(Rect(0, 0, tmp1.cols, tmp1.rows));
    tmp1.copyTo(roi1);
    Mat roi2 = canvas(Rect(tmp2.cols, 0, tmp2.cols, tmp2.rows));
    tmp2.copyTo(roi2);
    imshow(WINDOW_NAME, canvas);
    waitKey(0);
}

vector<Rect> expandBBoxes(vector<Rect> bboxes, const float MULT) {
    vector<Rect> ret;
    // Given boxes are returned with expanded width of constant MULT
    for(auto b : bboxes){
        Rect new_b(b.x-(b.width*MULT-b.width)/2, b.y-(b.width*MULT-b.width)/2, b.width*MULT, b.width*MULT);
        ret.push_back(new_b);
    }

    return ret;
}

vector<Rect> expandBBoxes2(vector<Rect> bboxes, const int DELTA) {
    vector<Rect> ret;
    // Given boxes are returned with expanded width of constant MULT
    for(auto b : bboxes){
        Rect new_b(b.x-DELTA/2, b.y-DELTA/2, b.width+DELTA, b.width+DELTA);
        ret.push_back(new_b);
    }

    return ret;
}