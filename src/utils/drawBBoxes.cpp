#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "utils/drawBBoxes.hpp"

using namespace cv;
using namespace std;

void drawBBoxes(Mat img, vector<Rect> bboxes) {
    if(img.data==NULL) {
        cout<<"Empty Image given";
        return;
    }

    Mat tmp = img.clone();

    for(Rect r : bboxes) {
        Point p1 = Point(r.x, r.y);
        Point p2 = Point(r.x+r.width, r.y+r.height);
        rectangle(tmp, p1, p2, Scalar(0,0,255), 2, LINE_AA);
    }   

    imshow("window", tmp);
    waitKey(0);
}