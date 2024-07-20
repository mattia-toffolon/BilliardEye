// AUTHOR: Toffolon Mattia

#include <iostream>
#include <opencv2/imgproc.hpp>
#include "utils/getTableColor.hpp"

using namespace std;
using namespace cv;

Vec3b getTableColor(Mat img) {
    int counter = 0;
    Scalar acc(0, 0, 0);
    // The table cloth color is estimated as the mean non-zero value pixel intensity and it is later returned
    for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if(img.at<Vec3b>(i, j) == Vec3b(0, 0, 0)) continue;
            counter++;
            acc[0] += img.at<Vec3b>(i, j)[0];
            acc[1] += img.at<Vec3b>(i, j)[1];
            acc[2] += img.at<Vec3b>(i, j)[2];
        }
    }
    if(counter==0) return Vec3b(0,0,0);
    return Vec3b(acc[0]/counter, acc[1]/counter, acc[2]/counter);
}