#include "utils/getTableColor.hpp"

using namespace cv;

// Returns the mean values of non-zero pixels
// note: should be used on a previously masked image to avoid considering the background
Vec3b getTableColor(Mat& img) {
    int counter = 0;
    Scalar acc(0, 0, 0);
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

// Same but for the only for the upper half
Vec3b getUpperTableColor(Mat& img) {
    int counter = 0;
    Scalar acc(0, 0, 0);
    for(int i=0; i<img.rows/2; i++) {
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

// Same but for the only for the lower half
Vec3b getLowerTableColor(Mat& img) {
    int counter = 0;
    Scalar acc(0, 0, 0);
    for(int i=img.rows/2; i<img.rows; i++) {
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