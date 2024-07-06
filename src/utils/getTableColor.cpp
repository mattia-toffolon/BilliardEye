#include <iostream>
#include <opencv2/imgproc.hpp>
#include "utils/getTableColor.hpp"


using namespace std;
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

vector<Vec3b> getTableColors(Mat& img, int levels) {
    vector<Vec3b> colors;
    int delta = (int)(img.rows/levels);
    Mat mask, roi;
    for(int i=0; i<levels; i++) {
        mask = Mat::zeros(img.size(), CV_8U);
        Point2d p1 = Point(0, i*delta);
        Point2d p2 = Point(img.cols-1, (i+1)*delta);
        rectangle(mask, p2, p1, Scalar(255), -1);
        roi = Mat::zeros(img.size(), CV_8U);
        img.copyTo(roi, mask);
        // imshow("window", roi);
        // waitKey(0);
        colors.push_back(getTableColor(roi));
    }
    return colors;
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

// returns a vector of color intensity (brightness) variations of the table color
// if the image is in HSV format, only the third channel (value=brightness) is altered
vector<Vec3b> getTableColorVariations(Mat& img, bool HSV) {
    vector<Vec3b> colors;
    Vec3b tc = getTableColor(img);
    // cout<<tc<<" "<<HSV<<endl;
    // colors.push_back(tc);
    // imshow("window", img);
    // waitKey(0);

    const int delta = 50;
    for(int i = -delta; i <= delta; i+=5) {
        if(HSV) {
            if(tc[2]+i<0 || tc[2]+i>255) continue;
            colors.push_back(Vec3b(tc[0], tc[1], tc[2]+i));
        }
        else {
            if(tc[0]+i<0 || tc[0]+i>255 || tc[1]+i<0 || tc[1]+i>255 || tc[2]+i<0 || tc[2]+i>255) continue;
            colors.push_back(Vec3b(tc[0]+i, tc[1]+i, tc[2]+i));
        }
    }

    return colors;
}