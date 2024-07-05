#include <cmath>
#include <cstdlib>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/features2d.hpp>
#include "recognition/ballIdentifier.h"
#include "recognition/side_recognition.hpp"
#include "rendering/render_table.hpp"
#include "segment/segBalls.h"
#include "segment/segTable.h"
#include "tracking/TrackBalls.hpp"
#include "utils/VideoReader.hpp"
#include "utils/drawBBoxes.hpp"
#include "utils/transformPoints.hpp"
using namespace cv;
const std::string WINDOW_NAME = "window_main";
int main(int argc, char** argv) {
    if(argc < 2){
        std::cout << "not enough an arguments provided";
        exit(1);
    }
    std::cout<<argv[1]<<std::endl;
    VideoReader vid(argv[1]);
    Mat img = vid.lastFrame();
    Mat mask;
    auto points = find_table(img, mask);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, gray, 100,300);
    bool rotate = isShortFirst(getRotatedborders(points, gray));
    imshow(WINDOW_NAME, mask);
    waitKey(0);
    Mat trans = transPoints(points, img.cols, img.rows,!rotate);
    Mat show;
    warpPerspective(img, show, trans, img.size());
    imshow(WINDOW_NAME,show);
    waitKey(0);

    img = vid.nextFrame();
    Mat segmentedTable = Mat::zeros(img.size(), img.type());
    img.copyTo(segmentedTable, mask);
    Mat sub = subtractTable(segmentedTable);
    cvtColor(sub, gray, COLOR_BGR2GRAY);

    std::vector<Vec3f> circles = circlesFinder(gray, HOUGH_GRADIENT, 1, gray.rows/32, 90, 12, 5, 15, false);
    std::vector<Rect> bboxes = bboxConverter(circles);
    std::vector<Rect2d> bboxes2;
    std::vector<Ball> balls = classifyBalls(img, bboxes);
    drawBBoxes(img, bboxes);
    TrackBalls tracker(img, balls);
    int width = img.cols;
    int height = img.rows;
    vid = VideoReader(argv[1]);
    TableRenderer rend(vid, tracker, balls, trans, width, height);
    auto vid2 = VideoReader(argv[1]);
    while(1){
        Mat fr = rend.nextFrame();
        if(fr.rows == 0){
            break;
        }
        imshow(WINDOW_NAME, fr);
        waitKey(0);
    }
    std::string filename = "/balls.txt";
    writeBallsFile(argv[2] + filename, rend.getBalls());
    std::string imgname = "/mask.png";
    cv::imwrite(argv[2] + imgname[2], mask);
    cv::Mat layer = cv::Mat::zeros(img.size(), CV_8UC3);
    std::vector<std::vector<Point>> poli;
    std::vector<Point> poli1;
    for(int i = 0; i < 4; i ++){
        poli1.push_back(points[i]);
    }
    Mat out;
    poli.push_back(poli1);
    cv::fillPoly(layer, poli, Scalar(0,0,255));
    cv::addWeighted(img, 0.5, layer, 0.5, 0, out);
    imshow(WINDOW_NAME,out);
    waitKey(0);
    return 0;
}
