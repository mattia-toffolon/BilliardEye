#include "utils/perfTesting.h"
#include "recognition/ballIdentifier.h"
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include <string>
#include <numeric>

using namespace cv;
using namespace std;

// Utility main written to produce some images to showcase classification

string truthPath = "data/game1_clip1/bounding_boxes/frame_first_bbox.txt";
string framePath = "data/game1_clip1/frames/frame_first.png";

int main()
{
    vector<Ball> trueBalls = readBallsFile(truthPath);
    Mat image = imread(framePath);

    Ball ball = trueBalls[2];

    Mat grayCrop;
    cvtColor(image(ball.bbox),grayCrop,COLOR_RGB2GRAY);
    Mat magnified = magnifyImg(grayCrop,20);

    imwrite("report/imgs/difficult_solid.png",magnified);

    namedWindow("W");
    imshow("W",magnified);
    waitKey(0);

    getBallType(image(ball.bbox));

    //imwrite("report/imgs/bad_otsu.png",otsued);
}