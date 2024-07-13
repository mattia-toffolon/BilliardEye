#include <iostream>
#include <string>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include "utils/drawBBoxes.hpp"
#include "utils/perfTesting.h"

using namespace cv;
using namespace std;

void fillTypes(const vector<Ball> &all, vector<Rect> &black, vector<Rect> &cue, vector<Rect> &fill, vector<Rect> &striped){
    for(auto b : all){
        switch (b.type) {
            case BallType::CUE : cue.push_back(b.bbox); 
            case BallType::EIGHT : black.push_back(b.bbox); 
            case BallType::SOLID : fill.push_back(b.bbox); 
            case BallType::STRIPED : striped.push_back(b.bbox); 
        }
    }
}
int main(int argc, char** argv) {
    if(argc < 3){
        std::cout << "not enough an arguments provided";
        exit(1);
    }
    Rect r1(0,0, 10, 10);
    Rect r2(6,6, 6, 6);
    std::cout << (r1 & r2) << std::endl;
    int samples = std::stoi(argv[2]);
    std::string directory = "/sample";
    std::string ground_mask = "/ground_mask";
    std::string predicted_mask = "/predicted_mask";
    std::string predicted_ball_last = "/predicted_balls_last";
    std::string ground_balls_last = "/ground_balls_last";
    std::string predicted_ball_first = "/predicted_balls_first";
    std::string ground_balls_first = "/ground_balls_first";
    std::string txt = ".txt";
    std::string png = ".png";
    
    std::vector<Mat> ground_truth_masks, predicted_masks;
    std::vector<std::vector<Ball>> ground_truth_balls_first, predicted_balls_first,ground_truth_balls_last, predicted_balls_last;
    for(int i = 0; i < samples; i++){
        ground_truth_masks.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask + png, IMREAD_GRAYSCALE));
        predicted_masks.push_back(imread(argv[1] + directory + std::to_string(i) + predicted_mask + png, IMREAD_GRAYSCALE));
        ground_truth_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_first + txt));
        predicted_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_first + txt));
        ground_truth_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_last + txt));
        predicted_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_last + txt));
    }
    //insert performance computations here
    double threshold = 0.5;
    std::vector<float> precisionsf, precisionsl, precisionMask;
    std::vector<float> ioufcue,ioufblack,iouffull,ioufstripe
                        ,ioulcue,ioulblack,ioulfull,ioulstripe;
    for(int i = 0; i < samples; i ++){
        std::vector<Ball> curft, curfp, curlt, curlp;
        std::vector<Rect> curftr, curfpr, curltr, curlpr;
        std::vector<Rect>
            cueft, cuefp, cuelt, cuelp,
            blackft, blackfp, blacklt, blacklp,
            stripefp, stripeft,stripelt, stripelp,
            fullft, fullfp, fulllt, fulllp;
        curft = ground_truth_balls_first[i];
        curfp = predicted_balls_first[i];
        curlt = ground_truth_balls_last[i];
        curlp = predicted_balls_last[i];
        fillTypes(curft, blackft, cueft, fullft, stripeft);
        fillTypes(curlt, blacklt, cuelt, fulllt, stripelt);
        fillTypes(curfp, blackfp, cuefp, fullfp, stripefp);
        fillTypes(curlp, blacklp, cuelp, fulllp, stripelp);

        for(int i = 0; i < curft.size(); i ++){
            curftr.push_back(curft[i].bbox);
            curfpr.push_back(curfp[i].bbox);
            curltr.push_back(curlt[i].bbox);
            curlpr.push_back(curlp[i].bbox);
        }

        drawBBoxes(imread(argv[1] + directory + std::to_string(i) + "/frame_first.png"), curftr);
        drawBBoxes(imread(argv[1] + directory + std::to_string(i) + "/frame_first.png"), curfpr);
        std::map<float, float> resf = precisionRecallCurve(curfpr, curftr, threshold);
        std::map<float, float> resl = precisionRecallCurve(curlpr, curltr, threshold);
        double curprecf =  averagePrecision(resf);
        double curprecl =  averagePrecision(resl);
        precisionsf.push_back(curprecf);
        precisionsl.push_back(curprecl);
        precisionMask.push_back(intersectionOverUnion(ground_truth_masks[i], predicted_masks[i]));
        //cue balls
        vector<float> cuefiou, cueliou;
        cuefiou = manyToManyIoU(cueft, cuefp);
        cueliou = manyToManyIoU(cuelt, cuelp);
        ioufcue.push_back(std::accumulate(cuefiou.begin(),cuefiou.end(), 0));
        ioulcue.push_back(std::accumulate(cueliou.begin(),cueliou.end(), 0));
        //black balls
        vector<float> blackfiou, blackliou;
        blackfiou = manyToManyIoU(blackft, blackfp);
        blackliou = manyToManyIoU(blacklt, blacklp);
        ioufblack.push_back(std::accumulate(blackfiou.begin(),blackfiou.end(), 0));
        ioulblack.push_back(std::accumulate(blackliou.begin(),blackliou.end(), 0));
        //full balls
        vector<float> fullfiou, fullliou;
        fullfiou = manyToManyIoU(fullft, fullfp);
        fullliou = manyToManyIoU(fulllt, fulllp);
        iouffull.push_back(std::accumulate(fullfiou.begin(),fullfiou.end(), 0));
        ioulfull.push_back(std::accumulate(fullliou.begin(),fullliou.end(), 0));
        //striped
        vector<float> stripefiou, stripeliou;
        stripefiou = manyToManyIoU(stripeft, stripefp);
        stripeliou = manyToManyIoU(stripelt, stripelp);
        ioufstripe.push_back(std::accumulate(stripefiou.begin(),stripefiou.end(), 0));
        ioulstripe.push_back(std::accumulate(stripeliou.begin(),stripeliou.end(), 0));
    }
    std::cout << "precision first " << precisionsf[0] << std::endl;
    std::cout << "precision last " << precisionsl[0] << std::endl;
    std::cout << "precision mask " << precisionMask[0] << std::endl;
}
