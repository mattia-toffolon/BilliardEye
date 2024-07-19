// AUTHOR: Artico Giovanni

#include <iostream>
#include <string>
#include <numeric>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include <vector>
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "utils/bboxesUtils.hpp"
#include "utils/perfTesting.h"

using namespace cv;
using namespace std;

void fillTypes(const vector<Ball> &all, vector<Rect> &eight, vector<Rect> &cue, vector<Rect> &solid, vector<Rect> &striped){
    for(auto b : all){
        switch (b.type) {
            case BallType::CUE : cue.push_back(b.bbox); break;
            case BallType::EIGHT : eight.push_back(b.bbox); break;
            case BallType::SOLID : solid.push_back(b.bbox); break;
            case BallType::STRIPED : striped.push_back(b.bbox); break;
        }
    }
}

Mat drawCircles(vector<Rect> balls, Size s, Scalar color = Scalar(255)){
    Mat ret = Mat::zeros(s, CV_8UC1);

    for(auto r : balls){
        Vec3f circ = toCircle(r);
        circle(ret, Point(circ[0],circ[1]), circ[2], color, FILLED);
    }

    return ret;
}

Mat drawCircles(vector<Ball> balls, Size s, Scalar color, Mat initial){
    Mat ret = initial.clone();

    for(auto r : balls){
        Vec3f circ = toCircle(r.bbox);
        circle(ret, Point(circ[0],circ[1]), circ[2], color, FILLED);
    }

    return ret;
}

float compute_average_precision(std::vector<Rect> prediction, std::vector<Rect> ground_truth, float threshold){
        std::map<float, float> res = precisionRecallCurve(prediction, ground_truth, threshold);
        return averagePrecision(res);
}

/*
 * letters in variable names:
 * f->First
 * l->Last
 * t->ground Truth
 * p->Predicted
 * ground refers to ground Truth
 * first and last refer to the respective frames
 */
//letters in variable names:
int main(int argc, char** argv) {
    if(argc < 3){
        std::cout << "not enough an arguments provided";
        exit(1);
    }
    int samples = std::stoi(argv[2]);
    std::string directory = "/sample";
    std::string ground_mask_first = "/ground_mask_first";
    std::string ground_mask_last = "/ground_mask_last";
    std::string predicted_mask = "/predicted_mask";
    std::string predicted_ball_last = "/predicted_balls_last";
    std::string ground_balls_last = "/ground_balls_last";
    std::string predicted_ball_first = "/predicted_balls_first";
    std::string ground_balls_first = "/ground_balls_first";
    std::string txt = ".txt";
    std::string png = ".png";
    
    std::vector<Mat> ground_truth_masks_first, ground_truth_masks_last, predicted_masks;
    std::vector<std::vector<Ball>> ground_truth_balls_first, predicted_balls_first,ground_truth_balls_last, predicted_balls_last;

    //read output files and ground truth
    for(int i = 0; i < samples; i++){
        ground_truth_masks_first.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask_first + png, IMREAD_GRAYSCALE));
        ground_truth_masks_last.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask_last + png, IMREAD_GRAYSCALE));
        predicted_masks.push_back(imread(argv[1] + directory + std::to_string(i) + predicted_mask + png, IMREAD_GRAYSCALE));
        ground_truth_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_first + txt));
        predicted_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_first + txt));
        ground_truth_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_last + txt));
        predicted_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_last + txt));
    }

    //iou threshold for map
    double threshold = 0.5;
    //metrics vectors

    std::vector<float> precisionsf, precisionsl,
        precisionMaskFirst,precisionMaskLast,
        precisionCueFirst,precisionCueLast,
        precisionEightFirst,precisionEightLast,
        precisionSolidFirst,precisionSolidLast,
        precisionStripedFirst,precisionStripedLast;

    std::vector<float> ioufcueseg,ioufeightseg,ioufsolidseg,ioufstripedseg
                        ,ioulcueseg,iouleightseg,ioulsolidseg,ioulstripedseg
                        ,ioufbackground, ioulbackground,
                        iouallBallsf, iouallBallsl;

    //compute performance for each sample
    for(int i = 0; i < samples; i ++){
        //current balls
        std::vector<Ball> curft, curfp, curlt, curlp;
        //current rectangles
        std::vector<Rect> curftr, curfpr, curltr, curlpr;
        //current rectangles divided by type
        std::vector<Rect>
            cueft, cuefp, cuelt, cuelp,
            eightft, eightfp, eightlt, eightlp,
            stripedfp, stripedft,stripedlt, stripedlp,
            solidft, solidfp, solidlt, solidlp;
        //initializing vectors vectors
        curft = ground_truth_balls_first[i];
        curfp = predicted_balls_first[i];
        curlt = ground_truth_balls_last[i];
        curlp = predicted_balls_last[i];
        fillTypes(curft, eightft, cueft, solidft, stripedft);
        fillTypes(curlt, eightlt, cuelt, solidlt, stripedlt);
        fillTypes(curfp, eightfp, cuefp, solidfp, stripedfp);
        fillTypes(curlp, eightlp, cuelp, solidlp, stripedlp);

        for(int i = 0; i < curft.size(); i ++){
            curftr.push_back(curft[i].bbox);
        }
        for(int i = 0; i < curfp.size(); i ++){
            curfpr.push_back(curfp[i].bbox);
        }
        for(int i = 0; i < curlt.size(); i ++){
            curltr.push_back(curlt[i].bbox);
        }
        for(int i = 0; i < curlp.size(); i ++){
            curlpr.push_back(curlp[i].bbox);
        }

        //compute iou for localization alone
        Mat ballsfpseg = drawCircles(curfpr, (predicted_masks[i]).size());
        Mat ballslpseg = drawCircles(curlpr, (predicted_masks[i]).size());
        
        Mat ballsftseg = (ground_truth_masks_first[i] >=1 & ground_truth_masks_first[i] <5 );
        Mat ballsltseg = (ground_truth_masks_last[i] >=1 & ground_truth_masks_last[i] <5 );
        waitKey(0);
        iouallBallsf.push_back(intersectionOverUnion(ballsfpseg, ballsftseg));
        iouallBallsl.push_back(intersectionOverUnion(ballslpseg, ballsltseg));
        //compute map for localization alone
        precisionsf.push_back(compute_average_precision(curfpr, curftr, threshold));
        precisionsl.push_back(compute_average_precision(curlpr, curltr, threshold));

        //compute iou background
        Mat antiBack_first = drawCircles(curfp, predicted_masks[i].size(), Scalar(1),predicted_masks[i]);
        Mat antiBack_last = drawCircles(curlp, predicted_masks[i].size(), Scalar(1),predicted_masks[i]);

        ioufbackground.push_back(intersectionOverUnion((ground_truth_masks_first[i] == 0), (antiBack_first == 0)));
        ioulbackground.push_back(intersectionOverUnion((ground_truth_masks_last[i] == 0), (antiBack_last == 0)));
        //compute iou for the table
        Mat predicted_mask_first, predicted_mask_last;
        //balls are removed from table mask for better segmentation
        predicted_mask_first = drawCircles(curfp, predicted_masks[i].size(), Scalar(0),predicted_masks[i]);
        predicted_mask_last = drawCircles(curlp, predicted_masks[i].size(), Scalar(0),predicted_masks[i]);
        precisionMaskFirst.push_back(intersectionOverUnion(ground_truth_masks_first[i] ==5, predicted_mask_first));
        precisionMaskLast.push_back(intersectionOverUnion(ground_truth_masks_last[i] == 5, predicted_mask_last));

        //cue balls
        
        vector<float> cuefiou, cueliou;

        //compute mean for segmentation, bounding boxes are considered as
        //perfectly circumscribing a circle
        Mat cueftseg, cueltseg, cuefpseg, cuelpseg;
        cueftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::CUE));
        cueltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::CUE));
        cuefpseg = drawCircles(cuefp, (predicted_masks[i]).size());
        cuelpseg = drawCircles(cuelp, (predicted_masks[i]).size());
        ioufcueseg.push_back(intersectionOverUnion(cueftseg, cuefpseg));
        ioulcueseg.push_back(intersectionOverUnion(cueltseg, cuelpseg));

        precisionCueFirst.push_back(compute_average_precision(cuefp, cueft, threshold));
        precisionCueLast.push_back(compute_average_precision(cuelp, cuelt, threshold));

        //eight balls

        Mat eightftseg, eightltseg, eightfpseg, eightlpseg;
        eightftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::EIGHT));
        eightltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::EIGHT));
        eightfpseg = drawCircles(eightfp, (predicted_masks[i]).size());
        eightlpseg = drawCircles(eightlp, (predicted_masks[i]).size());
        ioufeightseg.push_back(intersectionOverUnion(eightftseg, eightfpseg));
        iouleightseg.push_back(intersectionOverUnion(eightltseg, eightlpseg));
        
        precisionEightFirst.push_back(compute_average_precision(eightfp, eightft, threshold));
        precisionEightLast.push_back(compute_average_precision(eightlp, eightlt, threshold));

        //solid balls

        Mat solidftseg, solidltseg, solidfpseg, solidlpseg;
        solidftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::SOLID));
        solidltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::SOLID));
        solidfpseg = drawCircles(solidfp, (predicted_masks[i]).size());
        solidlpseg = drawCircles(solidlp, (predicted_masks[i]).size());
        ioufsolidseg.push_back(intersectionOverUnion(solidftseg, solidfpseg));
        ioulsolidseg.push_back(intersectionOverUnion(solidltseg, solidlpseg));

        precisionSolidFirst.push_back(compute_average_precision(solidfp, solidft, threshold));
        precisionSolidLast.push_back(compute_average_precision(solidlp, solidlt, threshold));

        //striped

        Mat stripeddftseg, stripeddltseg, stripeddfpseg, stripeddlpseg;
        stripeddftseg = (ground_truth_masks_first[i] == static_cast<char>(BallType::STRIPED));
        stripeddltseg = (ground_truth_masks_last[i] == static_cast<char>(BallType::STRIPED));
        stripeddfpseg = drawCircles(stripedfp, (predicted_masks[i]).size());
        stripeddlpseg = drawCircles(stripedlp, (predicted_masks[i]).size());
        ioufstripedseg.push_back(intersectionOverUnion(stripeddftseg, stripeddfpseg));
        ioulstripedseg.push_back(intersectionOverUnion(stripeddltseg, stripeddlpseg));

        precisionStripedFirst.push_back(compute_average_precision(stripedfp, stripedft, threshold));
        precisionStripedLast.push_back(compute_average_precision(stripedlp, stripedlt, threshold));
    }
    for(int i = 0; i < samples; i ++){
        std::cout << "sample " << i << std::endl;
        std::cout << "average precision localization (alone) first: " << precisionsf[i] << std::endl;
        std::cout << "average precision localization (alone) last: " << precisionsl[i] << std::endl;
        std::cout << "IOU localization (alone) first: " << iouallBallsf[i] << std::endl;
        std::cout << "IOU localization (alone) last: " << iouallBallsl[i] << std::endl;
        std::cout << "IoU segmentation cue first: " << ioufcueseg[i] << std::endl;
        std::cout << "IoU segmentation cue last: " << ioulcueseg[i] << std::endl;
        std::cout << "IoU segmentation eight first: " << ioufeightseg[i] << std::endl;
        std::cout << "IoU segmentation eight last: " << iouleightseg[i] << std::endl;
        std::cout << "IoU segmentation solid first: " << ioufsolidseg[i] << std::endl;
        std::cout << "IoU segmentation solid last: " << ioulsolidseg[i] << std::endl;
        std::cout << "IoU segmentation striped first: " << ioufstripedseg[i] << std::endl;
        std::cout << "IoU segmentation striped last: " << ioulstripedseg[i] << std::endl;
        std::cout << "IoU table first: " << precisionMaskFirst[i] << std::endl;
        std::cout << "IoU table last: " << precisionMaskLast[i] << std::endl;
        std::cout << "IoU background first: " << ioufbackground[i] << std::endl;
        std::cout << "IoU background last: " << ioulbackground[i] << std::endl;
        std::cout << "average precision cue first: " << precisionCueFirst[i] << std::endl;
        std::cout << "average precision cue last: " << precisionCueLast[i] << std::endl;
        std::cout << "average precision eight first: " << precisionEightFirst[i] << std::endl;
        std::cout << "average precision eight last: " << precisionEightLast[i] << std::endl;
        std::cout << "average precision solid first: " << precisionSolidFirst[i] << std::endl;
        std::cout << "average precision solid last: " << precisionSolidLast[i] << std::endl;
        std::cout << "average precision striped first: " << precisionStripedFirst[i] << std::endl;
        std::cout << "average precision striped last: " << precisionStripedLast[i] << std::endl;
        std::cout << "mIOU first: " << (ioufcueseg[i] + ioufeightseg[i] + ioufsolidseg[i] + ioufstripedseg[i] + precisionMaskFirst[i] + ioufbackground[i]) / 6.0 <<std::endl;
        std::cout << "mIOU last: " << (ioulcueseg[i] + iouleightseg[i] + ioulsolidseg[i] + ioulstripedseg[i] + precisionMaskLast[i] + ioulbackground[i]) / 6.0 <<std::endl;
        std::cout << "mAP first: " << (precisionCueFirst[i] + precisionEightFirst[i] + precisionSolidFirst[i] + precisionStripedFirst[i]) / 4.0 <<std::endl;
        std::cout << "mAP last: " << (precisionCueLast[i] + precisionEightLast[i] + precisionSolidLast[i] + precisionStripedLast[i]) / 4.0 <<std::endl;
        std::cout << std::endl;
    }
}
