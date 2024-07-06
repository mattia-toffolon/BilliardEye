#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <utils/balls.hpp>
#include <string>

using namespace cv;

int main(int argc, char** argv) {
    if(argc < 3){
        std::cout << "not enough an arguments provided";
        exit(1);
    }
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
        ground_truth_masks.push_back(imread(argv[1] + directory + std::to_string(i) + ground_mask + png));
        predicted_masks.push_back(imread(argv[1] + directory + std::to_string(i) + predicted_mask + png));
        ground_truth_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_fist + txt));
        predicted_balls_first.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_first + txt));
        ground_truth_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + ground_balls_last + txt));
        predicted_balls_last.push_back(readBallsFile(argv[1] + directory + std::to_string(i) + predicted_ball_last + txt));
    }
    //insert performance computations here
}
