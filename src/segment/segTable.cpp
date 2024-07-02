#include <cstdlib>
#include <iostream>


#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string>
#include <algorithm>

#include "segment/segTable.h"

using namespace cv;
const std::string WINDOW_NAME = "AD";



bool arelinessimilar(const struct linestr a, const struct linestr b, double thresh){
    //returns true only if the start and the end are both close
    //for this to work the two lines must be computed in the same way
    double dist1 = norm(a.start-b.start);
    double dist2 = norm(a.stop-b.start);
    double dist3 = norm(a.start-b.stop);
    double dist4 = norm(a.stop-b.stop);

    //account for inverted lines
    return (dist1 < thresh && dist4 < thresh) || (dist2 < thresh && dist3 < thresh);
}

//see nonbinarykmeans for full explanation
Mat simplekmeans(const Mat in, int k, char* colors){
    std::vector<Mat> bgr;
    Mat img(in);
    split(img, bgr);

    Mat p = Mat::zeros(img.cols*img.rows, 3, CV_32F);

    for(int i=0; i<img.cols*img.rows; i++) {
        p.at<float>(i,2) = bgr[0].data[i] / 255.0;
        p.at<float>(i,3) = bgr[1].data[i] / 255.0;
        p.at<float>(i,4) = bgr[2].data[i] / 255.0;
    }

    Mat clust = Mat::zeros(img.rows, img.cols, CV_8U);
    Mat labs,ctrs;
    kmeans(p, k, labs, TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, ctrs);

    for(int i=0; i<img.cols*img.rows; i++) {
        clust.at<char>(i/img.cols, i%img.cols) = static_cast<char>(colors[labs.at<int>(0,i)]);
    }

    return clust;
}

Mat nonbinarykmeans(const Mat in, int k, int blurSize){
    Mat img;
    
    //apply gaussianBlur to the image in input
    //this is done to remove most of the noise
    GaussianBlur(in,img, Size(blurSize,blurSize),0);

    Mat p = Mat::zeros(img.cols*img.rows, 5, CV_32F);

    std::vector<Mat> bgr;
    split(img, bgr);

    //flatten the image, transforming it in a series
    //of feature vectors
    for(int i=0; i<img.cols*img.rows; i++) {
        p.at<float>(i,0) = static_cast<float>(i%img.cols)/img.cols;
        p.at<float>(i,1) = static_cast<float>(i/img.cols)/img.rows;
        p.at<float>(i,2) = bgr[0].data[i] / 255.0;
        p.at<float>(i,3) = bgr[1].data[i] / 255.0;
        p.at<float>(i,4) = bgr[2].data[i] / 255.0;
    }

    Mat labs,ctrs;
    kmeans(p, k, labs, TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, ctrs);


    Point2f imgcenter = Point2f(img.rows/2,img.cols/2);

    //compute the cluster that is closest to the center
    std::vector<int> count(k,0);
    std::vector<float> points(k,0);
    for(int i=0; i<img.cols*img.rows; i++) {
        count[labs.at<int>(0,i)] +=1;
        points[labs.at<int>(0,i)] += std::pow(norm(Point2f(i/img.cols, i%img.cols)-imgcenter),2);
    }

    float min = points[0]/count[0];
    int index = 0;
    for(int i = 1; i < k;i++){
        float curdist = points[i]/count[i];
        if(curdist<min){
            index = i;
            min = curdist;
        }
    }

    //create the mask of the most centered cluster
    Mat clust = Mat::zeros(in.rows, in.cols, CV_8U);
    for(int i=0; i<img.cols*img.rows; i++) {
        if(labs.at<int>(0,i)==index){
            clust.at<char>(i/img.cols, i%img.cols) = static_cast<char>(255); 
        }
    }

    return clust;
}

Mat greatest_island(Mat input){
    Mat labels, stats, centroids;
    Mat in;

    //apply open in order to remove
    //weak connections
    int kerSize = 9;
    Mat kernel = Mat::ones(kerSize, kerSize, CV_8U);
    morphologyEx(input, in, MORPH_OPEN, kernel);
    //erode(input, in, kernel);

    //compute connected components statistics
    int nlabels = cv::connectedComponentsWithStats(in, labels, stats, centroids, 4, CV_32S);

    //compute then greatest connected area
    int max_label = 0;
    int max_area = 0;
    //0 is the background
    for(int i = 1; i<nlabels;i++){
        if(stats.at<int>(i, CC_STAT_AREA)>max_area){
            max_label = i;
            max_area =stats.at<int>(i, CC_STAT_AREA);
        }
    }

    Mat disp = Mat::zeros(in.size(), in.type());
    for(int col = 0; col<labels.cols; col++){
        for(int row = 0; row<labels.rows; row++){
            if(labels.at<int>(row,col)==max_label){
                disp.at<char>(row, col) = static_cast<char>(255);
            }
        }
    }

    return disp;
}

std::vector<struct linestr> line4line(Mat img, double thresh){
    //canny
    Mat disp;
    Mat kernl = Mat::ones(15,15,CV_8U); 
    //Close is used in order to remove eventual noise
    //such as the pool cue or in general the balls
    //on the table may create issues
    morphologyEx(img, disp, MORPH_CLOSE, kernl);

    //the parameters are taken from an opencv
    //tutorial but they shouldn't make much difference 
    //on a binary mask
    Canny(disp, disp, 128, 100);

    //hough
    std::vector<Vec3f> lines;
    HoughLines(disp, lines,1, CV_PI/90, 30);
    Mat show;
    cvtColor(disp, show, COLOR_GRAY2BGR);
    //lines that are not similar
    std::vector<struct linestr> good;
    //iterate along the lines as they are
    //ordered descending in confidence
    for(int i = 0; i < 10; i++ ){
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        struct linestr tmpline{pt1, pt2};
        bool sim = false;

        //verify if the current is similar to any lines
        //considered up to now
        for(struct linestr g : good){
            if(arelinessimilar(g, tmpline, img.cols/10.0)){
                sim = true;
                break;
            }
        }
        if(!sim){
            good.push_back(tmpline);
            line(show, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
        }
        if(good.size() >= 4){
            break;
        }
    }
    //imshow(WINDOW_NAME, show);
    //waitKey(0);
    return good;
}

//the code is an implementation of the equations from
// https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
std::vector<Point2f> find_vertices(std::vector<struct linestr> lines, int max_col, int max_row){
    std::vector<Point2f> good;
    for(int i = 0; i < lines.size()-1;i++){
        struct linestr line1 = lines[i];
        for(int j = i+1; j < lines.size();j++){
            struct linestr line2 = lines[j];
            double det1 = line1.start.x*line1.stop.y - line1.start.y*line1.stop.x;
            double det2 = line2.start.x*line2.stop.y - line2.start.y*line2.stop.x;
            double det3 = (line1.start.x - line1.stop.x)*(line2.start.y - line2.stop.y);
            double det4 = (line1.start.y - line1.stop.y)*(line2.start.x - line2.stop.x);
            double p_x = (det1*(line2.start.x-line2.stop.x) - det2*(line1.start.x-line1.stop.x))/(det3-det4);
            double p_y = (det1*(line2.start.y-line2.stop.y) - det2*(line1.start.y-line1.stop.y))/(det3-det4);
            if(p_x > 0 && p_x < max_col && p_y > 0 && p_y < max_row){
                good.push_back(Point2f(p_x,p_y));
            }
        }
    }
    return good;
}

std::vector<Point2f> order_points(std::vector<Point2f> point4){
    std::vector<Point2f> points_ord(point4.begin(),point4.end());
    //order points from uppermost
    std::sort(points_ord.begin(), points_ord.end(), [&](Point p1, Point p2){return p1.y<p2.y;});
    //invert the first two if the leftmost is second
    if(points_ord[0].x>points_ord[1].x){
        Point tmp = points_ord[0];
        points_ord[0] = points_ord[1];
        points_ord[1] = tmp;
    }
    //invert the second two if the rightmost is last
    if(points_ord[2].x < points_ord[3].x){
        Point tmp = points_ord[2];
        points_ord[2] = points_ord[3];
        points_ord[3] = tmp;
    }
    return points_ord;
}

Vec3b meanMask(Mat img, Mat mask){
    double b,g,r; 
    b = 0;
    g = 0;
    r = 0;
    double count = 0;
    for(int row = 0; row < img.rows; row++){
        for(int col = 0; col < img.cols; col++){
            if(mask.at<char>(row, col) != 0){
                count++;
                Vec3b cur = img.at<Vec3b>(row,col);
                b += cur[0];
                g += cur[1];
                r += cur[2];
            }
        }
    }
    return Vec3b(static_cast<char>(b/count),static_cast<char>(g/count),static_cast<char>(r/count));
}

Mat threshHue(const cv::Mat in, const cv::Vec3b color, int thresh){

    //find the hue of the color given
    //done this way to avoid not easily
    //detectable errors or imprecisions
    Mat x(1,1,CV_8UC3);
    x.at<Vec3b>(0,0) = color;
    cvtColor(x, x, COLOR_BGR2HSV);
    char hue = x.at<Vec3b>(0,0)[0];

    Mat img2,ret;
    std::vector<Mat> spl;
    cvtColor(in, img2, COLOR_BGR2HSV);

    split(img2, spl);
    img2 = spl[0];
    //apply the threshold only on the hue
    inRange(img2, Scalar(hue-thresh < 0? 0 :hue-thresh), Scalar(hue+thresh>254? 254 : hue+thresh), ret);
    return ret;
}

std::vector<cv::Point2f> find_table(cv::Mat in, Mat &mask){
    Mat img = in.clone();

    //first "rough" segmentation of the table
    Mat clust = nonbinarykmeans(img, 3, 15);
    Mat island = greatest_island(clust);

    //more refined, based on color
    Vec3b mean = meanMask(img, island);
    Mat better = threshHue(img, mean);
    better = greatest_island(better);
    mask = Mat::zeros(in.size(), CV_8U);

    //find the lines of the table sides
    std::vector<struct linestr> lines = line4line(better, 1);
    std::vector<Point2f> points = order_points(find_vertices(lines, img.cols, img.rows));

    //vector of vectors required by fillPoly
    std::vector<std::vector<Point>> all;
    //we need Point and not Point2f for fillPoly
    std::vector<Point> all1, ord;
    for(int i = 0; i < 4; i++){
        all1.push_back(points[i]);
    }
    all.push_back(all1);
    
    //draw the mask of the table
    fillPoly(mask, all, Scalar(255));

    return order_points(points); 
}
