#include <cstdlib>


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
    double dist1 = norm(a.start-b.start);
    double dist2 = norm(a.stop-b.start);
    double dist3 = norm(a.start-b.stop);
    double dist4 = norm(a.stop-b.stop);
    //std::cout << dist1 << " " <<dist2 << " "<<dist3 << " "<<dist4 << " "<<std::endl;
    return (dist1 < thresh && dist4 < thresh) || (dist2 < thresh && dist3 < thresh);
}


Mat binarykmeans(const Mat in){
    int blurSize=9;
    Mat img;
    GaussianBlur(in, img, Size(blurSize,blurSize), 0);
    Mat p = Mat::zeros(img.cols*img.rows, 5, CV_32F);

    std::vector<Mat> bgr;
    split(img, bgr);

    for(int i=0; i<img.cols*img.rows; i++) {
        p.at<float>(i,0) = static_cast<float>(i/img.cols) / img.rows;
        p.at<float>(i,1) = static_cast<float>(i%img.cols) / img.cols;
        p.at<float>(i,2) = bgr[0].data[i] / 255.0;
        p.at<float>(i,3) = bgr[1].data[i] / 255.0;
        p.at<float>(i,4) = bgr[2].data[i] / 255.0;
    }

    Mat labs,ctrs;
    kmeans(p, 2, labs, TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, ctrs);

    int colors[2] = {255,0};


    Mat clust = Mat(in.rows, in.cols, CV_32F);
    for(int i=0; i<img.cols*img.rows; i++) {
        clust.at<float>(i/img.cols, i%img.cols) = static_cast<float>(colors[labs.at<int>(0,i)]);
    }

    clust.convertTo(clust, CV_8U);
    //imshow(WINDOW_NAME, clust);
    //waitKey(0);
    return clust;
}

Mat greatest_island(Mat input){
    Mat labels, stats, centroids;
    Mat in;
    Mat kernel = Mat::ones(15, 15, CV_8U);
    morphologyEx(input, in, MORPH_CLOSE, kernel);
    int nlabels = cv::connectedComponentsWithStats(in, labels, stats, centroids, 4, CV_32S);
    int max_label = 0;
    int max_area = 0;
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
    //imshow(WINDOW_NAME, disp);
    //waitKey(0);
    return disp;
}

std::vector<struct linestr> line4line(Mat img, double thresh){
    //canny
    Mat disp;
    Canny(img, disp, 128, 100);
    //imshow(WINDOW_NAME, disp);
    //waitKey(0);
    //hough
    std::vector<Vec3f> lines;
    HoughLines(disp, lines,1, CV_PI/90, 30);
    Mat show;
    cvtColor(disp, show, COLOR_GRAY2BGR);
    std::vector<struct linestr> good;
    for( size_t i = 0; i < 10; i++ )
    {
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
        for(struct linestr g : good){
            if(arelinessimilar(g, tmpline, img.cols/10.0)){
                sim = true;
                break;
            }
        }
        if(!sim){
            good.push_back(tmpline);
            line(show, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
            //imshow(WINDOW_NAME, show);
            //waitKey(0);
        }
        if(good.size() >= 4){
            break;
        }
    }
    //imshow(WINDOW_NAME, show);
    //waitKey(0);
    return good;
}
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

std::vector<Point> order_points(std::vector<Point> point4){
    std::vector<Point> points_ord(point4.begin(),point4.end());
    std::sort(points_ord.begin(), points_ord.end(), [&](Point p1, Point p2){return p1.y<p2.y;});
    if(points_ord[0].x>points_ord[1].x){
        Point tmp = points_ord[0];
        points_ord[0] = points_ord[1];
        points_ord[1] = tmp;
    }
    if(points_ord[2].x < points_ord[3].x){
        Point tmp = points_ord[2];
        points_ord[2] = points_ord[3];
        points_ord[3] = tmp;
    }
    return points_ord;
}