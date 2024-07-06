#include "utils/perfTesting.h"

using namespace cv;
using std::vector;
using std::map;

float intersectionOverUnion(Rect region1, Rect region2)
{
    float areaInters = (region1 & region2).area();
    float areaUnion = region1.area() + region2.area() - areaInters;
    return areaInters / areaUnion;
}

float intersectionOverUnion(Mat mask1, Mat mask2)
{
    assert(mask1.size() == mask2.size());
    Mat intersection;
    bitwise_and(mask1,mask2,intersection);
    float areaInters = countNonZero(intersection);
    float areaUnion = countNonZero(mask1) + countNonZero(mask2) - areaInters;
    return areaInters/areaUnion;
}

float oneToManyIoU(Rect region, vector<Rect> candidates)
{
    float best = 0;
    for (Rect r : candidates)
    {
        best = std::max(best,intersectionOverUnion(region,r));
    }
    return best;
}

vector<float> manyToManyIoU(vector<Rect> regions1, vector<Rect> regions2)
{
    vector<float> ans;
    for (Rect r1 : regions1)
    {
        ans.push_back(oneToManyIoU(r1,regions2));
    }
    return ans;
}

map<float,float> precisionRecallCurve(
    vector<Rect> predictions, 
    vector<Rect> truths,
    float threshold
)
{
    map<float,float> ans;
    
    float truePos = 0;
    float falsePos = 0;
    float precision, recall;
    float numPrediction = predictions.size();
    float numTruths = truths.size();

    for (float iou : manyToManyIoU(predictions,truths))
    {
        if (iou >= threshold)
            truePos++;
        else
            falsePos++;

        precision = truePos / (truePos + falsePos);
        recall = truePos / numTruths;

        if (ans.count(recall))
        {
            ans[recall] = std::max(ans[recall],precision);
        }
        else
            ans[recall] = precision;
    }

    return ans;
}

float averagePrecision(
    map<float,float> prCurve,
    int steps
)
{
    // TODO
}