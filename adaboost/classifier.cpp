#include "classifier.h"

using std::vector;

double calcCharaValue(double ii[regWidth][regHeight], Feature feature)
{
    int left = feature.topLeft.x;
    int right = feature.bottomRight.x;
    int top = feature.topLeft.y;
    int bottom = feature.bottomRight.y;

    if (feature.s == 1 && feature.t == 2){
        int mid = (top + bottom) / 2;
        return (ii[left][top]-ii[right][top]-2*ii[left][mid]+2*ii[right][mid]+ii[left][bottom]-ii[right][bottom]);
    }
    else if (feature.s == 2 && feature.t == 1){
        int mid = (left + right) / 2;
        return (ii[left][top]-2*ii[mid][top]+ii[right][top]-ii[left][bottom]+2*ii[mid][bottom]-ii[right][bottom]);
    }
    else if (feature.s == 1 && feature.t == 3){
        int mid1 = (2*top + bottom) / 3;
        int mid2 = (top + 2*bottom) / 3;
        return (ii[left][top]-ii[right][top]-2*ii[left][mid1]+2*ii[right][mid1]
                +2*ii[left][mid2]-2*ii[right][mid2]-ii[left][bottom]+ii[right][bottom]);
    }
    else if (feature.s == 3 && feature.t == 1){
        int mid1 = (2*left + right) / 3;
        int mid2 = (left + 2*right) / 3;
        return (ii[left][top]-2*ii[mid1][top]+2*ii[mid2][top]-ii[right][top]
                -ii[left][bottom]+2*ii[mid1][bottom]-2*ii[mid2][bottom]+ii[right][bottom]);
    }
    else if (feature.s == 2 && feature.t == 2){
        int hmid = (left + right) / 2;
        int vmid = (top + bottom) / 2;
        return (ii[left][top]-2*ii[hmid][top]+ii[right][top]
                -2*ii[left][vmid]+4*ii[hmid][vmid]-2*ii[right][vmid]
                +ii[left][bottom]-2*ii[hmid][bottom]+ii[right][bottom]);
    }
    else
        return 0;
}

bool weakClassify(TrainSample &sample, Feature &feature)
{
    int charaValue = calcCharaValue(sample.ii, feature);
    if (charaValue*feature.p <= feature.threshold*feature.p)
        return true;
    else
        return false;
}

bool strongClassify(TrainSample &sample, StrongClassifier &sClassifier)
{
    double weightSum = 0.0;
    double weakSum = 0.0;
    for (vector<WeakClassifier>::iterator i = sClassifier.wClassifiers.begin(); i < sClassifier.wClassifiers.end(); ++ i){
        weightSum += (*i).weight;
        if (weakClassify(sample, (*i).feature))
            weakSum += (*i).weight;
    }
    if (weakSum >= weightSum/sClassifier.threshold)
        return true;
    else
        return false;
}

bool cascadeClassify(TrainSample &sample, CascadeClassifier &classifier)
{
    for (vector<StrongClassifier>::iterator i = classifier.sClassifiers.begin(); i != classifier.sClassifiers.end(); ++ i){
        if (!strongClassify(sample, *i))
            return false;
    }
    return true;
}


