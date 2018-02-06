#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <opencv2/opencv.hpp>

const int regWidth(21);
const int regHeight(21);
const CvSize regularSize{regWidth, regHeight};
//all training samples are regulated to regularSize: regWidth * regHeight

struct TrainSample{
    bool isFace;
    double ii[regWidth][regHeight];
    double charaValue;
    double weight;
    double error;
    short p;
};

struct Feature{
    CvPoint topLeft;
    CvPoint bottomRight;
    short s;          //s and t are used to describe a feature
    short t;
    double threshold;  //the optimum threshold of the feature
    short p;        //p is used to show the classifier direction
    double error;
    //double weight;  //alpha, weight in the strong classifier
};

struct WeakClassifier{
    Feature feature;
    double weight;
};

struct StrongClassifier{
    std::vector<WeakClassifier> wClassifiers;
    double threshold;
};

struct CascadeClassifier{
    std::vector<StrongClassifier> sClassifiers;
};

double calcCharaValue(double ii[regWidth][regHeight], Feature feature);
/*
 *Summary: calculate the character value according to the image and feature
 *Parameters:
 *  ii: the integral image
 *  feature: the feature
 *Return: the character value
 */
bool weakClassify(TrainSample &sample, Feature &feature);
/*
 *Summary: use a feature to classify a sample
 *Parameters:
 *  sample: the sample to be classified
 *  feature: the feature
 *Return: true if classified as a face, false otherwise
 */
bool strongClassify(TrainSample &sample, StrongClassifier &sClassifier);
/*
 *Summary: use a SrongClassifier to classify a sample
 *Parameters:
 *  sample: the sample to be classified
 *  sClassifier: the StrongClassifier
 *Return: true if classified as a face, false otherwise
 */

bool cascadeClassify(TrainSample &sample, CascadeClassifier &classifier);
#endif // CLASSIFIER_H
