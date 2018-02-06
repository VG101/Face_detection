#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "classifier.h"

/*const int regWidth(19);
const int regHeight(19);
const CvSize regularSize{regWidth, regHeight};
//all training samples are regulated to regularSize: regWidth * regHeight
const double scale_coef(1.25);

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
};*/

void addImg(std::vector<TrainSample> &samples, std::string imgName, bool isFace);
/*
 *Summary: add the image with imgName to samples
 *Parameters:
 *  samples: the series of train sample the image will be added to
 *  imgName: the name of the image to be added
 *  isFace: if the image is a face
 *Return: void
 */
IplImage* preProcessImg(IplImage *srcImg);
/*
 *Summary: resize the image and convert to grayscale image
 *Parameters:
 *  srcImg: the source image
 *Return: a new processed image
 */
double normalizePix(double pixTable[regWidth][regHeight]);
/*
 */
void getPixTab(IplImage *img, double pixTable[regWidth][regHeight]);
/*
 *Summary: get the grayscale of each pixel of the image
 *Parameters:
 *  img: the source image
 *  pixTable: the grayscale will be saved in the pixTable
 *Return: void
 */
void calcIntegralImage(double pixTable[regWidth][regHeight], double ii[regWidth][regHeight]);
/*
 *Summary: calculate the integral image of the image
 *Parameters:
 *  pixTable: the grayscale value of the image
 *  ii: the integral image will be saved in the ii
 *Return: void
 */
//double calcCharaValue(double ii[regWidth][regHeight], Feature feature);
/*
 *Summary: calculate the character value according to the image and feature
 *Parameters:
 *  ii: the integral image
 *  feature: the feature
 *Return: the character value
 */
void initializeFeatures(std::vector<Feature> &features);
/*
 *Summary: initialize all features
 *Parameters:
 *  features: the features vector to contain all features
 *Return: void
 */
void initFeaturesHelper(std::vector<Feature> &features, short s, short t);
/*
 *Summary: helper function to initializeFeatures
 *Parameters:
 *  features: the features vector to contain all features
 *  s: describe current feature
 *  t: describe current feature
 *Return: void
 */
double trainAFeature(std::vector<TrainSample> &samples, Feature &feature);
/*
 *Summary: train a feature, get its threshold and its error
 *Parameters:
 *  samples: training samples
 *  feature: the feature
 *Return: void
 */
void sortOnCharaValue(std::vector<TrainSample> &samples);
/*
 *Summary: sort the samples based on character value
 *Parameters:
 *  samples: the vector of samples to be sorted
 *Return: void
 */
bool compareTrainSample(TrainSample ts1, TrainSample ts2);
/*
 *Summary: the comparison function to be used in sortOnCharaValue
 *          compare the samples based on their character value
 *Parameters:
 *  ts1: one training sample
 *  ts2: another training sample
 *Return: true if ts1'charaValue <= ts2'charaValue, false otherwise
 */
WeakClassifier trainWeakClassifier(std::vector<TrainSample> &samples, std::vector<Feature> &features);
/*
 *Summary: train a weak classifier
 *Parameters:
 *  samples: the samples used in the training
 *  features: the features
 *Return: the trained WeakClassifier
 */
StrongClassifier trainStrongClassifier(std::vector<TrainSample> &samples, std::vector<Feature> &features, int round);
/*
 *Summary: train a strong classifier
 *Parameters:
 *  samples: the samples used in the training
 *  features: the features
 *  round: the number of WeakClassifier in the StrongClassifier
 *Return: the trained StrongClassifier
 */
void reduceFeature(std::vector<Feature> &features, double rate);
bool compareFeature(Feature f1, Feature f2);
void initializeWeight(std::vector<TrainSample> &samples, int face, int nface);
/*
 *Summary: initialize the weight of all training samples
 *Parameters:
 *  samples: the training samples
 *  face: number of samples that are faces
 *  nface: number of samples that are not faces
 *Return: void
 */
void normalizeWeight(std::vector<TrainSample> &samples);
/*
 *Summary: normalize the weight of all training samples
 *Parameters:
 *  samples: the training samples
 *Return: void
 */
void adjustWeight(std::vector<TrainSample> &samples, Feature &feature);
/*
 *Summary: adjust the weight of all training samples based on correctness of the feature
 *Parameters:
 *  samples: the training samples
 *  feature: the feature to detect
 *Return: void
 */
//bool weakClassify(TrainSample &sample, Feature &feature);
/*
 *Summary: use a feature to classify a sample
 *Parameters:
 *  sample: the sample to be classified
 *  feature: the feature
 *Return: true if classified as a face, false otherwise
 */
//bool strongClassify(TrainSample &sample, StrongClassifier &sClassifier);
/*
 *Summary: use a SrongClassifier to classify a sample
 *Parameters:
 *  sample: the sample to be classified
 *  sClassifier: the StrongClassifier
 *Return: true if classified as a face, false otherwise
 */
void setSCThreshold(StrongClassifier &sClassifer, double threshold);
CascadeClassifier trainCascadeClassifier(std::vector<TrainSample> &samples, std::vector<Feature> features);
//bool cascadeClassify(TrainSample &sample, CascadeClassifier &classifier);

void addSamples(std::vector<TrainSample> &samples, std::string imgName, CascadeClassifier &cc);

void getPix(std::vector<std::vector<double>> &pix, double cpix[regWidth][regHeight], int w, int h);
