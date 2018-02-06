#include "train.h"
#include "detect.h"
#include "quicksort.h"
#include <iostream>
#include <math.h>

using std::vector;
using std::string;
using std::cout;

void addSamples(vector<TrainSample> &samples, string imgName, CascadeClassifier &cc)
{
    IplImage *srcImg = cvLoadImage(imgName.c_str());
    IplImage *grayImg = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    cvCvtColor(srcImg, grayImg, CV_BGR2GRAY);
    CvSize cSize = {grayImg->width, grayImg->height};
    while (cSize.width >= regWidth && cSize.height >= regHeight){
        IplImage *img = cvCreateImage(cSize, grayImg->depth, grayImg->nChannels);
        cvResize(grayImg, img, CV_INTER_AREA);
        vector<vector<double>> pix;
        getPixData(img, pix);
        //int w = cSize.width / regWidth;
        //int h = cSize.height / regHeight;
        for (int i = 0; (i+regWidth-1) < cSize.width; i += 2){
            for (int j = 0; (j+regHeight-1) < cSize.height; j += 2){
                TrainSample ts;
                double pixTab[regWidth][regHeight];
                getPix(pix, pixTab, i, j);
                normalizePix(pixTab);
                calcIntegralImage(pixTab, ts.ii);
                ts.isFace = false;
                if (cascadeClassify(ts, cc))
                    samples.push_back(ts);
            }
        }
        cvReleaseImage(&img);
        cSize.width /= 2;
        cSize.height /= 2;
    }
    cout << imgName << " is added\n";
    cvReleaseImage(&srcImg);
    cvReleaseImage(&grayImg);
}

void getPix(std::vector<std::vector<double>> &pix, double cpix[regWidth][regHeight], int w, int h)
{
    for (int i = 0; i < regWidth; i ++)
        for (int j = 0; j < regHeight; j++)
            cpix[i][j] = pix[i+w][j+h];
}

void addImg(vector<TrainSample> &samples, string imgName, bool isFace)
{
    IplImage *srcImg = cvLoadImage(imgName.c_str());    //load the image
    IplImage* img = preProcessImg(srcImg);              //preprocess the image
    double pixTable[regWidth][regHeight];
    getPixTab(img, pixTable);
    /*for (int x = 0; x < regWidth; ++x){
        for (int y = 0; y < regHeight; ++y){
            cout << pixTable[x][y] << "\t";
        }
        cout <<"\n";
    }*/
    TrainSample sample;
    calcIntegralImage(pixTable, sample.ii);             //calculate the integral image of the sample
    sample.isFace = isFace;
    samples.push_back(sample);                          //add the sample to the vector
    //cout << imgName << " is added\n";
    cvReleaseImage(&srcImg);
    cvReleaseImage(&img);
}

IplImage* preProcessImg(IplImage *srcImg)
{
    IplImage *grayImg = cvCreateImage(cvGetSize(srcImg), IPL_DEPTH_8U, 1);
    cvCvtColor(srcImg, grayImg, CV_BGR2GRAY);                                           //convert to grayscale
    IplImage *dstImg = cvCreateImage(regularSize, grayImg->depth, grayImg->nChannels);
    cvResize(grayImg, dstImg, CV_INTER_AREA);                                           //resize
    cvReleaseImage(&grayImg);
    return dstImg;
}

double normalizePix(double pixTable[regWidth][regHeight])
{
    double sum = 0.0;
    for (int x = 0; x < regWidth; ++ x)
        for (int y = 0; y < regHeight; ++ y)
            sum += pixTable[x][y];
    double mean = sum / (regWidth*regHeight);
    double deviation = 0.0;
    for (int x = 0; x < regWidth; ++ x)
        for (int y = 0; y < regHeight; ++ y)
            deviation += pow((pixTable[x][y]-mean), 2);
    deviation = sqrt(deviation / (regWidth*regHeight));
    for (int x = 0; x < regWidth; ++ x)
        for (int y = 0; y < regHeight; ++ y){
            pixTable[x][y] = (pixTable[x][y]-mean)/deviation;
        }

    return deviation;
}

void getPixTab(IplImage *img, double pixTable[regWidth][regHeight])
{
    uchar *data = (uchar *)img->imageData;          //the pixel data of the image is saved in imageData
    int step = img->widthStep / sizeof(uchar);      //number of uchar per row

    for (int y = 0; y < img->height; ++ y)
        for (int x = 0; x < img->width; ++ x)
            pixTable[x][y] = (double)data[y * step + x];

    normalizePix(pixTable);
}

void calcIntegralImage(double pixTable[regWidth][regHeight], double ii[regWidth][regHeight])
{
    double s[regWidth][regHeight];
    s[0][0] = pixTable[0][0];
    ii[0][0] = s[0][0];
    for (int x = 1; x < regWidth; ++ x){
        s[x][0] = pixTable[x][0];
        ii[x][0] = ii[x-1][0] + s[x][0];
    }
    for (int y = 1; y < regHeight; ++ y){
        s[0][y] = s[0][y-1] + pixTable[0][y];
        ii[0][y] = s[0][y];
    }
    for (int x = 1; x < regWidth; ++ x)
        for (int y = 1; y < regHeight; ++ y){
            s[x][y] = s[x][y-1] + pixTable[x][y];
            ii[x][y] = ii[x-1][y] + s[x][y];
    }
}

void initializeFeatures(vector<Feature> &features)
{
    short s = 1;
    short t = 2;
    initFeaturesHelper(features, s, t);
    s = 2;
    t = 1;
    initFeaturesHelper(features, s, t);
    s = 1;
    t = 3;
    initFeaturesHelper(features, s, t);
    s = 3;
    t = 1;
    initFeaturesHelper(features, s, t);
    s = 2;
    t = 2;
    initFeaturesHelper(features, s, t);
}

void initFeaturesHelper(vector<Feature> &features, short s, short t)
{
    for (int x1 = 0; x1 < regWidth-s; ++ x1)
        for (int y1 = 0; y1 < regHeight-t; ++ y1)
            for (int x2 = x1+s; x2 < regWidth; x2 += s)
                for (int y2 = y1+t; y2 < regHeight; y2 += t){
                    Feature f;
                    f.topLeft = {x1, y1};
                    f.bottomRight = {x2, y2};
                    f.s = s;
                    f.t = t;
                    features.push_back(f);
                }
}

double trainAFeature(vector<TrainSample> &samples, Feature &feature)
{
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){  //calculate all samples' character value
        (*i).charaValue = calcCharaValue((*i).ii, feature);
    }
    sortOnCharaValue(samples);                                                          //sort samples based on character value
    double faceWeightSum(0), nfaceWeightSum(0);
    double preFaceWeightSum(0), preNfaceWeightSum(0);
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){
        if ((*i).isFace)
            faceWeightSum += (*i).weight;
        else
            nfaceWeightSum += (*i).weight;
    }
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){  //calculate error of all samples
        double sum1 = preFaceWeightSum+(nfaceWeightSum-preNfaceWeightSum);
        double sum2 = preNfaceWeightSum+(faceWeightSum-preFaceWeightSum);
        if (sum1 < sum2){
            (*i).error = sum1;
            (*i).p = -1;
        }
        else{
            (*i).error = sum2;
            (*i).p = 1;
        }
        if ((*i).isFace)
            preFaceWeightSum += (*i).weight;
        else
            preNfaceWeightSum += (*i).weight;
    }
    double mini = samples[0].error;
    int index = 0;
    for (int i = 1; i < samples.size(); ++ i){                                          //find the one with the smallest error
        if (samples[i].error < mini){
            mini = samples[i].error;
            index = i;
        }
    }
    feature.threshold = samples[index].charaValue;                                      //set the feature's threshold
    feature.p = samples[index].p;                                                       //set the feature's p

    double error = 0;
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){  //calculate weighted error of the feature
        if (weakClassify(*i, feature) != (*i).isFace)
            error += (*i).weight;
    }
    feature.error = error;
    return error;
}

void sortOnCharaValue(vector<TrainSample> &samples)
{
    quicksort(samples, 0, samples.size()-1, compareTrainSample);                          //use quicksort to sort the samples
}

bool compareTrainSample(TrainSample ts1, TrainSample ts2)
{
    return ts1.charaValue <= ts2.charaValue;
}

WeakClassifier trainWeakClassifier(vector<TrainSample> &samples, vector<Feature> &features)
{
    double minError = 1.0;
    WeakClassifier optWkClassifier;
    for (vector<Feature>::iterator i = features.begin(); i != features.end(); ++ i){
        double error = trainAFeature(samples, (*i));                                                //train each feature
        if (error < minError){                                                                      //find the one with smallest error
            minError = error;
            optWkClassifier.feature = *i;
        }
    }
    cout << "error: "<<minError<<"\n";
    optWkClassifier.weight = log((1-minError)/minError);  //calculate the weight of the WeakClassifier

    return optWkClassifier;
}

StrongClassifier trainStrongClassifier(vector<TrainSample> &samples, vector<Feature> &features, int round)
{
    StrongClassifier c;

    for (int i = 0; i < round; ++ i){
        normalizeWeight(samples);
        WeakClassifier wc = trainWeakClassifier(samples, features);
        cout << "w: " <<wc.weight << "\n";
        cout << wc.feature.s << " " << wc.feature.t <<"\n";
        cout << wc.feature.topLeft.x <<","<<wc.feature.topLeft.y<<"\n";
        cout << wc.feature.bottomRight.x <<","<<wc.feature.bottomRight.y<<"\n";
        c.wClassifiers.push_back(wc);

        if (i == 0){
            reduceFeature(features, 0.8);
            /*FILE *fp;
            fp = fopen("features/feature3", "wb");
            int fsize = features.size();
            fwrite(&fsize, sizeof(int), 1, fp);
            for (int j = 0; j < fsize; ++ j)
                fwrite(&features[j], sizeof(Feature), 1, fp);
            fclose(fp);
            reduceFeature(features, 0.8);*/
        }
        else
            reduceFeature(features, 0.01);
        adjustWeight(samples, wc.feature);
        cout << "round" << i+1 << " finished"<<"\n";
    }
    c.threshold = 2.0;
    return c;
}

void reduceFeature(vector<Feature> &features, double rate)
{
    quicksort(features, 0, features.size()-1, compareFeature);
    int removeNum = rate * features.size();
    for (int i = 0; i < removeNum; ++ i){
        features.pop_back();
    }
    cout << "feature size:" << features.size() <<"\n";
}

bool compareFeature(Feature f1, Feature f2)
{
    return f1.error <= f2.error;
}

void initializeWeight(vector<TrainSample> &samples, int face, int nface)
{
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){
        if ((*i).isFace)
            (*i).weight = 0.5 / face * 2;
        else
            (*i).weight = 0.5 / nface;
    }
}

void normalizeWeight(vector<TrainSample> &samples)
{
    double sum = 0;
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){
        sum += (*i).weight;
    }
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){
        ((*i).weight) = ((*i).weight) / sum;
    }
}

void adjustWeight(vector<TrainSample> &samples, Feature &feature)
{
    double beta = feature.error / (1 - feature.error);
    for (vector<TrainSample>::iterator i = samples.begin(); i != samples.end(); ++ i){
        if (weakClassify(*i, feature) == (*i).isFace)               //if the sample is detected correct
            (*i).weight *= beta;                                    //adjust the weight by multiplying a factor
    }
}


void setSCThreshold(StrongClassifier &sClassifier, double threshold)
{
    sClassifier.threshold = threshold;
}




