#include "detect.h"
#include <iostream>
using std::vector;
using std::cout;

void normalizePixel(vector<vector<double>>& pixTable, int width, int height)
{
    double sum = 0.0;
    for (int x = 0; x < width; ++ x)
        for (int y = 0; y < height; ++ y)
            sum += pixTable[x][y];
    double mean = sum / (width*height);
    double deviation = 0.0;
    for (int x = 0; x < width; ++ x)
        for (int y = 0; y < height; ++ y)
            deviation += pow((pixTable[x][y]-mean), 2);
    deviation = sqrt(deviation / (width*height));
    for (int x = 0; x < width; ++ x)
        for (int y = 0; y < height; ++ y){
            pixTable[x][y] = (pixTable[x][y]-mean)/deviation;
        }
}

void getPixData(IplImage *img, vector<vector<double>>& pixTable)
{
    pixTable.clear();
    pixTable.resize(img->width);
    for (int i = 0; i < img->width; ++ i)
        pixTable[i].resize(img->height);

    uchar *data = (uchar *)img->imageData;          //the pixel data of the image is saved in imageData
    int step = img->widthStep / sizeof(uchar);      //number of uchar per row

    for (int y = 0; y < img->height; ++ y)
        for (int x = 0; x < img->width; ++ x)
            pixTable[x][y] = (double)data[y * step + x];

    normalizePixel(pixTable, img->width, img->height);
}

void ccalcIntegralImage(vector<vector<double>>& pixTable, vector<vector<double>>& ii)
{
    ii.clear();
    int width = pixTable.size();
    int height = pixTable[0].size();

    vector<vector<double>> s(width);
    ii.resize(width);
    for (int i = 0; i < width; ++ i){
        s[i].resize(height);
        ii[i].resize(height);
    }
    s[0][0] = pixTable[0][0];
    ii[0][0] = s[0][0];
    for (int x = 1; x < width; ++ x){
        s[x][0] = pixTable[x][0];
        ii[x][0] = ii[x-1][0] + s[x][0];
    }
    for (int y = 1; y < height; ++ y){
        s[0][y] = s[0][y-1] + pixTable[0][y];
        ii[0][y] = s[0][y];
    }
    for (int x = 1; x < width; ++ x)
        for (int y = 1; y < height; ++ y){
            s[x][y] = s[x][y-1] + pixTable[x][y];
            ii[x][y] = ii[x-1][y] + s[x][y];
    }

}

void getIntegralImage(vector<vector<double>> &ii, double cii[regWidth][regHeight], int w, int h)
{
    for (int i = 0; i < regWidth; i ++)
        for (int j = 0; j < regHeight; j++)
            cii[i][j] = ii[i+w][j+h] + ii[w][h] - ii[i+w][h] - ii[w][j+h];
}

/*int detectFace(IplImage* srcImg, CascadeClassifier &classifier)
{
    IplImage *img = cvCreateImage(cvGetSize(srcImg), srcImg->depth, srcImg->nChannels);
    cvCopy(srcImg, img);

    vector<Area> faces;
    int width = img->width;
    int height = img->height;

    int faceCount = 0;
    double scale = 1.0;
    CvSize cSize = {width, height};
    while (cSize.width >= regWidth && cSize.height >= regHeight){
        IplImage *currImg = cvCreateImage(cSize, img->depth, img->nChannels);
        IplImage *currGImg = cvCreateImage(cSize, IPL_DEPTH_8U, 1);
        cvResize(img, currImg, CV_INTER_AREA);
        cvCvtColor(currImg, currGImg, CV_BGR2GRAY);
        vector<vector<double>> pix;
        getPixData(currGImg, pix);
        //vector<vector<double>>ii;
        //ccalcIntegralImage(pix, ii);
        for (int x = 0; (x+regWidth-1) < cSize.width; x += step){
            for (int y = 0; (y+regHeight-1) < cSize.height; y += step){
                TrainSample ts;
                double pixTable[regWidth][regHeight];
                getPix(pix, pixTable, x, y);
                normalizePix(pixTable);
                calcIntegralImage(pixTable, ts.ii);
                //getIntegralImage(ii, ts.ii, x, y);
                if (cascadeClassify(ts, classifier)){
                    Area face;
                    face.leftTop = {x*scale, y*scale};
                    face.rightBottom = {(x+regWidth-1)*scale, (y+regHeight-1)*scale};
                    faces.push_back(face);
                    faceCount ++;

                }
            }
        }

        cvReleaseImage(&currImg);
        cvReleaseImage(&currGImg);
        scale *= scale_coef;
        cSize = {width/scale, height/scale};
    }
    cvNamedWindow("win");
    for (vector<Area>::iterator i = faces.begin(); i != faces.end(); ++ i){
        cvRectangle(img, (*i).leftTop, (*i).rightBottom, CV_RGB(255,255,255));
    }
    cvShowImage("win", img);
    cvWaitKey(0);

    return faceCount;
}*/

bool faceExist(vector<Area> &faces, Area &face)
{
    int left = face.leftTop.x;
    int top = face.leftTop.y;
    int right = face.rightBottom.x;
    int bottom = face.rightBottom.y;
    bool exist = false;

    for (vector<Area>::iterator i = faces.begin(); i != faces.end(); ++ i){
        int left1 = (*i).leftTop.x;
        int top1 = (*i).leftTop.y;
        int right1 = (*i).rightBottom.x;
        int bottom1 = (*i).rightBottom.y;

        if (left >= left1 && left <= right1 && top >= top1 && top <= bottom1)
            exist = true;
        else if (right >= left1 && right <= right1 && bottom >= top1 && bottom <= bottom1)
            exist = true;
        else if (left >= left1 && left <= right1 && bottom >= top1 && bottom <= bottom1)
            exist = true;
        else if (right >= left1 && right <= right1 && top >= top1 && top <= bottom1)
            exist = true;
        else if (left1 >= left && left1 <= right && top1 >= top && top1 <= bottom)
            exist = true;
        else if (right1 >= left && right1 <= right && bottom1 >= top && bottom1 <= bottom)
            exist = true;
        else if (left1 >= left && left1 <= right && bottom1 >= top && bottom1 <= bottom)
            exist = true;
        else if (right1 >= left && right1 <= right && top1 >= top && top1 <= bottom)
            exist = true;
        if (exist){
            face.leftTop.x = (left1 * (*i).count + left * face.count)/((*i).count + face.count);
            face.leftTop.y = (top1 * (*i).count + top * face.count)/((*i).count + face.count);
            face.rightBottom.x = (right1 * (*i).count + right * face.count)/((*i).count + face.count);
            face.rightBottom.y = (bottom1 * (*i).count + bottom * face.count)/((*i).count+ face.count);
            face.count += (*i).count;
            faces.erase(i);
            if (!faceExist(faces, face))
                faces.push_back(face);
            return true;
        }
    }

    return false;
}


int detectFace(IplImage* srcImg, CascadeClassifier &classifier)
{
    CvSize csize = {srcImg->width, srcImg->height};
    /*if (csize.width >= 500 && csize.height >= 500){
        csize.width /= 1.5;
        csize.height /= 1.5;
    }*/
    IplImage *img = cvCreateImage(csize, srcImg->depth, srcImg->nChannels);
    IplImage *img2 = cvCreateImage(cvGetSize(img), srcImg->depth, srcImg->nChannels);
    IplImage *img3 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *img4 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvResize(srcImg, img, CV_INTER_AREA);

    cvCvtColor(img, img2, CV_BGR2YCrCb);
    CvScalar s;
    CvScalar b = CV_RGB(0, 0, 0);
    CvScalar w = CV_RGB(255, 255, 255);
    for (int y = 0; y < img2->height; ++y){
        for (int x = 0; x < img2->width; ++x){
            s = cvGet2D(img2, y, x);
            if (s.val[1] <= 140 || s.val[1] >= 180 || s.val[2] <= 90 || s.val[2] >= 130)
                cvSet2D(img3, y, x, b);
            else
                cvSet2D(img3, y, x, w);
        }
    }
    cvCopy(img3, img4);
    //cvSmooth(img3, img3, CV_MEDIAN, 21, 21, 0, 0);
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvSeq *contours = 0;
    cvFindContours(img3, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL);
    cvReleaseImage(&img2);
    cvReleaseImage(&img3);
    vector<Area> faces;
    int count = 0;
    /*CvRect rect = {0, 0, img->width, img->height};
    count = detectArea(faces, img, rect, classifier);*/
    for( ; contours != NULL; contours = contours->h_next)
    {
        CvRect rect = cvBoundingRect(contours, 0);
        if (rect.width < 21 || rect.height < 21)
            continue;
        if (rect.x - 5 >= 0)
            rect.x -= 5;
        else
            rect.x = 0;
        if (rect.y - 5 >= 0)
            rect.y -= 5;
        else
            rect.y = 0;
        if (rect.x + rect.width + 10 < srcImg->width)
            rect.width += 10;
        else
            rect.width = srcImg->width - rect.x - 1;
        if (rect.y + rect.height + 10 < srcImg->height)
            rect.height += 10;
        else
            rect.height = srcImg->height - rect.y - 1;
        if (rect.width >= regWidth && rect.height >= regHeight){
            CvSize aSize = cvSize(rect.width, rect.height);
            cvSetImageROI(img, rect);
            IplImage *aImg = cvCreateImage(aSize, srcImg->depth, srcImg->nChannels);
            cvCopy(img, aImg);
            cvResetImageROI(img);
            count += detectArea(faces, aImg, rect, classifier);
            cvReleaseImage(&aImg);
        }
    }
    colorDetect(faces, img4);
    areaCombine(faces);

    /*for (vector<Area>::iterator i = faces.begin(); i != faces.end(); ++i){
        findEye(*i, img);
        CvRect rect = cvRect((*i).leftTop.x, (*i).leftTop.y, (*i).rightBottom.x-(*i).leftTop.x, (*i).rightBottom.y-(*i).leftTop.y);
        cvSetImageROI(img, rect);
        IplImage *aImg = cvCreateImage(cvSize(rect.width, rect.height), srcImg->depth, srcImg->nChannels);
        cvCopy(img, aImg);
        cvResetImageROI(aImg);
        IplImage *gImg = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
        cvCvtColor(aImg, gImg, CV_BGR2GRAY);
        IplImage *bImg = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
        cvThreshold(gImg, bImg, 100, 255, CV_THRESH_BINARY);
        cvNamedWindow("w1");
        cvNamedWindow("w2");
        cvShowImage("w1", aImg);
        cvShowImage("w2", bImg);
        cvWaitKey(0);
        cvDestroyAllWindows();
    }*/

    cvNamedWindow("win");
    for (vector<Area>::iterator i = faces.begin(); i != faces.end(); ++ i){
        cvRectangle(srcImg, (*i).leftTop, (*i).rightBottom, CV_RGB(255,0,0), 2);
    }
    cvShowImage("win", srcImg);
    cvWaitKey(0);

    return faces.size();

}

int detectArea(vector<Area> &faces, IplImage* img, CvRect &rect, CascadeClassifier &classifier)
{
    int count = 0;
    int width = img->width;
    int height = img->height;

    int faceCount = 0;
    double scale = 1.0;
    CvSize cSize = {width, height};
    while (cSize.width >= regWidth && cSize.height >= regHeight){
        IplImage *currImg = cvCreateImage(cSize, img->depth, img->nChannels);
        IplImage *currGImg = cvCreateImage(cSize, IPL_DEPTH_8U, 1);
        cvResize(img, currImg, CV_INTER_AREA);
        cvCvtColor(currImg, currGImg, CV_BGR2GRAY);
        vector<vector<double>> pix;
        getPixData(currGImg, pix);
        for (int x = 0; (x+regWidth-1) < cSize.width; x += step){
            for (int y = 0; (y+regHeight-1) < cSize.height; y += step){
                TrainSample ts;
                double pixTable[regWidth][regHeight];
                getPix(pix, pixTable, x, y);
                double deviation = normalizePix(pixTable);
                if (deviation <= 0.35)
                    continue;
                calcIntegralImage(pixTable, ts.ii);
                if (cascadeClassify(ts, classifier)){
                    Area face;
                    face.leftTop = {rect.x + x*scale, rect.y + y*scale};
                    face.rightBottom = {rect.x + (x+regWidth-1)*scale, rect.y + (y+regHeight-1)*scale};
                    count ++;
                    face.count = 1;
                    faces.push_back(face);
                }
            }
        }

        cvReleaseImage(&currImg);
        cvReleaseImage(&currGImg);
        scale *= scale_coef;
        cSize = {width/scale, height/scale};
    }
    /*IplImage *gimg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCvtColor(img, gimg, CV_BGR2GRAY);
    IplImage *bimg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvThreshold(gimg, bimg, 150, 255, CV_THRESH_BINARY);
    cvNamedWindow("w1");
    cvNamedWindow("w2");
    cvShowImage("w1", bimg);
    cvShowImage("w2", img);
    cvWaitKey(0);
    cvDestroyAllWindows();*/
    return count;
}

void areaCombine(vector<Area> &faces)
{
    vector<Area> nfaces;
    for (int i = 0; i < faces.size(); i ++){
        if (!faceExist(nfaces, faces[i]))
            nfaces.push_back(faces[i]);
    }
    faces.clear();
    for (int i = 0; i < nfaces.size(); i ++){
        faces.push_back(nfaces[i]);
    }
    nfaces.clear();
}

void colorDetect(vector<Area> &faces, IplImage *img)
{
    int sum;
    int total;
    for (vector<Area>::iterator i = faces.begin(); i != faces.end(); ){
        CvScalar s;
        sum = 0;
        total = 0;
        for (int y = (*i).leftTop.y; y <= (*i).rightBottom.y; ++y){
            for (int x = (*i).leftTop.x; x <= (*i).rightBottom.x; ++x){
                s = cvGet2D(img, y, x);
                if (s.val[0] == 255)
                    sum ++;
                total ++;
            }
        }
        if (sum < total / 3)
            i = faces.erase(i);
        else
            i++;
    }
}

bool findEye(Area area, IplImage *img)
{
    equalizeHist(img);
    CvRect rect = cvRect(area.leftTop.x, area.leftTop.y, area.rightBottom.x-area.leftTop.x, area.rightBottom.y-area.leftTop.y);
    cvSetImageROI(img, rect);
    IplImage *aImg = cvCreateImage(cvSize(rect.width, rect.height), img->depth, img->nChannels);
    cvCopy(img, aImg);
    cvResetImageROI(img);
    IplImage *gImg = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
    cvCvtColor(aImg, gImg, CV_BGR2GRAY);
    CvScalar s;
    int mean1 = 0;
    int mean2 = 0;
    int count1 = 0;
    int count2 = 0;
    int thres = 0;
    int nthres = 0;
    for (int y = 0; y < gImg->height; ++y){
        for (int x = 0; x < gImg->width; ++x){
            s = cvGet2D(gImg, y, x);
            mean1 += s.val[0];
        }
    }
    nthres = mean1 / (gImg->height * gImg->width);
    do{
        thres = nthres;
        mean1 = mean2 = count1 = count2 = 0;
        for (int y = 0; y < gImg->height; ++y){
            for (int x = 0; x < gImg->width; ++x){
                s = cvGet2D(gImg, y, x);
                if (s.val[0] < thres){
                    count1 ++;
                    mean1 += s.val[0];
                }
                else{
                    count2 ++;
                    mean2 += s.val[0];
                }
            }
        }
        mean1 /= count1;
        mean2 /= count2;
        nthres = (mean1 + mean2)/2;
    } while(abs(thres-nthres)>1);

    IplImage *bImg = cvCreateImage(cvSize(rect.width, rect.height), IPL_DEPTH_8U, 1);
    cvThreshold(gImg, bImg, nthres*0.8, 255 ,CV_THRESH_BINARY);

    IplImage *img4 = cvCreateImage(cvGetSize(bImg), bImg->depth, bImg->nChannels);
    cvCopy(bImg, img4);
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvSeq *contours = 0;
    int c = cvFindContours(img4, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL);

    for( ; contours != NULL; contours = contours -> h_next )
    {
        CvRect rect = cvBoundingRect( contours, 0 );
        cvRectangle( bImg, cvPoint( rect.x, rect.y ),cvPoint( rect.x + rect.width, rect.y + rect.height ), cvScalar(255,255,255), 0 );
    }

    cvNamedWindow("w1");
    cvNamedWindow("w2");
    cvShowImage("w1", gImg);
    cvShowImage("w2", bImg);
    cvWaitKey(0);
    cvDestroyAllWindows();
}

void equalizeHist(IplImage* img)
{
    int i;
	IplImage* imgChannel[4] = { 0, 0, 0, 0 };

	if(img)
	{
		for(i = 0; i < img->nChannels; i++ )
		{
			imgChannel[i] = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );  //要求单通道图像才能直方图均衡化
		}
		//通道分离
		cvSplit(img, imgChannel[0], imgChannel[1], imgChannel[2], imgChannel[3] );//BGRA
		for( i = 0; i < img->nChannels; i++ )
		{
			//直方图均衡化，原始图像和目标图像必须是单通道
			cvEqualizeHist( imgChannel[i], imgChannel[i] );
		}

		//通道组合
		cvMerge( imgChannel[0], imgChannel[1], imgChannel[2], imgChannel[3], img );

		//释放资源
		for( i = 0; i < img->nChannels; i++ )
		{
			if( imgChannel[i] )
			{
				cvReleaseImage( &imgChannel[i] );
				//imgChannel[i] = 0;
			}
		}
	}

}
