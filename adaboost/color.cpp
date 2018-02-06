#include <opencv2/opencv.hpp>
#include <iostream>
#include "detect.h"

using namespace std;


int main4()
{
    //CvScalar total = cvScalar(0, 0, 0);
    /*int totals[4]={0,0,0,0};
    for (int i = 1; i <= 23; i++){
        stringstream ss;
        ss << "tt/color/1 (" << i << ").jpg";
        IplImage *img = cvLoadImage(ss.str().c_str());
        IplImage *img2 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
        cvCvtColor(img, img2, CV_BGR2YUV);
        CvScalar s;
        CvScalar sr;
        //CvScalar sum = cvScalar(0, 0, 0);
        int sums[4]={0,0,0,0};
        for (int i = 0; i < img2->height; i++){
            for (int j = 0; j < img2->width; j++){
                s = cvGet2D(img2, i, j);
                sr = cvGet2D(img, i, j);
                sums[0] += s.val[0];
                sums[1] += s.val[1];
                sums[2] += s.val[2];
                sums[3] += (sr.val[1] - s.val[0]);
            }
        }
        int num = img->height * img->width;
        sums[0] /= num;
        sums[1] /= num;
        sums[2] /= num;
        sums[3] /= num;
        cout << sums[0] << ", "<<sums[1]<<", "<<sums[2]<<", "<<sums[3]<<"\n";
        totals[0] += sums[0];
        totals[1] += sums[1];
        totals[2] += sums[2];
        totals[3] += sums[3];
    }
    totals[0] /= 23;
    totals[1] /= 23;
    totals[2] /= 23;
    totals[3] /= 23;
    cout <<"a: "<< totals[0] << ", "<<totals[1]<<", "<<totals[2]<<", "<<totals[3]<<"\n";*/

    IplImage *img = cvLoadImage("test/myDataBase/13.jpg");
    IplImage *img2 = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
    IplImage *img3 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCvtColor(img, img2, CV_BGR2YCrCb);
    CvScalar s;
    CvScalar sg;
    CvScalar b = CV_RGB(0, 0, 0);
    CvScalar w = CV_RGB(255, 255, 255);
    for (int y = 0; y < img2->height; ++y){
        for (int x = 0; x < img2->width; ++x){
            s = cvGet2D(img2, y, x);
            sg = cvGet2D(img, y, x);
            if (s.val[1] <= 140 || s.val[1] >= 180 || s.val[2] <= 90 || s.val[2] >= 130)
                cvSet2D(img3, y, x, b);
            else
                cvSet2D(img3, y, x, w);
        }
    }
    //cvSmooth(img3, img3, CV_MEDIAN, 9, 9, 0, 0);
    //cvDilate(img3,img3);
    IplImage *img4 = cvCreateImage(cvGetSize(img), img3->depth, img3->nChannels);
    cvCopy(img3, img4);
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvSeq *contours = 0;
    int c = cvFindContours(img3, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL);

    for( ; contours != NULL; contours = contours -> h_next )
    {
        CvRect rect = cvBoundingRect( contours, 0 );
        if (rect.height >= 21 && rect.width >= 21){
            cvRectangle( img4, cvPoint( rect.x, rect.y ),cvPoint( rect.x + rect.width, rect.y + rect.height ), cvScalar(255,255,255), 0 );
            cvRectangle( img, cvPoint( rect.x, rect.y ),cvPoint( rect.x + rect.width, rect.y + rect.height ), cvScalar(255,255,255), 0 );
        }
    }
    //cvNot(img4,img4);
    cvNamedWindow("w1");
    cvNamedWindow("w2");
    cvShowImage("w1", img);
    cvShowImage("w2", img4);
    cvWaitKey(0);

    return 0;
}

/*int main()
{
    IplImage *img = cvLoadImage("test/myDataBase/7.jpg");
    IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    IplImage *img3 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
    cvCvtColor(img, img2, CV_BGR2GRAY);
    cvEqualizeHist(img2, img3);
    cvNamedWindow("w1");
    cvNamedWindow("w2");
    cvShowImage("w1", img2);
    cvShowImage("w2", img3);
    cvWaitKey(0);
}*/

/*int main()
{
	int i;
	IplImage* src = cvLoadImage("test/myDataBase/1.jpg");
	IplImage* imgChannel[4] = { 0, 0, 0, 0 };
	IplImage* dst = cvCreateImage( cvGetSize( src ), IPL_DEPTH_8U, 3 );

	if( src )
	{
		for( i = 0; i < src -> nChannels; i++ )
		{
			imgChannel[i] = cvCreateImage( cvGetSize( src ), IPL_DEPTH_8U, 1 );  //要求单通道图像才能直方图均衡化
		}
		//通道分离
		cvSplit( src, imgChannel[0], imgChannel[1], imgChannel[2], imgChannel[3] );//BGRA
		for( i = 0; i < dst -> nChannels; i++ )
		{
			//直方图均衡化，原始图像和目标图像必须是单通道
			cvEqualizeHist( imgChannel[i], imgChannel[i] );
		}

		//通道组合
		cvMerge( imgChannel[0], imgChannel[1], imgChannel[2], imgChannel[3], dst );
		cvNamedWindow( "src", 1 );
		cvShowImage( "src", src );
		cvNamedWindow( "Equalize", 1 );
		cvShowImage( "Equalize", dst );

		cvWaitKey(0);
		//释放资源
		for( i = 0; i < src -> nChannels; i++ )
		{
			if( imgChannel[i] )
			{
				cvReleaseImage( &imgChannel[i] );
				//imgChannel[i] = 0;
			}
		}
		cvReleaseImage( &dst );
	}

	return 0;
}*/


