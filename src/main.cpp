#include "stdafx.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Config.h"
#include <iostream>
#include <stdio.h>
#include <ctime>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat& frame);
void GammaCorrection(Mat& src, Mat& dst, float fGamma)

{

	unsigned char lut[256];

	for (int i = 0; i < 256; i++)

	{

		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

	}

	dst = src.clone();

	const int channels = dst.channels();

	switch (channels)

	{

	case 1:

	{

		MatIterator_<uchar> it, end;

		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)

			*it = lut[(*it)];

		break;

	}

	case 3:

	{

		MatIterator_<Vec3b> it, end;

		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)

		{

			(*it)[0] = lut[((*it)[0])];

			(*it)[1] = lut[((*it)[1])];

			(*it)[2] = lut[((*it)[2])];

		}

		break;

	}

	}

}
/** Global variables */
String face_cascade_name, eyes_cascade_name, smile_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier smile_cascade;
String window_name = "Capture - Face detection";

void MovingAverage(Mat& inputImage,Mat &outputImage, const int k = 1)
{
	outputImage = inputImage;
	unsigned char *input = (unsigned char*)(inputImage.data);
	unsigned char *output = (unsigned char*)(outputImage.data);
	int i, j, r, g, b;
	for (int i = k; i < inputImage.cols-k; i++) 
	{
		for (int j = k; j < inputImage.rows-k; j++) 
		{
			for (int u = -k; u < k; u++)
			{
				for (int v = -k; v < k; v++)
				{
					output[inputImage.cols * j + i] += input[inputImage.cols * (i + u) + j + v];
					output[inputImage.cols * j + i+1] += input[inputImage.cols * (i + u) + j + v+1];
					output[inputImage.cols * j + i+2] += input[inputImage.cols * (i + u) + j + v+2];
				}
			}
			output[inputImage.cols * j + i] /= (2*k+1);
			output[inputImage.cols * j + i+1] /= (2 * k + 1);
			output[inputImage.cols * j + i+2] /= (2 * k + 1);
		}
	}
}
Mat equalizeIntensity(Mat& inputImage)
{
	GaussianBlur(inputImage, inputImage, Size(51,51), 31,31);

	return inputImage;
}
void ApplyFilter(Mat& frame)
{
	equalizeIntensity(frame);
	//cv::transform(frame, frame, kern);
}
/** @function main */
int main(int argc, const char** argv)
{
	face_cascade_name = "../data/haarcascades/haarcascade_frontalface_alt.xml";
	eyes_cascade_name ="../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
	smile_cascade_name ="../data/haarcascades/haarcascade_smile.xml";
	VideoCapture capture(0);
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	if (!smile_cascade.load(smile_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };

	//-- 2. Read the video stream
	TickMeter tm;

	while (true)
	{
#if defined(SHOW_FPS)
		tm.stop();
		tm.start();
#endif
		capture.read(frame);
#if defined(SHOW_FPS)
		if (tm.getCounter() > 0)
		{
			//print framerate
			cv::putText(frame, cv::format("Average FPS=%d", cvRound(tm.getCounter() / tm.getTimeSec())), cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
		}
#endif
		detectAndDisplay(frame);
		//-- Show what you got
		ApplyFilter(frame);
		imshow(window_name, frame);
		char c = (char)waitKey(1);
		if (c == 27) { break; } // escape
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat& frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		//Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			cv::Vec3b pixelColor(255, 0, 0);
		}
	}
}
