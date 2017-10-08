#include <jni.h>
#include <string>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
int test();
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_manta_myapplication2_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    test();
    return env->NewStringUTF(hello.c_str());
}
using namespace std;
using namespace cv;
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
Mat equalizeIntensity(Mat& inputImage)
{
    Mat frame_gray;

    cvtColor(inputImage, inputImage, COLOR_BGR2GRAY);
    test();
    equalizeHist(inputImage, inputImage);
    GammaCorrection(inputImage, inputImage, 1);
    return inputImage;
}
void ApplyFilter(Mat& frame)
{
    equalizeIntensity(frame);
    //cv::transform(frame, frame, kern);
}
/** @function main */
int test()
{
    cv::Mat inframe = cv::Mat();
    cv::VideoCapture mCamera;
    mCamera.open(1);
    mCamera.set(CV_CAP_PROP_FRAME_WIDTH, 400);
    mCamera.set(CV_CAP_PROP_FRAME_HEIGHT, 300);

// mCamera.set("scene mode", "beach"); // <-- looking for a way

    while (mCamera.isOpened()) {
        bool grab = mCamera.grab();
        if (grab) {
            mCamera.retrieve(inframe, 2);

            // To do something for the iframe

        } else {
          int x=0;
            x++;
        }
    }
    mCamera.release();
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
