/* 

 DETECT DIFFERENT BODY PARTS USING HAAR-LIKE FEATURES

 AUTHOR: PRATEEK JOSHI

*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Face detection
void detectFace( Mat frame );

// Eye detection
void detectEyes( Mat frame );

// Ear detection
void detectEars( Mat frame );

// Mouth detection
void detectMouth( Mat frame );

// Nose detection
void detectNose( Mat frame );

// Nose detection
void detectSmile( Mat frame );

// Upper body detection
void detectUpperBody( Mat frame );

// Nose detection
void detectLowerBody( Mat frame );

// Nose detection
void detectFullBody( Mat frame );