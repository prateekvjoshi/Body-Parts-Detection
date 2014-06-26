/* 

 DETECT DIFFERENT BODY PARTS USING HAAR-LIKE FEATURES

 AUTHOR: PRATEEK JOSHI

*/

#include "detectBodyParts.h"

using namespace std;
using namespace cv;

String face_cascade_name = "CascadeFiles/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "CascadeFiles/haarcascade_eye_tree_eyeglasses.xml";
String leftear_cascade_name = "CascadeFiles/haarcascade_mcs_leftear.xml";
String rightear_cascade_name = "CascadeFiles/haarcascade_mcs_rightear.xml";
String mouth_cascade_name = "CascadeFiles/haarcascade_mcs_mouth.xml";
String nose_cascade_name = "CascadeFiles/haarcascade_mcs_nose.xml";
String smile_cascade_name = "CascadeFiles/haarcascade_smile.xml";
String upperbody_cascade_name = "CascadeFiles/haarcascade_mcs_upperbody.xml";
String lowerbody_cascade_name = "CascadeFiles/haarcascade_lowerbody.xml";
String fullbody_cascade_name = "CascadeFiles/haarcascade_fullbody.xml";

// Face detection using Haar-like features
void detectFace( Mat frame )
{
    // Load the cascade
    CascadeClassifier face_cascade;
    if( !face_cascade.load( face_cascade_name ) ) { cout << "Error loading face cascade file\n" << endl; return; };
    
    vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    imshow( "Face Detection", frame );
}


// Eye detection using Haar-like features
void detectEyes( Mat frame )
{
    // Load the cascades
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    if( !face_cascade.load( face_cascade_name ) ) { cout << "Error loading face cascade file\n" << endl; return; };
    if( !eyes_cascade.load( eyes_cascade_name ) ) { cout << "Error loading eyes cascade file\n" << endl; return; };
    
    vector<Rect> faces;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        
        // In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, center, radius, Scalar( 255, 255, 255 ), 4, 8, 0 );
        }
    }
    
    imshow( "Eye Detection", frame );
}


// Ear detection using Haar-like features
void detectEars( Mat frame )
{
    // Load the cascade
    CascadeClassifier leftear_cascade;
    CascadeClassifier rightear_cascade;
    if( !leftear_cascade.load( leftear_cascade_name ) ) { cout << "Error loading left ear cascade file\n" << endl; return; };
    if( !rightear_cascade.load( rightear_cascade_name ) ) { cout << "Error loading right ear cascade file\n" << endl; return; };
    
    vector<Rect> leftears, rightears;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect left ears
    leftear_cascade.detectMultiScale( frame_gray, leftears, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    // Detect right ears
    rightear_cascade.detectMultiScale( frame_gray, rightears, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < leftears.size(); i++ )
    {
        Point center( leftears[i].x + leftears[i].width*0.5, leftears[i].y + leftears[i].height*0.5 );
        ellipse( frame, center, Size( leftears[i].width*0.5, leftears[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 0 ), 4, 8, 0 );
    }
    
    for( size_t i = 0; i < rightears.size(); i++ )
    {
        Point center( rightears[i].x + rightears[i].width*0.5, rightears[i].y + rightears[i].height*0.5 );
        ellipse( frame, center, Size( rightears[i].width*0.5, rightears[i].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 4, 8, 0 );
    }
    
    imshow( "Ear Detection", frame );
}


// Mouth detection using Haar-like features
void detectMouth( Mat frame )
{
    // Load the cascade
    CascadeClassifier mouth_cascade;
    if( !mouth_cascade.load( mouth_cascade_name ) ) { cout << "Error loading mouth cascade file\n" << endl; return; };
    
    vector<Rect> mouths;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect mouth
    mouth_cascade.detectMultiScale( frame_gray, mouths, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < mouths.size(); i++ )
    {
        Point center( mouths[i].x + mouths[i].width*0.5, mouths[i].y + mouths[i].height*0.5 );
        ellipse( frame, center, Size( mouths[i].width*0.5, mouths[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    imshow( "Mouth Detection", frame );
}


// Nose detection using Haar-like features
void detectNose( Mat frame )
{
    // Load the cascade
    CascadeClassifier nose_cascade;
    if( !nose_cascade.load( nose_cascade_name ) ) { cout << "Error loading nose cascade file\n" << endl; return; };
    
    vector<Rect> noses;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect noses
    nose_cascade.detectMultiScale( frame_gray, noses, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < noses.size(); i++ )
    {
        Point center( noses[i].x + noses[i].width*0.5, noses[i].y + noses[i].height*0.5 );
        ellipse( frame, center, Size( noses[i].width*0.5, noses[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    imshow( "Nose Detection", frame );
}


// Smile detection using Haar-like features
void detectSmile( Mat frame )
{
    // Load the cascade
    CascadeClassifier smile_cascade;
    if( !smile_cascade.load( smile_cascade_name ) ) { cout << "Error loading smile cascade file\n" << endl; return; };
    
    vector<Rect> smiles;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    smile_cascade.detectMultiScale( frame_gray, smiles, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    for( size_t i = 0; i < smiles.size(); i++ )
    {
        Point center( smiles[i].x + smiles[i].width*0.5, smiles[i].y + smiles[i].height*0.5 );
        ellipse( frame, center, Size( smiles[i].width*0.5, smiles[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    imshow( "Smile Detection", frame );
}


// Upper body detection using Haar-like features
void detectUpperBody( Mat frame )
{
    // Load the cascade
    CascadeClassifier upperbody_cascade;
    if( !upperbody_cascade.load( upperbody_cascade_name ) ) { cout << "Error loading upper body cascade file\n" << endl; return; };
    
    vector<Rect> upperbody;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    upperbody_cascade.detectMultiScale( frame_gray, upperbody, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    
    for( size_t i = 0; i < upperbody.size(); i++ )
    {
        Rect temp = upperbody[i];
        temp.y += 100;
        rectangle(frame, temp, Scalar(255,255,255), 4, 8);
    }
    
    imshow( "Upper Body Detection", frame );
}


// Lower body detection using Haar-like features
void detectLowerBody( Mat frame )
{
    // Load the cascade
    CascadeClassifier lowerbody_cascade;
    if( !lowerbody_cascade.load( lowerbody_cascade_name ) ) { cout << "Error loading lower body cascade file\n" << endl; return; };
    
    vector<Rect> lowerbody;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    lowerbody_cascade.detectMultiScale( frame_gray, lowerbody, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    
    for( size_t i = 0; i < lowerbody.size(); i++ )
    {
        Point center( lowerbody[i].x + lowerbody[i].width*0.5, lowerbody[i].y + lowerbody[i].height*0.5 );
        ellipse( frame, center, Size( lowerbody[i].width*0.5, lowerbody[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    imshow( "Lower Body Detection", frame );
}


// Full body detection using Haar-like features
void detectFullBody( Mat frame )
{
    // Load the cascade
    CascadeClassifier fullbody_cascade;
    if( !fullbody_cascade.load( fullbody_cascade_name ) ) { cout << "Error loading full body cascade file\n" << endl; return; };
    
    vector<Rect> fullbody;
    Mat frame_gray;
    
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    // Detect faces
    fullbody_cascade.detectMultiScale( frame_gray, fullbody, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100) );
    
    for( size_t i = 0; i < fullbody.size(); i++ )
    {
        Point center( fullbody[i].x + fullbody[i].width*0.5, fullbody[i].y + fullbody[i].height*0.5 );
        ellipse( frame, center, Size( fullbody[i].width*0.5, fullbody[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
    
    imshow( "Full Body Detection", frame );
}
