Body-Parts-Detection
====================

You can detect different body parts using Haar-like features. Using this code, you can detect the following: Face, Eyes, Ears, Mouth, Nose, Upper Body, Lower Body, and Full Body. You can detect smiles too.

There is a folder called "CascadeFiles" which includes all the necessary files required for the Haar classifiers. There is a file called "CMakeLists.txt", which will help you build the project. If you didn't use cmake to build OpenCV, just use the .cpp file in your project and build it. To build using command line, follow the steps below to get it up and running:

	$ cmake .
	$ make
	$ ./main 

All the required functions are included in the file "detectBodyParts.cpp". The usage is shown in "main.cpp". In "main.cpp", you will see a bunch of macros defined at the beginning. You can enable detection of different body parts by enabling or disabling these macros. It will read the live input from your webcam and the output will be displayed on a bunch of windows.


