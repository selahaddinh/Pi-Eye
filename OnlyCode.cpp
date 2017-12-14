extern "C" {
  #include "arduino-serial-lib.h"
}

#include<stdio.h>
#include<string.h>
#include<unistd.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <time.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <math.h>

using namespace std;
using namespace cv;

//This method works for pinHole webcam. That means there is no focus point
//You need to update Positions of the n points according to target for different targets
//You need to update pcZ value for different webcams
//You need to update k and j value for different resolutions
//It returns orientation in radians and position in millimeter
void posEstimation(vector<Point2f> &pointsOnScene,int faceID,double *&pos,double *&rot,const int numberOfPoints,int CamResolutionX,int CamResolutionY);
void pathPlanning(double *&pos,double *&rot,double loopTime,int faceID,double *&motorVelocity,int &previousPathPlanningMode,double stateX[],double stateZ[],double stateW[],double velocityX[],double velocityZ[],double velocityW[]);

//Webcam Resolution can be changed from here
//To switch between multiple webcam, change webcamID
//You need to update images and numberOfFace if target changes
//You may need to change address of serial port of arduino
int main()
{
	//START of DEFINING VARIABLES
	const int numberOfFace = 4;	//Number Of Target's faces
	const int numberOfPoints = 5;	//Number of points that used for posEstimation
	const int minimalKeyPoints = 8;	//Minimal Keypoint to assume we found the object	
	const int findNearestNeighbors = 2;	//Value for Brute force matching
	const float nndrRatio = 0.7f;		//Value for Brute force matching
	//Values to put information on screen
	const int pointOfInfoX = 5;		//X axes of position of first information
	const int pointOfInfoY = 20;		//Y axes of position of first information
	const int nextInformation = 25;	//Difference between informations
	const double scaleOfText = 0.5;
	const Scalar colorOfText = {0, 0, 0};
	const int thicknessOfText = 1;
	
	const double scaleOfPoints = 1;
	const Scalar colorOfPoints = {128, 128, 128};
	const int thicknessOfPoints = 2;
	
	const double radiusOfCircle = 4;
	const Scalar colorOfCircle = {0, 0, 255};
	const int thicknessOfCircle = 4;
	//Webcam Resolutions
	int CamResolutionX = 640;
	int CamResolutionY = 480;
	int webcamID = 0;			//It determine which webcam to use between multiple webcam. webcamID=1 uses second webcam on computer
	VideoCapture capWebcam;		//Opens Webcam
	
	const Point2f pnt(CamResolutionX/2,CamResolutionY/2);	// To show middle point on the screen
	char charCheckForEscKey = 0;	//To stop program press ESC
	
	Mat imgOriginal[numberOfFace];//Reads images of objects
	Mat imgScene;	//Reads image of scene
	vector<KeyPoint> objectKeypoints[numberOfFace];	//Keeps Keypoints of images
	Mat objectDescriptors[numberOfFace];			//Keeps definations of keypoints
	vector<Point2f> *pointsOnObject = new vector<Point2f>[numberOfFace];	//Holds desired points on objects
	vector<Point2f> pointsOnObject0(numberOfPoints);
	vector<Point2f> pointsOnObject1(numberOfPoints);
	vector<Point2f> pointsOnObject2(numberOfPoints);
	vector<Point2f> pointsOnObject3(numberOfPoints);
	//Creates BRISK objects to detect feature points and exracts their descriptions
	Ptr<FeatureDetector> detector = BRISK::create();
	Ptr<cv::DescriptorExtractor> extractor = BRISK::create();
	//Brute Force Descriptor. It matches feature points
	BFMatcher matcher(NORM_HAMMING);
	//These are for serial communication with arduino. Used for precompiled code from external source
	int fd;
	int baudrate = 9600;
	//Holds position and orientation information of target according to chaser and motorVelocities of chaser
	double *pos = new double[3];
	double *rot = new double[3];
	double *motorVelocity =new double[3];
	double loopTime = 50;
	
	int faceID=-1;				// 2=right 1=back 0=left 3=front -1=no face found
	int previousPathPlanningMode=0;//To memorize previous path planning. It provides continuance for path planning
	double stateX[10] = {0,0,0,0,0,0,0,0,0,0};
	double stateZ[10] = {0,0,0,0,0,0,0,0,0,0};
	double stateW[10] = {0,0,0,0,0,0,0,0,0,0};
	double velocityX[10] = {0,0,0,0,0,0,0,0,0,0};
	double velocityZ[10] = {0,0,0,0,0,0,0,0,0,0};
	double velocityW[10] = {0,0,0,0,0,0,0,0,0,0};
	//Measures the time of the loop
	auto start_time = chrono::high_resolution_clock::now();
	auto end_time = chrono::high_resolution_clock::now();
	//END of DEFINING VARIABLES
/*
	//Reads images of target
	imgOriginal[0] = imread("./box0.jpg",IMREAD_GRAYSCALE);
	imgOriginal[1] = imread("./box1.jpg",IMREAD_GRAYSCALE);
	imgOriginal[2] = imread("./box2.jpg",IMREAD_GRAYSCALE);
	imgOriginal[3] = imread("./box3.jpg",IMREAD_GRAYSCALE);
*/
	//Reads images of target
	imgOriginal[0] = imread("./box0.jpg");
	imgOriginal[1] = imread("./box1.jpg");
	imgOriginal[2] = imread("./box2.jpg");
	imgOriginal[3] = imread("./box3.jpg");

	//Desired Points on Object to be used in poseEstimation. Later will be transformed in homography
	pointsOnObject[0] = pointsOnObject0;
	pointsOnObject[1] = pointsOnObject1;
	pointsOnObject[2] = pointsOnObject2;
	pointsOnObject[3] = pointsOnObject3;
	for(int tempFace=0; tempFace<numberOfFace; tempFace++){
		pointsOnObject[tempFace][0] = cvPoint(imgOriginal[tempFace].cols/2, 0);
		pointsOnObject[tempFace][1] = cvPoint(imgOriginal[tempFace].cols/2, imgOriginal[tempFace].rows);
		pointsOnObject[tempFace][2] = cvPoint(imgOriginal[tempFace].cols/2, imgOriginal[tempFace].rows/2);
		pointsOnObject[tempFace][3] = cvPoint(0, imgOriginal[tempFace].rows/2);
		pointsOnObject[tempFace][4] = cvPoint(imgOriginal[tempFace].cols, imgOriginal[tempFace].rows/2);	
	}
	
	//Initialize Webcam
	capWebcam.open(webcamID);
	if(capWebcam.isOpened() == false){
		cout << "error: Webcam is not available"<<endl;
		return 0;
	}
	//Sets webcam's resolution
	capWebcam.set(CAP_PROP_FRAME_WIDTH,CamResolutionX);
	capWebcam.set(CAP_PROP_FRAME_HEIGHT,CamResolutionY);

	//Initialize Serial Port
	fd = serialport_init("/dev/ttyUSB0", baudrate);
	if( fd == -1 ) cout<<"serial port not opened";
	cout<<"fd value: "<<fd<<endl;
	serialport_flush(fd);
	
	//Extracting keypoints of object and their descriptions 
	for(int i=0; i<numberOfFace; i++){
		//Detects Brisk Keypoints of images of target
		detector->detect(imgOriginal[i], objectKeypoints[i]);
		//Extracts descriptions of the keypoints
		extractor->compute(imgOriginal[i], objectKeypoints[i], objectDescriptors[i]);
	}

	start_time = chrono::high_resolution_clock::now();
	//GENERAL LOOP STARTS. take picture, estimate position and planning the path. Stop if there is error on webcam or ESC is pressed
	while (charCheckForEscKey != 27 && capWebcam.read(imgScene)){
//		capWebcam.read(imgScene);
//		cvtColor(imgScene, imgScene, CV_BGR2GRAY);
		//Take care of cam delay!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//Those values will be destructed in each loop
		vector<KeyPoint> sceneKeypoints;	//Keeps Keypoints of images
		Mat sceneDescriptors;			//Keeps definations of keypoints
		vector<Point2f> matchedPointsOfObject[numberOfFace];	//Holds position of object keypoints which are matched with scene keypoints
		vector<Point2f> matchedPointsofScene[numberOfFace];	//Holds position of scene keypoints which are matched with object keypoints
		vector<vector <DMatch> >matches[numberOfFace]; 		//Holds all matches between object and scene
		vector<Point2f> pointsOnScene(numberOfPoints);		//Holds desired points on objects on the scene
		int numberOfGoodMatches[numberOfFace]={0, 0, 0, 0};
		bool scanAllFaces=false;
		ostringstream SerialCommunicationText; //To send velocities to arduino
		//Informations to show
		Mat img_matches;	//Modified versions of scene photo. Used to show information on screen
		String faceInformationText;
		String PathPlanningText;
		ostringstream positionText;
		ostringstream orientationText;
		ostringstream numberOfGoodMatchesText;
		ostringstream loopTimeText;
		
		//Detects keypoints of scene and extracts their descriptions
		detector->detect(imgScene, sceneKeypoints);
		extractor->compute(imgScene, sceneKeypoints, sceneDescriptors);
		
		//START of FINDING FACE and its match points
		if(faceID != -1){
			//Finding matches between scene and the face
			matcher.knnMatch(objectDescriptors[faceID], sceneDescriptors, matches[faceID], findNearestNeighbors);
			//Finding positions of matched points and counting how many there are
			for(unsigned int i=0; i<matches[faceID].size(); ++i){
				// Apply NNDR (Nearest Neighbor Distance Ratio)
				if(matches[faceID].at(i).size() == 2 && matches[faceID].at(i).at(0).distance <= nndrRatio * matches[faceID].at(i).at(1).distance){
					//Adds the points of object keypoints which are well matched with the scene keypoints
					matchedPointsOfObject[faceID].push_back(objectKeypoints[faceID].at(matches[faceID].at(i).at(0).queryIdx).pt);
					//Adds the points of scene keypoints which are well matched with the object keypoints
					matchedPointsofScene[faceID].push_back(sceneKeypoints.at(matches[faceID].at(i).at(0).trainIdx).pt);
					//Counts number of good matches for specified face
					numberOfGoodMatches[faceID]++;
				}
			}
		}
		
		if(faceID == -1){
			scanAllFaces = true;
		}
		else if(numberOfGoodMatches[faceID] < minimalKeyPoints){
			scanAllFaces = true;
		}
		else{
			scanAllFaces = false;
		}
		
		if(scanAllFaces){
			for(int tempFaceId=0; tempFaceId<numberOfFace; tempFaceId++){
				if(tempFaceId != faceID){
					//Finding matches between scene and the face
					matcher.knnMatch(objectDescriptors[tempFaceId], sceneDescriptors, matches[tempFaceId], findNearestNeighbors);
					//Finding positions of matched points and counting how many there are
					for(unsigned int i=0; i<matches[tempFaceId].size(); ++i){
						// Apply NNDR (Nearest Neighbor Distance Ratio)
						if(matches[tempFaceId].at(i).size() == 2 && matches[tempFaceId].at(i).at(0).distance <= nndrRatio * matches[tempFaceId].at(i).at(1).distance){
							//Adds the points of object keypoints which are well matched with the scene keypoints
							matchedPointsOfObject[tempFaceId].push_back(objectKeypoints[tempFaceId].at(matches[tempFaceId].at(i).at(0).queryIdx).pt);
							//Adds the points of scene keypoints which are well matched with the object keypoints
							matchedPointsofScene[tempFaceId].push_back(sceneKeypoints.at(matches[tempFaceId].at(i).at(0).trainIdx).pt);
							//Counts number of good matches for specified face
							numberOfGoodMatches[tempFaceId]++;
						}
					}
				}
			}
			
			faceID=0;
			for(int tempID=1; tempID<numberOfFace; tempID++){
				if(numberOfGoodMatches[tempID]>numberOfGoodMatches[faceID]){
					faceID = tempID;
				}
			}
			
			if(numberOfGoodMatches[faceID] < minimalKeyPoints){
				faceID=-1;
			}
		}
		//END of FINDING FACE and its match points
		
		//POSE ESTIMATION
		if(faceID != -1){
			// Finding Homography to find desired points on object on scene
			Mat H = findHomography(matchedPointsOfObject[faceID], matchedPointsofScene[faceID], RANSAC, 1.0);

			if(!H.empty()){
				//Finding Desired points on object on scene to be used in poseEstimation
				perspectiveTransform(pointsOnObject[faceID], pointsOnScene, H);
				
				//Calling posEstimation
				posEstimation(pointsOnScene,faceID,pos,rot,numberOfPoints,CamResolutionX,CamResolutionY);
				
				//START of setting rotation value between -pi and +pi
				if(faceID==0){
					rot[1]+=M_PI/2;
				}
				else if(faceID==1){
					rot[1]+=M_PI;
				}
				else if(faceID==2){
					rot[1]-=M_PI/2;
				}

				if(rot[1]>M_PI){
					rot[1]-=2*M_PI;
				}
				else if(rot[1]<=-M_PI){
					rot[1]+=2*M_PI;
				}
				//END of setting rotation value between -pi and +pi

				//Modifing pos to measured from center of target to center of chaser
				pos[0]=pos[0]+65*sin(rot[1]);
				pos[2]=pos[2]+65*cos(rot[1]);
				pos[2]=pos[2]+60;
			}
			
			else{
				faceID = -1;
				faceInformationText = "Homography not exist!";
			}//END OF POS ESTIMATION

		}
		else{
			faceID = -1;
			faceInformationText = "Object is not found!";
		}
		
		//Measuring loop time
		end_time = chrono::high_resolution_clock::now();
		loopTime = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
		loopTimeText << "Total Loop Time: " << loopTime << " ms";
		start_time = chrono::high_resolution_clock::now();
		
		//Calling pathPlanningcout<<"fonksiyon çıkışı"<<endl;
		pathPlanning(pos,rot,loopTime,faceID,motorVelocity,previousPathPlanningMode,stateX,stateZ,stateW,velocityX,velocityZ,velocityW);
		
		//Sending motorVelocity values to chaser
		SerialCommunicationText << motorVelocity[0] << "," << motorVelocity[1] << "," << motorVelocity[2] << "T";
		if(serialport_write(fd,SerialCommunicationText.str().c_str())==-1) cout<<"error in Serial Communication";
		
		//Wait to get key press, if any (ms)
		charCheckForEscKey = waitKey(2);
		
		//PUTTING INFO	
		if(faceID == 0) faceInformationText = "LEFT";
		else if(faceID == 1) faceInformationText = "BACK";
		else if(faceID == 2) faceInformationText = "RIGHT";
		else if(faceID == 3) faceInformationText = "FRONT";
		
		if(previousPathPlanningMode==0) PathPlanningText = "Path planning : Linear+Circular";
		else if(previousPathPlanningMode==1) PathPlanningText = "Path planning : Linear";
		else if(previousPathPlanningMode==2) PathPlanningText = "Path planning : Circular Correction";
		else if(previousPathPlanningMode==3) PathPlanningText = "Path planning : Circular";
		else if(previousPathPlanningMode==4) PathPlanningText = "Path planning : DOCKING!!!";
		else if(previousPathPlanningMode==5) PathPlanningText = "Path planning : DANGER!!!";
		else PathPlanningText = "Path planning : Searching";

		positionText << "POSITION: " << (int)(pos[0]/10) << " " << (int)(pos[1]/10) << " " << (int)(pos[2]/10) << "(cm)";
		orientationText << "ORIENTATION: " << (int)(rot[0]*180/M_PI) << " " << (int)(rot[1]*180/M_PI) << " " << (int)(rot[2]*180/M_PI) << "(deg)";
		numberOfGoodMatchesText << "Number of Good Matches: " << numberOfGoodMatches[faceID];
/*
		//SHOW INFO ON SCREEN
		//drawKeypoints(imgScene, sceneKeypoints, img_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		img_matches = imgScene;

		putText(img_matches, faceInformationText, {pointOfInfoX,pointOfInfoY}, FONT_HERSHEY_SIMPLEX, scaleOfText, colorOfText, thicknessOfText);
		putText(img_matches, positionText.str(), {pointOfInfoX,pointOfInfoY+nextInformation}, FONT_HERSHEY_SIMPLEX, scaleOfText, colorOfText, thicknessOfText);
		putText(img_matches, orientationText.str(), {pointOfInfoX,pointOfInfoY+2*nextInformation}, FONT_HERSHEY_SIMPLEX, scaleOfText, colorOfText, thicknessOfText);
		putText(img_matches, numberOfGoodMatchesText.str(), {pointOfInfoX,pointOfInfoY+3*nextInformation}, FONT_HERSHEY_SIMPLEX, scaleOfText, colorOfText, thicknessOfText);
		putText(img_matches, PathPlanningText, {pointOfInfoX,pointOfInfoY+4*nextInformation}, FONT_HERSHEY_SIMPLEX, scaleOfText, colorOfText, thicknessOfText);
		putText(img_matches, loopTimeText.str(), {pointOfInfoX,pointOfInfoY+5*nextInformation}, FONT_HERSHEY_SIMPLEX, scaleOfText, colorOfText, thicknessOfText);

		putText(img_matches, "0", pointsOnScene[0], FONT_HERSHEY_SIMPLEX, scaleOfPoints, colorOfPoints, thicknessOfPoints);
		putText(img_matches, "1", pointsOnScene[1], FONT_HERSHEY_SIMPLEX, scaleOfPoints, colorOfPoints, thicknessOfPoints);
		putText(img_matches, "2", pointsOnScene[2], FONT_HERSHEY_SIMPLEX, scaleOfPoints, colorOfPoints, thicknessOfPoints);
		putText(img_matches, "3", pointsOnScene[3], FONT_HERSHEY_SIMPLEX, scaleOfPoints, colorOfPoints, thicknessOfPoints);
		putText(img_matches, "4", pointsOnScene[4], FONT_HERSHEY_SIMPLEX, scaleOfPoints, colorOfPoints, thicknessOfPoints);

		circle(img_matches, pointsOnScene[0], radiusOfCircle, colorOfCircle, thicknessOfCircle);
		circle(img_matches, pointsOnScene[1], radiusOfCircle, colorOfCircle, thicknessOfCircle);
		circle(img_matches, pointsOnScene[2], radiusOfCircle, colorOfCircle, thicknessOfCircle);
		circle(img_matches, pointsOnScene[3], radiusOfCircle, colorOfCircle, thicknessOfCircle);
		circle(img_matches, pointsOnScene[4], radiusOfCircle, colorOfCircle, thicknessOfCircle);
		circle(img_matches, pnt , 5, Scalar(0,0,0), 4, 3);

		line(img_matches, pointsOnScene[0] , pointsOnScene[1] , Scalar(0, 255, 0), 4);
		line(img_matches, pointsOnScene[3] , pointsOnScene[4] , Scalar(0, 255, 0), 4);

		imshow("Docking System", img_matches);
*/
		//SHOW INFO ON COMMAND LINE
		cout << string(70, '\n');
		cout << faceInformationText << endl;
		cout << positionText.str() << endl;
		cout << orientationText.str() << endl;
		cout << numberOfGoodMatchesText.str() << endl;
		cout << PathPlanningText << endl;
		cout << loopTimeText.str() << endl;
	}
	//END OF LOOP

	//Stopping chaser
	motorVelocity[0] = 0;
	motorVelocity[1] = 0;
	motorVelocity[2] = 0;
	
	ostringstream ss;
	ss << motorVelocity[0] << "," << motorVelocity[1] << "," << motorVelocity[2] << "T";
	if(serialport_write(fd,ss.str().c_str())==-1) cout<<"error in Serial Communication";

	//closing port
	int serialport_close(fd);

	return 0;
}

void pathPlanning(double *&pos,double *&rot,double loopTime,int faceID,double *&motorVelocity,int &previousPathPlanningMode,double stateX[],double stateZ[],double stateW[],double velocityX[],double velocityZ[],double velocityW[])
{
	const double ERROR_TOLERANCE=30;// 3 cm error tolerance
	const double safeDistance=600;
	const double dockDistance=440;
	const double r=120;	//radius of chaser robot

	double state[3];	//Holds x,z pozition and orientation around y axis
	double limitOrientation=0;//if orientation of target relative to position of chaser is more than limitOrientation, there will be circular path
	double netDistance=0;//net distance between target and chaser
	double chaserOrientation=0;//chaser's orientation relative to targets position
	double maxVelocity; //it is used to normalize motor velocities
	double Vx=0;
	double Vz=0;
	double w=0;
	double avarageStateW = 0;
	double distanceLeft;
	//velocity is calculated in mm/s
	double c=2.887;
	double T=23.25;
	
	distanceLeft = 0;

	state[0]=pos[0];//x distance of target (mm)
	state[1]=pos[2];//z distance of target (mm)
	state[2]=rot[1];//orientation of target relative to chaser (rad)

	//Calculating the variables
	netDistance=sqrt((state[0]*state[0])+(state[1]*state[1]));
	chaserOrientation=atan(state[0]/state[1]);
	/*
	avarageStateW = 0;
	for(int i=0; i<10; i++){
		avarageStateW = avarageStateW + stateW[i];
	}
	avarageStateW = avarageStateW / 10;
	*/
	//Docking Process
	if(fabs(state[0]) < ERROR_TOLERANCE && state[1] < safeDistance+ERROR_TOLERANCE && fabs(state[2]) < 20*M_PI/180 && faceID!=-1){
		previousPathPlanningMode=4;
		
		Vz=state[1]-dockDistance;
		w=4*(state[2]-3*M_PI/180);
		Vx=4*state[0]-w*dockDistance;
		/*
		w=state[2]+chaserOrientation;
		Vz=-state[2]*netDistance*sin(chaserOrientation)+state[1]-netDistance*cos(chaserOrientation);
		Vx=-state[2]*netDistance*cos(chaserOrientation)+state[0]-netDistance*sin(chaserOrientation);
		
		w = w/time;
		Vz = Vz/time;
		Vx = Vx/time;
		
		Vx = T+Vx*c;
		Vz = T+Vz*c;
		w = T+w*c;
		
		if(state[2] > dockDistance) Vz = Vz + 3;
		*/
		//calculating the ratio of motor velocities according to desired Vx,Vy and w velocities
		
		motorVelocity[0]=(Vx/2)-Vz*(sqrt(3)/2)+w*r;
		motorVelocity[1]=-Vx+w*r;
		motorVelocity[2]=(Vx/2)+Vz*(sqrt(3)/2)+w*r;	

		//Finding maximum velocity between motor velocities
		maxVelocity=fabs(motorVelocity[0]);
		for(int i=1;i<3;i++){
			if(fabs(motorVelocity[i])>maxVelocity)
				maxVelocity=fabs(motorVelocity[i]);
		}
		if(maxVelocity!=0){
			//maxVelocity=maxVelocity*1.8;
			maxVelocity=maxVelocity/((5.0/9) + ((netDistance-dockDistance)/2340));

			motorVelocity[0]=motorVelocity[0]/maxVelocity;
			motorVelocity[1]=motorVelocity[1]/maxVelocity;
			motorVelocity[2]=motorVelocity[2]/maxVelocity;
			
			//According to motor shield max value 255
			motorVelocity[0]=(int) (motorVelocity[0]*255);
			motorVelocity[1]=(int) (motorVelocity[1]*255);
			motorVelocity[2]=(int) (motorVelocity[2]*255);
		}
	}
	//No Face Found
	else{
		if(faceID == -1){
			previousPathPlanningMode=6;
			Vx=0;
			Vz=0;
			if(state[0] < 0)
				w=-1;
			else
				w=1;
			w=0;
		}
		//If netDistance is more than safeDistance+ERROR_TOLERANCE it will use linear path planning
		//If netDistance is greater than safeDistance it will go linear if previous path planning is linear. It provides continuity
		//Linear Path Planning
		else if(netDistance-ERROR_TOLERANCE>=safeDistance || (netDistance>safeDistance && previousPathPlanningMode<2 ) ){
			limitOrientation=acos(safeDistance/netDistance);
			//linear+circular path planning
			if(fabs(state[2]-chaserOrientation)>limitOrientation){
				previousPathPlanningMode=0;
				if((state[2]-chaserOrientation<0 && state[2]-chaserOrientation>-M_PI) || state[2]-chaserOrientation>M_PI){	//decide whether go from left or right
					limitOrientation=-limitOrientation;
				}

				Vx=state[0]-safeDistance*sin(limitOrientation+chaserOrientation);
				Vz=state[1]-safeDistance*cos(limitOrientation+chaserOrientation);
				w=limitOrientation+chaserOrientation;
				distanceLeft = sqrt(Vx*Vx+Vz*Vz) + fabs((state[2]-limitOrientation)*safeDistance);
			}
			//Just linear path planning
			else{
				previousPathPlanningMode=1;
				Vx=state[0]-safeDistance*sin(state[2]);
				Vz=state[1]-safeDistance*cos(state[2]);
				w=state[2];
				distanceLeft = sqrt(Vx*Vx+Vz*Vz);
			}
		}
		//Just circular path planning
		else if(fabs(netDistance-safeDistance)<=ERROR_TOLERANCE){
			//Chaser just rotating around itself
			if((fabs(chaserOrientation)>=15*M_PI/180) || (fabs(chaserOrientation)>=10*M_PI/180 && previousPathPlanningMode==1)){
				previousPathPlanningMode=2;
				w=chaserOrientation;
				Vz=0;
				Vx=0;
			}
			//chaser rotate around target while looking at target
			else{
				previousPathPlanningMode=3;
				w=state[2]+chaserOrientation;
				Vz=-state[2]*netDistance*sin(chaserOrientation)+state[1]-safeDistance*cos(chaserOrientation);
				Vx=-state[2]*netDistance*cos(chaserOrientation)+state[0]-safeDistance*sin(chaserOrientation);
				distanceLeft = fabs((state[2]-chaserOrientation)*safeDistance);
			}
		}
		//DANGER. running away from target
		else{
			previousPathPlanningMode=5;
			Vx=-state[0];
			Vz=-state[1];
			w=chaserOrientation;
			distanceLeft = sqrt(Vx*Vx+Vz*Vz);
		}
		

		//calculating the ratio of motor velocities according to desired Vx,Vy and w velocities
		motorVelocity[0]=(Vx/2)-Vz*(sqrt(3)/2)+w*r;
		motorVelocity[1]=-Vx+w*r;
		motorVelocity[2]=(Vx/2)+Vz*(sqrt(3)/2)+w*r;	

		//Finding maximum velocity between motor velocities
		maxVelocity=fabs(motorVelocity[0]);
		for(int i=1;i<3;i++){
			if(fabs(motorVelocity[i])>maxVelocity)
				maxVelocity=fabs(motorVelocity[i]);
		}

		if(maxVelocity!=0){
			if(distanceLeft < 2*ERROR_TOLERANCE && previousPathPlanningMode!=6)
				maxVelocity=maxVelocity/((6.0/9) + ((distanceLeft)/180));

			motorVelocity[0]=motorVelocity[0]/maxVelocity;
			motorVelocity[1]=motorVelocity[1]/maxVelocity;
			motorVelocity[2]=motorVelocity[2]/maxVelocity;

			//According to motor shield max value 255
			motorVelocity[0]=(int) (motorVelocity[0]*255);
			motorVelocity[1]=(int) (motorVelocity[1]*255);
			motorVelocity[2]=(int) (motorVelocity[2]*255);
		}
	
		//Calculating normalized Vx,Vy and w velocities
		Vx = (motorVelocity[0] - 2*motorVelocity[1] + motorVelocity[2]) / 3;
		Vz = (motorVelocity[2] - motorVelocity[0]) / sqrt(3);
		w = (motorVelocity[0] + motorVelocity[1] + motorVelocity[2]) / (3*r);
	}

	for(int i=8; i>=0; i--){
		stateX[i+1] = stateX[i];
		stateZ[i+1] = stateZ[i];
		stateW[i+1] = stateW[i];
		velocityX[i+1] = velocityX[i];
		velocityZ[i+1] = velocityZ[i];
		velocityW[i+1] = velocityW[i];
	}
	stateX[0] = state[0];
	stateZ[0] = state[1];
	stateW[0] = state[2];
	velocityX[0] = Vx;
	velocityZ[0] = Vz;
	velocityW[0] = w;

//	cout<<"avarage w: " << avarageStateW<<endl;
}


void posEstimation(vector<Point2f> &pointsOnScene,int faceID,double *&pos,double *&rot,const int numberOfPoints,int CamResolutionX,int CamResolutionY)
{
	//Positions of those n points according to target
	double p1x[numberOfPoints]={0, 0, 0, -65, 65};
	double p1y[numberOfPoints]={-145, 145, 0, 0, 0};
	double p1z[numberOfPoints]={0,0,0,0,0};
	//Positions of those n points on webcam
	double pcX[numberOfPoints];
	double pcY[numberOfPoints];
	double pcZ[numberOfPoints];
	//Positions of those n points according to chaser
	double p0x[numberOfPoints];
	double p0y[numberOfPoints];
	double p0z[numberOfPoints];
	//Square of L12 and L45
	double l12=pow((p1x[0]-p1x[1]),2)+pow((p1y[0]-p1y[1]),2)+pow((p1z[0]-p1z[1]),2);
	double l45=pow((p1x[3]-p1x[4]),2)+pow((p1y[3]-p1y[4]),2)+pow((p1z[3]-p1z[4]),2);
	//p0z[2] = a*p0z[1] and p0z[4] = b*p0z[3]
	double a;
	double b;
	//Unit Vectors of target according to chaser
	double unitx[3];
	double unity[3];
	double unitz[3];
	double normalizeUnitx;
	double normalizeUnity;
	//Orientation of target according to chaser
	double roll; //Around x axes. [-pi,+pi]
	double pitch; //Around y axes. [-pi,+pi]
	double yaw; //Around z axes. [-pi/2,+pi/2]

	//Positions of points on webcam
	pcX[0]=pointsOnScene[0].x-CamResolutionX/2;
	pcX[1]=pointsOnScene[1].x-CamResolutionX/2;
	pcX[2]=pointsOnScene[2].x-CamResolutionX/2;
	pcX[3]=pointsOnScene[3].x-CamResolutionX/2;
	pcX[4]=pointsOnScene[4].x-CamResolutionX/2;

	pcY[0]=pointsOnScene[0].y-CamResolutionY/2;
	pcY[1]=pointsOnScene[1].y-CamResolutionY/2;
	pcY[2]=pointsOnScene[2].y-CamResolutionY/2;
	pcY[3]=pointsOnScene[3].y-CamResolutionY/2;
	pcY[4]=pointsOnScene[4].y-CamResolutionY/2;

	for(int j=0;j<numberOfPoints;j++){
	pcZ[j]=814,3157;
	}

	//p0z[2] = a*p0z[1] and p0z[4] = b*p0z[3]
	a = ((pcY[0]+pcX[0]) - (pcY[2]+pcX[2])) / ((pcY[2]+pcX[2]) - (pcY[1]+pcX[1]));
	b = ((pcY[3]+pcX[3]) - (pcY[2]+pcX[2])) / ((pcY[2]+pcX[2]) - (pcY[4]+pcX[4]));

	//Positions of those n points according to chaser
	p0z[0]=sqrt((l12*pow(pcZ[0],2)) / (pow((pcZ[0]-a*pcZ[1]),2) + pow((pcY[0]-a*pcY[1]),2) + pow((pcX[0]-a*pcX[1]),2)));
	p0z[1]=a*p0z[0];
	p0z[2]=(p0z[0]+p0z[1])/2;
	p0z[3]=sqrt(l45*pow(pcZ[3],2) / (pow((pcZ[3]-a*pcZ[4]),2) + pow((pcY[3]-a*pcY[4]),2) + pow((pcX[3]-a*pcX[4]),2)));
	p0z[4]=b*p0z[3];

	p0x[0]=p0z[0]*pcX[0]/pcZ[0];
	p0x[1]=p0z[1]*pcX[1]/pcZ[1];
	p0x[2]=p0z[2]*pcX[2]/pcZ[2];
	p0x[3]=p0z[3]*pcX[3]/pcZ[3];
	p0x[4]=p0z[4]*pcX[4]/pcZ[4];

	p0y[0]=p0z[0]*pcY[0]/pcZ[0];
	p0y[1]=p0z[1]*pcY[1]/pcZ[1];
	p0y[2]=p0z[2]*pcY[2]/pcZ[2];
	p0y[3]=p0z[3]*pcY[3]/pcZ[3];
	p0y[4]=p0z[4]*pcY[4]/pcZ[4];

	//Unit Vectors of target according to chaser
	normalizeUnitx = sqrt(pow(p0x[4]-p0x[3],2)+pow(p0y[4]-p0y[3],2)+pow(p0z[4]-p0z[3],2));
	unitx[0]=(p0x[4]-p0x[3])/normalizeUnitx;
	unitx[1]=(p0y[4]-p0y[3])/normalizeUnitx;
	unitx[2]=(p0z[4]-p0z[3])/normalizeUnitx;

	normalizeUnity = sqrt(pow(p0x[1]-p0x[0],2)+pow(p0y[1]-p0y[0],2)+pow(p0z[1]-p0z[0],2));
	unity[0]=(p0x[1]-p0x[0])/normalizeUnity;
	unity[1]=(p0y[1]-p0y[0])/normalizeUnity;
	unity[2]=(p0z[1]-p0z[0])/normalizeUnity;

	unitz[0]=unitx[1]*unity[2]-unitx[2]*unity[1];
	unitz[1]=unitx[2]*unity[0]-unitx[0]*unity[2];
	unitz[2]=unitx[0]*unity[1]-unitx[1]*unity[0];

	//Orientation of target according to chaser
	roll = atan2(unity[2],unitz[2]);	//Around x axes. [-pi,+pi]
	pitch = asin(-1*unitx[2]);			//Around y axes. [-pi,+pi]
	yaw = atan2(unitx[1],unitx[0]);		//Around z axes. [-pi/2,+pi/2]

	//Result for position and orientation of target according to chaser
	pos[0]=p0x[2];
	pos[1]=p0y[2];
	pos[2]=p0z[2];

	rot[0]=roll;
	rot[1]=pitch;
	rot[2]=yaw;
}
