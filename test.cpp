#include<stdio.h>
#include<string.h>
#include<unistd.h>
#include<sys/socket.h>
#include<sys/types.h>
#include<netdb.h>
#include<arpa/inet.h>



#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <time.h>
#include <cstdio>
#include <ctime>
#include <sstream>

using namespace std;
using namespace cv;

void updateKalmanFilter( cv::KalmanFilter &KF, cv::Mat &measurement,
                        cv::Mat &translation_estimated, cv::Mat &rotation_estimated,cv::Mat &velocity,cv::Mat &angVelocity )
{
    // First predict, to update the internal statePre variable
    cv::Mat prediction = KF.predict();
    // The "correct" phase that is going to use the predicted value and our measurement
    cv::Mat estimated = KF.correct(measurement);
    // Estimated translation
    translation_estimated.at<double>(0) = estimated.at<double>(0);
    translation_estimated.at<double>(1) = estimated.at<double>(1);
    translation_estimated.at<double>(2) = estimated.at<double>(2);
    // Estimated euler angles
    cv::Mat eulers_estimated(3, 1, CV_64F);
    eulers_estimated.at<double>(0) = estimated.at<double>(6);
    eulers_estimated.at<double>(1) = estimated.at<double>(7);
    eulers_estimated.at<double>(2) = estimated.at<double>(8);
	//345 and 9,10,11 is get for velocity calculations
	velocity.at<double>(0) = estimated.at<double>(3);
    velocity.at<double>(1) = estimated.at<double>(4);
    velocity.at<double>(2) = estimated.at<double>(5);
	//Estimation of angular velocity
	angVelocity.at<double>(0) = estimated.at<double>(9);
    angVelocity.at<double>(1) = estimated.at<double>(10);
    angVelocity.at<double>(2) = estimated.at<double>(11);
    // Convert estimated quaternion to rotation matrix
    rotation_estimated = eulers_estimated;
}
void initKalmanFilter(cv::KalmanFilter &KF, int nStates, int nMeasurements, int nInputs, double dt)
{
    KF.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
    cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));    // set process noise//Q
    KF.measurementNoiseCov.at<double>(0,0) = 0.0033;
    KF.measurementNoiseCov.at<double>(0,1) = 0.0020;
    KF.measurementNoiseCov.at<double>(0,2) =   -0.0169;
    KF.measurementNoiseCov.at<double>(0,3) =    0.0161;
    KF.measurementNoiseCov.at<double>(0,4) =   -0.0074;
    KF.measurementNoiseCov.at<double>(0,5) =    0.0037;
    KF.measurementNoiseCov.at<double>(1,0) = 0.0020;
    KF.measurementNoiseCov.at<double>(1,1) =    0.0046;
    KF.measurementNoiseCov.at<double>(1,2) =   -0.0119 ;
    KF.measurementNoiseCov.at<double>(1,3) =   0.0379 ;
    KF.measurementNoiseCov.at<double>(1,4) =   0.0255 ;
    KF.measurementNoiseCov.at<double>(1,5) =   0.0167;
    KF.measurementNoiseCov.at<double>(2,0)  = -0.0169 ;
    KF.measurementNoiseCov.at<double>(2,1) =  -0.0119  ;
    KF.measurementNoiseCov.at<double>(2,2) =  0.1532   ;
    KF.measurementNoiseCov.at<double>(2,3) = 0.3089   ;
    KF.measurementNoiseCov.at<double>(2,4) = 0.1754  ;
    KF.measurementNoiseCov.at<double>(2,5) = -0.0046;
    KF.measurementNoiseCov.at<double>(3,0) =  0.0161;
    KF.measurementNoiseCov.at<double>(3,1) =    0.0379 ;
    KF.measurementNoiseCov.at<double>(3,2) =   0.3089  ;
    KF.measurementNoiseCov.at<double>(3,3) = 14.6365  ;
    KF.measurementNoiseCov.at<double>(3,4) =  1.0490  ;
    KF.measurementNoiseCov.at<double>(3,5) =  0.2065;
    KF.measurementNoiseCov.at<double>(4,0) = -0.0074  ;
    KF.measurementNoiseCov.at<double>(4,1) =  0.0255  ;
    KF.measurementNoiseCov.at<double>(4,2) = 0.1754   ;
    KF.measurementNoiseCov.at<double>(4,3) = 1.0490   ;
    KF.measurementNoiseCov.at<double>(4,4) = 1.2979 ;
    KF.measurementNoiseCov.at<double>(4,5) =   0.2683;
    KF.measurementNoiseCov.at<double>(5,0) = 0.0037  ;
    KF.measurementNoiseCov.at<double>(5,1) =  0.0167  ;
    KF.measurementNoiseCov.at<double>(5,2) = -0.0046  ;
    KF.measurementNoiseCov.at<double>(5,3) =  0.2065  ;
    KF.measurementNoiseCov.at<double>(5,4) =  0.2683  ;
    KF.measurementNoiseCov.at<double>(5,5) =  0.1090;
    cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));             // error covariance//p_init
    /* DYNAMIC MODEL */
    //  [1 0 0 dt  0  0 0  0  0   0   0   0]
    //  [0 1 0  0 dt  0 0  0  0   0   0   0]
    //  [0 0 1  0  0 dt 0  0  0   0   0   0]
    //  [0 0 0  1  0  0 0  0  0   0   0   0]
    //  [0 0 0  0  1  0 0  0  0   0   0   0]
    //  [0 0 0  0  0  1 0  0  0   0   0   0]
    //  [0 0 0  0  0  0 1  0  0  dt   0   0]
    //  [0 0 0  0  0  0 0  1  0   0  dt   0]
    //  [0 0 0  0  0  0 0  0  1   0   0  dt]
    //  [0 0 0  0  0  0 0  0  0   1   0   0]
    //  [0 0 0  0  0  0 0  0  0   0   1   0]
    //  [0 0 0  0  0  0 0  0  0   0   0   1]
    // position
    KF.transitionMatrix.at<double>(0,3) = dt;
    KF.transitionMatrix.at<double>(1,4) = dt;
    KF.transitionMatrix.at<double>(2,5) = dt;
    // orientation
    KF.transitionMatrix.at<double>(6,9) = dt;
    KF.transitionMatrix.at<double>(7,10) = dt;
    KF.transitionMatrix.at<double>(8,11) = dt;
    /* MEASUREMENT MODEL */
    //  [1 0 0 0 0 0 0 0 0 0 0 0 ]
    //  [0 1 0 0 0 0 0 0 0 0 0 0 ]
    //  [0 0 1 0 0 0 0 0 0 0 0 0 ]
    //  [0 0 0 0 0 0 1 0 0 0 0 0 ]
    //  [0 0 0 0 0 0 0 1 0 0 0 0 ]
    //  [0 0 0 0 0 0 0 0 1 0 0 0 ]
    KF.measurementMatrix.at<double>(0,0) = 1;  // x
    KF.measurementMatrix.at<double>(1,1) = 1;  // y
    KF.measurementMatrix.at<double>(2,2) = 1;  // z
    KF.measurementMatrix.at<double>(3,6) = 1;  // roll
    KF.measurementMatrix.at<double>(4,7) = 1; // pitch
    KF.measurementMatrix.at<double>(5,8) = 1; // yaw
}






cv::Mat rot2euler(const cv::Mat & rotationMatrix)
{
  cv::Mat euler(3,1,CV_64F);

  double m00 = rotationMatrix.at<double>(0,0);
  double m02 = rotationMatrix.at<double>(0,2);
  double m10 = rotationMatrix.at<double>(1,0);
  double m11 = rotationMatrix.at<double>(1,1);
  double m12 = rotationMatrix.at<double>(1,2);
  double m20 = rotationMatrix.at<double>(2,0);
  double m22 = rotationMatrix.at<double>(2,2);

  double x, y, z;

  // Assuming the angles are in radians.
  if (m10 > 0.998) { // singularity at north pole
    x = 0;
    y = CV_PI/2;
    z = atan2(m02,m22);
  }
  else if (m10 < -0.998) { // singularity at south pole
    x = 0;
    y = -CV_PI/2;
    z = atan2(m02,m22);
  }
  else
  {
    x = atan2(-m12,m11);
    y = asin(m10);
    z = atan2(-m20,m00);
  }

  euler.at<double>(0) = x*57.29;
  euler.at<double>(1) = y*57.29;
  euler.at<double>(2) = z*57.29;

  return euler;
}

/////////////////////////////////////////////////////////////////////////////


bool checkerboardMimic(vector<Point2f> &pointbuf,cv::KalmanFilter &KF,string &resultText)
{

    int n=6;

    if(pointbuf.empty()==true)
    {
        cout<<"Point Buff is Empty"<<endl;
        return false;
    }

    //P1x = [-86 86 0 -86 86 0]*420/297;
        //P1y = [-53.75 -53.75 -53.75 53.75 53.75 53.75]*420/297;
        //P1z = [0 0 0 0 0 0]*420/297;
    //double p1x[n]={-90, 90, 0, -90, 90,0};
    //double p1y[n]={-150, -150, -150, 150, 150, 150};
    double p1y[n]={-90, -90, -90, +90, +90,+90};
    double p1x[n]={-150, 150, 0, -150, 150, 0};
    double p1z[n]={0,0,0,0,0,0};

    //L12 = (P1x(1)-P1x(2))^2 + (P1y(1)-P1y(2))^2 + (P1z(1)-P1z(2))^2;
        //L45 = (P1x(4)-P1x(5))^2 + (P1y(4)-P1y(5))^2 + (P1z(4)-P1z(5))^2;
    double l12=pow((p1x[0]-p1x[1]),2)+pow((p1y[0]-p1y[1]),2)+pow((p1z[0]-p1z[1]),2);
    double l45=pow((p1x[3]-p1x[4]),2)+pow((p1y[3]-p1y[4]),2)+pow((p1z[3]-p1z[4]),2);


    double pcX[6]={0,0,0,0,0,0};
    double k=319;
    double j=239;
    //pcX[1]=pointbuff


    pcX[1]=pointbuf[0].x-k;
    pcX[2]=pointbuf[1].x-k;
    pcX[0]=pointbuf[2].x-k;
    pcX[4]=pointbuf[3].x-k;
    pcX[5]=pointbuf[4].x-k;
    pcX[3]=pointbuf[5].x-k;

    cout<<"******************************"<<endl;
    cout<<"PCX "<<endl;
    for(int j=0;j<n;j++){
        cout<<pcX[j]<<endl;
    }

    double pcY[6];
    pcY[1]=pointbuf[0].y-j;
    pcY[2]=pointbuf[1].y-j;
    pcY[0]=pointbuf[2].y-j;
    pcY[4]=pointbuf[3].y-j;
    pcY[5]=pointbuf[4].y-j;
    pcY[3]=pointbuf[5].y-j;

    cout<<"PCY "<<endl;
    for(int j=0;j<n;j++){
        cout<<pcY[j]<<endl;
    }
    cout<<"**************************"<<endl;


    double pcZ[n];
    for(int j=0;j<n;j++){
    pcZ[j]=814,3157;
    }


    double a = sqrt((pow((pcY[2]-pcY[0]),2)+pow((pcX[2]-pcX[0]),2))/(pow((pcY[2]-pcY[1]),2)+pow((pcX[2]-pcX[1]),2)));
    double b = sqrt((pow((pcY[5]-pcY[3]),2)+pow((pcX[5]-pcX[3]),2))/(pow((pcY[5]-pcY[4]),2)+pow((pcX[5]-pcX[4]),2)));
    cout<<"**************************"<<endl;
    cout<<"a: "<<a<<" b: "<<b<<endl;
    cout<<"**************************"<<endl;
    double p0z[6];
    //P0z(1) = sqrt(L12*Pcz(1)^2 / ((Pcz(1)-a*Pcz(2))^2 + (Pcy(1)-a*Pcy(2))^2 + (Pcx(1)-a*Pcx(2))^2));
        //P0z(2) = a*P0z(1);
        //P0z(3) = (P0z(1)+P0z(2))/2;
        //P0z(4) = sqrt(L45*Pcz(4)^2 / ((Pcz(4)-b*Pcz(5))^2 + (Pcy(4)-b*Pcy(5))^2 + (Pcx(4)-b*Pcx(5))^2));
        //P0z(5) = b*P0z(4);
        //P0z(6) = (P0z(4)+P0z(5))/2;
    p0z[0]=sqrt( l12*pow(pcZ[0],2)/(  pow((pcZ[0]-a*pcZ[1]),2) + pow((pcY[0]-a*pcY[1]),2) + pow((pcX[0]-a*pcX[1]),2)  )     );
    p0z[1]=a*p0z[0];
    p0z[2]=(p0z[0]+p0z[1])/2;
    p0z[3]=sqrt(l45*pow(pcZ[3],2)/  (   pow((pcZ[3]-a*pcZ[4]),2) + pow((pcY[3]-a*pcY[4]),2) + pow((pcX[3]-a*pcX[4]),2)  )   );
    p0z[4]=b*p0z[3];
    p0z[5]=(p0z[3]+p0z[4])/2;


    //P0x = [P0z(1)*Pcx(1)/Pcz(1) P0z(2)*Pcx(2)/Pcz(2) P0z(3)*Pcx(3)/Pcz(3) P0z(4)*Pcx(4)/Pcz(4) P0z(5)*Pcx(5)/Pcz(5) P0z(6)*Pcx(6)/Pcz(6)];
    double p0x[6];
    p0x[0]=p0z[0]*pcX[0]/pcZ[0];
    p0x[1]=p0z[1]*pcX[1]/pcZ[1];
    p0x[2]=p0z[2]*pcX[2]/pcZ[2];
    p0x[3]=p0z[3]*pcX[3]/pcZ[3];
    p0x[4]=p0z[4]*pcX[4]/pcZ[4];
    p0x[5]=p0z[5]*pcX[5]/pcZ[5];



    //P0y = [P0z(1)*Pcy(1)/Pcz(1) P0z(2)*Pcy(2)/Pcz(2) P0z(3)*Pcy(3)/Pcz(3) P0z(4)*Pcy(4)/Pcz(4) P0z(5)*Pcy(5)/Pcz(5) P0z(6)*Pcy(6)/Pcz(6)];
    double p0y[6];
    p0y[0]=p0z[0]*pcY[0]/pcZ[0];
    p0y[1]=p0z[1]*pcY[1]/pcZ[1];
    p0y[2]=p0z[2]*pcY[2]/pcZ[2];
    p0y[3]=p0z[3]*pcY[3]/pcZ[3];
    p0y[4]=p0z[4]*pcY[4]/pcZ[4];
    p0y[5]=p0z[5]*pcY[5]/pcZ[5];





    //unitx = [P0x(2)-P0x(1) P0y(2)-P0y(1) P0z(2)-P0z(1)] / sqrt((P0x(2)-P0x(1))^2 + (P0y(2)-P0y(1))^2 + (P0z(2)-P0z(1))^2);
    double coeff = sqrt(pow(p0x[1]-p0x[0],2)+pow(p0y[1]-p0y[0],2)+pow(p0z[1]-p0z[0],2));
    double unitx[3];
    unitx[0]=(p0x[1]-p0x[0])/coeff;
    unitx[1]=(p0y[1]-p0y[0])/coeff;
    unitx[2]=(p0z[1]-p0z[0])/coeff;


    //unity = [P0x(6)-P0x(3) P0y(6)-P0y(3) P0z(6)-P0z(3)] / sqrt((P0x(6)-P0x(3))^2 + (P0y(6)-P0y(3))^2 + (P0z(6)-P0z(3))^2);
        double coeff2 = sqrt(pow(p0x[5]-p0x[2],2)+pow(p0y[5]-p0y[2],2)+pow(p0z[5]-p0z[2],2));
    double unity[3];
    unity[0]=(p0x[5]-p0x[2])/coeff2;
    unity[1]=(p0y[5]-p0y[2])/coeff2;
    unity[2]=(p0z[5]-p0z[2])/coeff2;



    //unitz = cross(unitx, unity);
    double unitz[3];
    unitz[0]=unitx[1]*unity[2]-unitx[2]*unity[1];
    unitz[1]=unitx[2]*unity[0]-unitx[0]*unity[2];
    unitz[2]=unitx[0]*unity[1]-unitx[1]*unity[0];



    //yaw = atan2d(unitx(2), unitx(1)); %-180 +180
        //roll = atan2d(unity(3), unitz(3));%-180 +180
        //pitch = asind(-unitx(3)); %-90 +90
    double yaw = atan2(unitx[1],unitx[0]);
    double roll = atan2(unity[2],unitz[2]);
    double pitch = asin(-1*unitx[2]);
    yaw=yaw*180/3.14;
    roll=roll*180/3.14;
    pitch=pitch*180/3.14;

    //%RESULTS
        //POSITION = [(P0x(3)+P0x(6))/2 (P0y(3)+P0y(6))/2 (P0z(3)+P0z(6))/2]/10; %unit is cm
        //ORIENTATION = int16([roll pitch yaw]); % degree
//DAHA SONRA BUNLAR POINTER A DONUSTURULUP ARGUMANDAN VERILECEK SIZE BILGISI DE DONDURMEYI UNUTMA
	double position[3] = {(p0x[2]+p0x[5])/20, (p0y[2]+p0y[5])/20, (p0z[2]+p0z[5])/20};
	double orientation[3]= {roll, pitch, yaw};
	cout<<"POSIZYONLAR"<<endl;
	cout<<position[0];
	cout<<" ";

	cout<<position[1];
    cout<<" ";
	cout<<position[2];
    cout<<" "<<endl;

	cout<<"OIYANTASYON :"<<endl;
	cout<<orientation[0];
	cout<<" ";
	cout<<orientation[1];
	cout<<" ";
	cout<<orientation[2]<<endl;

	cv::Mat translation_estimated(3, 1, CV_64F);
	cv::Mat rotation_estimated(3, 1, CV_64F);
	cv::Mat measurements(6, 1, CV_64F);

	// update the Kalman filter with good measurements

	measurements.at<double>(0)=position[0];
	measurements.at<double>(1)=position[1];
	measurements.at<double>(2)=position[2];
	measurements.at<double>(3)=orientation[0];
	measurements.at<double>(4)=orientation[1];
	measurements.at<double>(5)=orientation[2];

	cv::Mat velocity(3, 1, CV_64F);
	cv::Mat angVelocity(3, 1, CV_64F);


	updateKalmanFilter( KF, measurements,
                   translation_estimated, rotation_estimated,velocity,angVelocity );
	cout<<" Kalman sonrası"<<endl;
    stringstream wrt2;
	wrt2 << translation_estimated.at<double>(0,0) << " " << translation_estimated.at<double>(1,0) << " " << translation_estimated.at<double>(2,0) << " " << 			rotation_estimated.at<double>(0,0) << " "<< rotation_estimated.at<double>(1,0) << " " <<rotation_estimated.at<double>(2,0)<<" "<<velocity.at<double>(0,0) << " " << velocity.at<double>(1,0) << " " << velocity.at<double>(2,0)<<" "<<angVelocity.at<double>(0,0) << " " << angVelocity.at<double>(1,0) << " " << angVelocity.at<double>(2,0);
    resultText=wrt2.str();
	cout<<"************************************"<<endl;
	cout<<"position Kalman"<<endl;
	cout<<translation_estimated<<endl;
	cout<<"Oriantation Kalman :"<<endl;
	cout<<rotation_estimated<<endl;
	cout<<"************************************"<<endl;

	return true;
}




void * get_in_addr(struct sockaddr * sa)
{
	if(sa->sa_family == AF_INET)
	{
		return &(((struct sockaddr_in *)sa)->sin_addr);
	}

	return &(((struct sockaddr_in6 *)sa)->sin6_addr);
}

int main(int argc, char * argv[])
{
	// Variables for writing a server.
	/*
	1. Getting the address data structure.
	2. Openning a new socket.
	3. Bind to the socket.
	4. Listen to the socket.
	5. Accept Connection.
	6. Receive Data.
	7. Close Connection.
	*/
	int status;
	struct addrinfo hints, * res;
	int listner;


	// Before using hint you have to make sure that the data structure is empty
	memset(& hints, 0, sizeof hints);
	// Set the attribute for hint
	hints.ai_family = AF_UNSPEC; // We don't care V4 AF_INET or 6 AF_INET6
	hints.ai_socktype = SOCK_STREAM; // TCP Socket SOCK_DGRAM
	hints.ai_flags = AI_PASSIVE;

	// Fill the res data structure and make sure that the results make sense.
	status = getaddrinfo(NULL, "8888" , &hints, &res);
	if(status != 0)
	{
		fprintf(stderr,"getaddrinfo error: %s\n",gai_strerror(status));
	}

	// Create Socket and check if error occured afterwards
	listner = socket(res->ai_family,res->ai_socktype, res->ai_protocol);
	if(listner < 0 )
	{
		fprintf(stderr,"socket error: %s\n",gai_strerror(status));
	}

	// Bind the socket to the address of my local machine and port number
	status = bind(listner, res->ai_addr, res->ai_addrlen);
	if(status < 0)
	{
		fprintf(stderr,"bind: %s\n",gai_strerror(status));
	}

	status = listen(listner, 10);
	if(status < 0)
	{
		fprintf(stderr,"listen: %s\n",gai_strerror(status));
	}

	// Free the res linked list after we are done with it
	freeaddrinfo(res);


	// We should wait now for a connection to accept
	int new_conn_fd;
	struct sockaddr_storage client_addr;
	socklen_t addr_size;
	char s[INET6_ADDRSTRLEN]; // an empty string

	// Calculate the size of the data structure
	addr_size = sizeof client_addr;

	printf("I am now accepting connections ...\n");

	while(1){
		// Accept a new connection and return back the socket desciptor
		new_conn_fd = accept(listner, (struct sockaddr *) & client_addr, &addr_size);
		if(new_conn_fd < 0)
		{
			fprintf(stderr,"accept: %s\n",gai_strerror(new_conn_fd));
			continue;
		}

		inet_ntop(client_addr.ss_family, get_in_addr((struct sockaddr *) &client_addr),s ,sizeof s);
		printf("I am now connected to %s \n",s);
        //KODU BURAYA AL



////////////////////////////////////////////////////////////


		Point2f pnt(320,240);

        cv::KalmanFilter KF;         // instantiate Kalman Filter
        int nStates = 12;            // the number of states
        int nMeasurements = 6;       // the number of measured states
        int nInputs = 0;             // the number of action control
        double dt = 0.125;           // time between measurements (1/FPS)
        initKalmanFilter(KF, nStates, nMeasurements, nInputs, dt);    // init function

///////////////////////////////////////////////////////////////////////////////////////////////
        VideoCapture capWebcam(0);            // declare a VideoCapture object and associate to webcam, 0 => use 1st webcam

        if (capWebcam.isOpened() == false) {      // check if VideoCapture object was associated to webcam successfully
		cout << "error: capWebcam not accessed successfully\n\n";      // if not, print error message to std out
		                                             // may have to modify this line if not using Windows
		return(0);                                                          // and exit program
        }


        capWebcam.set(CAP_PROP_FRAME_WIDTH,640);
        capWebcam.set(CAP_PROP_FRAME_HEIGHT,480);

        Mat imgOriginal = imread("./box.jpg");
        Mat imgScene;// = imread("./box_in_scene.jpg");
        char charCheckForEscKey = 0;
        vector<KeyPoint> objectKeypoints;

        Mat objectDescriptors;



        int k = 2; // find the 2 nearest neighbors
        bool useBFMatcher = true; // SET TO TRUE TO USE BRUTE FORCE MATCHER
        float nndrRatio = 0.7f;


        Ptr<FeatureDetector> detector = BRISK::create(); // create ORB features
        detector->detect(imgOriginal, objectKeypoints); // detect

        Ptr<cv::DescriptorExtractor> extractor = BRISK::create();
        extractor->compute(imgOriginal, objectKeypoints, objectDescriptors);

        bool blnFrameReadSuccessfully;

        while (charCheckForEscKey != 27 && capWebcam.isOpened()) {            // until the Esc key is pressed or webcam connection is lost
            blnFrameReadSuccessfully = capWebcam.read(imgScene);            // get next frame
            if (!blnFrameReadSuccessfully || imgScene.empty()) {                 // if frame not read successfully                // print error message to std out
                break;                                                              // and jump out of while loop
            }
            //namedWindow("imgOriginal", CV_WINDOW_NORMAL);       // note: you can use CV_WINDOW_NORMAL which allows resizing the window	        	// CV_WINDOW_AUTOSIZE is the default
            //imshow("imgOriginal", imgOriginal);                 // show windows
            Mat sceneDescriptors;
            Mat img_keypoints_1;
            Mat results;
            Mat dists;
            vector<KeyPoint> sceneKeypoints;
            std::clock_t start;
                start = std::clock();

            detector->detect(imgScene, sceneKeypoints);
            //*************
            //drawKeypoints(imgOriginal, objectKeypoints, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            //-- Show detected (drawn) keypoints
            //namedWindow("imgOriginal", CV_WINDOW_NORMAL);
            //imshow("Keypoints 1", img_keypoints_1);

            extractor->compute(imgScene, sceneKeypoints, sceneDescriptors);
            vector<vector <DMatch> >matches;
            Mat img_matches;
            if (objectDescriptors.type() == CV_8U)
            {
                // Binary descriptors detected (from ORB, Brief, BRISK, FREAK)

                if (useBFMatcher)
                {
                    BFMatcher matcher(NORM_HAMMING); // use cv::NORM_HAMMING2 for ORB descriptor with WTA_K == 3 or 4 (see ORB constructor)
                    matcher.knnMatch(objectDescriptors, sceneDescriptors, matches, k);
                    //matcher.match(objectDescriptors, sceneDescriptors, matches);
                }
                else
                {
                    // Create Flann LSH index
                    flann::Index flannIndex(sceneDescriptors, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

                    // search (nearest neighbor)
                    flannIndex.knnSearch(objectDescriptors, results, dists, k, cv::flann::SearchParams());
                }
            }
            else
            {
                // assume it is CV_32F
                if (useBFMatcher)
                {
                    BFMatcher matcher(NORM_L2);
                    matcher.knnMatch(objectDescriptors, sceneDescriptors, matches, k);
                    //matcher.match(objectDescriptors, sceneDescriptors, matches);
                }
                else
                {
                    // Create Flann KDTree index
                    cv::flann::Index flannIndex(sceneDescriptors, cv::flann::KDTreeIndexParams(), cvflann::FLANN_DIST_EUCLIDEAN);

                    // search (nearest neighbor)
                    flannIndex.knnSearch(objectDescriptors, results, dists, k, cv::flann::SearchParams());
                }
            }
            if (dists.type() == CV_32S)
            {
                cv::Mat temp;
                dists.convertTo(temp, CV_32F);
                dists = temp;
            }

            ////////////////////////////
            // PROCESS NEAREST NEIGHBOR RESULTS
            ////////////////////////////
            // Find correspondences by NNDR (Nearest Neighbor Distance Ratio)


            double max_dist = 0; double min_dist = 100;

            //-- Quick calculation of max and min distances between keypoints

            for (int i = 0; i < objectDescriptors.rows; i++)
            {
                //Checks if no matches found
                if(matches.at(i).size()!=0)
                {
                double dist = matches.at(i).at(0).distance;
                if (dist < min_dist) min_dist = dist;
                if (dist > max_dist) max_dist = dist;
                }

            }



            //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
            vector< DMatch > good_matches;
            for (int i = 0; i < objectDescriptors.rows && matches.at(i).size()!=0; i++)
            {
                if (matches.at(i).at(0).distance < 3 * min_dist)
                {
                    good_matches.push_back(matches.at(i).at(0));
                }
            }

            //drawMatches(imgOriginal, objectKeypoints, imgScene, sceneKeypoints, good_matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

            vector<cv::Point2f> mpts_1, mpts_2; // Used for homography
            vector<int> indexes_1, indexes_2; // Used for homography
            vector<uchar> outlier_mask;  // Used for homography
            vector<Point2f> obj_corners(4);
            vector<Point2f> scene_corners(4);
            // Check if this descriptor matches with those of the objects
            if (!useBFMatcher)
            {
                for (int i = 0; i<objectDescriptors.rows; ++i)
                {
                    // Apply NNDR
                    //printf("q=%d dist1=%f dist2=%f\n", i, dists.at<float>(i,0), dists.at<float>(i,1));
                    if (results.at<int>(i, 0) >= 0 && results.at<int>(i, 1) >= 0 &&
                        dists.at<float>(i, 0) <= nndrRatio * dists.at<float>(i, 1))
                    {
                        mpts_1.push_back(objectKeypoints.at(i).pt);
                        indexes_1.push_back(i);

                        mpts_2.push_back(sceneKeypoints.at(results.at<int>(i, 0)).pt);
                        indexes_2.push_back(results.at<int>(i, 0));
                    }
                }
            }
            else
            {
                for (int i = 0; i < good_matches.size(); i++)
                {
                    //-- Get the keypoints from the good matches
                    mpts_1.push_back(objectKeypoints[good_matches[i].queryIdx].pt);
                    mpts_2.push_back(sceneKeypoints[good_matches[i].trainIdx].pt);
                }

                /*for (unsigned int i = 0; i<good_matches.size(); ++i)
                {
                    // Apply NNDR
                    //printf("q=%d dist1=%f dist2=%f\n", matches.at(i).at(0).queryIdx, matches.at(i).at(0).distance, matches.at(i).at(1).distance);
                    if (good_matches.size() == 2 &&
                        good_matches.at(0).distance <= nndrRatio * good_matches.at(1).distance)
                    {
                        mpts_1.push_back(objectKeypoints.at(good_matches.at(i).queryIdx).pt);
                        indexes_1.push_back(good_matches.at(i).queryIdx);

                        mpts_2.push_back(sceneKeypoints.at(good_matches.at(i).trainIdx).pt);
                        indexes_2.push_back(good_matches.at(i).trainIdx);
                        cout << good_matches.size() << endl;
                    }
                }*/
            }

            //drawMatches(imgOriginal, objectKeypoints, imgScene, sceneKeypoints, matches, img_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
            drawKeypoints(imgScene, sceneKeypoints, img_matches, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

            // FIND HOMOGRAPHY
            int minMatches=2;
            int minInliers = 250;
			bool isFound;
			string sendMessage;
            if (mpts_1.size() <= minInliers && good_matches.size()>=minMatches)
            {
                Mat H = findHomography(mpts_1,
                    mpts_2,
                    cv::RANSAC,
                    1.0,
                    outlier_mask);

                    cout<<" BURDAMI "<<H.empty()<<endl;;

                int inliers = 0, outliers = 0;
                for (unsigned int k = 0; k < mpts_1.size(); ++k)
                {
                    if (outlier_mask.at(k))
                    {
                        ++inliers;
                    }
                    else
                    {
                        ++outliers;
                    }
                }
                obj_corners[0] = cvPoint(0, 0);
                obj_corners[1] = cvPoint(imgOriginal.cols, 0);
                obj_corners[2] = cvPoint(imgOriginal.cols, imgOriginal.rows);
                obj_corners[3] = cvPoint(0, imgOriginal.rows);

                if(!H.empty())
                {
                    perspectiveTransform(obj_corners, scene_corners, H);


                    //////////////////////////
                    //FORT HE BOIS

                    vector<Point2f> pointbuf(6);

                    pointbuf[0]=scene_corners[0];
                    pointbuf[1]=cvPoint((scene_corners[0].x+scene_corners[3].x)/2, (scene_corners[0].y+scene_corners[3].y)/2  );
                    pointbuf[2]=scene_corners[3];
                    pointbuf[3]=scene_corners[1];
                    pointbuf[4]=cvPoint((scene_corners[1].x+scene_corners[2].x)/2, (scene_corners[1].y+scene_corners[2].y)/2  );
                    pointbuf[5]=scene_corners[2];

                    cout<<"**************************************"<<endl;
                    cout<<"VERİLEN İNPUT :"<<pointbuf<<endl;
                    cout<<"*************************************"<<endl;
                    

					bool found=checkerboardMimic(pointbuf,KF,sendMessage);



                    line(img_matches, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0), 4);
                    line(img_matches, scene_corners[1] , scene_corners[2] , Scalar(0, 255, 0), 4);
                    line(img_matches, scene_corners[2] , scene_corners[3] , Scalar(0, 255, 0), 4);
                    line(img_matches, scene_corners[3] , scene_corners[0] , Scalar(0, 255, 0), 4);


                    //pnt ekranın ortası

                    //Center
                    Point2f center((pointbuf[1].x+pointbuf[4].x)/2, (pointbuf[1].y+pointbuf[4].y)/2);


                    Point2f pnt2(center.x,pnt.y);
                    Point2f pnt4(pnt.x,center.y);

                    line(img_matches, center , pnt2 , Scalar(175, 255, 100), 2);
                    line(img_matches, pnt2 , pnt , Scalar(175, 255, 100), 2);
                    line(img_matches, pnt , pnt4 , Scalar(175, 255, 100), 2);
                    line(img_matches, pnt4 , center , Scalar(175, 255, 100), 2);


                    line(img_matches, scene_corners[0] , scene_corners[1] , Scalar(0, 255, 0), 4);
                    line(img_matches, scene_corners[1] , scene_corners[2] , Scalar(0, 255, 0), 4);
                    line(img_matches, scene_corners[2] , scene_corners[3] , Scalar(0, 255, 0), 4);
                    line(img_matches, scene_corners[3] , scene_corners[0] , Scalar(0, 255, 0), 4);



                    putText(img_matches, "0", pointbuf[0], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0));
                    putText(img_matches, "1", pointbuf[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0));
                    putText(img_matches, "2", pointbuf[2], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0));
                    putText(img_matches, "3", pointbuf[3], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0));
                    putText(img_matches, "4", pointbuf[4], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0));
                    putText(img_matches, "5", pointbuf[5], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0));

                    circle(img_matches, pointbuf[0], 5, Scalar(0,0,50), 4, 3);
                    circle(img_matches, pointbuf[1], 5, Scalar(0,0,100), 4, 3);
                    circle(img_matches, pointbuf[2], 5, Scalar(0,0,150), 4, 3);
                    circle(img_matches, pointbuf[3], 5, Scalar(0,0,200), 4, 3);
                    circle(img_matches, pointbuf[4], 5, Scalar(0,0,250), 4, 3);
                    circle(img_matches, pointbuf[5], 5, Scalar(200,0,0), 4, 3);
					isFound=found;

                }
                else
                {
					isFound=false;
                    cout<<"Homography not exist!"<<endl;
                }



            }
            else{
                    //BULAMAMA DURUMU
                cout << "0";
                cout<<" ";
                cout << "0";
                cout<<" ";
                cout << "0";cout<<" ";
                cout << "0";cout<<" ";
                cout << "0";cout<<" ";
                cout << "0"<< endl;
				isFound=false;
            }
            //-- Show detected matches
            double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
            //double tickk=1/duration;
            //cout<<" FPS : "<< tickk <<endl;
            //cout<<tickk<<endl;



            circle(img_matches, pnt , 5, Scalar(0,0,0), 4, 3);

            stringstream msgsizess;

            //SEND
            /////////////////////
            Mat imgSent=img_matches;

            imgSent=(imgSent.reshape(0,1));

            int imgSize=imgSent.total()*imgSent.elemSize();

            ////////image

            //SENDD
					
			if(!isFound)
			{
		        stringstream wrt2;
				wrt2 << 0 << " " << 0 << " " << 0 << " " << 0 << " "<< 0 << " " <<0<<endl;
				sendMessage=wrt2.str();
			}

			stringstream end;
			end<<sendMessage<<" "<<duration<<endl;
			sendMessage=end.str();

            int size=sendMessage.size();
            msgsizess<< setfill('0') << setw(3) << size;
            string msgSize=msgsizess.str();
            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(new_conn_fd,&rfds);
            // timer thingy
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec =5;
            int rv = select(new_conn_fd+1,&rfds,NULL,NULL,&tv);
            if(rv==0)
            {
                cout<< "nothing to read";
            }
            else
            //if(FD_ISSET(0,&new_conn_fd)==1)
            {
                char reply;
                recv(new_conn_fd, &reply, 1, 0);
                cout<<"reply: "<<reply<<endl;

                if(reply=='1')
                {
                    send(new_conn_fd,msgSize.c_str(), 3,0);
                    char reply;
                    recv(new_conn_fd, &reply, 1, 0);
                    if(reply=='2')
                    {
                        send(new_conn_fd,sendMessage.c_str(), size,0);
                        send(new_conn_fd,imgSent.data,imgSize,0);
                    }
                    else if(reply=='1')
                    {
                        send(new_conn_fd,"000", 3,0);
                    }
                }
                else if(reply=='2')
                {
                    send(new_conn_fd,"e", 1,0);
                }

            }
            //SENDDD
            imshow("Good Matches & Object detection", img_matches);
            charCheckForEscKey = waitKey(10);        // delay (in ms) and get key press, if any
            //waitKey(0);
        }
//////////////////////////////////////////////////////////

		status = send(new_conn_fd,"Welcome", 7,0);
		if(status == -1)
		{
			close(new_conn_fd);
			_exit(4);
		}

	}
	// Close the socket before we finish
	close(new_conn_fd);

	return 0;
}
