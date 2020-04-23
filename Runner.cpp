

#include "Runner.h"
#include <stdio.h>
#include <iostream> 

#include "testbench.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>
#include "ImageStitcherFAST.h"
#include "ORB.h"


int main()
{ 
	/*

	
	cv::Mat left_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\left.jpg");
	cv::Mat middle_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\middle.jpg");
	cv::Mat right_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\right.jpg");


	cv::Mat left_image_gray;
	cv::Mat middle_image_gray;
	cv::Mat right_image_gray;

	ImageStitcherFAST stitcher;
	
	//resize images since the calculation takes lots of time !!!
	
	int scale_percent = 10;
	stitcher.scale_image(left_image, left_image, scale_percent);
	stitcher.scale_image(middle_image, middle_image, scale_percent);
	stitcher.scale_image(right_image, right_image, scale_percent);
	//-----------------------------------------------------------------

	cv::cvtColor(left_image, left_image_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(middle_image, middle_image_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(right_image, right_image_gray, cv::COLOR_BGR2GRAY);


	//std::cout << middle_image.size() << std::endl;

	cv::Mat left_flipped_gray;
	cv::Mat middle_flipped_gray;
	cv::flip(left_image_gray, left_flipped_gray, 1);


	cv::flip(middle_image_gray, middle_flipped_gray, 1);
	
	std::cout << middle_image_gray.type() << std::endl;

	//--------------------------------------------------------------------


	// -- lets try changing the sides since we flipped
	cv::Mat H12 = stitcher.calculate_h_matrix(left_flipped_gray, middle_flipped_gray);
	//--------------------------------------------------------------------

	
	cv::Mat H23 = stitcher.calculate_h_matrix(right_image_gray, middle_image_gray);

	//Stitch right_image and middle_image and saved in img
	cv::Mat img = stitcher.stitch_image(right_image, middle_image, H23);

	cv::Mat left_flipped;
	cv::Mat middle_flipped;
	cv::flip(left_image, left_flipped, 1);
	cv::flip(middle_image, middle_flipped, 1);
	//Stitch middle_image and left_image and saved in img2
	cv::Mat img2 = stitcher.stitch_image(left_flipped, middle_flipped, H12);

	//cv::imshow("Image 2 ", img2);
	///cv::waitKey(0);

	//cv::imshow("Image 1 ", img);
	//cv::waitKey(0);
	
	cv::flip(img2, img2, 1);
	//cv::imshow("Image 2 ", img2);
	//cv::waitKey(0);


	/*
	cv::Mat image_temp1 = img(cv:: Rect(0, 0, img.cols, img.rows / 2)).clone();
	cv::Mat image_temp2 = img(cv::Rect(0, img.rows / 2, img.cols, img.rows / 2)).clone();
	
	cv::Mat result(img.rows, img.cols);
	image_temp1.copyTo(result(Rect(0, 0, image.cols, image.rows / 2)));
	image_temp2.copyTo(result(Rect(0, image.rows / 2, image.cols, image.rows / 2));
	
	


	cv::Mat result(middle_image.rows, left_image.cols + middle_image.cols + right_image.cols, CV_8UC3);
	img2.copyTo(result.colRange(0,left_image.cols + middle_image.cols));
	img.copyTo(result.colRange(left_image.cols, left_image.cols + middle_image.cols + right_image.cols));
	
	//std::cout << result.size() << std::endl;
	cv::imshow("result ", result);
	cv::waitKey(0);
	
	*/

    
	testbench v;
	//v.CheckRANSACMatching();
	//v.findhomographymatix();
	//v.NormalizeMatrix();
	//v.isCollinear();
	//v.ComputeH();
	//v.CheckHomography();
	//v.ComputeAngle();
	
	v.TestFull();
	return 0;

	
	
}


