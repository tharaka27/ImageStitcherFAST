

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

int main()
{
	
	cv::Mat left_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\left.jpg");
	cv::Mat middle_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\middle.jpg");
	cv::Mat right_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\right.jpg");

	ImageStitcherFAST stitcher;
	
	//resize images since the calculation takes lots of time !!!
	
	int scale_percent = 10;
	stitcher.scale_image(left_image, left_image, scale_percent);
	stitcher.scale_image(middle_image, middle_image, scale_percent);
	stitcher.scale_image(right_image, right_image, scale_percent);
	//-----------------------------------------------------------------


	cv::Mat left_flipped;
	cv::Mat middle_flipped;
	cv::flip(left_image, left_flipped, 1);


	cv::flip(middle_image, middle_flipped, 1);


	//--------------------------------------------------------------------


	// -- lets try changing the sides since we flipped
	cv::Mat H12 = stitcher.calculate_h_matrix(left_flipped, middle_flipped);
	//--------------------------------------------------------------------

	
	//cv::Mat H23 = stitcher.calculate_h_matrix(right_image, middle_image);

	//Stitch right_image and middle_image and saved in img
	//cv::Mat img = stitcher.stitch_image(right_image, middle_image, H23);


	//Stitch middle_image and left_image and saved in img2
	//cv::Mat img2 = stitcher.stitch_image(left_flipped, middle_flipped, H12);

	//cv::imshow("Final Image ", img2);
	//cv::waitKey(0);

	
	//cv::flip(img2, img2, 1);


	//Stitch(left_image and middle_image) and (right_image and middle_image)
	//cv::Mat H123 = stitcher.calculate_h_matrix(img, img2);
	//cv::Mat img4 = stitcher.stitch_image(img, img2, H123);

	//cv::imshow("Final Image ", img4);
	//cv::waitKey(0);

	
	
	/*
	testbench v;
	v.findhomographymatix();
	v.NormalizeMatrix();
	*/

	return 0;
}


