#include <stdio.h>
#include <iostream> 

#include "testbench.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;


cv::Mat calculate_h_matrix(cv::Mat gray_image1, cv::Mat gray_image2)
{
	//-- Step 1: Detect the keypoints using FAST Detector
	vector<KeyPoint> keypoints_object, keypoints_scene;
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
	detector->detect(gray_image1, keypoints_object, Mat());
	detector->detect(gray_image2, keypoints_scene, Mat());
	
	cout << "Detecting keypoints using FAST detector is completed" << endl;

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;
	Ptr<ORB> extractor = ORB::create();
	extractor->compute(gray_image1, keypoints_object, descriptors_object);
	extractor->compute(gray_image2, keypoints_scene, descriptors_scene);

	cout << "Calculating descriptors using ORB is completed" << endl;


	//-- Step 3: Matching descriptor vectors using FLANN matcher

	cv::Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	//cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	std::vector< DMatch > matches;
	matcher->match(descriptors_object, descriptors_scene, matches);
	matcher->match(descriptors_object, descriptors_scene, matches);
	
	cout << "Matching features are completed" << endl;
	
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints 
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist: %f \n", max_dist);
	printf("-- Min dist: %f \n", min_dist);


	/* Exceptional case
	   For the first time min_dist became 0 
	   since 3*min_distance = 0 we could not find any good match
	   hence the default value was set to 3 if min_dis = 0
	*/
	if (min_dist < 10) {
		min_dist = 3;
	}

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;
	cv::Mat result;
	// cv::Mat result23;
	cv::Mat H;
	// cv::Mat H23;
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	std::vector< Point2f > obj;
	std::vector< Point2f > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	if (obj.empty()|| scene.empty()) {
		cout << "No matches found at all....." << endl;

	}
	else {
		cout << "Enough matches found" << endl;

	}

	// Find the Homography Matrix for img 1 and img2
	H = findHomography(obj, scene, RANSAC);

	cout << "Homography Calculation successfull" << endl;

	return H;
}



Mat stitch_image(Mat image1, Mat image2, Mat H)
{
	cv::Mat result;
	// cv::Mat result23;
	warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, image1.rows));
	cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);
	return result;
}

int main()
{
	cv::Mat left_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\left.jpg");
	cv::Mat middle_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\middle.jpg");
	cv::Mat right_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\right.jpg");
	
	/*
	  resize images since the calculation takes lots of time !!!
	*/
	resize(left_image,left_image,Size(640,480));
	resize(middle_image, middle_image, Size(640, 480));
	resize(right_image, right_image, Size(640, 480));


	cv::imshow("left image", left_image);
	waitKey(0);

	cv::imshow("right image", middle_image);
	waitKey(0);



	Mat H12 = calculate_h_matrix(middle_image, left_image);
	
	
	Mat H23 = calculate_h_matrix(right_image,middle_image);
	
	//Stitch right_image and middle_image and saved in img
	Mat img = stitch_image(right_image, middle_image, H23);
	

	//Stitch middle_image and left_image and saved in img2
	Mat img2 = stitch_image(middle_image, left_image, H12);

	//Show img
	cv::imshow("Hist_Equalized_Result of right_image and middle_image", img);
	waitKey(0);

	//Show img2
	cv::imshow("Hist_Equalized_Result of middle_image and left_image", img2);
	waitKey(0);


	//Stitch(left_image and middle_image) and (right_image and middle_image)
	Mat H123 = calculate_h_matrix(img, img2);
	Mat img4 = stitch_image(img, img2, H123);

	cv::imshow("Final Image ", img4);
	waitKey(0);


	
	return 0;
}