#include <stdio.h>
#include <iostream> 

#include "testbench.h"
#include "ImageStitcherFAST.h"

#include "homography.h"
//#include "RANSAC_algo.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d.hpp>
#include "RANSAC_algo.h"

using namespace std;
using namespace cv;




cv::Mat ImageStitcherFAST::calculate_h_matrix(cv::Mat gray_image1, cv::Mat gray_image2)
{
	//-- Step 1: Detect the keypoints using FAST Detector
	vector<KeyPoint> keypoints_object, keypoints_scene;

	/*
	Ptr<ORB> detector = ORB::create();
	detector->detect(gray_image1, keypoints_object);
	detector->detect(gray_image2, keypoints_scene);
	*/
	
	
	Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
	detector->detect(gray_image1, keypoints_object);
	detector->detect(gray_image2, keypoints_scene);
	

	cout << "no of keypoints of object " << keypoints_object.size() << endl;
	cout << "no of keypoints of scene " << keypoints_scene.size() << endl;
	cout << "Detecting keypoints using FAST detector is completed" << endl << endl;

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;
	Ptr<ORB> extractor = ORB::create();
	extractor->compute(gray_image1, keypoints_object, descriptors_object);
	extractor->compute(gray_image2, keypoints_scene, descriptors_scene);


	cout << "no of desciptors of object " << descriptors_object.size()<< endl;
	cout << "no of desciptors of scene " << descriptors_scene.size() << endl;
	cout << "Calculating descriptors using ORB is completed" << endl << endl;


	//-- Step 3: Matching descriptor vectors using FLANN matcher
	std::vector< DMatch > matches;
	
	
	cv::Ptr<DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors_object, descriptors_scene, matches);
	//matcher->match(descriptors_object, descriptors_scene, matches);
	
	/*
	cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
	matcher.match(descriptors_object, descriptors_scene, matches);
	matcher.match(descriptors_object, descriptors_scene, matches);
	*/
	cout << "matches " << matches.size() << endl;
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
	if (min_dist < 15) {
		min_dist = 15;
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

	cout << "number of good matches " << good_matches.size() << endl;


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
	H = findHomography(obj, scene, cv::RANSAC);
	//cout << "Homography Calculation successfull" << endl << endl << endl<< endl;

	cout << "Calculated homography using OPENCV:::" << endl;
	std::cout << H<< std::endl;

	/*
	================================================================================

									ALERT !!!!!!!
	================================================================================
	Extra candidate
	
	void findHomography(cv::Point2f src[4], cv::Point2f dst[4], float* h[3][3]);
	
	*/

	
	Point2f src[4];
	Point2f dst[4];

	src[0] = obj[0];
	src[1] = obj[3];
	src[2] = obj[2];
	src[3] = obj[5];

	dst[0] = scene[0];
	dst[1] = scene[3];
	dst[2] = scene[2];
	dst[3] = scene[5];

	RANSAC_algo c;
	//c.findHomography(src, dst);
	cv::Mat V;
	V = c.computeHomography_RANSAC(obj,scene);
	
	//cout << "Calculated homography using My Impelementation:::" << endl;
	//std::cout << V << std::endl;

	/*
	================================================================================

									ALERT !!!!!!!
	================================================================================
	*/

	return V;
}



Mat ImageStitcherFAST::stitch_image(Mat image1, Mat image2, Mat H)
{
	cv::Mat result;
	// cv::Mat result23;
	warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, image1.rows));
	cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);
	return result;
}



void ImageStitcherFAST::scale_image(cv::Mat &InputImage, cv::Mat &OutputImage, int scale_percent)
{
	int width = int(InputImage.cols * scale_percent / 100);
	int height = int(InputImage.rows * scale_percent / 100);
	cv::resize(InputImage, OutputImage, cv::Size(width, height));
}

