/*


   Copyright (c) 2020, Tharaka Ratnayake, email: tharakasachintha.17@cse.mrt.ac.lk
   All rights reserved. https://github.com/tharaka27/ImageStitcherFAST

   Some algorithms used in this code referred to:
   1. OpenCV: http://opencv.org/



   Revision history:
	  March 30th, 2020: initial version.
*/



#include "testbench.h"
#include <iostream>

#include "homography.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2\calib3d.hpp>
#include "RANSAC_algo.h"
#include "ImageStitcherFAST.h"
#include "ORBExtractor.h"
#include "ORBMatcher.h"

#include "ORBExtractorHLS.h"

void testbench::print_hello_world() {
	std::cout << "hello world " << std::endl;
}



void testbench::image_test() {
	cv::Mat img = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\middle.jpg");
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", img);
	cv::waitKey(0);
}


void testbench::findhomographymatix() {

	cv::Point2d src[4];
	cv::Point2d dst[4];

	//
	//       Dataset 1
	//
	src[0] = cv::Point2f(141, 131);
	src[1] = cv::Point2f(480, 159);
	src[2] = cv::Point2f(493, 630);
	src[3] = cv::Point2f(64, 601);

	dst[0] = cv::Point2f(318, 256);
	dst[1] = cv::Point2f(534, 372);
	dst[2] = cv::Point2f(316, 670);
	dst[3] = cv::Point2f(73, 473);
	


	homography homo;

	cv::Mat h = homo.findHomography_(src,dst);

	cv::Mat obj = (cv::Mat_<float>(4, 2) << 141, 131, 480, 159, 493, 630, 64, 601);
	cv::Mat scene = (cv::Mat_<float>(4, 2) << 318, 256, 534, 372, 316, 670, 73, 473);

	cv::Mat H = findHomography(obj, scene, cv::RANSAC);
	std::cout << "homograph Matrix from opencv RANSAC implementation" << std::endl;
	std::cout << H << std::endl;
	std::cout << "homograph Matrix from gussian elemenation implementation" << std::endl;
	std::cout << h << std::endl;

}


struct norm {
	cv::Mat_<double> Normalization_matrix;
	std::vector<cv::Point2f> points3d;
};


void testbench::NormalizeMatrix() {

	std::vector<cv::Point2f> pointsVec;
	
	//RANSAC_algo r;
	
	pointsVec.push_back(cv::Point2f(1, 1));
	pointsVec.push_back(cv::Point2f(0, 0));
	pointsVec.push_back(cv::Point2f(2, 2));
	pointsVec.push_back(cv::Point2f(4, 4));
	
	struct norm d;

	std::cout << d.Normalization_matrix << std::endl;
	std::cout << d.points3d.at(0) << std::endl;
	std::cout << d.points3d.at(1) << std::endl;
	std::cout << d.points3d.at(2) << std::endl;
	std::cout << d.points3d.at(3) << std::endl;
	}

/*
void testbench::isCollinear() {

	
	RANSAC_homo r;
	
	cv::Point2f p[3] = { cv::Point2f(1,2), cv::Point2f(2,4) , cv::Point2f(3,6) };
	std::cout <<"points (1,2), (2,4), (3,6) are collinear : "<< r.isColinear(3,p) << std::endl;

	cv::Point2f d[3] = { cv::Point2f(1,2), cv::Point2f(2,4) , cv::Point2f(3,5) };
	std::cout << "points (1,2), (2,4), (3,5) are collinear : " << r.isColinear(3, d) << std::endl;

	cv::Point2f t[3] = { cv::Point2f(1,2), cv::Point2f(2,4) , cv::Point2f(1,3) };
	std::cout << "points (1,2), (2,4), (1,3) are collinear : " << r.isColinear(3, t) << std::endl;


	cv::Point2f v[4] = { cv::Point2f(1,2), cv::Point2f(2,4) , cv::Point2f(1,3), cv::Point2f(3,6) };
	std::cout << "points (1,2), (2,4), (1,3), (3,6) are collinear : " << r.isColinear(4, v) << std::endl;
}
*/



void testbench::CheckHomography() {


	




}

void scale_image(cv::Mat& InputImage, cv::Mat& OutputImage, int scale_percent)
{
	int width = int(InputImage.cols * scale_percent / 100);
	int height = int(InputImage.rows * scale_percent / 100);
	cv::resize(InputImage, OutputImage, cv::Size(width, height));
}



void testbench::CheckRANSACMatching()
{

	cv::Mat left_flipped;
	cv::Mat middle_flipped;


	cv::Mat gray_image1 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\left.jpg");
	cv::Mat gray_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\middle.jpg");


	ImageStitcherFAST stitcher;
	int scale_percent = 10;
	scale_image(gray_image1, gray_image1, scale_percent);
	scale_image(gray_image2, gray_image2, scale_percent);



	//-- Step 1: Detect the keypoints using FAST Detector
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;


	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	detector->detect(gray_image1, keypoints_object);
	detector->detect(gray_image2, keypoints_scene);


	//-- Step 2: Calculate descriptors (feature vectors)
	cv::Mat descriptors_object, descriptors_scene;
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();
	extractor->compute(gray_image1, keypoints_object, descriptors_object);
	extractor->compute(gray_image2, keypoints_scene, descriptors_scene);


	//-- Step 3: Matching descriptor vectors using FLANN matcher
	std::vector< cv::DMatch > matches;


	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors_object, descriptors_scene, matches);
	

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints 
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	/* Exceptional case
	   For the first time min_dist became 0
	   since 3*min_distance = 0 we could not find any good match
	   hence the default value was set to 3 if min_dis = 0
	*/
	if (min_dist < 15) {
		min_dist = 15;
	}

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< cv::DMatch > good_matches;
	cv::Mat result;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}


	std::vector< cv::Point2d > obj;
	std::vector< cv::Point2d > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}
	

	RANSAC_algo ra;
	struct returnRANSAC r = ra.computeHomography_RANSAC(obj, scene);
	std::vector< cv::DMatch > ransac_matches;



	for (int i = 0; i < r.InlierMask.rows; i++)
	{
		if (r.InlierMask.at<double>(i, 0) == 1)
		{
			ransac_matches.push_back(good_matches[i]);
		}
	}

	std::cout << "number of candidate matches found after brute force hamming: " << good_matches.size() << std::endl;
	std::cout << "number of ransac chosen matches: " << ransac_matches.size() << std::endl;

	cv::Mat output_image_all;
	cv::drawMatches(gray_image1, keypoints_object, gray_image2, keypoints_scene, good_matches, output_image_all);
	cv::imshow("good points", output_image_all);

	cv::Mat output_image;
	cv::drawMatches(gray_image1, keypoints_object, gray_image2, keypoints_scene, ransac_matches, output_image);

	cv::imshow("RANSAC chosen points", output_image);
	cv::waitKey(0);

}

void testbench::CheckRANSACHMatrix()
{

	cv::Mat left_flipped;
	cv::Mat middle_flipped;


	cv::Mat gray_image1 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\left.jpg");
	cv::Mat gray_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\middle.jpg");


	ImageStitcherFAST stitcher;
	int scale_percent = 10;
	scale_image(gray_image1, gray_image1, scale_percent);
	scale_image(gray_image2, gray_image2, scale_percent);


	//-- Step 1: Detect the keypoints using FAST Detector
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;


	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	detector->detect(gray_image1, keypoints_object);
	detector->detect(gray_image2, keypoints_scene);


	std::cout << "no of keypoints of object " << keypoints_object.size() << std::endl;
	std::cout << "no of keypoints of scene " << keypoints_scene.size() << std::endl;
	std::cout << "Detecting keypoints using FAST detector is completed" << std::endl << std::endl;

	//-- Step 2: Calculate descriptors (feature vectors)
	cv::Mat descriptors_object, descriptors_scene;
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();
	extractor->compute(gray_image1, keypoints_object, descriptors_object);
	extractor->compute(gray_image2, keypoints_scene, descriptors_scene);


	std::cout << "no of desciptors of object " << descriptors_object.size() << std::endl;
	std::cout << "no of desciptors of scene " << descriptors_scene.size() << std::endl;
	std::cout << "Calculating descriptors using ORB is completed" << std::endl << std::endl;


	//-- Step 3: Matching descriptor vectors using FLANN matcher
	std::vector< cv::DMatch > matches;


	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors_object, descriptors_scene, matches);


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
	std::vector< cv::DMatch > good_matches;
	cv::Mat result;
	// cv::Mat result23;

	// cv::Mat H23;
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}


	std::vector< cv::Point2d > obj;
	std::vector< cv::Point2d > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	if (obj.empty() || scene.empty()) {
		std::cout << "No matches found at all....." << std::endl;

	}
	else {
		std::cout << "Enough matches found" << std::endl;

	}
	/* --------------------------------------------------------------------------------

				OpenCV RANSACK implementation

	---------------------------------------------------------------------------------
	*/
	cv::Mat mask;
	cv::Mat cv_H_Matrix;
	cv_H_Matrix = findHomography(obj, scene, mask, cv::RANSAC);


	/* --------------------------------------------------------------------------------

				  RANSAC implementation

	---------------------------------------------------------------------------------
	*/
	RANSAC_algo ra;
	struct returnRANSAC r = ra.computeHomography_RANSAC(obj, scene);
	std::vector< cv::DMatch > ransac_matches;




	std::cout << "Calculated homography using OPENCV:::" << std::endl;
	std::cout << cv_H_Matrix << std::endl;

	std::cout << "Calculated homography using RANSAC implementation" << std::endl;
	std::cout << r.Hmatrix << std::endl;


}

typedef std::vector<bool> DescType; // type of descriptor, 256 bools



void testbench::ORBImplementation() {



	// load image
	cv::Mat left_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_2\\left.jpg");    // load grayscale image
	cv::Mat middle_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_2\\middle.jpg");
	//cv::Mat second_image = cv::imread(second_file, 0);  // load grayscale image

	cv::Mat left_image;
	cv::Mat middle_image;


	cv::resize(left_image2, left_image2, cv::Size(), 0.1, 0.1);
	cv::resize(middle_image2, middle_image2, cv::Size(), 0.1, 0.1);

	cv::cvtColor(left_image2, left_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(middle_image2, middle_image, cv::COLOR_BGR2GRAY);


	std::cout << "Channels: " << left_image2.channels() << " .Type: " << left_image2.type() << std::endl;
	std::cout << "Channels: " << left_image.channels() << " .Type: " << left_image.type() << std::endl;
	

	


	
	cv::Mat descriptors_object, descriptors_scene;

	//detect FAST keypoints using threshold = 40
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
	//cv::FAST(left_image, keypoints_object, 31);

	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	detector->detect(left_image, keypoints_object);
	detector->detect(middle_image, keypoints_scene);

	cv::Ptr<TORB::ORBExtractor> de = new TORB::ORBExtractor(5000, 1.2f, 8, 31, 20); //(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST)
	(*de)(left_image, cv::Mat(), keypoints_object, descriptors_object);
	

	// plot the keypoints
	cv::Mat image_show;
	//cv::drawKeypoints(left_image, keypoints, image_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::imshow("features", image_show);
	// cv::imwrite("feat1.png", image_show);
	//cv::waitKey(0);
	//cout << "First image completed" << endl;
	//cout << "Descriptors" << endl;

	//cout << descriptors << endl;
	
	
	//cv::FAST(middle_image, keypoints_scene, 40);

	//cv::Ptr<TORB::ORBExtractor> de = new TORB::ORBExtractor(10000, 1.2f, 8, 40, 20); //(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST)
	(*de)(middle_image, cv::Mat(), keypoints_scene, descriptors_scene);


	std::vector< cv::DMatch > matches;

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors_object, descriptors_scene, matches);



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


	// Exceptional case
	  // For the first time min_dist became 0
	  // since 3*min_distance = 0 we could not find any good match
	  // hence the default value was set to 3 if min_dis = 0

	if (min_dist < 15) {
		min_dist = 15;
	}

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< cv::DMatch > good_matches;
	
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	std::cout << good_matches.size() << std::endl;

	cv::drawMatches(left_image, keypoints_object, middle_image, keypoints_scene, matches, image_show);
	cv::imshow("matches", image_show);
	//cv::imwrite("matches.png", image_show);
	cv::waitKey(0);



	std::vector< cv::Point2d > obj;
	std::vector< cv::Point2d > scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	if (obj.empty() || scene.empty()) {
		std::cout << "No matches found at all....." << std::endl;

	}
	else {
		std::cout << "Enough matches found" << std::endl;

	}

	cv::Mat mask;
	cv::Mat H = findHomography(obj, scene, mask, cv::RANSAC);
	std::cout << H << std::endl;



	//
	//   
	//   OpenCV implementation
	//
	//

	//cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	detector->detect(left_image, keypoints_object);
	detector->detect(middle_image, keypoints_scene);

	//cv::Mat descriptors_object, descriptors_scene;
	cv::Ptr<cv::ORB> extractor = cv::ORB::create();
	extractor->compute(left_image, keypoints_object, descriptors_object);
	extractor->compute(middle_image, keypoints_scene, descriptors_scene);

	

	//cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors_object, descriptors_scene, matches);



	//double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints 
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist: %f \n", max_dist);
	printf("-- Min dist: %f \n", min_dist);


	// Exceptional case
	  // For the first time min_dist became 0
	  // since 3*min_distance = 0 we could not find any good match
	  // hence the default value was set to 3 if min_dis = 0

	if (min_dist < 15) {
		min_dist = 15;
	}

	//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< cv::DMatch > goodcv_matches;

	for (int i = 0; i < descriptors_object.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			goodcv_matches.push_back(matches[i]);
		}
	}

	std::cout << goodcv_matches.size() << std::endl;

	cv::drawMatches(left_image, keypoints_object, middle_image, keypoints_scene, goodcv_matches, image_show);
	cv::imshow("matches", image_show);
	//cv::imwrite("matches.png", image_show);
	cv::waitKey(0);


	std::vector< cv::Point2d > objcv;
	std::vector< cv::Point2d > scenecv;

	for (int i = 0; i < goodcv_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		objcv.push_back(keypoints_object[goodcv_matches[i].queryIdx].pt);
		scenecv.push_back(keypoints_scene[goodcv_matches[i].trainIdx].pt);
	}

	if (objcv.empty() || scenecv.empty()) {
		std::cout << "No matches found at all....." << std::endl;

	}
	else {
		std::cout << "Enough matches found" << std::endl;

	}

	//cv::Mat mask;
	cv::Mat Hcv = findHomography(objcv, scenecv, mask, cv::RANSAC);

	std::cout << Hcv << std::endl;

}


void testbench::ORBMatcherTest() {



	// load image
	cv::Mat left_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_2\\left.jpg");    // load grayscale image
	cv::Mat middle_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_2\\middle.jpg");
	//cv::Mat second_image = cv::imread(second_file, 0);  // load grayscale image

	cv::Mat left_image;
	cv::Mat middle_image;


	cv::resize(left_image2, left_image2, cv::Size(), 0.1, 0.1);
	cv::resize(middle_image2, middle_image2, cv::Size(), 0.1, 0.1);

	cv::cvtColor(left_image2, left_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(middle_image2, middle_image, cv::COLOR_BGR2GRAY);


	cv::Mat descriptors_object, descriptors_scene;

	//detect FAST keypoints using threshold = 40
	std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
	//cv::FAST(left_image, keypoints_object, 31);

	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	detector->detect(left_image, keypoints_object);
	detector->detect(middle_image, keypoints_scene);

	cv::Ptr<TORB::ORBExtractor> de = new TORB::ORBExtractor(5000, 1.2f, 8, 31, 20); //(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST)
	(*de)(left_image, cv::Mat(), keypoints_object, descriptors_object);
	(*de)(middle_image, cv::Mat(), keypoints_scene, descriptors_scene);

	
	cv::Mat image_show;
	


	std::vector< cv::DMatch > matches;
	std::vector< cv::DMatch > CVmatches;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors_object, descriptors_scene, CVmatches);

	std::cout << "Opencv matcher size: "  << CVmatches.size() << std::endl ;

   ORBMatcher orb;
   orb.MatchDescriptors(descriptors_object, descriptors_scene, &matches);


   std::cout << "My matcher size: " << matches.size() << std::endl;

   for (int i = 0; i < matches.size(); i++) {
   
	   std::cout << "distance: " << matches[i].distance;
	   std::cout << " queryIdx: " << matches[i].queryIdx;
	   std::cout << " trainIdx: " << matches[i].trainIdx << std::endl;
   }

   std::cout << "End........"<< std::endl;

   for (int i = 0; i < CVmatches.size(); i++) {

	   std::cout << "distance: " << CVmatches[i].distance;
	   std::cout << " queryIdx: " << CVmatches[i].queryIdx;
	   std::cout << " trainIdx: " << CVmatches[i].trainIdx << std::endl;
   }

}

void testbench::ORBHLS()
{

	cv::Mat image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_FPGA\\middle_r.jpg",0);

	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoints_vector;

	detector->detect(image, keypoints_vector);

	const int NUM_KEYPOINTS = 1000 ;
	cv::KeyPoint keypoints_array[1000];
	for (int i = 0; i < 1000; i++) {
		keypoints_array[i] = keypoints_vector.at(i);
	}

	cv::Mat descriptor;
	
	TORBHLS::ORBExtractorHLS<1000, 8, 31, 20, 1000>(image, keypoints_array, descriptor, 1.2);


}




















static void
HarrisResponses(const cv::Mat& img, const std::vector<cv::Rect>& layerinfo, std::vector<cv::KeyPoint>& pts, int blockSize, float harris_k)
{
	CV_Assert(img.type() == CV_8UC1 && blockSize * blockSize <= 2048);

	size_t ptidx, ptsize = pts.size();

	const uchar* ptr00 = img.ptr<uchar>();
	int step = (int)(img.step / img.elemSize1());
	int r = blockSize / 2;

	float scale = 1.f / ((1 << 2) * blockSize * 255.f);
	float scale_sq_sq = scale * scale * scale * scale;

	cv::AutoBuffer<int> ofsbuf(blockSize * blockSize);
	int* ofs = ofsbuf.data();
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i * blockSize + j] = (int)(i * step + j);

	for (ptidx = 0; ptidx < ptsize; ptidx++)
	{
		int x0 = cvRound(pts[ptidx].pt.x);
		int y0 = cvRound(pts[ptidx].pt.y);
		int z = pts[ptidx].octave;

		const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y) * step + x0 - r + layerinfo[z].x;
		int a = 0, b = 0, c = 0;

		for (int k = 0; k < blockSize * blockSize; k++)
		{
			const uchar* ptr = ptr0 + ofs[k];
			int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
			int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
			a += Ix * Ix;
			b += Iy * Iy;
			c += Ix * Iy;
		}
		pts[ptidx].response = ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
	}
}




void testbench::response()
{

	cv::Mat image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_FPGA\\middle_r.jpg", 0);

	std::cout << image.type();
	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	std::vector<cv::KeyPoint> keypoints_vector;

	detector->detect(image, keypoints_vector);
	for (int i = 0; i < 10; i++) {
		std::cout <<"response "<< keypoints_vector[i].response << "\n";
		std::cout << "octave "<< keypoints_vector[i].octave << "\n";
	}


	std::vector<cv::KeyPoint> keypoints_r;
	for (int i = 0; i < keypoints_vector.size(); i++) {
		cv::KeyPoint j;
		j.octave = keypoints_vector[i].octave;
		j.pt.x = keypoints_vector[i].pt.x;
		j.pt.y = keypoints_vector[i].pt.y;
		keypoints_r.push_back(j);
	}

	for (int i = 0; i < 25; i++) {
		std::cout << keypoints_vector[i].pt.x << "," << keypoints_vector[i].pt.y << "  " << keypoints_r[i].pt.x << "," << keypoints_r[i].pt.y << "\n";

	}


	std::vector<cv::Rect> layerinfo;

	cv::Rect linfo(0,0, image.cols, image.rows);
	layerinfo.push_back(linfo);
	std::vector<cv::KeyPoint> pts;
	//pts.push_back(keypoints_r);
	//HarrisResponses(image, layerinfo,  pts, 7, 0.04f);
	HarrisResponses(image, layerinfo, keypoints_r, 7, 0.04f);
	for (int i = 0; i < 10; i++) {
		std::cout << "response " << keypoints_r[i].response << "\n";
		std::cout << "octave " << keypoints_r[i].octave << "\n";
	}

	for (int i = 0; i < 25; i++) {
		std::cout << keypoints_vector[i].response << "  " << keypoints_r[i].response << "\n";

	}

}
