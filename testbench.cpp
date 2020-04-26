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
#include "ORB.h"
#include "ORBExtractor.h"
#include "ORBMatcher.h"


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
	
	RANSAC_algo r;
	
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

typedef vector<bool> DescType; // type of descriptor, 256 bools


void testbench::ComputeAngle() {

	/*
	cv::Mat left_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\left.jpg");
	int scale_percent = 10;
	int width = int(left_image.cols * scale_percent / 100);
	int height = int(left_image.rows * scale_percent / 100);
	cv::resize(left_image, left_image, cv::Size(width, height));

	std::vector<cv::KeyPoint> keypoints_object;
	cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
	detector->detect(left_image, keypoints_object);


	std::cout << "Fast detected keypoints : " << keypoints_object.size() << std::endl;

	//cv::Mat image = cv::Mat_<double>(17, 17, double(1));
	

	std::cout << "First keypoint : (" << keypoints_object[500].pt.x <<"," << keypoints_object[500].pt.y << ")" << std::endl;

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	keypoints.push_back(keypoints_object[500]);

	cv::Ptr<cv::ORB> extractor = cv::ORB::create();
	extractor->compute(left_image, keypoints, descriptors);
	std::cout << descriptors << std::endl;
	std::cout << keypoints_object[500].angle << std::endl;
	*/


	cv::Mat image = cv::Mat_<uchar>(15, 15, uchar(2));
	
	for (int i = 0; i < 15; i++) {
		for (int j = 0; j <15; j++) {
			image.at<uchar>(i, j) = i*i;
		}
	}
	std::vector<DescType> descriptors;
	ORB orb;
	std::vector<cv::KeyPoint> keypoints;
	keypoints.push_back(cv::KeyPoint(8,8,1));
	orb.computeAngle(image, keypoints);
	std::cout << keypoints[0].angle << std::endl;

	cv::Moments M = cv::moments(image);
	std::cout << M.m01 / M.m10 << std::endl;

}



void testbench::TestORB() {



	// load image
	cv::Mat left_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_2\\left.jpg");    // load grayscale image
	cv::Mat middle_image2 = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_2\\middle.jpg");
	//cv::Mat second_image = cv::imread(second_file, 0);  // load grayscale image

	cv::Mat left_image;
	cv::Mat middle_image;


	cv::resize(left_image2, left_image2, cv::Size(), 0.2, 0.2);
	cv::resize(middle_image2, middle_image2, cv::Size(), 0.2, 0.2);

	cv::cvtColor(left_image2, left_image, cv::COLOR_BGR2GRAY);
	cv::cvtColor(middle_image2, middle_image, cv::COLOR_BGR2GRAY);


	std::cout << "Channels: " << left_image2.channels() << " .Type: " << left_image2.type() << std::endl;
	std::cout << "Channels: " << left_image.channels() << " .Type: " << left_image.type() << std::endl;
	ImageStitcherFAST stitcher;


	//return 0;

	// plot the image
	cv::imshow("first image", left_image);
	// cv::imshow("second image", second_image);
	cv::waitKey(0);

	// detect FAST keypoints using threshold=40
	vector<cv::KeyPoint> keypoints;
	cv::FAST(left_image, keypoints, 40);
	cout << "keypoints: " << keypoints.size() << endl;

	// compute angle for each keypoint
	ORB orb;
	orb.computeAngle(left_image, keypoints);



	// compute ORB descriptors
	vector<DescType> descriptors;
	orb.computeORBDesc(left_image, keypoints, descriptors);


	// plot the keypoints
	cv::Mat image_show;
	cv::drawKeypoints(left_image, keypoints, image_show, cv::Scalar::all(-1),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("features", image_show);
	// cv::imwrite("feat1.png", image_show);
	cv::waitKey(0);
	cout << "First image completed" << endl;

	// we can also match descriptors between images
	// same for the second
	vector<cv::KeyPoint> keypoints2;
	cv::FAST(middle_image, keypoints2, 40);
	cout << "keypoints: " << keypoints2.size() << endl;

	// compute angle for each keypoint
	orb.computeAngle(middle_image, keypoints2);

	// compute ORB descriptors
	vector<DescType> descriptors2;
	orb.computeORBDesc(middle_image, keypoints2, descriptors2);
	cout << "Second image completed" << endl;

	// find matches
	vector<cv::DMatch> matches;
	orb.bfMatch(descriptors, descriptors2, matches);
	cout << "matches: " << matches.size() << endl;

	//cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	//matcher->match(descriptors, descriptors2, matches);

	// plot the matches
	cv::drawMatches(left_image, keypoints, middle_image, keypoints2, matches, image_show);
	cv::imshow("matches", image_show);
	cv::imwrite("matches.png", image_show);
	cv::waitKey(0);

}



void testbench::TestFull() {



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
	ImageStitcherFAST stitcher;


	//return 0;

	// plot the image
	cv::imshow("first image", left_image);
	// cv::imshow("second image", second_image);
	cv::waitKey(0);

	// detect FAST keypoints using threshold=40
	vector<cv::KeyPoint> keypoints;
	cv::FAST(left_image, keypoints, 40);
	cout << "keypoints: " << keypoints.size() << endl;

	// compute angle for each keypoint
	ORB orb;
	orb.computeAngle(left_image, keypoints);



	// compute ORB descriptors
	vector<DescType> descriptors;
	orb.computeORBDesc(left_image, keypoints, descriptors);


	// plot the keypoints
	cv::Mat image_show;
	cv::drawKeypoints(left_image, keypoints, image_show, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::imshow("features", image_show);
	// cv::imwrite("feat1.png", image_show);
	cv::waitKey(0);
	cout << "First image completed" << endl;

	// we can also match descriptors between images
	// same for the second
	vector<cv::KeyPoint> keypoints2;
	cv::FAST(middle_image, keypoints2, 40);
	cout << "keypoints: " << keypoints2.size() << endl;

	// compute angle for each keypoint
	orb.computeAngle(middle_image, keypoints2);

	// compute ORB descriptors
	vector<DescType> descriptors2;
	orb.computeORBDesc(middle_image, keypoints2, descriptors2);
	cout << "Second image completed" << endl;

	// find matches
	vector<cv::DMatch> matches;


	orb.bfMatch(descriptors, descriptors2, matches);
	cout << "matches: " << matches.size() << endl;

	//cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	//matcher->match(descriptors, descriptors2, matches);

	// plot the matches
	cv::drawMatches(left_image, keypoints, middle_image, keypoints2, matches, image_show);
	cv::imshow("matches", image_show);
	cv::imwrite("matches.png", image_show);
	cv::waitKey(0);



	std::vector< cv::Point2d > obj;
	std::vector< cv::Point2d > scene;

	for (int i = 0; i < matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints[matches[i].queryIdx].pt);
		scene.push_back(keypoints2[matches[i].trainIdx].pt);
	}

	cv::Mat H;
	H = findHomography(obj, scene,cv::RANSAC);

	std::cout << "Implemented ORB algoritham generated H matrix" << std::endl;
	std::cout << H << std::endl;



	//
	//  As a test we compare the H matrix generated by our implementation vs the H
	//  Matrix generated by the OpenCV implementation.
	//
	//

    left_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\left.jpg");
	middle_image = cv::imread("C:\\Users\\ASUS\\Desktop\\sem 5 project\\ImageStitcherSIFT\\Data_3\\middle.jpg");
	

	cv::Mat left_image_gray;
	cv::Mat middle_image_gray;
	

	//ImageStitcherFAST stitcher;

	//resize images since the calculation takes lots of time !!!

	int scale_percent = 10;
	stitcher.scale_image(left_image, left_image, scale_percent);
	stitcher.scale_image(middle_image, middle_image, scale_percent);
	
	//-----------------------------------------------------------------

	cv::cvtColor(left_image, left_image_gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(middle_image, middle_image_gray, cv::COLOR_BGR2GRAY);
	

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

	std::cout << "OpenCV ORB algoritham generated H matrix" << std::endl;
	std::cout << H12 << std::endl;




}


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
	vector<cv::KeyPoint> keypoints_object, keypoints_scene;
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

	cout << good_matches.size() << endl;

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
		cout << "No matches found at all....." << endl;

	}
	else {
		cout << "Enough matches found" << endl;

	}

	cv::Mat mask;
	cv::Mat H = findHomography(obj, scene, mask, cv::RANSAC);
	cout << H << endl;



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

	cout << goodcv_matches.size() << endl;

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
		cout << "No matches found at all....." << endl;

	}
	else {
		cout << "Enough matches found" << endl;

	}

	//cv::Mat mask;
	cv::Mat Hcv = findHomography(objcv, scenecv, mask, cv::RANSAC);

	cout << Hcv << endl;

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
	vector<cv::KeyPoint> keypoints_object, keypoints_scene;
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

	cout << "Opencv matcher size: "  << CVmatches.size() << endl ;

   ORBMatcher orb;
   orb.MatchDescriptors(descriptors_object, descriptors_scene, &matches);


   cout << "My matcher size: " << matches.size() << endl;

   for (int i = 0; i < matches.size(); i++) {
   
	   cout << "distance: " << matches[i].distance;
	   cout << " queryIdx: " << matches[i].queryIdx;
	   cout << " trainIdx: " << matches[i].trainIdx << endl;
   }

   cout << "End........"<< endl;

   for (int i = 0; i < CVmatches.size(); i++) {

	   cout << "distance: " << CVmatches[i].distance;
	   cout << " queryIdx: " << CVmatches[i].queryIdx;
	   cout << " trainIdx: " << CVmatches[i].trainIdx << endl;
   }

}