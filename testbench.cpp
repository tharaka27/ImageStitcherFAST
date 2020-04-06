#include "testbench.h"
#include <iostream>

#include "homography.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2\calib3d.hpp>
#include "RANSAC_algo.h"


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

	cv::Point2f src[4];
	cv::Point2f dst[4];
	src[0] = cv::Point2f(141, 131);
	src[1] = cv::Point2f(480, 159);
	src[2] = cv::Point2f(493, 630);
	src[3] = cv::Point2f(64, 601);

	dst[0] = cv::Point2f(318, 256);
	dst[1] = cv::Point2f(534, 372);
	dst[2] = cv::Point2f(316, 670);
	dst[3] = cv::Point2f(73, 473);


	homography homo;
	cv::Mat V = homo.findHomographySVD(src, dst);
	cv::Mat h = homo.findHomography_(src,dst);
	//cv::Mat V = homo.findHomographySVD(src, dst);

	cv::Mat obj = (cv::Mat_<float>(4, 2) << 141, 131, 480, 159, 493, 630, 64, 601);
	cv::Mat scene = (cv::Mat_<float>(4, 2) << 318, 256, 534, 372, 316, 670, 73, 473);

	cv::Mat H = findHomography(obj, scene, cv::RANSAC);
	std::cout << "homograph Matrix from opencv RANSAC implementation" << std::endl;
	std::cout << H << std::endl;
	std::cout << "homograph Matrix from gussian elemenation implementation" << std::endl;
	std::cout << h << std::endl;
	std::cout << "homograph Matrix from SVD implementation" << std::endl;
	std::cout << V << std::endl;

}


struct norm {
	cv::Mat_<double> Normalization_matrix;
	std::vector<cv::Point2f> points3d;
};


void testbench::NormalizeMatrix() {

	std::vector<cv::Point2f> pointsVec;
	//std::vector<cv::Mat> points3d; 
	RANSAC_algo r;
	//cv::Mat_<double> n;
	
	pointsVec.push_back(cv::Point2f(1, 1));
	pointsVec.push_back(cv::Point2f(0, 0));
	pointsVec.push_back(cv::Point2f(2, 2));
	pointsVec.push_back(cv::Point2f(4, 4));
	
	struct norm d;

	d = r.normalize_matrix(pointsVec);

	std::cout << d.Normalization_matrix << std::endl;
	std::cout << d.points3d.at(0) << std::endl;
	std::cout << d.points3d.at(1) << std::endl;
	std::cout << d.points3d.at(2) << std::endl;
	std::cout << d.points3d.at(3) << std::endl;
	}

