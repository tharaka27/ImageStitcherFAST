#pragma once

#include "opencv2/core/core.hpp"


class homography
{
public:
	//void gaussian_elimination(float* input, int n);
	//void findHomography(cv::Point2f src[4], cv::Point2f dst[4]);

	void gaussian_elimination(double* input, int n);

	cv::Mat findHomography_(cv::Point2f src[4], cv::Point2f dst[4]);

	void findHomography(std::vector<cv::Point2f> src, std::vector<cv::Point2f> dst);
	cv::Mat findHomographySVD(cv::Point2f src[4], cv::Point2f dst[4]);
};
	

