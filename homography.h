#pragma once

#include "opencv2/core/core.hpp"


class homography
{
public:

	void gaussian_elimination(double* input, int n);

	cv::Mat findHomography_(cv::Point2d src[4], cv::Point2d dst[4]);

};
	

