#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"


class ImageStitcherFAST
{
public:
	
	cv::Mat calculate_h_matrix(cv::Mat gray_image1, cv::Mat gray_image2);
	cv::Mat stitch_image(cv::Mat image1, cv::Mat image2, cv::Mat H);
	void scale_image(cv::Mat& InputImage, cv::Mat& OutputImage, int scale_percent);
	
};