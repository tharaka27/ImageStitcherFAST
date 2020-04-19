#pragma once

struct returnRANSAC {

	cv::Mat Hmatrix;
	cv::Mat InlierMask;

};


class RANSAC_algo
{

public :
	
	returnRANSAC computeHomography_RANSAC(std::vector<cv::Point2d> obj, std::vector<cv::Point2d> scene );

};

