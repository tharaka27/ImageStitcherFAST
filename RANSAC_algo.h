#pragma once
class RANSAC_algo
{
private:

	
	

	double GetDistance(cv::Point2f  test_src, cv::Point2f test_dst, cv::Mat H);

public :
	cv::Mat computeHomography_RANSAC(std::vector<cv::Point2f> obj, std::vector<cv::Point2f> scene);
	//cv::Mat_<double> normalize_matrix(std::vector<cv::Point2f> pointsVec, std::vector<cv::Mat> points3d);
	struct norm normalize_matrix(std::vector<cv::Point2f> pointsVec);
};

