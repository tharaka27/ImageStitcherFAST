#pragma once


#include "opencv2/core/core.hpp"

class ORBMatcher
{
public:
	ORBMatcher();

	static int ComputeOrbDistance(cv::Mat vector1, cv::Mat vector2);

	static std::vector<int> MatchDescriptors(cv::Mat descriptors1, cv::Mat descriptors2, std::vector<cv::DMatch>* matches);

	//void MatchDescriptors(cv::InputArray queryDescriptors, cv::InputArray trainDescriptors, std::vector<cv::DMatch>& matches);

private:
	static const int TH_HAMMING_DIST = 80;
};



