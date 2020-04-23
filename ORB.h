#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
const double pi = 3.1415926;    // pi
typedef vector<bool> DescType; // type of descriptor, 256 bools

class ORB
{
public:
	/**
	 * compute the angle for ORB descriptor
	 * @param [in] image input image
	 * @param [in|out] detected keypoints
	 */
	void computeAngle(const cv::Mat& image, vector<cv::KeyPoint>& keypoints);


	/**
	 * compute ORB descriptor
	 * @param [in] image the input image
	 * @param [in] keypoints detected keypoints
	 * @param [out] desc descriptor
	 */
	 
	void computeORBDesc(const cv::Mat& image, vector<cv::KeyPoint>& keypoints, vector<DescType>& desc);


	/**
	 * brute-force match two sets of descriptors
	 * @param desc1 the first descriptor
	 * @param desc2 the second descriptor
	 * @param matches matches of two images
	 */
	void bfMatch(const vector<DescType>& desc1, const vector<DescType>& desc2, vector<cv::DMatch>& matches);

};