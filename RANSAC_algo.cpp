/* 


   Copyright (c) 2020, Tharaka Ratnayake, email: tharakasachintha.17@cse.mrt.ac.lk
   All rights reserved. https://github.com/tharaka27/ImageStitcherFAST
   
   Some algorithms used in this code referred to:
   1. OpenCV: http://opencv.org/
   
   
   Got some of the RANSAC basics from this lecture slides for homography calculation using RANSAC
	   http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf

   Revision history:
	  March 30th, 2020: initial version.
	  April  7th, 2020: Added new error calculations schemes
	  April 10th, 2020: Added Symmetric transfer error as the error calculation mechanism
	  April 12th, 2020: Code refactored.

*/

#include <cmath>
#include <array>
#include <vector>
#include <iostream>
#include <ctime>

#include <numeric>


#include "opencv2/core/core.hpp"

#include "RANSAC_algo.h"
#include "homography.h"


#define T_DIST 20  // thres. for distance in RANSAC algorithm 



struct norm {
	cv::Mat_<double> Normalization_matrix;
	std::vector<cv::Point2f> points3d;
};


/*
 @Check colinearity of a set of pts.

 @Param num Number of points in the array.
 @Param p Pointer to the array of cv::Point2f points.

The function check whether there are any collinear points in the given set of points.

It optionally returns true if there is/are collinear point/s, false is no colinear
point/s.

Consider 3 points such that A =(x1,y1) , B =(x2,y2), C=(x3,y3)
				computing cross product between A and B gives us the line between them.
				We can take the dot product with line and C. if the are collinear resulting
				value will be zero
				A * B = |i     j   k|
						|x1   y1   1|
						|x2   y2   1|

					  = i(y1-y2) -j(x1-x2) + k(x1y2 - x2y1)

				C . (A*B) = (x3,y3,1) . (y1-y2, x2-x1, x1y2 - x2y1)
						  = x3 (y1 - y2) + y3 (x1-x2) + (x1y2 - x2y1)
 */

/*
bool isColinear(int num, cv::Point2d* p) {
	int i, j, k;
	bool iscolinear;
	float value;

	iscolinear = false;
	// check for each 3 points combination
	for (i = 0; i < num - 2; i++) {
		for (j = i + 1; j < num - 1; j++) {
			for (k = j + 1; k < num; k++) {

				value = p[k].x * (p[i].y - p[j].y) - p[k].y * (p[i].x - p[j].x)
					+ (p[i].x * p[j].y - p[j].x * p[i].y);

				//std::cout << value << std::endl;

				if (abs(value) < 10e-2) {
					iscolinear = true;
					break;
				}
			}
			if (iscolinear == true) break;
		}
		if (iscolinear == true) break;
	}

	return iscolinear;
}

*/

//**********************************************************************
// finding the normalization matrix x' = T*x, where 
//  T={ s,0,tx, 
//      0,s,ty, 
//      0,0, 1 }
// compute T such that the centroid of x' is the coordinate origin (0,0)T
// and the average distance of x' to the origin is sqrt(2)
// we can derive that tx = -scale*mean(x), ty = -scale*mean(y),
// scale = sqrt(2)/(sum(sqrt((xi-mean(x)^2)+(yi-mean(y))^2))/n)
// where n is the total number of points
// input: num (ttl number of pts)
// p (pts to be normalized)
// output: T (normalization matrix)
// p (normalized pts)
// NOTE: because of the normalization process, the pts coordinates should
// has accurcy as "float" or "double" instead of "int"
//**********************************************************************
struct norm normalize_matri(std::vector<cv::Point2d> pointsVec) {
	
	// Averaging
	double count = (double)pointsVec.size();
	double xAvg = 0;
	double yAvg = 0;
	for (auto& member : pointsVec) {
		xAvg = xAvg + member.x;
		yAvg = yAvg + member.y;
	}
	xAvg = xAvg / count;
	yAvg = yAvg / count;

	// Normalization
	std::vector<cv::Point2f> points3d;
	std::vector<double> distances;
	for (auto& member : pointsVec) {

		double distance = (std::sqrt(std::pow((member.x - xAvg), 2) + std::pow((member.y - yAvg), 2)));
		distances.push_back(distance);
	}
	double xy_norm = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

	// Create a matrix transforming the points into having mean (0,0) and mean distance to the center equal to sqrt(2)
	cv::Mat_<double> Normalization_matrix(3, 3);
	double diagonal_element = sqrt(2) / xy_norm;
	double element_13 = -sqrt(2) * xAvg / xy_norm;
	double element_23 = -sqrt(2) * yAvg / xy_norm;

	Normalization_matrix << diagonal_element, 0, element_13, 0, diagonal_element, element_23, 0, 0, 1;

	// Multiply the original points with the normalization matrix
	for (auto& member : pointsVec) {
		cv::Mat triplet = (cv::Mat_<double>(3, 1) << member.x, member.y, 1);
		//points3d.emplace_back(Normalization_matrix * triplet);

		cv::Mat r = Normalization_matrix * triplet;
		cv::Point2f sample;
		sample.x = r.at<double>(0, 0);
		sample.y = r.at<double>(1, 0);
		points3d.emplace_back(sample);
	}

	struct norm r;
	r.Normalization_matrix = Normalization_matrix;
	r.points3d = points3d;
	return r;
}



/*
 @Compute the least square of the given dataset.

 @Param obj std::vector to the array of cv::Point2f obj.
 @Param scene std::vector to the array of cv::Point2f scene.
 @Param numOfIn number of Inliers in the inlier mask.
 @Param InliersMat cv::Mat of the inlier mask matrix.

 Compute number of inliers by computing distance under a perticular H
		distance = d(Hx, x') + d(invH x', x)

 return: number of inliers
 */
cv::Mat leastSquare(std::vector< cv::Point2d > obj,
	std::vector< cv::Point2d > scene, int numOfIn, cv::Mat InliersMat) {


	std::vector< cv::Point2d > obj_inliers;
	std::vector< cv::Point2d > scene_inliers;

	for (int i = 0; i < obj.size(); i++) {

		if (InliersMat.at<double>(i,0) == 1) {
			obj_inliers.push_back(obj[i]);
			scene_inliers.push_back(scene[i]);
		}
	}

	/*
		Normalize the coordinate system to stabilize the Least Squares algorithm
	*/
	struct norm norm_obj;
	struct norm norm_scene;
	norm_obj = normalize_matri(obj_inliers);
	norm_scene = normalize_matri(scene_inliers);
	/*
		let's create a opencv matrix to store X and Y values
	*/
	cv::Mat X = cv::Mat_<double>(2 * numOfIn, 9, double(0));

	int j = 0;

	std::cout << norm_scene.points3d.size() << std::endl;

	for (int i = 0; i < scene_inliers.size(); i++) {

		cv::Mat X_ = (cv::Mat_<double>(1, 9) << -norm_obj.points3d[i].x, -norm_obj.points3d[i].y, -1, 0, 0, 0,
			norm_obj.points3d[i].x * norm_scene.points3d[i].x, norm_obj.points3d[i].y * norm_scene.points3d[i].x, norm_scene.points3d[i].x);
		X.row(j) += X_;

		X_ = (cv::Mat_<double>(1, 9) << 0, 0, 0, -norm_obj.points3d[i].x, -norm_obj.points3d[i].y, -1,
			norm_obj.points3d[i].x * norm_scene.points3d[i].y, norm_obj.points3d[i].y * norm_scene.points3d[i].y, norm_scene.points3d[i].y);
		X.row(j + 1) += X_;


		j = j + 2;


	}

	cv::Mat U = cv::Mat_<double>(2 * numOfIn, 9, double(0));
	cv::Mat W = cv::Mat_<double>(9, 9, double(0));
	cv::Mat Vt = cv::Mat_<double>(9, 9, double(0));

	cv::SVDecomp(X, W, U, Vt);

	cv::Mat V = cv::Mat_<double>(9, 9, double(0));
	V = Vt.t();
	cv::Mat H = (cv::Mat_<double>(3, 3) << V.at<double>(0, 8), V.at<double>(1, 8), V.at<double>(2, 8),
		V.at<double>(3, 8), V.at<double>(4, 8), V.at<double>(5, 8), V.at<double>(6, 8), V.at<double>(7, 8), V.at<double>(8, 8));

	H = norm_scene.Normalization_matrix.inv() * H * norm_obj.Normalization_matrix;
	H = H / H.at<double>(2, 2);

	return H;
}


/*
 @Compute the number of Inliers using symmetric transfer error.

 @Param num Number of points in the array.
 @Param H Pointer to the homography matrix.
 @Param p1 std::vector to the array of cv::Point2f obj.
 @Param p2 std::vector to the array of cv::Point2f scene.
 @Param inlier_mask Pointer to the inlier mask matrix.
 @Param dist_std Pointer to the std of the distance among all inliers.

 Compute number of inliers by computing distance under a perticular H
		distance = d(Hx, x') + d(invH x', x)

 return: number of inliers
 */
int ComputeNumberOfInliers(int num, cv::Mat H, std::vector< cv::Point2d> obj, 
	std::vector< cv::Point2d> scene, cv::Mat* inlier_mask, double* dist_std) {
	
	int i, num_inlier;
	double curr_dist, sum_dist, mean_dist;
	cv::Point2f tmp_pt;

	cv::Mat dist = cv::Mat(num, 1, CV_64FC1);

	cv::Mat Hin = cv::Mat(3, 1, CV_64FC1);
	H.copyTo(Hin);
	cv::Mat invH = cv::Mat(3, 1, CV_64FC1);

	invH = Hin.inv();

	// check each correspondence
	sum_dist = 0;
	num_inlier = 0;


	//cvZero(inlier_mask);
	for (int i = 0; i < inlier_mask->rows; i++) {
		inlier_mask->at<double>(i, 0) = 0;
	}


	for (i = 0; i < num; i++) {

		// initial point x
		cv::Mat x = (cv::Mat_<double>(3, 1) << obj[i].x, obj[i].y, 1);

		// initial point x'
		cv::Mat xp = (cv::Mat_<double>(3, 1) << scene[i].x, scene[i].y, 1);


		cv::Mat pt = cv::Mat(3, 1, CV_64FC1);
		// d(Hx, x')
		pt = Hin * x;
		tmp_pt.x = pt.at<double>(0, 0) / pt.at<double>(2, 0);
		tmp_pt.y = pt.at<double>(1, 0) / pt.at<double>(2, 0);
		curr_dist = pow(tmp_pt.x - scene[i].x, 2.0) + pow(tmp_pt.y - scene[i].y, 2.0);

		// d(x, invH x')
		pt = invH * xp;
		tmp_pt.x = pt.at<double>(0, 0) / pt.at<double>(2, 0);
		tmp_pt.y = pt.at<double>(1, 0) / pt.at<double>(2, 0);
		curr_dist += pow(tmp_pt.x - obj[i].x, 2.0) + pow(tmp_pt.y - obj[i].y, 2.0);

		if (curr_dist < T_DIST) {
			// an inlier
			num_inlier++;
			inlier_mask->at<double>(i, 0) = 1;
			dist.at<double>(i, 0) = curr_dist;
			sum_dist += curr_dist;
			//inliers->at(i) = true;
		}
	}


	// Compute the standard deviation of the distance
	mean_dist = sum_dist / (double)num_inlier;


	*dist_std = 0;
	for (i = 0; i < num; i++) {
		if (inlier_mask->at<double>(i, 0) == 1)
			*dist_std += pow(dist.at<double>(i, 0) - mean_dist, 2.0);
	}

	*dist_std /= (double)(num_inlier - 1);
	return num_inlier;
}



/*
 @Compute the homogrphy using RANSAC Algorithm

 @Param obj std::vector to the array of points of object
 @Param scene std::vector to the array of points of scene

 Compute Homography matrix and Inlier mask by minimizing the
 distance under a perticular H
 distance = d(Hx, x') + d(invH x', x)

 return: structure containing Homography matrix and inlier mask
 */
returnRANSAC RANSAC_algo::computeHomography_RANSAC(std::vector< cv::Point2d > obj, std::vector< cv::Point2d > scene) {

	
	// required probability that a result will be generated from inliers-only
	double Confidence = 0.99995;

	// threshold on error metric 
	double Threshold = 1;

	// approximate expected inlier fraction
	double ApproximateInlierFraction = 0.5;

	// Output variables ... 
	std::vector<bool> Inliers;
	bool bSuccess(false);

	cv::Mat inlier_mask  = cv::Mat_<double>(obj.size(), 1, double(0));
	



	//randomize the random number generator
	srand(1235);


	// initialize matrices need for homography calculation (exact)
	// and for inlier detection
	cv::Point2d src[4];
	cv::Point2d dst[4];

	cv::Point2d good_src[4];
	cv::Point2d good_dst[4];

	
	cv::Point2d test_src;
	cv::Point2d test_dst;

	cv::Point2d max_src;
	cv::Point2d max_dst;
	
	std::vector<bool> TInliers;

	//lets create a variable to count how many of the points are inliners
	int numOfIn = 0;

	// and another variable to store the max number of inliners acheived by 
	// the last selection of points
	int maxNumOfIn = 0;

	//number of inliners 
	double w = 0.00;

	//number of data points
	double p = 0.00;

	//Set the number of iterations to do based on the 
	//confidence required and the approximated inliner fraction

	// 4 is used since we need to find 4 data points to calculate homography correctly
	int k = log(1 - Confidence) / log(1 - pow(ApproximateInlierFraction, 4));
	k = 1000;
	std::cout << "iteration is :" << k << std::endl;
	
	
	bool iscolinear;


	for (int i = 0; i <= k; i++) {

        /*
		iscolinear = true;
		while (iscolinear == true) {
			iscolinear = false;
			for (i = 0; i < 4; i++) {
				// randomly select an index
				curr_idx[i] = rand() % num;
				for (j = 0; j < i; j++) {
					if (curr_idx[i] == curr_idx[j]) {
						iscolinear = true;
						break;
					}
				}
				if (iscolinear == true) break;
				curr_m1[i].x = m1[curr_idx[i]].x;
				curr_m1[i].y = m1[curr_idx[i]].y;
				curr_m2[i].x = m2[curr_idx[i]].x;
				curr_m2[i].y = m2[curr_idx[i]].y;

				//std::cout << curr_m1[i].x << " "<< curr_m1[i].y << " "<< curr_m2[i].x <<" " <<curr_m2[i].y << std::endl;
			}
			// Check whether these points are colinear
			if (iscolinear == false)
				iscolinear = isColinear(s, curr_m1);
		}
		*/

		// choose four random points from the sample points
		// using a randomised random number generator
		int testP0 = (rand() % obj.size());
		src[0] = obj[testP0];
		dst[0] = scene[testP0];

		int testP1 = (rand() % obj.size());
		src[1] = obj[testP1];
		dst[1] = scene[testP1];

		int testP2 = (rand() % obj.size());
		src[2] = obj[testP2];
		dst[2] = scene[testP2];

		int testP3 = (rand() % obj.size());
		src[3] = obj[testP3];
		dst[3] = scene[testP3];


		// let's calculate the homography matirx (exact) using the four random points.
		homography c;
		cv::Mat H = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
			
		H = c.findHomography_(src, dst);
		
		//reset the number of inliner for the nex iteration
		numOfIn = 0;

		//clear the list on inliers for this iteration
		TInliers.clear();
		
		cv::Mat Temp_inlier_mask = cv::Mat_<double>(obj.size(), 1 , double(0)) ;

		double dist_std ;

		numOfIn = ComputeNumberOfInliers( obj.size(),H, obj, scene, &Temp_inlier_mask, &dist_std);

		//save the maximum number of inliners found in the data set and the points that form them
		if (maxNumOfIn < numOfIn) {

			maxNumOfIn = numOfIn;
			
			good_src[0] = obj[testP0];
			good_dst[0] = scene[testP0];

			good_src[1] = obj[testP1];
			good_dst[1] = scene[testP1];

			good_src[2] = obj[testP2];
			good_dst[2] = scene[testP2];

			good_src[3] = obj[testP3];
			good_dst[3] = scene[testP3];

			//copy the inliners to the Output inliners
			Inliers = TInliers;
			inlier_mask = Temp_inlier_mask;
		}
	}

	
	//it is always a success because it is running for a certain
	//numner of iterations
	bSuccess = true;

	// if successful, write the line parameterized as two points, 
	// and each input point along with its inlier status

	cv::Mat H = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);


	
	H = leastSquare(obj, scene, maxNumOfIn, inlier_mask);
	
	
	struct returnRANSAC r;
	r.Hmatrix = H;
	r.InlierMask = inlier_mask;

	return r;
}



