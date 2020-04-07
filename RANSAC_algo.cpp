
#include <cmath>
#include <array>
#include <vector>
#include <iostream>
#include <ctime>

#include <numeric>


#include "opencv2/core/core.hpp"

#include "RANSAC_algo.h"
#include "homography.h"
/*
------------------------------------------------------------------------
File: ransac.cpp
 
Description: A purely combinatorial implementation of RANSAC
 
Input : a set of 2d points with unknown uncertainties, and any required algorithm parameters.
        A file containing a set of test points will be provided.
Outputs : input points are inliers.

------------------------------------------------------------------------
*/

struct norm{
	cv::Mat_<double> Normalization_matrix;
	std::vector<cv::Point2f> points3d;
};

struct norm normalize_matri(std::vector<cv::Point2f> pointsVec) {
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

struct norm RANSAC_algo::normalize_matrix(std::vector<cv::Point2f> pointsVec) {
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

cv::Mat leastSquare_transpose(std::vector<bool> Inliers, std::vector< cv::Point2f > obj,
	std::vector< cv::Point2f > scene, int numOfIn) {

	/*
		let's create a opencv matrix to store X and Y values
	*/
	cv::Mat X = cv::Mat_<double>(numOfIn, 3, double(0));
	cv::Mat Y = cv::Mat_<double>(numOfIn, 3, double(0));
	cv::Mat Xt = cv::Mat_<double>(numOfIn, 3, double(0));
	cv::Mat Yt = cv::Mat_<double>(numOfIn, 3, double(0));

	int j = 0;

	for (int i = 0; i < obj.size(); i++) {

		if (Inliers[i]) {
			cv::Mat X_ = (cv::Mat_<double>(1, 3) << obj[i].x, obj[i].y, 1 );
			X.row(j) += X_;

			//std::cout << X_ << std::endl;

			cv::Mat Y_ = (cv::Mat_<double>(1, 3) << scene[i].x, scene[i].y, 1);
			Y.row(j) += Y_;

			j = j + 1;
		}
	}

	std::cout << "[" << X.rows << "," << X.cols << "]" << std::endl;
	//std::cout << X << std::endl << std::endl;
	//std::cout << Y << std::endl;

	cv::Mat Xt_X = cv::Mat_<double>(3, 3);
	cv::Mat Xt_X_inv = cv::Mat_<double>(3, 3);

	cv::transpose(X, Xt);
	cv::transpose(Y, Yt);

	Xt_X = 	Xt * X;
	//std::cout << Xt_X << std::endl << std::endl;
	cv::Mat Xt_Y = cv::Mat_<double>(3, 3);
	Xt_Y = Xt * Y;
	//std::cout << Xt_Y << std::endl << std::endl;
	cv::Mat  H = cv::Mat_<double>(3, 3);

	cv::invert(Xt_X, Xt_X_inv);
	H = Xt_X_inv * Xt_Y;
	std::cout << H.t() << std::endl << std::endl;
	return H;
}


cv::Mat leastSquare(std::vector<bool> Inliers, std::vector< cv::Point2f > obj,
	std::vector< cv::Point2f > scene, int numOfIn) {


	std::vector< cv::Point2f > obj_inliers;
	std::vector< cv::Point2f > scene_inliers;

	for (int i = 0; i < obj.size(); i++) {

		if (Inliers[i]) {
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
	cv::Mat X = cv::Mat_<double>(2*numOfIn, 9,double(0));
	//cv::Mat X =  cv::Mat::zeros(cv::Size(2 * numOfIn, 9), CV_64FC3);
	//cv::Mat X = Mat(1, 1, CV_64F, double(0));
	
	int j = 0;

	std::cout << norm_scene.points3d.size() << std::endl;

	for (int i = 0; i < scene_inliers.size(); i++) {
		
		//cv::Mat X_ = (cv::Mat_<double>(1, 9) << -obj[i].x, -obj[i].y, -1, 0, 0, 0, obj[i].x* scene[i].x, obj[i].y *scene[i].x, scene[i].x );
		//X.row(j) += X_;
			
		//std::cout << X_ << std::endl;

		//X_ = (cv::Mat_<double>(1, 9) << 0, 0, 0, -obj[i].x, -obj[i].y, -1, obj[i].x * scene[i].y, obj[i].y * scene[i].y, scene[i].y);
		//X.row(j+1) += X_;

		cv::Mat X_ = (cv::Mat_<double>(1, 9) << -norm_obj.points3d[i].x, -norm_obj.points3d[i].y, -1, 0, 0, 0, 
			norm_obj.points3d[i].x * norm_scene.points3d[i].x, norm_obj.points3d[i].y * norm_scene.points3d[i].x, norm_scene.points3d[i].x);
		X.row(j) += X_;

		X_ = (cv::Mat_<double>(1, 9) << 0, 0, 0, -norm_obj.points3d[i].x, -norm_obj.points3d[i].y, -1, 
			norm_obj.points3d[i].x * norm_scene.points3d[i].y, norm_obj.points3d[i].y * norm_scene.points3d[i].y, norm_scene.points3d[i].y);
		X.row(j + 1) += X_;


		j= j+2;
		
		
	}

	//std::cout << X << std::endl;
	cv::Mat U = cv::Mat_<double>(2 * numOfIn, 9, double(0));
	cv::Mat W = cv::Mat_<double>(9, 9, double(0));
	cv::Mat Vt = cv::Mat_<double>(9, 9, double(0));
	//std::cout << "[" << X.rows << "," << X.cols << "]" << std::endl;
	//std::cout << X << std::endl;


	cv::SVDecomp(X, W, U, Vt);
	//std::cout <<"W:"<< "[" << W.rows << "," << W.cols << "]" << std::endl;
	//std::cout <<"U:"<<"[" << U.rows << "," << U.cols << "]" << std::endl;
	//std::cout <<"Vt:"<<"[" << Vt.rows << "," << Vt.cols << "]" << std::endl;
	//cv::SVDecomp(X, W, U, Vt, cv::SVD::FULL_UV);
	//std::cout << W << std::endl << std::endl;
	//std::cout << Vt << std::endl << std::endl;
	cv::Mat V = cv::Mat_<double>(9, 9, double(0));
	V  = Vt.t();
	cv::Mat H = (cv::Mat_<double>(3, 3) << V.at<double>(0, 8), V.at<double>(1, 8), V.at<double>(2, 8), 
		V.at<double>(3, 8), V.at<double>(4,8), V.at<double>(5, 8), V.at<double>(6, 8), V.at<double>(7, 8), V.at<double>(8, 8));
	
	//H = H / H.at<double>(2, 2);
	//std::cout << H << std::endl;

	//std::cout << "norm_obj.Normalization_matrix:" << "[" << norm_obj.Normalization_matrix.rows << "," << norm_obj.Normalization_matrix.cols << "]" << std::endl;
	//std::cout << "H:" << "[" << H.rows << "," << H.cols << "]" << std::endl;
	//std::cout << "norm_scene.Normalization_matrix:" << "[" << norm_scene.Normalization_matrix.rows << "," << norm_scene.Normalization_matrix.cols << "]" << std::endl;

	//H = norm_obj.Normalization_matrix.inv() * H * norm_scene.Normalization_matrix;
	H = norm_scene.Normalization_matrix.inv() * H * norm_obj.Normalization_matrix;
	H = H / H.at<double>(2, 2);
	std::cout << H << std::endl;
	return H;
}


void gradientDescent(std::vector<bool> Inliers, std::vector< cv::Point2f > obj, 
	std::vector< cv::Point2f > scene, cv::Mat H, int numOfIn,
	cv::Point2f max_src,cv::Point2f max_dst) {

	float alpha = 0.000001;
	int max_iterations = 100;
	

	/*
	
		let's create a opencv matrix to store X and Y values
	*/
	
	cv::Mat X = cv::Mat_<double>(numOfIn, 3, double(0));
	cv::Mat Y = cv::Mat_<double>(numOfIn, 3, double(0));

	int j = 0;

	//cv::Mat hypothesis = cv::Mat_<double>(numOfIn, 3, double(0));
	
	for (int i = 0; i < obj.size(); i++) {

		if (Inliers[i]) {
			cv::Mat X_ = (cv::Mat_<double>(1, 3) << obj[i].x/max_src.x, obj[i].y/max_src.y, 1);
			X.row(j) += X_;

			//std::cout << X_ << std::endl;

			cv::Mat Y_ = (cv::Mat_<double>(1, 3) << scene[i].x/ max_dst.x, scene[i].y/ max_dst.y, 1);
			Y.row(j) += Y_;

			j = j + 1;
		}
	}
	
	//std::cout << "[" << X.rows << "," << X.cols << "]" << std::endl;
	//std::cout <<  X << std::endl;
	std::cout << Y << std::endl;
	
	
	
	cv::Mat Ht = cv::Mat_<double>(3, 3, double(0));
	cv::Mat Xt = cv::Mat_<double>(numOfIn, 3, double(0));
	cv::transpose(X, Xt);

	for (int i = 0; i < 100; i++) {
		
		cv::transpose(H, Ht);

		cv::Mat hypothesis = cv::Mat_<double>(numOfIn, 3, double(0));
		//cv::transpose(H, H);
		
		hypothesis += X * Ht;

		//std::cout << hypothesis << std::endl << std::endl;
		//std::cout << "okay till here" << std::endl;

		cv::Mat delta = cv::Mat_<double>(3, 3);
		
		delta = Xt * (hypothesis - Y);
		delta = delta / numOfIn;
		//std::cout << alpha << std::endl << std::endl;
		//std::cout << delta << std::endl << std::endl;
		
		H = H - alpha * delta;
	}

	std::cout << "homography from gradient descent" << std::endl;
	std::cout << H << std::endl;
	
}


void tuneParameters(std::vector<bool> Inliers, std::vector< cv::Point2f > obj, std::vector< cv::Point2f > scene, cv::Mat V) {

	cv::Mat H = (cv::Mat_<double>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

	float alpha = 0.0001;
	int iterations = 100;
	int num_inliers = Inliers.size();
	while(iterations > 0) {

			//decrease the variable only when a inlier is used
			iterations--;

			// Tuning Paramter H11
			double cost_h11 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h11 += (-scene[i].x + H.at<double>(0, 0) * obj[i].x + H.at<double>(0, 1) * obj[i].y + H.at<double>(0, 2)) * obj[i].x;
				}
			}
			H.at<double>(0, 0) = H.at<double>(0, 0) - alpha * cost_h11/ num_inliers;
			
			

			// Tuning Paramter H12
			double cost_h12 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h12 += (scene[i].x - H.at<double>(0, 0) * obj[i].x - H.at<double>(0, 1) * obj[i].y - H.at<double>(0, 2)) * obj[i].y;
				}
			}
			H.at<double>(0, 1) = H.at<double>(0, 1) - alpha * cost_h12 / num_inliers;

			
			// Tuning Paramter H13
			double cost_h13 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h13 += (scene[i].x - H.at<double>(0, 0) * obj[i].x - H.at<double>(0, 1) * obj[i].y - H.at<double>(0, 2));
				}
			}
			H.at<double>(0, 2) = H.at<double>(0, 2) - alpha * cost_h13 / num_inliers;



			// Tuning Paramter H21
			double cost_h21 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h21 += (scene[i].y - H.at<double>(1, 0) * obj[i].x - H.at<double>(1, 1) * obj[i].y - H.at<double>(1, 2)) * obj[i].x;
				}
			}
			H.at<double>(1, 0) = H.at<double>(1, 0) - alpha * cost_h21 / num_inliers;


			// Tuning Paramter H22
			double cost_h22 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h22 += (scene[i].y - H.at<double>(1, 0) * obj[i].x - H.at<double>(1, 1) * obj[i].y - H.at<double>(1, 2)) * obj[i].y;
				}
			}
			H.at<double>(1, 1) = H.at<double>(1, 1) - alpha * cost_h22 / num_inliers;




			// Tuning Paramter H23
			double cost_h23 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h23 += (scene[i].y - H.at<double>(1, 0) * obj[i].x - H.at<double>(1, 1) * obj[i].y - H.at<double>(1, 2));
				}
			}
			H.at<double>(1, 2) = H.at<double>(1, 2) - alpha * cost_h23 / num_inliers;


			float athal = 0.1;

			// Tuning Paramter H31
			double cost_h31 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h31 += (athal*H.at<double>(2, 0) * obj[i].x + athal*H.at<double>(2, 1) * obj[i].y ) * obj[i].x;
				}
			}
			H.at<double>(2, 0) = H.at<double>(2, 0) - alpha * cost_h31 / num_inliers;



			// Tuning Paramter H32
			double cost_h32 = 0;
			for (int i = 0; i < obj.size(); i++) {
				if (Inliers[i]) {
					cost_h32 += (athal * H.at<double>(2, 0) * obj[i].x + athal * H.at<double>(2, 1) * obj[i].y ) * obj[i].y;
				}
			}
			H.at<double>(2, 1) = H.at<double>(2, 1) - alpha * cost_h32 / num_inliers;
			
			//std::cout << "iteration : " << (100 - iterations) << " "<<  cost_h31 << " , " << cost_h32 << std::endl;

			//std::cout << "Homography matrix after parameter tuning : " << iterations << " times " << std::endl;
			//std::cout << H << std::endl;
		
	}

	std::cout << "Homography matrix after parameter tuning : " << iterations << " times "<< std::endl;
	std::cout << H << std::endl;


}
	

double RANSAC_algo::GetDistance(cv::Point2f  test_src, cv::Point2f test_dst, cv::Mat H){
	double distance = 0;
	
	cv::Mat A = (cv::Mat_<double>(3, 1) << test_src.x, test_src.y, 1);
	
	//cv::Mat A_ = (cv::Mat_<double>(3, 1) << test_dst.x, test_dst.y, 1);
	
	cv::Mat mul = (cv::Mat_<double>(3, 1) << 0,0,0);

	//------------------------------------------------------------------------------------------
	//			Error estimation method - 1 
	//-------------------------------------------------------------------------------------------
	//mul = H * A;

	//cv::Mat error = (cv::Mat_<double>(3, 1) << 0,0,0);
	
	//error = A_ - mul;
	// let's perform the transformation

	//double error_x =   error.at<double>(0, 0) / mul.at<double>(0, 0);
	//double error_y = error.at<double>(1, 0) / mul.at<double>(1, 0);
	//double error_const = error.at<double>(2, 0) / mul.at<double>(2, 0);

	//std::cout << error_x << " " << error_y << " "<< error_const << std::endl;

	//distance = (double) std::pow( pow(error.at<double>(0, 0), 2) + pow(error.at<double>(1, 0), 2) + pow(error.at<double>(2, 0), 2), 1/3);
	
	//distance = (double)std::pow( pow(error_x, 2) + pow(error_y, 2) + pow(error_const, 2), 1 / 2);

	//distance = std::abs(error_x) + std::abs(error_y) + std::abs(error_const);

	//std::cout << error.at<double>(0, 0) << " " << error.at<double>(1, 0) << " " << error.at<double>(2, 0) << " " << std::endl;
	//std::cout << A_ << std::endl;
	//std::cout << mul << std::endl;
	//std::cout << error << std::endl;

	//std::cout << std::endl ;
	//------------------------------------------------------------------------------------------
	//			Error estimation method - 2 
	//------------------------------------------------------------------------------------------
	
	//mul = H * A;
	//distance = sqrt((mul.at<double>(0, 0) - A.at<double>(0, 0)) * (mul.at<double>(0, 0) - A.at<double>(0, 0)) + 
	//	(mul.at<double>(1, 0) - A.at<double>(1, 0)) * (mul.at<double>(1, 0) - A.at<double>(1, 0)));
    



	//------------------------------------------------------------------------------------------
	//			Error estimation method - 3
	//------------------------------------------------------------------------------------------
	//
	// The following method was created considering homogeneous coordinate system and euclidean coordinate
	//  system. First the homogeneous coordinates were calculated using the data in H matrix
	//  then the coordinate system was changed to euclidean.
	// After that euclidean distance was calculated. 
	//
	
	double homo_x = H.at<double>(0, 0) * test_src.x + H.at<double>(0, 1) * test_src.y + H.at<double>(0, 2);
	double homo_y = H.at<double>(1, 0) * test_src.x + H.at<double>(1, 1) * test_src.y + H.at<double>(1, 2);
	double homo_z = H.at<double>(2, 0) * test_src.x + H.at<double>(2, 1) * test_src.y + H.at<double>(2, 2);

	double est_x = homo_x / homo_z;
	double est_y = homo_y / homo_z;

	distance = std::sqrt( (test_dst.x - est_x)*(test_dst.x - est_x) + (test_dst.y - est_y)*(test_dst.y - est_y));

	//std::cout << est_x << "," << test_dst.x << " | " << est_y << "," << test_dst.y << std::endl;



	//------------------------------------------------------------------------------------------
	//			Error estimation method - 4  Geometric distance
	//------------------------------------------------------------------------------------------
	//
	//
	//
	//cv::Mat A = (cv::Mat_<double>(3, 1) << test_src.x, test_src.y, 1);

	//cv::Mat A_ = (cv::Mat_<double>(3, 1) << test_dst.x, test_dst.y, 1);

	//cv::Mat mul = (cv::Mat_<double>(3, 1) << 0,0,0);


	//mul = H * A;
	//distance = (mul.at<double>(0, 0) - A_.at<double>(0, 0)) * (mul.at<double>(0, 0) - A_.at<double>(0, 0)) +
	//	(mul.at<double>(1, 0) - A_.at<double>(1, 0)) * (mul.at<double>(1, 0) - A_.at<double>(1, 0)) +
	//	(mul.at<double>(2, 0) - A_.at<double>(2, 0)) * (mul.at<double>(2, 0) - A_.at<double>(2, 0));
	
	return distance;
}


cv::Mat RANSAC_algo::computeHomography_RANSAC(std::vector< cv::Point2f > obj, std::vector< cv::Point2f > scene) {

	

	// required probability that a result will be generated from inliers-only
	double Confidence = 0.99995;

	// threshold on error metric 
	double Threshold = 1;

	// approximate expected inlier fraction
	double ApproximateInlierFraction = 0.5;

	// Output variables ... 
	std::vector<bool> Inliers;
	bool bSuccess(false);


	/*
	Got some of the RANSAC basics from this lecture slides for homography calculation using RANSAC

	 		http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf

	*/



	 //randomize the random number generator
	 //I am using the time because it is a number that changes almost every
	 //time the program is initialised
	srand(time(NULL));


	// initialize matrices need for homography calculation (exact)
	// and for inlier detection
	cv::Point2f src[4];
	cv::Point2f dst[4];

	cv::Point2f good_src[4];
	cv::Point2f good_dst[4];

	
	cv::Point2f test_src;
	cv::Point2f test_dst;

	cv::Point2f max_src;
	cv::Point2f max_dst;
	
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
	
	std::cout << "iteration is :" << k << std::endl;

	
	for (int i = 0; i <= k; i++) {
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
		//H = c.findHomographySVD(src, dst);

		//reset the number of inliner for the nex iteration
		numOfIn = 0;

		//clear the list on inliers for this iteration
		TInliers.clear();

		for (size_t Index = 0; Index != obj.size(); ++Index)
		{
			//test the choosen test point
			test_src = obj[Index];
			test_dst = scene[Index];

			if (GetDistance(test_src, test_dst, H) <= Threshold) {
				//if the point is within the desired distance to the line
				//add assing a true to the same index of the inliers vector
				TInliers.push_back(true);
				//and add to the count of number of inliers for this line model
				numOfIn++;

				max_src.x = 0;
				max_src.y = 0;
				max_dst.x = 0;
				max_dst.y = 0;

				if (max_src.x < test_src.x) {
					max_src.x = test_src.x;
				}

				if (max_src.y < test_src.y) {
					max_src.y = test_src.y;
				}

				if (max_dst.x < test_dst.x) {
					max_dst.x = test_dst.x;
				}

				if (max_dst.y < test_dst.y) {
					max_dst.y = test_dst.y;
				}
			}
			else TInliers.push_back(false);


			/*
				let's find the maximum of points
			*/

		}

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


			//TInliers[testP0] = true;
			//TInliers[testP1] = true;
			//TInliers[testP2] = true;
			//TInliers[testP3] = true;
			
			//print the current champion
			//std::cout << "X1: " << Point0[0] << " Y1: " << Point0[1] << "  X2: " << Point1[0] << " Y2: " << Point1[1] << '\n';
			//copy the inliners to the Output inliners
			Inliers = TInliers;
		}
	}
	//it is always a success because it is running for a certain
	//numner of iterations
	bSuccess = true;

	maxNumOfIn = maxNumOfIn ;
	//print the number of inliners detected
	std::cout << maxNumOfIn << '\n';
	// if successful, write the line parameterized as two points, 
	// and each input point along with its inlier status

	homography c;
	cv::Mat H = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);


	H = c.findHomography_(good_src, good_dst);
	
	cv::Mat V = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
	V = H + 0;

	//tuneParameters(Inliers, obj, scene, H);
	
	//leastSquare_transpose(Inliers, obj, scene, maxNumOfIn);
	H = leastSquare(Inliers, obj, scene, maxNumOfIn);
	//std::cout << "Found from ransac algorithm good value" << std::endl;
	//std::cout << V << std::endl;
	return H;
}



