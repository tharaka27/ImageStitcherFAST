#include "homography.h"


#include "opencv2/core/core.hpp"

#include <math.h>
#include <iostream>


void homography::gaussian_elimination(double* input, int n) {
	// ported to c from pseudocode in
	// http://en.wikipedia.org/wiki/Gaussian_elimination

	double* A = input;
	int i = 0;
	int j = 0;
	int m = n - 1;
	while (i < m && j < n) {
		// Find pivot in column j, starting in row i:
		int maxi = i;
		for (int k = i + 1; k < m; k++) {
			if (fabs(A[k * n + j]) > fabs(A[maxi * n + j])) {
				maxi = k;
			}
		}
		if (A[maxi * n + j] != 0) {
			//swap rows i and maxi, but do not change the value of i
			if (i != maxi)
				for (int k = 0; k < n; k++) {
					double aux = A[i * n + k];
					A[i * n + k] = A[maxi * n + k];
					A[maxi * n + k] = aux;
				}
			//Now A[i,j] will contain the old value of A[maxi,j].
			//divide each entry in row i by A[i,j]
			double A_ij = A[i * n + j];
			for (int k = 0; k < n; k++) {
				A[i * n + k] /= A_ij;
			}
			//Now A[i,j] will have the value 1.
			for (int u = i + 1; u < m; u++) {
				//subtract A[u,j] * row i from row u
				double A_uj = A[u * n + j];
				for (int k = 0; k < n; k++) {
					A[u * n + k] -= A_uj * A[i * n + k];
				}
				//Now A[u,j] will be 0, since A[u,j] - A[i,j] * A[u,j] = A[u,j] - 1 * A[u,j] = 0.
			}

			i++;
		}
		j++;
	}

	//back substitution
	for (int i = m - 2; i >= 0; i--) {
		for (int j = i + 1; j < n - 1; j++) {
			A[i * n + m] -= A[i * n + j] * A[j * n + m];
			//A[i*n+j]=0;
		}
	}
}

cv::Mat homography::findHomography_(cv::Point2f src[4], cv::Point2f dst[4]) {
	

	// create the equation system to be solved
	//
	// from: Multiple View Geometry in Computer Vision 2ed
	//       Hartley R. and Zisserman A.
	//
	// x' = xH
	// where H is the homography: a 3 by 3 matrix
	// that transformed to inhomogeneous coordinates for each point
	// gives the following equations for each point:
	//
	// x' * (h31*x + h32*y + h33) = h11*x + h12*y + h13
	// y' * (h31*x + h32*y + h33) = h21*x + h22*y + h23
	//
	// as the homography is scale independent we can let h33 be 1 (indeed any of the terms)
	// so for 4 points we have 8 equations for 8 terms to solve: h11 - h32
	// after ordering the terms it gives the following matrix
	// that can be solved with gaussian elimination:

	double P[8][9] = {
			{-src[0].x, -src[0].y, -1,   0,   0,  0, src[0].x * dst[0].x, src[0].y * dst[0].x, -dst[0].x }, // h11
			{  0,   0,  0, -src[0].x, -src[0].y, -1, src[0].x * dst[0].y, src[0].y * dst[0].y, -dst[0].y }, // h12

			{-src[1].x, -src[1].y, -1,   0,   0,  0, src[1].x * dst[1].x, src[1].y * dst[1].x, -dst[1].x }, // h13
			{  0,   0,  0, -src[1].x, -src[1].y, -1, src[1].x * dst[1].y, src[1].y * dst[1].y, -dst[1].y }, // h21

			{-src[2].x, -src[2].y, -1,   0,   0,  0, src[2].x * dst[2].x, src[2].y * dst[2].x, -dst[2].x }, // h22
			{  0,   0,  0, -src[2].x, -src[2].y, -1, src[2].x * dst[2].y, src[2].y * dst[2].y, -dst[2].y }, // h23

			{-src[3].x, -src[3].y, -1,   0,   0,  0, src[3].x * dst[3].x, src[3].y * dst[3].x, -dst[3].x }, // h31
			{  0,   0,  0, -src[3].x, -src[3].y, -1, src[3].x * dst[3].y, src[3].y * dst[3].y, -dst[3].y }, // h32
	};

	homography::gaussian_elimination(&P[0][0], 9);
	
	/*
	cv::Mat w, u, vt;
	cv::Mat input_mat = (cv::Mat_<float>(8, 9) << 
		-src[0].x, -src[0].y, -1, 0, 0, 0, src[0].x * dst[0].x, src[0].y * dst[0].x, -dst[0].x,
		0, 0, 0, -src[0].x, -src[0].y, -1, src[0].x * dst[0].y, src[0].y * dst[0].y, -dst[0].y,
		-src[1].x, -src[1].y, -1, 0, 0, 0, src[1].x * dst[1].x, src[1].y * dst[1].x, -dst[1].x,
		0, 0, 0, -src[1].x, -src[1].y, -1, src[1].x * dst[1].y, src[1].y * dst[1].y, -dst[1].y,
		-src[2].x, -src[2].y, -1, 0, 0, 0, src[2].x * dst[2].x, src[2].y * dst[2].x, -dst[2].x,
		0, 0, 0, -src[2].x, -src[2].y, -1, src[2].x * dst[2].y, src[2].y * dst[2].y, -dst[2].y,
		-src[3].x, -src[3].y, -1, 0, 0, 0, src[3].x * dst[3].x, src[3].y * dst[3].x, -dst[3].x,
		0, 0, 0, -src[3].x, -src[3].y, -1, src[3].x * dst[3].y, src[3].y * dst[3].y, -dst[3].y
		);
	cv::SVD::compute(input_mat, w, u, vt);
	cv::transpose(vt, vt);
	*/


	/*
	std::cout << "size of vt" << std::endl;
	std::cout << vt.size() << std::endl;

	std::cout << "transpose of vt" << std::endl;
	std::cout << vt << std::endl;
	*/

	/*

	float matrix[4][4];

	matrix[0][0] = P[0][8]; //h11
	matrix[0][1] = P[1][8]; //h12
	matrix[0][2] = 0;
	matrix[0][3] = P[2][8]; //h13

	matrix[1][0] = P[3][8]; //h21
	matrix[1][1] = P[4][8]; //h22
	matrix[1][2] = 0;
	matrix[1][3] = P[5][8]; //h23

	matrix[2][0] = 0;
	matrix[2][1] = 0;
	matrix[2][2] = 0;
	matrix[2][ 3] = 0;

	matrix[3][0] = P[6][8]; //h31
	matrix[3][1] = P[7][8]; //h32
	matrix[3][2] = 0;
	matrix[3][ 3] = 1;		//h33

	*/
	
	//float h[] = { P[0][8] , P[1][8] ,P[2][8], P[3][8] ,P[4][8] ,P[5][8] ,P[6][8],P[7][8], 1 };
	
	/*
	std::cout << P[0][8] << " " << P[1][8] << " "<< P[2][8] << std::endl;
	std::cout << P[3][8] << " " << P[4][8] << " " << P[5][8] << std::endl;
	std::cout << P[6][8] << " " << P[7][8] << " " << 1 << std::endl;
	*/
	
	cv::Mat A = (cv::Mat_<double>(3, 3) << P[0][8], P[1][8], P[2][8], P[3][8], P[4][8], P[5][8], P[6][8], P[7][8], 1);
	/* 
	h[0][0] = P[0][8]; //h11
	h[0][1] = P[1][8]; //h12
	h[0][1] = P[2][8]; //h13

	h[1][0] = P[3][8]; //h21
	h[1][1] = P[4][8]; //h22
	h[1][1] = P[5][8]; //h23

	h[2][0] = P[6][8]; //h31
	h[2][1] = P[7][8]; //h32
	h[2][1] = 1;	    //h33
	*/

	/*
	std::cout << "homograph Matrix from gussian elimination" << std::endl;
	std::cout << A << std::endl;
	*/

	return A;
}

void homography::findHomography(std::vector< cv::Point2f > obj, std::vector< cv::Point2f > scene) {
	cv::Mat c = (cv::Mat_<float>(3, 3) << 0,0,0,0,0,0,0,0,0 );

	std::cout << obj.size() << std::endl ;
	for (int i = 0; i < 10; i++) {
		
		cv::Point2f src[4];
		cv::Point2f dst[4];

		src[0] = obj.at(i    );
		src[1] = obj.at(i + 1);
		src[2] = obj.at(i + 2);
		src[3] = obj.at(i + 3);

		dst[0] = scene.at(i);
		dst[1] = scene.at(i+1);
		dst[2] = scene.at(i+2);
		dst[3] = scene.at(i+3);

		cv::Mat A = findHomography_(src, dst);
		c = c + A;

		//if (i <  10) {
		//	std::cout << A << std::endl;
		//}

	}
	cv::Mat Result = cv::Mat_<float>(3, 3);
	Result = c * 0.1;

	std::cout << "Final" << std::endl;
	std::cout << "Calculated homography using Function:::" << std::endl;
	std::cout << Result << std::endl;
}

cv::Mat homography::findHomographySVD(cv::Point2f src[4], cv::Point2f dst[4]) {

	/*
		let's create a opencv matrix to store X 
	*/
	cv::Mat X = cv::Mat_<double>(8, 9, double(0));

	int j = 0;

	for (int i = 0; i < 4; i++) {

		cv::Mat X_ = (cv::Mat_<double>(1, 9) << -src[i].x, -src[i].y, -1, 0, 0, 0, src[i].x * dst[i].x, src[i].y * dst[i].x, dst[i].x);
		X.row(2*i) += X_;

		X_ = (cv::Mat_<double>(1, 9) << 0, 0, 0, -src[i].x, -src[i].y, -1, src[i].x * dst[i].y, src[i].y * dst[i].y, dst[i].y);
		X.row(2*i+1) += X_;

		j = j + 2;

	}
	std::cout << X << std::endl;
	cv::Mat U = cv::Mat_<double>(8, 8, double(0));
	cv::Mat W = cv::Mat_<double>(8, 1, double(0));
	cv::Mat Vt = cv::Mat_<double>(8, 9, double(0));

	cv::SVDecomp(X, W, U, Vt);
	//cv::SVDecomp(X, W, U, Vt, cv::SVD::FULL_UV);
	std::cout << "W:" << "[" << W.rows << "," << W.cols << "]" << std::endl;
	std::cout << "U:" << "[" << U.rows << "," << U.cols << "]" << std::endl;
	std::cout << "Vt:" << "[" << Vt.rows << "," << Vt.cols << "]" << std::endl;

	cv::Mat V = cv::Mat_<double>(9, 8, double(0));
	

	V = Vt.t();
	std::cout << W << std::endl;
	std::cout << V << std::endl;
	cv::Mat H = (cv::Mat_<double>(3, 3) << V.at<double>(0, 7), V.at<double>(1, 7), 
		V.at<double>(2, 7),V.at<double>(3, 7), V.at<double>(4, 7), V.at<double>(5, 7), 
		V.at<double>(6, 7), V.at<double>(7, 7), V.at<double>(8, 7));

	//H = H / H.at<double>(2, 2);
	

	return H;
}