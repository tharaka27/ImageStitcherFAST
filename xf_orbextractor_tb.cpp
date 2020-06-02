#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include "ap_int.h"
#include "hls_stream.h"

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"


#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "common/xf_sw_utils.h"

#include "xf_orbextractor.h"

int main(int argc, char** argv){
	//std::cout << "Tharaka" ;

	if(argc != 2)
	{
		fprintf(stderr,"Invalid Number of Arguments!\nUsage:\n");
		fprintf(stderr,"<Executable Name> <input image path> \n");
		return -1;
	}

	cv::Mat in_img,in_img1,out_img,ocv_ref;
	cv::Mat in_gray,in_gray1,diff;


	in_gray = cv::imread("C:\\Users\\ASUS\\Desktop\\EagleEye\\FastFPGA\\fastFPGA\\left_r.jpg", 0);

	if (in_gray.data == NULL)
	{
		fprintf(stderr,"Cannot open image at %s\n", argv[1]);
		return 0;
	}

	ocv_ref.create(in_gray.rows,in_gray.cols,CV_8UC1);
	out_img.create(in_gray.rows,in_gray.cols,CV_8UC1);
	diff.create(in_gray.rows,in_gray.cols,CV_8UC1);

	std::vector<cv::KeyPoint> kp;
	cv::FAST(in_gray, kp, 20, true);

	for(int i=0; i < 5; i++){
		std::cout << kp[i].pt.x << "," << kp[i].pt.y << "\n";
	}

	std::cout << "\nNumber of Keypoints found using OpenCV FAST: "<< kp.size();

	/////////////////////	End of OpenCV reference	 ////////////////

	static xf::Mat<XF_8UC1, 480, 640, XF_NPPC1> imgInput(in_gray.rows,in_gray.cols);
	static xf::Mat<XF_8UC1, 480, 640, XF_NPPC1> imgOutput(in_gray.rows,in_gray.cols);

	imgInput.copyTo(in_gray.data);

	FAST_accel(imgInput, imgOutput);

	xf::imwrite("hls_out.jpg",imgOutput);


	//generated output checking
	cv::Mat out_gen = cv::imread("C:\\Users\\ASUS\\Desktop\\EagleEye\\FastFPGA\\fastFPGA\\solution1\\csim\\build\\hls_out.jpg", 0);

	//std::cout << out_gen;
	//for(int i = 0;i<5; i++){
	//	for(int j = 0;j<5; j++){
	//		std::cout << out_gen.at<uchar>(i, j) << "\n";
	//	}
	//}
	std::vector<cv::Point2f> d;
	for(int i = 0; i<480; i++){
			for(int j = 0; j< 640; j++){
				if(out_gen.at<uchar>(i, j) >= 250){
					cv::Point2f temp;
					temp.x = i;
					temp.y = j;
					d.push_back(temp);
				}
			}
		}
	std::cout << "\nNumber of Keypoints found using FPGA FAST: "<< d.size();

	return 0;
}
