#ifndef _ORBEXTRACTORHLS_H
#define _ORBEXTRACTORHLS_H

#include "opencv2/core/core.hpp""

namespace TORBHLS {

	template<int _nfeatures, int _nlevels, int _iniThFAST, int _minThFAST, int NUM_KEYPOINTS>
	void ORBExtractorHLS(cv::Mat _image, cv::KeyPoint _keypoints[NUM_KEYPOINTS], cv::Mat _descriptors, float _scaleFactor) {

		if (_image.empty())
			return;

		cv::Mat image = _image;
		assert(image.type() == CV_8UC1);

		std::cout << "good to go :D\n";

	}

	


}


#endif // _ORBEXTRACTORHLS_H
