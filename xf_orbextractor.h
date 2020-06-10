
#ifndef _XF_ORBEXTRACTOR_H_
#define _XF_ORBEXTRACTOR_H_

#include "hls_stream.h"
#include "ap_int.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "features/xf_fast.hpp"


#define WIDTH 	640
#define HEIGHT	480
#define NPC1 XF_NPPC1
#define TYPE XF_8UC1


// This function apply FAST algorithm and mark keypoints
void FAST_accel(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_src,xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_dst);


// This function generate 8 level of the image for the scale space
void computePyramid(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_src);


#endif //_XF_ORBEXTRACTOR_H_
