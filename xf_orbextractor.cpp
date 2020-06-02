
#include "xf_orbextractor.h"



void FAST_accel(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_src,xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &_dst){

	unsigned char _threshold = 20;
	const int NMS = 1;
	xf::fast<NMS,XF_8UC1,HEIGHT,WIDTH,NPC1>(_src,_dst,_threshold);

}
