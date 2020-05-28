# Using xfOpenCV in HLS
The xfOpenCV library can be used to build applications in Vivado HLS. In this context, this document provides details on how the xfOpenCV library components can be integrated into a design in Vivado HLS. This section of the document provides steps on how to run a single library component through the Vivado HLS use flow which includes, C-simulation, C-synthesis, C/RTL cosimulation and exporting the RTL as an IP.
The user needs to make the following changes to facilitate proper functioning of the use model in Vivado HLS.

## 1. Define xf::Mat objects as static in the testbench
In order to use xfOpenCV functions in standalone HLS mode, all the xf::Mat objects in the testbench must be declared with static keyword.
Ex: ```static xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> imgInput(in_img.rows,in_img.cols);```

## 2. Use of appropriate compile-time options
When using the xfOpenCV functions in HLS, the following options need to be provided at the time of compilation : __XFCV_HLS_MODE__, -std=c++0x
Note: When using Vivado-HLS in Windows OS, 
provide the ```-std=c++0x``` flag only for C-Sim and Co-Sim. Do not include the flag when performing Synthesis.

## 3. Specifying interface pragmas to the interface level arguments
For the functions with top level interface arguments as pointers (with more than one read/write access) must have m_axi Interface pragma specified.
Ex: 
```
void lut_accel(xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &imgInput, xf::Mat<TYPE, HEIGHT, WIDTH, NPC1> &imgOutput, unsigned char *lut_ptr)
{
#pragma HLS INTERFACE m_axi depth=256 port=lut_ptr offset=direct bundle=lut_ptr
xf::LUT< TYPE, HEIGHT, WIDTH, NPC1>(imgInput,imgOutput,lut_ptr);
}
```
