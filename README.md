# ImageStitcherFAST
 
Image stitching or photo stitching is the process of combining multiple photographic images with overlapping fields of view to produce a segmented panorama or high-resolution image. This project was intended to create image stitching algorithm in FPGA to get high throughput

# Design decisions
## 1. Use FAST feature detector instead of SIFT  [ SIFT | SURF | FAST ]
SIFT (patented) is slow compared to FAST algorithm (source : Homography Estimation by Elan Dubrofsky (Master’s essay submission))
FAST algorithm is already implemented in vivado xfopencv library 

## 2. Use ORB feature descriptor [ SIFT | SURF | BRISK | BRIEF | ORB ]
FAST algorithm lacks orientation component so the algorithm is not invariant for orientation. ORB add orientation component to the FAST features.
ORB is using BRISK keypoint descriptor which is relatively speed and contains less data relative to the SIFT descriptor 

## 3. Use Brute force feature Matcher [ BRUTE FORCE | FLANN based ]
FLANN based matcher used random sampling. Since we are using FPGA all the feature sizes are fixed size, hence take only fixed number of clock cycles which can be calculated in compilation time so planned to proceed with brute force matcher.


# Implementation
In this repository we have implemented following functions
1. ORB Extractor
2. ORB Matcher brute force
3. RANSAC algorithm
4. Homography estimation

In following section we present the API and functionality of the above functions

### ORB Extractor

ORB extractor generate keypoints and descriptors. ORB stands for Oriented FAST Rotated Brief. In this algorithm first we find keypoints using FAST in different scales (to make scale invariant). Then we assign orientation to each keypoint. After that descriptors are calculate(Based on BRIEF)
API:

```cpp
void ORBExtractor::operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints, 
                                   cv::OutputArray _descriptors)
```

Ex Usage : 
```cpp
TORB::ORBExtractor orbextractor(1000, 1.2, 8, 31, 20);

	orbextractor(gray_image1, cv::Mat(), keypoints_object, descriptors_object);
	orbextractor(gray_image2, cv::Mat(), keypoints_scene, descriptors_scene);

```
 ### ORB Matcher
 
 ORB matcher matches to descriptors and return DMatches using brute force Hamming.
 API:

```cpp
std::vector<int> ORBMatcher::MatchDescriptors(cv::Mat descriptors1, cv::Mat descriptors2, std::vector<cv::DMatch> *matches)
```
 
Ex Usage : 

```cpp
ORBMatcher orb;
orb.MatchDescriptors(descriptors_object, descriptors_scene, &matches);	
```
 ### RANSAC Algorithm
 
1. Extract features
2. Compute a set of potential matches
3. do
 - select minimal sample (i.e. 7 matches)
 - compute solution(s) for F
 - determine inliers until a large enough set of the matches become inliers
4. Compute F based on all inliers
5. Look for additional matches
6. Refine F based on all correct matches


API:

```cpp
 /*
 @Compute the homogrphy using RANSAC Algorithm
 @Param obj std::vector to the array of points of object
 @Param scene std::vector to the array of points of scene
 Compute Homography matrix and Inlier mask by minimizing the
 distance under a perticular H
 distance = d(Hx, x') + d(invH x', x)
 return: structure containing Homography matrix and inlier mask
 */
 
 returnRANSAC RANSAC_algo::computeHomography_RANSAC(std::vector< cv::Point2d > obj, std::vector< cv::Point2d > scene)

```


Ex Usage : 
```cpp
RANSAC_algo ra;
struct returnRANSAC r = ra.computeHomography_RANSAC(obj, scene);
```

 ### Homography 
 

finding the homography using 4 given points<br>
[ X ] &ensp; [ h11 , h12, h13] [ x ]<br>
[ Y ]  = [ h21 , h22, h23]  [ y ]<br>
[ 1 ] &ensp;   [ h31 , h32,   1] [ 1 ]

Computer hij values which satisfy above equation

src  cv::Point2d  4 source points 
dst  cv::Point2d  4 destination points


NOTE:  Currently gaussian elimination is used. New mechanism
      will be required for optimized algorithm
      
 
 API :
```cpp
cv::Mat homography::findHomography_(cv::Point2d src[4], cv::Point2d dst[4]) 
```
