/*

11.26 update:
1/ fix the unaccurate pupil center problem

11.23 update:
1/ append ellipse fitting method
2/ state mechine algorithm updated
3/ output image updated

11月9日更新：
1、瞳孔Mask加入了阈值范围判定，加上快启动，慢恢复规则。。。
2、眨眼通过光斑判定
3、不再输出直方图

11月2日更新：
1、修改了参数
2、修改了count的更新方式，去除了多个瞳孔区域造成瞳孔不准，无瞳孔区域造成光斑和瞳孔中心重合的现象

10月31日第三次更新：
1、修改了一些参数

10月31日第二次更新：
1/修改了过滤面积时count的更新方式

10月31日更新：
1/修改了一些参数（还有一些参数有待实验）
2/修改了输出图像
3/增加状态机法的状态数
4/写完注释

10月30日更新：
1/改了输出图像上的位数限制

10月29日第三次更新：
1/修改了眨眼判断成功时，没有更新pupilMask

10月29日第二次更新：
1/修改了眨眼显示的BUG


10月29日更新：
1/增加输出中间结果
2/利用椭圆判断，精确中心点计算refinedPupilCenter,修正了光斑在瞳孔外，瞳孔边缘时的错：误

10月28日第二次更新：
1/修改kalman filtering 参数
2/修改pupilTHresholding 眨眼

10月28日更新：
1/卡尔曼滤波对光斑中心和瞳孔中心过滤
2/问题：开头几帧眨眼问题未解决

10月27日第二次更新：
1/直方图分割增加阈值判定，状态机更新
2/对pupilMask连通域个数进行判定
3/refinePupilCenter精准瞳孔中心点
cost:3ms+-1ms

10月27日更新：
1/直方图分割瞳孔，增加：平滑直方图，求极小值自动阈值
2/光斑中心增加距离判断
3/分割后瞳孔稀释，求质心


10月15更新：
1/增加otsu算法
2/优化求瞳孔面积算法，改为半径法
3/修改了irisSeg接口，增加了面积输出
4/修改了光斑中心求法，增加了面积过：滤

10月5日更新：
1/求光斑中心点的mask和消除光斑inpaint的mask 分开计算，减少假光斑产生。
初步的做法，无法消除带眼镜时多光斑产生的光斑中心的误差，有待更：新

9月29日更新：
1/ cen_X的错误。原来的输出的名称错误为centerX, 现改正为float cen_X
2/ 灰度值判断处更：改
3/ 初步的眨眼检测, 正确的眨眼有效，但是没有排除掉错误检测的情况
4/ 修改了连通域检测，但还是有一点问题，在继续修改
*/

#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <iomanip>
#include <opencv2/core/core.hpp> //OpenCV 2.4.13 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>


using namespace std;
using namespace cv;

#define M_PI 3.14159265358979323846
#undef max
#undef min

#define CEN_THRE 20 

/** no globbing in win32 mode **/
int _CRT_glob = 0;


/*
* 卡尔曼滤波
*/
void kalmanfiltering(float &xk, float &pk, float xk_1, float z, float pk_1, float Q = .01, float R = .000001) {
	float x_ = xk_1;
	float p_ = pk_1 + Q;
	float k = p_ / (p_ + R);
	xk = x_ + k * (z - x_);
	pk = (1 - k) * p_;
}



/** ------------------------------- OpenCV helpers ------------------------------- **/

/**
* Visualizes a CV_32FC1 image (using peak normalization)
*
* filename: name of the file to be stored
* norm: CV_32FC1 image
*
* returning imwrite result code
*/


bool imwrite2f(const string filename, const Mat& src) {
	Mat norm(src.rows, src.cols, CV_8UC1);
	MatConstIterator_<float> it;
	MatIterator_<uchar> it2;
	float maxAbs = 0;
	for (it = src.begin<float>(); it < src.end<float>(); it++) {
		if (std::abs(*it) > maxAbs) maxAbs = std::abs(*it);
	}
	for (it = src.begin<float>(), it2 = norm.begin<uchar>(); it < src.end<float>(); it++, it2++) {
		*it2 = saturate_cast<uchar>(cvRound(127 + (*it / maxAbs) * 127));
	}
	return imwrite(filename, norm);
}

/**
* Show a Mat in a Window
* src: Mat
* n: Window name
*/
void show(Mat src, const char *n = "temp") {
#define SAVE_IMG
#ifdef SAVE_IMG
	static int ns = 0;
	ns++;
	stringstream ss;
	string s;
	ss << ns++;
	ss >> s;
	s = ".\\" + s+".jpg";
	imwrite(s.c_str(), src);
#endif
	
		cvNamedWindow(n, CV_WINDOW_AUTOSIZE);
		IplImage *img = &IplImage(src);
		cvShowImage(n, img);
		cvWaitKey(0);
		cvDestroyWindow(n);
	
}
void show2(Mat src, const char *n = "temp") {
#define SAVE_IMG
#ifdef SAVE_IMG
	static int ns = 0;
	ns++;
	stringstream ss;
	string s;
	ss << ns++;
	ss >> s;
	s = ".\\" + s + ".jpg";
	imwrite(s.c_str(), src);
#endif
	cvNamedWindow(n, CV_WINDOW_AUTOSIZE);
	IplImage *img = &IplImage(src);
	cvShowImage(n, img);
	cvWaitKey(0);
	cvDestroyWindow(n);
}
/**
* Calculate a standard uniform upper exclusive lower inclusive 256-bin histogram for range [0,256]
*
* src: CV_8UC1 image
* histogram: CV_32SC1 1 x 256 histogram matrix
*/
void hist2u(const Mat& src, Mat& histogram) {
	histogram.setTo(0);
	MatConstIterator_<uchar> s = src.begin<uchar>();
	MatConstIterator_<uchar> e = src.end<uchar>();
	int * p = (int *)histogram.data;
	for (; s != e; s++) {
		p[*s]++;
	}
}


void drawHist(Mat &histogram, int mark = -1, int mark2 = -1) {
	int i = 0;
	int n = 0;
	MatIterator_<int> p = histogram.begin<int>();
	MatIterator_<int> e = histogram.end<int>();
	Mat hist2show(256, 256, CV_8UC1);
	for (; p != e; p++, n++) {
		auto s = hist2show.ptr<uchar>(n);
		for (i = 0; i < 256; i++) {
			s[i] = (n == mark)? 255:
				(n==mark2)?180:
				   (i < *p)? 255: 0;
		}
	}
	show(hist2show, "HIST");
}
/**
* Calculate a standard uniform upper exclusive lower inclusive 256-bin histogram for range [0,256]
*
* src: CV_32FC1 image
* histogram: CV_32SC1 1 x 256 histogram matrix
* min: minimal considered value (inclusive)
* max: maximum considered value (exclusive)
*/
//hist2f(mag, hist, 0, histmax);
void hist2f(const Mat& src, Mat& histogram, const float min = 0, const float max = 256) {
	histogram.setTo(0);
	MatConstIterator_<float> s = src.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	int bins = histogram.rows * histogram.cols;
	float binsize = (max - min) / bins;
	int * p = (int *)histogram.data;
	for (; s != e; s++) {
		if (*s >= min) {
			if (*s < max) {
				int idx = cvFloor((*s - min) / binsize);
				p[(idx < 0) ? 0 : ((idx > bins - 1) ? bins - 1 : idx)]++;
			}
		}
	}
}

/**
* Computate upper exclusive lower inclusive uniform histogram quantile, i.e. quantile * 100% are
* less than returned value, and c(1-quantile) * 100 % are greater or equal than returned value.
*
* histogram: CV_32SC1 1 x 256 histogram matrix
* count: histogram member count
* quantile:  quantile between 0 and 1
*
* returning quantile bin between 0 and 256
* e.g. float minval = histquant2u(hist, width*height, (100 - roiPercent)*0.01) * histmax / histbins;
*/
int histquant2u(const Mat& histogram, const int count, const float quantile) {
	int * s = (int *)histogram.data;
	int left = max(0, min(count, cvRound(quantile * count)));
	int sum = 0;
	int p = 0;
	for (; sum < left; p++) {
		sum += s[p];
	}
	if (p > 0 && (sum - left > left - sum + s[p - 1])) p--;
	return p;
}

/** ------------------------------- Clahe (Contrast limited adaptive histogram equalization)------------------------------- **/

/*
* Retrieves the (bilinear) interpolated byte from 4 bytes
*
* x: distance to left byte
* y: distance to right byte
* r: distance to upper byte
* s: distance to lower byte
* b1: upper left byte
* b2: upper right byte
* b3: lower left byte
* b4: lower right byte
*/
uchar interp(const double x, const double y, const double r, const double s, const uchar b1, const uchar b2, const uchar b3, const uchar b4) {
	double w1 = (x + y);
	double w2 = x / w1;
	w1 = y / w1;
	double w3 = (r + s);
	double w4 = r / w3;
	w3 = s / w3;
	return saturate_cast<uchar>(w3 * (w1 * b1 + w2 * b2) + w4 * (w1 * b3 + w2 * b4));
}


/** ------------------------------- Mask generation ------------------------------- **/

/*
* Masks a region of interest within a floating point matrix
*
* src: CV_32FC1 matrix
* dst: CV_32FC1 matrix
* mask: CV_8UC1 region of interest matrix
* onread: for any p: dst[p] := set if mask[p] = onread, otherwise dst[p] = src[p]
* set: set value for onread in mask, see onread
*/
void maskValue(const Mat& src, Mat& dst, const Mat& mask, const uchar onread = 0, const uchar set = 0) {

	MatConstIterator_<float> s = src.begin<float>();
	MatIterator_<float> d = dst.begin<float>();
	MatConstIterator_<float> e = src.end<float>();
	MatConstIterator_<uchar> r = mask.begin<uchar>();
	for (; s != e; s++, d++, r++) {
		*d = (*r == onread) ? set : *s;
	}
}

/**
* Generates destination regions map from source image
*
* src: CV_8UC1 image
* dst: CV_32SC1 regions map image (same size as src)
* count: outputs number of regions
*/
void regionsmap(const Mat& src, Mat& dst, int& count) {
	int width = src.cols;
	int height = src.rows;
	int labelsCount = 0;
	int maxRegions = ((width / 2) + 1) * ((height / 2) + 1) + 1;
	Mat regsmap(1, maxRegions, CV_32SC1);
	int * map = (int *)regsmap.data;
	for (int i = 0; i< maxRegions; i++) map[i] = i; // identity mapping
	uchar * psrc = src.data;
	int * pdst = (int *)(dst.data);
	int srcoffset = src.step - src.cols;
	int srcline = src.step;
	int dstoffset = dst.step / sizeof(int) - dst.cols;
	int dstline = dst.step / sizeof(int);
	dst.setTo(0);
	// 1) processing first row
	if (*psrc != 0) *pdst = ++labelsCount;
	if (width > 1) {
		psrc++; pdst++;
	}
	for (int x = 1; x < width; x++, psrc++, pdst++) {
		if (*psrc != 0) {// if pixel is a region pixel, check left neightbor
			if (psrc[-1] != 0) { // label like left neighbor
				*pdst = pdst[-1];
			} else { // label new region
				*pdst = ++labelsCount;
			}
		}
	}
	if (height > 1) {
		psrc += srcoffset;
		pdst += dstoffset;
	}
	// 2) for all other rows
	for (int y = 1; y < height; y++, psrc += srcoffset, pdst += dstoffset) {
		// first pixel in row only checks upper and upper-right pixels
		if (*psrc != 0) {
			if (psrc[-srcline] != 0) { // check upper pixel
				*pdst = pdst[-dstline];
			} else if (1 < width && psrc[1 - srcline] != 0) { // check upper right
				*pdst = pdst[1 - dstline];
			} else {
				*pdst = ++labelsCount;
			}
		}
		if (width > 1) {
			psrc++;
			pdst++;
		}
		// all other pixels in the row check for left and three upper pixels
		for (int x = 1; x < width - 1; x++, psrc++, pdst++) {
			if (*psrc != 0) {
				if (psrc[-1] != 0) {// check left neighbor
					*pdst = pdst[-1];
				} else if (psrc[-1 - srcline] != 0) {// label like left upper
					*pdst = pdst[-1 - dstline];
				} else if (psrc[-srcline] != 0) {// check upper
					*pdst = pdst[-dstline];
				}
				if (psrc[1 - srcline] != 0) {
					if (*pdst == 0) { // label pixel as the above right
						*pdst = pdst[1 - dstline];
					} else {
						int label1 = *pdst;
						int label2 = pdst[1 - dstline];
						if ((label1 != label2) && (map[label1] != map[label2])) {
							if (map[label1] == label1) { // map unmapped to already mapped
								map[label1] = map[label2];
							} else if (map[label2] == label2) { // map unmapped to already mapped
								map[label2] = map[label1];
							} else { // both values are already mapped
								map[map[label1]] = map[label2];
								map[label1] = map[label2];
							}
							// reindexing
							for (int i = 1; i <= labelsCount; i++) {
								if (map[i] != i) {
									int j = map[i];
									while (j != map[j]) {
										j = map[j];
									}
									map[i] = j;
								}
							}
						}
					}
				}
				if (*pdst == 0) {
					*pdst = ++labelsCount;
				}
			}
		}
		if (*psrc != 0) {
			if (psrc[-1] != 0) {// check left neighbor
				*pdst = pdst[-1];
			} else if (psrc[-1 - srcline] != 0) {// label like left upper
				*pdst = pdst[-1 - dstline];
			} else if (psrc[-srcline] != 0) {// check upper
				*pdst = pdst[-dstline];
			} else {
				*pdst = ++labelsCount;
			}
		}
		psrc++;
		pdst++;
	}
	Mat regsremap(1, maxRegions, CV_32SC1);
	int * remap = (int *)regsremap.data;
	count = 0;
	for (int i = 1; i <= labelsCount; i++) {
		if (map[i] == i) {
			remap[i] = ++count;
		}
	}
	remap[0] = 0;
	// complete remapping
	for (int i = 1; i <= labelsCount; i++) {
		if (map[i] != i) remap[i] = remap[map[i]];
	}
	pdst = (int *)(dst.data);
	for (int y = 0; y < height; y++, pdst += dstoffset) {
		for (int x = 0; x < width; x++, pdst++) {
			*pdst = remap[*pdst];
		}
	}
}

/**
* Filters out too large or too small binary large objects (regions) in a region map
*
* regmap:  CV_32SC1 regions map (use regionsmap() to calculate this object)
* mask:    CV_8UC1 output mask with filtered regions (same size as regmap)
* count:   number of connected components in regmap
* minSize: only regions larger or equal than minSize are kept
* maxSize: only regions smaller or equal than maxSize are kept
*/
void maskRegsize(const Mat& regmap, Mat& mask,  const int count, int &aftercount, const int minSize = INT_MIN, const int maxSize = INT_MAX) {

	//show(mask);
	Mat regs(1, count + 1, CV_32SC1);
	int * map = (int *)regs.data;
	// resetting map to now count region size
	for (int i = 0; i<count + 1; i++) map[i] = 0;
	int width = regmap.cols;
	int height = regmap.rows;
	int * pmap = (int *)(regmap.data);
	int mapoffset = regmap.step / sizeof(int) - regmap.cols;
	for (int y = 0; y < height; y++, pmap += mapoffset) {
		for (int x = 0; x < width; x++, pmap++) {
			if (*pmap > 0) {
				map[*pmap]++;
			}
		}
	}
	// delete too large and too small regions
	pmap = (int *)(regmap.data);
//	..vector<bool> isDeleted(count + 1, false);

	vector<int> isDeleted(count + 1, 0);
	
	uchar * pmask = mask.data;
	int maskoffset = mask.step - mask.cols;
	for (int y = 0; y < height; y++, pmap += mapoffset, pmask += maskoffset) {
		for (int x = 0; x < width; x++, pmap++, pmask++) {
			int tmp = *pmap;
			if (*pmap > 0) {
				int size = map[*pmap];
				if (size < minSize || size > maxSize) { 
					*pmask = 0; 
					*pmap = 0;
					//if (isDeleted[*pmap] == false)
						//isDeleted[*pmap] = true;
					isDeleted[tmp] |= 1;
				} else {
					*pmask = 255;
				}
			} else *pmask = 0;
		}

	}
	int tmp = 0;
	for (int i = 1; i < isDeleted.size(); i++) {
		if (isDeleted[i] == 1) {
			tmp++;
		}
	}
	aftercount = count - tmp;
}
/**
* Computes mask for reflections in image
*
* src: CV_8UC1 input image
* mask: CV_8UC1 output image (same size as src)
* roiPercent: parameter for the number of highest pixel intensities in percent
* maxSize: maximum size of reflection region between 0 and 1
* dilateSize: size of circular structuring element for dilate operation
* dilateIterations: iterations of dilate operation
*/
void createReflectionMask(const Mat& src, Mat& mask, Mat &regions, int &count, const float roiPercent = 20, const float maxSizePercent = 3,
	const int dilateSize = 4, const int dilateIterations = 1) {
	//show(src, "SRC");
	CV_Assert(src.type() == CV_8UC1);
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(mask.size() == src.size());
//	Mat src2(src.rows,src.cols,CV_8UC1);
//	blur(src,src2,Size(3,3));
	adaptiveThreshold(src, mask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, -60);
	//show(mask2, "MASK2");
	//show(mask, "MASK1");
	regionsmap(mask, regions, count);
	int tmp; 
	maskRegsize(regions, mask, count, tmp, 4, 100);
	Mat kernel(dilateSize, dilateSize, CV_8UC1);
	kernel.setTo(0);
	circle(kernel, Point(dilateSize / 2, dilateSize / 2), dilateSize / 2, Scalar(255), CV_FILLED);
	dilate(mask, mask, kernel);
	//show(mask, "MASK4");
}




/**
* Main eye mask selecting pupillary and limbic boundary pixels
*
* src: CV_8UC1 image
* mask: CV_8UC1 mask (same size as src)
* gradX: CV_32FC1 gradient image in x-direction
* gradY: CV_32FC1 gradient image in y-direction
* mag: CV_32FC1 gradient magnitude
*/
void createBoundaryMask(const Mat& src, Mat& mask, const Mat& gradX, const Mat& gradY, const Mat& mag) {
	const float roiPercent = 20; // was 20
	const int histbins = 1000;
	int width = mask.cols;
	int height = mask.rows;
	int cellWidth = width / 30;
	int cellHeight = height / 30;
	int gridWidth = width / cellWidth + (width % cellWidth == 0 ? 0 : 1);
	int gridHeight = height / cellHeight + (height % cellHeight == 0 ? 0 : 1);
	MatConstIterator_<float> pmag = mag.begin<float>();
	MatConstIterator_<float> emag = mag.end<float>();
	float max = 0;
	for (; pmag != emag; pmag++) {
		if (*pmag > max) max = *pmag;
	}
	Mat hist(1, histbins, CV_32SC1);
	float histmax = max + max / histbins;
	//cout << max << " " << histbins;
	hist2f(mag, hist, 0, histmax);
	float minval = histquant2u(hist, width*height, (100 - roiPercent)*0.01) * histmax / histbins;
	//cout << " minval: " << minval << endl;
	MatIterator_<uchar> smask = mask.begin<uchar>();
	pmag = mag.begin<float>();
	for (; pmag != emag; pmag++, smask++) {
		*smask = (*pmag >= minval ) ? 255 : 0;
	}

	//show(mask, "INIT_BOUNDARYMASK");
	int stepx = (gradX.step / sizeof(float));
	int stepy = (gradY.step / sizeof(float));
	int stepmag = (mag.step / sizeof(float));
	for (int y = 0; y < gridHeight; y++) {
		for (int x = 0; x < gridWidth; x++) {
			int cX = x*cellWidth;
			int cY = y*cellHeight;
			int cWidth = min(cellWidth, width - x*cellWidth);
			int cHeight = min(cellHeight, height - y*cellHeight);
			float * pgradX = ((float *)(gradX.data)) + stepx*cY + cX;
			float * pgradY = ((float *)(gradY.data)) + stepy*cY + cX;
			float * pmag = ((float *)(mag.data)) + stepmag*cY + cX;
			int gradXCellOffset = stepx - cWidth;
			int gradYCellOffset = stepy - cWidth;
			int magoffset = stepmag - cWidth;
			double sumX = 0;
			double sumY = 0;
			double sumMag = 0;
			for (int b = 0; b < cHeight; b++, pmag += magoffset, pgradX += gradXCellOffset, pgradY += gradYCellOffset) {
				for (int a = 0; a < cWidth; a++, pmag++, pgradX++, pgradY++) {
					if (*pmag >= minval) {
						sumX += *pgradX;
						sumY += *pgradY;
						sumMag += *pmag;
					}
				}
			}
			if (sumMag > 0) {
				sumX /= sumMag;
				sumY /= sumMag;
			}
			bool is_significant = ((sumX * sumX + sumY * sumY) > 0.5);
			uchar * pmask = mask.data + (mask.step)*cY + cX;
			int maskoffset = mask.step - cWidth;
			for (int b = 0; b < cHeight; b++, pmask += maskoffset) {
				for (int a = 0; a < cWidth; a++, pmask++) {
					if (!is_significant && *pmask > 0) *pmask = 0;
				}
			}
		}
	}

	//show(mask, "AFTER_BOUNDARYMASK");
}

/** ------------------------------- Center detection ------------------------------- **/

/**
* Type for a bi-directional ray with originating point and direction
*
* x: x-coordinate of origin
* y: y-coordinate of origin
* fx: x-direction
* fy: y-direction
* mag: ray weight (magnitude)
*/
struct BidRay {
	float x;
	float y;
	float fx;
	float fy;
	float mag;
	BidRay(float _x, float _y, float _fx, float _fy, float _mag) {
		x = _x;
		y = _y;
		fx = _fx;
		fy = _fy;
		mag = _mag;
	}
};

/**
* Calculates determinant of vectors (x1, y1) and (x2, y2)
*
* x1: first vector's x-coordinate
* y1: first vector's x-coordinate
* x2: second vector's y-coordinate
* y2: second vector's y-coordinate
*
* returning: determinant
*/
inline float det(const float &x1, const float &y1, const float &x2, const float &y2) {
	return x1*y2 - y1*x2;
}

/**
* Intersects two lines (x1, y1) + s*(fx1, fy1) and (x2, y2) + t*(fx2, fy2)
*
* x1: x-coordinate of point on line 1
* y1: y-coordinate of point on line 1
* fx1: direction-vector x-coordinate of line 1
* fy1: direction-vector y-coordinate of line 1
* x2: x-coordinate of point on line 2
* y2: y-coordinate of point on line 2
* fx2: direction-vector x-coordinate of line 2
* fy2: direction-vector y-coordinate of line 2
* sx: intersection point x-coordinate
* sy: intersection point y-coordinate
*
* returning: 1 if they intersect, 0 if they are parallel, -1 is they are equal
*/
int intersect(const float &x1, const float &y1, const float &fx1, const float &fy1, const float &x2, const float &y2, const float &fx2, const float &fy2, float &sx, float &sy) {
	if (det(fx1, fy1, fx2, fy2) == 0) {
		if (det(fx1, fy1, x2 - x1, y2 - y1) == 0) {
			sx = x1;
			sy = y1;
			return -1; // equal
		}
		sx = NAN;
		sy = NAN;
		return 0; // parallel
	}
	float Ds = det(x2 - x1, y2 - y1, -fx2, -fy2);
	float D = det(fx1, fy1, -fx2, -fy2);
	float s = Ds / D;
	sx = x1 + s*fx1;
	sy = y1 + s*fy1;
	return 1;
}

/**
* Intersects a line (x1, y1) + s*(fx1, fy1) with an axis parallel to the x-axis
*
* x1: x-coordinate of point on line 1
* y1: y-coordinate of point on line 1
* fx1: direction-vector x-coordinate of line 1
* fy1: direction-vector y-coordinate of line 1
* y2: y-coordinate of point on axis parallel to x-axis
* sx: intersection point x-coordinate (sy is always equal y2)
*
* returning:  1 if the line intersects, 0 if it is parallel to the x-axis, -1 if it is the x-axis
*/
int intersectX(const float &x1, const float &y1, const float &fx1, const float &fy1, const float &y2, float &sx) {
	if (fy1 == 0) {
		if (y2 - y1 == 0) {
			sx = x1;
			return -1; // equal
		}
		sx = NAN;
		return 0; // parallel
	}
	sx = x1 + ((y2 - y1)*fx1 / fy1);
	return 1;
}

/**
* Intersects a line (x1, y1) + s*(fx1, fy1) with an axis parallel to the y-axis
*
* x1: x-coordinate of point on line 1
* y1: y-coordinate of point on line 1
* fx1: direction-vector x-coordinate of line 1
* fy1: direction-vector y-coordinate of line 1
* x2: x-coordinate of point on axis parallel to y-axis
* sy: intersection point x-coordinate (sx is always equal x2)
*
* returning:  1 if the line intersects, 0 if it is parallel to the y-axis, -1 if it is the y-axis
*/
int intersectY(const float &x1, const float &y1, const float &fx1, const float &fy1, const float &x2, float &sy) {
	if (fx1 == 0) {
		if (x2 - x1 == 0) {
			sy = y1;
			return -1; // equal
		}
		sy = NAN;
		return 0; // parallel
	}
	sy = y1 + ((x2 - x1)*fy1 / fx1);
	return 1;
}

/**
* intersects a line (x, y) + s*(fx, fy) with an axis parallel rectangle
*
* x: x-coordinate of point on line
* y: y-coordinate of point on line
* fx: direction-vector x-coordinate of line
* fy: direction-vector y-coordinate of line
* left: left coordinate of rectangle
* top: top coordinate of rectangle
* right: right coordinate of rectangle
* bottom: bottom coordinate of rectangle
* px: first intersection point x-coordinate
* py: first intersection point y-coordinate
* qx: first intersection point x-coordinate
* qy: first intersection point y-coordinate
*
* returning:  1 if the line intersects in 2 points, 0 if it does not intersect, -1 if it corresponds to a side of the rectangle
*/
int intersectRect(const float &x, const float &y, const float &fx, const float &fy, const float &left, const float &top, const float &right, const float &bottom, float &px, float &py, float &qx, float &qy) {
	float leftY, bottomX, rightY, topX;
	int lefti = intersectY(x, y, fx, fy, left, leftY);
	bool leftHit = (lefti != 0) && leftY >= top && leftY <= bottom;
	int topi = intersectX(x, y, fx, fy, top, topX);
	bool topHit = (topi != 0) && topX >= left && topX <= right;
	int righti = intersectY(x, y, fx, fy, right, rightY);
	bool rightHit = (righti != 0) && rightY >= top && rightY <= bottom;
	int bottomi = intersectX(x, y, fx, fy, bottom, bottomX);
	bool bottomHit = (bottomi != 0) && bottomX >= left && bottomX <= right;
	if (leftHit) {
		if (bottomHit) {
			if (rightHit) {
				if (topHit) {
					// left, bottom, right, top
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return -1;
				} else {
					// left, bottom, right
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return -1;
				}
			} else {
				if (topHit) {
					// left, bottom, top
					px = topX;
					py = top;
					qx = bottomX;
					qy = bottom;
					return -1;
				} else {
					// left, bottom
					px = left;
					py = leftY;
					qx = bottomX;
					qy = bottom;
					return 1;
				}
			}
		} else {
			if (rightHit) {
				if (topHit) {
					// left, right, top
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return -1;
				} else {
					// left, right
					px = left;
					py = leftY;
					qx = right;
					qy = rightY;
					return 1;
				}
			} else {
				if (topHit) {
					// left, top
					px = left;
					py = leftY;
					qx = topX;
					qy = top;
					return 1;
				}
			}
		}
	} else {
		if (bottomHit) {
			if (rightHit) {
				if (topHit) {
					// bottom, right, top
					px = topX;
					py = top;
					qx = bottomX;
					qy = bottom;
					return -1;
				} else {
					// bottom, right
					px = bottomX;
					py = bottom;
					qx = right;
					qy = rightY;
					return 1;
				}
			} else {
				if (topHit) {
					// bottom, top
					px = topX;
					py = top;
					qx = bottomX;
					qy = bottom;
					return 1;
				}
			}
		} else {
			if (rightHit) {
				if (topHit) {
					// right, top
					px = topX;
					py = top;
					qx = right;
					qy = rightY;
					return 1;
				}
			}
		}
	}
	px = NAN;
	py = NAN;
	qx = NAN;
	qy = NAN;
	return 0;
}

/**
* Draws a line onto accumulator matrix using Bresenham's algorithm:
* Increases a rectangular accumulator by adding a given value to all points on a line
*
* line: line to be drawn
* accu: floating point canvas (accumulator)
* border: outer accu boundary rectangle in user space coordinates
*
* returning: true, if values are added to the accu
*/
bool drawLine(const BidRay& line, Mat_<float>& accu, const float borderX, 
	const float borderY, const float borderWidth, const float borderHeight) {
	// intersect line with border
	float cellWidth = borderWidth / accu.cols;
	float cellHeight = borderHeight / accu.rows;
	float lx = borderX, ly = borderY;
	float rx = borderX + borderWidth, ry = borderY + borderHeight;
	float px, py, qx, qy;
	float incValue = line.mag / 1000;
	int accuLine = (accu.step / sizeof(float));

	int res = intersectRect(line.x, line.y, line.fx, line.fy, lx, ly, rx, ry, px, py, qx, qy);
	if (res != 0) {
		int x1 = min(max(cvRound((px - lx) / cellWidth), 0), accu.cols - 1);
		int y1 = min(max(cvRound((py - ly) / cellHeight), 0), accu.rows - 1);
		int x2 = min(max(cvRound((qx - lx) / cellWidth), 0), accu.cols - 1);
		int y2 = min(max(cvRound((qy - ly) / cellHeight), 0), accu.rows - 1);
		// line intersects with border, so draw line onto accu
		float * p = (float *)(accu.data);
		int t, dx, dy, incx, incy, pdx, pdy, ddx, ddy, es, el, err;
		dx = x2 - x1;
		dy = y2 - y1;
		incx = (dx > 0) ? 1 : (dx < 0) ? -1 : 0;
		incy = (dy > 0) ? accuLine : (dy < 0) ? -accuLine : 0;
		if (dx<0) dx = -dx;
		if (dy<0) dy = -dy;
		if (dx>dy) {
			pdx = incx; // parallel step
			pdy = 0;
			ddx = incx; // diagonal step
			ddy = incy;
			es = dy; // error step
			el = dx;
		} else {
			pdx = 0; // parallel step
			pdy = incy;
			ddx = incx; // diagonal step
			ddy = incy;
			es = dx; // error step
			el = dy;
		}
		p += x1 + y1*accuLine;
		err = el / 2;
		// setPixel
		*p += incValue;
		// Calculate pixel
		for (t = 0; t<el; ++t) {// t counts Pixels, el is also count
			// update error
			err -= es;
			if (err<0) {
				// make error term positive
				err += el;
				// step towards slower direction
				p += ddx + ddy;
			} else {
				// step towards faster direction
				p += pdx + pdy;
			}
			*p += incValue;

		}
		return true;

	}
	return false;
}

/**
* Returns a gaussian 2D kernel
* kernel: output CV_32FC1 image of specific size
* sigma: gaussian sigma parameter
*/
void gaussianKernel(cv::Mat& kernel, float sigma = 1.4) {
	CV_Assert(kernel.type() == CV_32FC1);
	CV_Assert(kernel.cols % 2 == 1 && kernel.rows % 2 == 1);
	float * p = (float *)kernel.data;
	int width = kernel.cols;
	int height = kernel.rows;
	int offset = kernel.step / sizeof(float) - width;
	int rx = width / 2;
	int ry = height / 2;
	float sqrsigma = sigma*sigma;
	for (int y = -ry, i = 0; i<height; y++, i++, p += offset) {
		for (int x = -rx, j = 0; j<width; x++, j++, p++) {
			*p = std::exp((x*x + y*y) / (-2 * sqrsigma)) / (2 * M_PI*sqrsigma);
		}
	}
}

/*
* Calculates circle center in source image.
*
* gradX: CV_32FC1 image, gradient in x direction
* gradY: CV_32FC1 image, gradient in y direction
* mask: CV_8UC1 mask image to exclude wrong points for gradient extraction (same size as gradX, gradY)
* center: center point of main circle in source image
* accuPrecision: stop condition for accuracy of center
* accuSize: size of the accumulator array
*/


//detectEyeCenter(gradX, gradY, mag, boundaryEdges, centerX, centerY, .5f, 101);
void detectEyeCenter(const Mat& gradX, const Mat& gradY, const Mat& mag, const Mat& mask, 
		float& centerx, float& centery, uchar ave_grey, const int threshold = 20, const float accuPrecision = 5, const int accuSize = 10) {
	// initial declarations

	int width = mask.cols;
	int height = mask.rows;
	int accuScaledSize = (accuSize + 1) / 2;
	float rectX = -0.5, rectY = -0.5, rectWidth = width, rectHeight = height;
	Mat gauss(accuScaledSize, accuScaledSize, CV_32FC1);
	gaussianKernel(gauss, accuScaledSize / 3);
	Mat_<float> accu(accuSize, accuSize);//Accumulator
	Mat_<float> accuScaled(accuScaledSize, accuScaledSize);//Scaled Accumulator

	// create candidates list
	list<BidRay> candidates;
	float * px = (float *)(gradX.data);
	float * py = (float *)(gradY.data);
	float * pmag = (float *)(mag.data);
	uchar * pmask = (uchar *)(mask.data);
	int xoffset = gradX.step / sizeof(float) - width;//Mat::step  Ã¿Ò»Î¬ÔªËØµÄ´óÐ¡£¬µ¥Î»×Ö½Ú
	//cout << xoffset<< " " << gradX.step<<" "<<width<<" "<<sizeof(float);
	int yoffset = gradY.step / sizeof(float) - width;
	int magoffset = mag.step / sizeof(float) - width;
	int maskoffset = mask.step - width;

	for (int y = 0; y < height; y++, px += xoffset, py += yoffset, pmask += maskoffset, pmag += magoffset) {
		for (int x = 0; x < width; x++, px++, py++, pmask++, pmag++) {
			if (*pmask > 0) {
				candidates.push_back(BidRay(x, y, *px, *py, *pmag));
			}
		}
	}
	while (rectWidth > accuPrecision || rectHeight > accuPrecision) {
		accu.setTo(0);
		bool isIn = true;
		if (candidates.size() > 0) {
			for (list<BidRay>::iterator it = candidates.begin(); it != candidates.end(); (isIn) ? ++it : it = candidates.erase(it)) {
				isIn = drawLine(*it, accu, rectX, rectY, rectWidth, rectHeight);//»­Ïß
			}
		}
		pyrDown(accu, accuScaled);
		multiply(accuScaled, gauss, accuScaled, 1, CV_32FC1);

		float * p = (float *)(accuScaled.data);
		float maxCellValue = 0;
		int maxCellX = accuScaled.cols / 2;
		int maxCellY = accuScaled.rows / 2;
		int accuOffset = accuScaled.step / sizeof(float) - accuScaled.cols;
		for (int y = 0; y < accuScaledSize; y++, p += accuOffset) {
			for (int x = 0; x < accuScaledSize; x++, p++) {
				if (*p > maxCellValue && ((*p - ave_grey < threshold) || (ave_grey - *p < threshold))){
					maxCellX = x;
					maxCellY = y;
					maxCellValue = *p;//×î´óÖµµã,ÔÚÕâÀï¿ÉÒÔ×öµã¸Ä±ä£¬ÐèÒªÂú×ã»Ò¶ÈÖµ²Å¸üÐÂ,ÕâÀïÐèÒª¿¼ÂÇÇå³þ
				}
			}
		}
		rectX += (((maxCellX + 0.5) * rectWidth) / accuScaledSize) - (rectWidth * 0.25); //std::min( accuRect.width * 0.5,((maxCellX * accuRect.width + 0.5) / accuScaledSize) - (accuRect.width * 0.25));
		rectY += (((maxCellY + 0.5) * rectHeight) / accuScaledSize) - (rectHeight * 0.25); //std::min( accuRect.height * 0.5,((maxCellY * accuRect.height + 0.5) / accuScaledSize) - (accuRect.height * 0.25));
		rectWidth /= 2;
		rectHeight /= 2;

	}
	centerx = rectX + rectWidth / 2;
	centery = rectY + rectHeight / 2;
}


/*
* 寻找255-0矩阵的质心，返回所有255值处坐标的平均值
* Mat mask: 矩阵（CV_8UC1）
* Vec2f &c: 坐标
*/
void findCentroid(Mat mask, Vec2f& c) {
	CV_Assert(mask.type() == CV_8UC1);
	int sum = 0;
	int offset = mask.step - mask.cols;
	int xcount = 0, ycount = 0;
	int x, y;
	uchar *p = (uchar *)mask.data;
	for (y = 0; y < mask.rows; y ++, p+=offset) {
		for (x = 0; x < mask.cols; x++,p++) {
			if (*p == 255) {
				xcount += x;
				ycount += y;
				sum++;
			}
		}
	}
	if (sum != 0) {
		c[0] = static_cast<float>(xcount)/ static_cast<float>(sum);
		c[1] = static_cast<float>(ycount)/ static_cast<float>(sum);
	} else {
		c[0] = -1;
		c[1] = -1;
	}
}


/*
* 平滑直方图，消除锯齿
* Mat src: 原始直方图（CV_32SC1）
* Mat &dst: 输出直方图（CV_32SC1)
* int step: 平滑参数，步长
*/
void smoothHist(Mat src, Mat &dst, int step) {
	
	CV_Assert(src.type() == CV_32SC1);
	int start = step - step / 2;
	int end = 256 - step / 2;
	double temp = 0;
	int *dp = (int *)dst.data;
	int *sp = (int *)src.data;
	for (int i = start; i < end; i++) {
		temp = 0;
		for (int j = 0 - step / 2; j < step / 2; j++) {
			temp += sp[i + j];
		}
		temp /= step;
		dp[i] = (int)temp;
	}
}


/*
* 状态机法判断第一个直方图波谷，找到阈值
* Mat pupilhist: 直方图(CV_32SC1)
* int start: 开始检测的灰度，默认为3
*/
int calThreshold1(Mat pupilhist, int start = 3) {
	CV_Assert(pupilhist.type() == CV_32SC1);
	int *pp = (int *)pupilhist.data + start;
	const int S0 = 0, S1 = 1, S2 = 2, S3 = 3;
	int state = S0;
	int downval = 0, upval = 0, T1 = 5, T2 = 3;// T1 : threshold for ascend, T2: threshold for decline
	for (int i = start; i < 256; i++, pp++) {
		switch (state) {
		case S0:
			upval += pp[1] - *pp;
			state = (upval < T1) ? S0 : S1;
			//cout <<"S0:"<< *pp << endl;
			break;
		case S1:
		//	cout <<"S1 :"<< *pp << endl;

			state = (*pp <= pp[1]) ? S1 : S2;
			if (state == S2)
				T2 = *pp / 3;
			break;
		case S2:

			downval += *pp - pp[1];
			//cout << "S2: " << *pp <<"DV: "<<downval<< endl;

			state = (downval < T2) ? S2 : S3;
			break;
		case S3:
			//cout <<"S3:"<< *pp << endl;

			if (pp[1] > *pp)
				return i;
		default:
			break;
		}
	}
	return -1;
}


/*
* 状态机法判断第一个直方图波谷，找到阈值
* Mat pupilhist: 直方图(CV_32SC1)
* int start: 开始检测的灰度，默认为3
*/
int calThreshold2(Mat pupilhist, int start = 3) {
	CV_Assert(pupilhist.type() == CV_32SC1);
	int *pp = (int *)pupilhist.data;
	int valley = 0;
	const int s0 = 0, s1 = 1, s2 = 2, s3 = 3, s4 = 4, s5 = 5, s6 = 6;
	int state = s0;
	pp += start;
	for (int i = start; i < 256; i++, pp++) {
		switch (state) {
		case s0:
			state = (*pp < pp[1]) ? s1 : s0;
			break;
		case s1:
			state = (*pp < pp[1]) ? s3 :
				(*pp == pp[1]) ? s1 : s0;
			break;
			/*case s2:
			state = (*pp < pp[1]) ? s3 :
			(*pp == pp[1]) ? s2 : s1;
			break;
			*/
		case s3:
			state = (*pp > pp[1]) ? s4 : s3;
			break;

		case s4:
			state = (*pp > pp[1]) ? s5 :
				(*pp == pp[1]) ? s4 : s3;
			break;
			/*
			case s5:
			state = (*pp > pp[1]) ? s6 :
			(*pp == pp[1]) ? s5 : s4;
			break;
			*/
		case s5:
			if (*pp < pp[1])
				return i;
		default:
			break;
		}
	}
	return -1;
}

void dshit() {
	cout << "shit";
}

/*
* 通过阈值分割，找到瞳孔中心点
* Mat img, 原始眼睛图像
* Mat &pupilMask: 瞳孔mask，255-0矩阵，瞳孔处为255，其余为0
* int &ave_val: 平均阈值
* int &val: 当前阈值
* float &cen_X, &cen_Y: 瞳孔中心
* bool &isBlink: 输出值，判断是否眨眼
* int manthreshold: 是否人工设定阈值，小于0时为自动阈值，大于0为人工设定的阈值
*/
void thresholdingPupil(Mat img, Mat &pupilMask, int &val, int manthreshold) {

	CV_Assert(pupilMask.type() == CV_8UC1);
	CV_Assert(img.type() == CV_8UC1);
	int height = img.rows; int width = img.cols;
	Mat pupilhisttmp(1, 256, CV_32SC1);
	Mat pupilhist(1, 256, CV_32SC1);
	Mat pupilregion(height, width, CV_32SC1);
//	cout << val << "hahahahah" << endl;
	if (manthreshold >= 0) {
		val = manthreshold;
	} else {
		hist2u(img, pupilhisttmp);
		smoothHist(pupilhisttmp, pupilhist, 4);
		val = calThreshold1(pupilhist);
	//	cout << val << endl;
		if (val == -1 || val > 80){
			val = calThreshold2(pupilhist);
		}
		val += 5;
		if (val == -1) {
			cerr << "Threshold error!" << endl;
			val = 30;
		}
		
	}
	//drawHist(pupilhist, val);
	
	MatIterator_<uchar> p = pupilMask.begin<uchar>();
	MatIterator_<uchar> e = pupilMask.end<uchar>();
	MatIterator_<uchar> pp = img.begin<uchar>();
	MatIterator_<uchar> ee = img.end<uchar>();
	for (; p != e && pp!=ee; p++, pp++) {
		*p = (*pp < val) ? 255 : 0;
	}
	int indexcount = 0;
	//show(pupilMask);
	regionsmap(pupilMask, pupilregion, indexcount);
	//cout << count << endl;
	int count;
	maskRegsize(pupilregion, pupilMask, indexcount, count, 50, 800);
	//show(pupilMask,"after");
	//cout << "after: " <<count << endl;
	int accendvalue = 5;
	//通过accendvalue，模仿tcp，ip中的快启动原理启动阈值
	//cout << val << "shit" << endl;
	//cout << count << endl;
	//show(pupilMask);
	while (count == 0 && val <= 80) {
		//show(pupilMask);

		//cout << val << endl;
		//drawHist(pupilhist, val, ave_val);
		val += accendvalue;
		if (accendvalue / 2 > 1) accendvalue /= 2;
		else accendvalue = 1;
		indexcount = 0;
		p = pupilMask.begin<uchar>();
		pp = img.begin<uchar>();
		for (; p != e && pp != ee; p++, pp++) {
			*p = (*pp < val) ? 255 : 0;
		}
		regionsmap(pupilMask, pupilregion, indexcount);
		maskRegsize(pupilregion, pupilMask, indexcount, count, 50, 800);
	}
	//慢恢复
	while (((count > 1 || count == 0) && val >5) ){
		val--;
		indexcount = 0;
		p = pupilMask.begin<uchar>();
		pp = img.begin<uchar>();
		for (; p != e && pp != ee; p++, pp++) {
			*p = (*pp < val) ? 255 : 0;
		}
		regionsmap(pupilMask, pupilregion, indexcount);
		maskRegsize(pupilregion, pupilMask, indexcount, count, 50, 800);
		//cout << val << endl;
	}
	//cout << val << "haha";
	int dilateSize = 4;
	Mat kernel(dilateSize, dilateSize, CV_8UC1);
	kernel.setTo(0);
	circle(kernel, Point(dilateSize / 2, dilateSize / 2), dilateSize / 2, Scalar(255), CV_FILLED);
	dilate(pupilMask, pupilMask, kernel);
}
/*
* 过滤光斑，只保留离瞳孔中心最近的亮斑
* Mat &mask: 光斑矩阵
* Mat regions: 区域矩阵
* int count: 区域数
* float cen_X, cen_Y: 瞳孔中心
*/
void filterReflection(Mat &mask, Mat regions, int count, float cen_X, float cen_Y) {
	CV_Assert(mask.type() == CV_8UC1);
#define DIST(x1,y1,x2,y2) sqrt(((x1)-(x2))*((x1)-(x2))+((y1)-(y2))*((y1)-(y2)))
	int i = 0, x = 0, y = 0;
	int height = mask.rows;
	int width = mask.cols;
	Mat distance(1, count + 1, CV_32SC1);
	distance.setTo(height + width);
	int *dp = (int *)distance.data;
	int *rp = (int *)regions.data;
	int rpOffset = regions.step / sizeof(int) - regions.cols;
	int dpOffset = distance.step / sizeof(int) - distance.cols;
	//show(mask);
	for (y = 0; y < height; y++, rp += rpOffset) {
		for (x = 0; x < width; x++, rp++) {
			int tmpDist = DIST(x, y, cen_X, cen_Y);
			if (*rp > 0 && tmpDist < dp[*rp]) {
				dp[*rp] = tmpDist;
			}
		}
	}
	dp = (int *)distance.data;
	int minDistance = width + height;
	int minReg = 0;
	for (i = 1; i < count + 1; i++) {
		if (dp[i] < minDistance) {
			minDistance = dp[i];
			minReg = i;
		}
	}

	uchar *mp = (uchar *)mask.data;
	rp = (int *)regions.data;
	int mpOffset = mask.step - mask.cols;
	for (y = 0; y < height; y++, mp+=mpOffset, rp += rpOffset) {
		for (x = 0; x < width; x++, mp++, rp++) {
			*mp = (*rp > 0 && *rp == minReg) ? 255 : 0;
		}
	}
	//show(mask);
}



/*
* 精确瞳孔mask，画出在原瞳孔mask矩阵中的内接椭圆，在椭圆内的光斑值加入矩阵
* Mat reflectionMask: 光斑矩阵
* Mat pmask: 瞳孔矩阵
*/
void refinePupilMask(Mat reflectionMask, Mat &pmask) {
	CV_Assert(reflectionMask.type() == CV_8UC1);
	CV_Assert(pmask.type() == CV_8UC1);
	int rwidth = reflectionMask.cols;
	int rheight = reflectionMask.rows;
	uchar *rp = reflectionMask.data;

	int width = pmask.cols;
	int height = pmask.rows;
	uchar *pm = (uchar *)pmask.data;
	int offset = pmask.step - pmask.cols;

	int upX = 0, upY = 0, downX = width + height, downY = width + height;
	int x, y;
	for (y = 0; y < height; y++, pm += offset) {
		for (x = 0; x < width; x++, pm++) {
			if (*pm == 255) {
				if (x>upX)upX = x;
				if (x < downX)downX = x;
				if (y > upY)upY = y;
				if (y < downY)downY = y;
			}
		}
	}
	float blockwidth = upX - downX;
	float blockheight = upY - downY;

	float a = blockwidth / 2;
	float b = blockheight / 2;
	pm = (uchar *)pmask.data;
	rp = (uchar *)reflectionMask.data;
	for (int y = 0; y < height; y++, rp += offset, pm += offset) {
		for (int x = 0; x < width; x++, rp++, pm++) {
			if (*rp == 255) {
				float X =x- (upX+downX)/2;
				float Y =y- (upY+downY)/2;
				*pm = ((X*X) / (a*a) + (Y*Y) / (b*b) <= 1) ? 255 : 0;
			}
		}
	}

}
/*
* 计算瞳孔的面积
* Mat pmask: 瞳孔矩阵
* int &area: 返回的面积
*/
void calPupilArea(Mat pmask, int &area) {
	CV_Assert(pmask.type() == CV_8UC1);
	area = 0;
	int width = pmask.cols;
	int height = pmask.rows;
	uchar *pm = (uchar *)pmask.data;
	int offset = pmask.step - pmask.cols;
	for (int y = 0; y < height; y++, pm += offset) {
		for (int x = 0; x < width; x++, pm++) {
			if (*pm == 255)area++;
		}
	}
}



/*
* 将直方图与分割阈值画在结果图上
* Mat &hist: 直方图
* Mat &dst: 结果图
* int val: 分割阈值
*/
void drawHistOnDst(Mat &hist, Mat &dst, int val) {
	CV_Assert(hist.type() == CV_32SC1);
	CV_Assert(dst.type() == CV_8UC1);
	int x, y;
	uchar *dp = (uchar *)dst.data;
	MatIterator_ <int> p = hist.begin<int>();
	int dwidth = dst.cols;
	int dheight = dst.rows;
	int doffset = dst.step - dst.cols;
	int Yoffset = 200, Xoffset = dst.cols - 320;
	int HistHeight = 200;
	for (y = Yoffset; y < HistHeight+Yoffset; y++) {
		for (x = Xoffset; x < 256 + Xoffset; x++) {
			dp[x + (dwidth + doffset)*y] =
				(y == Yoffset || x == Xoffset || y == HistHeight - 1 + Yoffset || x == 255 + Xoffset) ? 255 :
				(x - Xoffset == val) ? 150 :
				(HistHeight + Yoffset - y <= p[x - Xoffset]) ? 255 : 0;
		}
	}

}

/*
* 将瞳孔矩阵画在结果图上
* Mat &mask: 瞳孔矩阵
* Mat &dst: 结果图
*/
void drawMaskOnDst(Mat &mask, Mat &dst) {
	CV_Assert(mask.type() == CV_8UC1);
	CV_Assert(dst.type() == CV_8UC1);
	int mwidth = mask.cols;
	int mheight = mask.rows;
	int width = dst.cols;
	int height = dst.rows;
	uchar *mp = (uchar *)mask.data;
	uchar *dp = (uchar *)dst.data;
	int moffset = mask.step - mask.cols;
	int doffset = dst.step - dst.cols;
	int x, y;
	int Yoffset = 50, Xoffset = 80;
	for (y = 0; y < height; y++, dp += doffset) {
		for (x = 0; x < width; x++, dp++) {
			if (x < width - Xoffset && x >= width - mwidth - Xoffset 
					&& y < mheight + Yoffset && y >= Yoffset) {
				*dp = (x == width - mwidth - Xoffset) ? 255:
					  (y == mheight - 1 + Yoffset) ? 255:
					  (y == Yoffset) ? 255:
				 	  (x == width - Xoffset - 1) ? 255: 
							mp[x - width + mwidth + Xoffset + mwidth * (y-Yoffset)+ moffset * (y-Yoffset)];
			}
		}
	}
}
bool findCenterByFittingEllipse(Mat pupilMask, RotatedRect &box,float &cen_X, float &cen_Y){
	vector<vector<Point> > contours;
	Mat copypM;
	pupilMask.copyTo(copypM);
	findContours(copypM, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	if (contours.size() == 0)
		return false;
	Mat pointsf;
	Mat(contours[contours.size() - 1]).convertTo(pointsf, CV_32F);
	box = fitEllipse(pointsf);
	cen_X = box.center.x;
	cen_Y = box.center.y;

	//	ellipse(pupilMask, box, 255, 1, CV_AA);
		//show(pupilMask);
	return true;
 }

/*
* 输入src图像，输入前一帧的中心点坐标和光斑坐标，输出标记了中心点后的图像dst，输出当前帧的中心坐标和光斑坐标
* Mat src: 当前帧图像
* Mat &dst: 标记了中心点后的图像
* float &cen_X: 当前帧的中心点的X坐标
* float &cen_Y: 当前帧的中心点的Y坐标
* Vec2f &reflection: 当前帧的光斑坐标,二维向量
* bool &isBlink: 是否眨眼
* int &area: 瞳孔面积
* std::vector<int> block: 眼球矩阵块，四位向量：（X_mask,Y_mask,width_mask,height_mask）
* int manthreshold: 如果小于0，则自动阈值，如果大于等于0，则阈值为manthreshold
*/
void irisSeg(Mat src, Mat &dst, float &cen_X, float &cen_Y, Vec2f &reflection, bool& isBlink, int &area, std::vector<int> block, int manthreshold) {
	//show(src);
	//float scale = 2.0f;
	//resize(src, src, Size(src.cols * scale, src.rows * scale),0,0,1);
	//block[0] *= scale; block[1] *= scale; block[2] *= scale; block[3] *= scale;
	Mat orig;
	src.copyTo(orig);
	CV_Assert(block.size() == 4);
	CV_Assert(block[0] > 0 && block[1] > 0 && block[2] > 0 && block[3] > 0);
	CV_Assert(block[0] + block[2] <= src.cols && block[1] + block[3] <= src.rows);
	
	int X_mask = block[0];
	int Y_mask = block[1];
	int width_mask = block[2];
	int height_mask = block[3];
	isBlink = false;
	static int flag = 0;
	Rect masktest(X_mask, Y_mask, width_mask, height_mask);
	double t = 0, start_t;

	Mat img = src(masktest);
	CV_Assert(img.data != 0);
	int outWidth = width_mask, outHeight = height_mask;
	int width = img.cols;
	int height = img.rows;
	int regCount = 0; /*¹â°ßÇøÓò¸öÊý*/
	Mat reMask(height, width, CV_8UC1);
	Mat pupilMask(height, width, CV_8UC1);
	Mat pupilHist(1, 256, CV_32SC1);
	Mat regions(reMask.rows, reMask.cols, CV_32SC1);
	bool fittingSuccess = false;
	int cur_val = 0;
	Scalar RED = Scalar(0, 0, 255);
	Scalar GREEN = Scalar(0, 255, 0);
	Scalar BLUE = Scalar(255, 0, 0);
	RotatedRect box;

	try {

		//show(src);
		//show(img);
		createReflectionMask(img, reMask, regions, regCount);
		//show(reMask);
		inpaint(img, reMask, img, 1.5, CV_INPAINT_TELEA);
		hist2u(img, pupilHist);
		//drawHist(pupilHist);
		smoothHist(pupilHist, pupilHist, 4);
		//drawHist(pupilHist);
		thresholdingPupil(img, pupilMask, cur_val, manthreshold);
		//show(img);
		//show(pupilMask);
		refinePupilMask(reMask, pupilMask);
		//show(pupilMask);
		fittingSuccess = findCenterByFittingEllipse(pupilMask, box, cen_X, cen_Y);
		if (!fittingSuccess){
			Vec2f tmpcen;
			findCentroid(pupilMask, tmpcen);
			cen_X = tmpcen[0];
			cen_Y = tmpcen[1];
		}
		filterReflection(reMask, regions, regCount, cen_X, cen_Y);
		
			findCentroid(reMask, reflection);
		if (reflection[0] == -1 || reflection[1] == -1) {
		//	cerr << "Warning: Reflection Center Wrong!" << endl;
		//	show2(reMask);
		//	cout << regCount << endl;
			reflection[0] = width / 2;
			reflection[1] = height / 2;
			isBlink = true;
		}
	//	refinePupilMask(reMask, pupilMask);

		if (!fittingSuccess){
			Vec2f tmpcen;
			findCentroid(pupilMask, tmpcen);
			cen_X = tmpcen[0];
			cen_Y = tmpcen[1];
		}

		if (cen_X < 0 || cen_Y < 0 || cen_X >= width || cen_Y >= height) {
		//	cerr << "Warning: Center Wrong!" << endl;
			cen_X = width / 2;
			cen_Y = height / 2;
		}
		calPupilArea(pupilMask, area);


#define KALMANFILTERING
#ifdef KALMANFILTERING
		static float pxk_1 = 1;
		static float pyk_1 = 1;
		static float xk_1  = 0;
		static float yk_1 = 0;
		if (xk_1 == 0) {
			xk_1 = cen_X;
			yk_1 = cen_Y;
		} else {
			kalmanfiltering(cen_X, pxk_1, xk_1, cen_X, pxk_1);
			kalmanfiltering(cen_Y, pyk_1, yk_1, cen_Y, pyk_1);
		}
		
		static float rpxk_1 = 1;
		static float rpyk_1 = 1;
		static float rxk_1 = 0;
		static float ryk_1 = 0;
		float tmprx = reflection[0], tmpry = reflection[1];
		if (rxk_1 == 0) {
			rxk_1 = reflection[0];
			ryk_1 = reflection[1];
		} else {
			kalmanfiltering(tmprx, rpxk_1, rxk_1, tmprx, rpxk_1);
			kalmanfiltering(tmpry, rpyk_1, ryk_1, tmpry, rpyk_1);
		}
		reflection[0] = tmprx; reflection[1] = tmpry;
#endif


		cen_X += X_mask;
		cen_Y += Y_mask;
		reflection[0] += X_mask;
		reflection[1] += Y_mask;

		
		/*Drawing Area, Center, Histogram*/
		dst = orig;

		drawMaskOnDst(pupilMask, dst);

		
	//	smoothHist(pupilHist, pupilHist, 4);
		drawHistOnDst(pupilHist, dst, cur_val);
		cvtColor(dst, dst, CV_GRAY2BGR);
		line(dst, Point2f(cen_X , 0), Point2f(cen_X, dst.rows), Scalar(0,0,255), 1);
		line(dst, Point2f(0, cen_Y), Point2f(dst.cols-321, cen_Y), Scalar(0,0,255), 1);
		double xx = reflection[0], yy = reflection[1];
		line(dst, Point2f(xx, Y_mask), Point2f(xx, Y_mask+height_mask), Scalar(0,255,0), 1);
		line(dst, Point2f(X_mask, yy), Point2f(X_mask+width_mask, yy), Scalar(0,255,0), 1);
		circle(dst, Point2f(cen_X, cen_Y), 2, RED, -1, CV_AA);
		circle(dst, Point2f(xx, yy), 2, GREEN, -1, CV_AA);
		rectangle(dst, masktest, BLUE, 2, CV_AA);
		RotatedRect boxx = RotatedRect(Point2f(masktest.x, masktest.y) + box.center, box.size, box.angle);
		ellipse(dst, boxx, RED, 1, CV_AA);


		string pupilMessage;
		stringstream ss1;
		ss1 << fixed << setprecision(2);
		ss1 << "Pupil: (" << cen_X << ", " << cen_Y << ")  " << "Reflection: (" << reflection[0] << ", " << reflection[1] << ")";
		pupilMessage = ss1.str();
		putText(dst, pupilMessage, Point(40, 40), FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(0, 255, 255), 1, CV_AA);

		stringstream ss2;
		string refMessage;
		float vecx = cen_X - reflection[0];
		float vecy = cen_Y - reflection[1];
		ss2 << fixed << setprecision(2);

		ss2 << "Vector: (" << vecx << ", " << vecy << ")";
		refMessage = ss2.str();
		putText(dst, refMessage, Point(40, 60), FONT_HERSHEY_COMPLEX_SMALL, .8, Scalar(0, 255, 255), 1, CV_AA);

		line(dst, Point(2, 2), Point(abs(100 * vecx), 2), RED, 2);
		line(dst, Point(2, 7), Point(abs(100 * vecy), 5), GREEN, 2);

		stringstream ss3;
		string areaMessage;
		ss3 <<"Pupil Area: "<< area;
		areaMessage = ss3.str();
		putText(dst, areaMessage, Point(dst.cols - pupilMask.cols-100, 35), FONT_HERSHEY_COMPLEX_SMALL, .8, Scalar(0, 255, 255), 1, CV_AA);
		
		stringstream ss4;
		string threMsg;
		ss4 << "Threshold:" << cur_val ;
		threMsg = ss4.str();
		putText(dst, threMsg, Point(dst.cols - 300, 180), FONT_HERSHEY_COMPLEX_SMALL, .8, Scalar(0, 255, 255), 1, CV_AA);
		

		putText(dst, (isBlink) ? "BLINK !" : "Eye Opening", Point(40, 90), FONT_HERSHEY_COMPLEX_SMALL, .8, Scalar(0, 255, 255), 1, CV_AA);

		stringstream ss5;
		string blockMsg;
		ss5 << "Block: [" << block[0] << "," << block[1] << "," << block[2] << "," << block[3] << "]";
		blockMsg = ss5.str();
		putText(dst, blockMsg, Point(40, 120), FONT_HERSHEY_COMPLEX_SMALL, .8, Scalar(0, 255, 255), 1, CV_AA);

		
		if (isBlink)
			circle(dst, Point(130,85), 11, GREEN, -1, CV_AA);
		if (fittingSuccess){
			RotatedRect box2 = RotatedRect(Point2f(dst.cols -pupilMask.cols- 80, 50) + box.center, box.size, box.angle);
			ellipse(dst, box2, RED, 1, CV_AA);
		}
		//resize(dst, dst, Size(dst.cols / scale, dst.rows / scale));
		//cout << cen_X - xx << ", " << cen_Y - yy << endl;
		//show(dst);
		//show(src);
	} catch (...) {
		cerr << "Exit with errors."<< endl;
		system("pause");
		exit(EXIT_FAILURE);
	}
}
