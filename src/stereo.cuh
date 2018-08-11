#ifndef STEREO_CUH_
#define STEREO_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#define SCALE 4
#define WinSize 5
using namespace cv;

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef unsigned char uchar;

__constant__  uchar imLcst[130*450];

void fillData(const Mat &img, uchar *data);

void fillImage(const uchar *data, Mat &img);

__device__ int devAbs(const int x);

__device__ int SAD(const int *w1, const int *w2);

__device__ void fillWin(int *win, uchar *img, int imX, int x, int y);

__global__ void getDepthMap(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY);

__global__ void getDepthMap_shared(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY);

__global__ void getDepthMap_constant(
	uchar *imR, uchar *imD,
	const int imX, const int imY);

__global__ void getDepthMap_stream(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY);

__global__ void getDepthMap_Dynamic(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY);

__global__ void getOptX(int *winL, uchar *imR, int imX, int y);




#endif