#include "stereo.cuh"



void fillData(const Mat &img, uchar *data) {
	int imX = img.cols, imY = img.rows, idx = 0;
	for (int y = 0; y < imY; y++) {
		for (int x = 0; x < imX; x++) {
			data[idx++] = img.at<uchar>(y, x);
		}
	}
	return;
}

void fillImage(const uchar *data, Mat &img) {
	int imX = img.cols, imY = img.rows, idx = 0;
	for (int y = 0; y < imY; y++) {
		for (int x = 0; x < imX; x++) {
			img.at<uchar>(y, x) = data[idx++];
		}
	}
	return;
}

__device__ int devAbs(const int x) {
	return x >= 0 ? x : -x;
}

__device__ int SAD(const int *w1, const int *w2) {
	int ret = 0;
	for (int i = 0; i < WinSize*WinSize; i++) {
		ret += devAbs(w1[i] - w2[i]);
	}
	return ret;
}

__device__ void fillWin(int *win, uchar *img, int imX, int x, int y) {
	int r = WinSize / 2;
	for (int j = 0; j < WinSize; j++) {
		for (int i = 0; i < WinSize; i++) {
			win[j * WinSize + i] = img[(y - r + j)*imX + (x - r + i)];
		}
	}
	return;
}


__global__ void getDepthMap(uchar *imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int r = WinSize / 2;
	if (x >= r && x < imX - r && y >= r && y < imY - r) {
		int winL[WinSize*WinSize], winR[WinSize *WinSize];
		int xopt, S, max = 0x7fffffff;
		fillWin(winL, imL, imX, x, y);
		for (int i = x; i < imX - r; i++) {
			fillWin(winR, imR, imX, i, y);
			S = SAD(winL, winR);
			if (S < max) {
				xopt = i;
				max = S;
			}
		}
		imD[y*imX + x] = (xopt - x)* SCALE;//计算视差,并乘以SCALE
	}
}

__global__ void getDepthMap_constant(
	uchar *imR, uchar *imD,
	const int imX, const int imY) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int r = WinSize / 2;
	if (x >= r && x < imX - r && y >= r && y < imY - r) {
		int winL[WinSize*WinSize], winR[WinSize *WinSize];
		int xopt, S, max = 0x7fffffff;
		fillWin(winL, imLcst, imX, x, y);
		for (int i = x; i < imX - r; i++) {
			fillWin(winR, imR, imX, i, y);
			S = SAD(winL, winR);
			if (S < max) {
				xopt = i;
				max = S;
			}
		}
		if (y == 5) printf("%d\n", imLcst[y*imX + x]);
		imD[y*imX + x] = imLcst[y*imX + x];//(xopt - x)* SCALE;//计算视差,并乘以SCALE
	}
}

__global__ void getDepthMap_shared(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY) {
	int x = threadIdx.x, y = blockIdx.x, r = WinSize/2;
	__shared__ uchar im[WinSize*450];
	if (x >= r && x < imX - r && y >= r && y < imY - r) {
		for (int i = -r; i <= r; i++) {
			im[(i + r)*imX + x] = imR[(y + i)*imX + x];
		}
		__syncthreads();
		int winL[WinSize*WinSize], winR[WinSize *WinSize];
		int xopt, S, max = 0x7fffffff;
		fillWin(winL, imL, imX, x, y);
		for (int i = x; i < imX - r; i++) {
			fillWin(winR, im, imX, i, r);
			S = SAD(winL, winR);
			if (S < max) {
				xopt = i;
				max = S;
			}
		}
		imD[y*imX + x] = (xopt - x)* SCALE;//计算视差,并乘以SCALE
	}
}

__global__ void getDepthMap_shared_const(
	uchar *imR, uchar *imD,
	const int imX, const int imY) {
	int x = threadIdx.x, y = blockIdx.x, r = WinSize / 2;
	__shared__ uchar im[WinSize * 450];
	if (x >= r && x < imX - r && y >= r && y < imY - r) {
		for (int i = -r; i <= r; i++) {
			im[(i + r)*imX + x] = imR[(y + i)*imX + x];
		}
		__syncthreads();
		int winL[WinSize*WinSize], winR[WinSize *WinSize];
		int xopt, S, max = 0x7fffffff;
		fillWin(winL, imLcst, imX, x, y);
		for (int i = x; i < imX - r; i++) {
			fillWin(winR, im, imX, i, r);
			S = SAD(winL, winR);
			if (S < max) {
				xopt = i;
				max = S;
			}
		}
		imD[y*imX + x] = (xopt - x)* SCALE;//计算视差,并乘以SCALE
	}
}


__global__ void getDepthMap_stream(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int r = WinSize / 2;
	if (x >= r && x < imX - r && y >= r && y < imY - r) {
		int winL[WinSize*WinSize], winR[WinSize *WinSize];
		int xopt, S, max = 0x7fffffff;
		fillWin(winL, imL, imX, x, y);
		for (int i = x; i < imX - r; i++) {
			fillWin(winR, imR, imX, i, y);
			S = SAD(winL, winR);
			if (S < max) {
				xopt = i;
				max = S;
			}
		}
		imD[y*imX + x] = (xopt - x)* SCALE;//计算视差,并乘以SCALE
	}

}


/*
__global__ void getOptX_serial(int *winL, uchar *imR, int imX, int y, int *xopt) {
	__shared__ int S[450];
	int winR[WinSize *WinSize], r = WinSize / 2;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x >= r && x < imX - r) {
		fillWin(winR, imR, imX, x, y);
		S[x] = SAD(winL, winR);
	}
	__syncthreads();
	if (x == 0) {
		int min = 0x7fffffff;
		for (int i = r; i < imX - r; i++) {
			if (S[i] < min) {
				min = S[i];
				*xopt = i;
			}
		}
	}
}

__global__ void getOptX_reduction(int *winL, uchar *imR, int imX, int y, int *xopt) {
	__shared__ int S[450], idx[450];
	int winR[WinSize *WinSize], r = WinSize / 2;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	idx[x] = x;//保存下标
	if (x >= r && x < imX - r) {
		fillWin(winR, imR, imX, x, y);
		S[x] = SAD(winL, winR);
	}else   S[x] = 0x7fffffff;//将剩下的置为最大
	__syncthreads();

	int step = 450;
	do {
		step = (step + 1) / 2;
		if (x + step < 450 && S[x] > S[x + step]) {
			S[x] = S[x + step];
			idx[x] = idx[x + step];
		}	
	} while (step != 1);
	if (idx == 0) *xopt = idx[0];
}

__global__ void getDepthMap_Dynamic(uchar* imL,
	uchar *imR, uchar *imD,
	const int imX, const int imY) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int r = WinSize / 2;
	if (x >= r && x < imX - r && y >= r && y < imY - r) {
		int winL[WinSize*WinSize], winR[WinSize *WinSize], *xopt;
		fillWin(winL, imL, imX, x, y);
		//getOptX_serial <<<1, 450 >>>(winL, imR, imX， y, xopt);
		getOptX_reduction <<<1, 450 >>>(winL, imR, imX， y, xopt);
		imD[y*imX + x] = (*xopt - x)* SCALE;//计算视差,并乘以SCALE
	}
}
*/