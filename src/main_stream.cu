#include "stereo.cuh"


int main_stream()
{
	float elapseTime;
	const int streamSize = 5;
	cudaEvent_t  start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	Mat imL = imread("imr.png", IMREAD_GRAYSCALE);//左图
	Mat imR = imread("iml.png", IMREAD_GRAYSCALE);//右图
	Mat imD = Mat::zeros(imL.size(), imL.type());//视差图
	printf("%d %d\n", imL.cols, imL.rows);
	if (imL.empty() || imR.empty()) {
		perror("load image\n");
		return -1;
	}
	uchar *data[3], *devData[streamSize][3];
	int dataSize = imL.cols*imL.rows, Row = imL.rows, Col = imL.cols;
	///////////开始计时
	HANDLE_ERROR(cudaEventRecord(start, 0));
	//内存分配
	for (int i = 0; i < streamSize; i++) {
		for (int j = 0; j < 3; j++) {
			HANDLE_ERROR(cudaMalloc((void**)&devData[i][j], dataSize/streamSize * sizeof(uchar)+WinSize*Col));
		}
	}
	//HANDLE_ERROR(cudaEventRecord(start, 0));
	for (int i = 0; i < 3; i++) {
		HANDLE_ERROR(cudaHostAlloc((void**)&data[i], dataSize*sizeof(uchar), cudaHostAllocDefault));//page-locked host memory
	}
	fillData(imL, data[0]);
	fillData(imR, data[1]);

	//kernel stream
	cudaStream_t  stream[streamSize];
	for (int i = 0; i < streamSize; i++) {
		HANDLE_ERROR(cudaStreamCreate(&stream[i]));
	}
	int step = Row / streamSize, r = WinSize/2;
	for (int i = 0; i < streamSize; i++) {
		int beg, end, start;
		if (i == 0) {
			beg = 0;	end = step;    start = 0;
		}else if (i == streamSize - 1) {
			beg = i*step - r;   end = Row - r;   start = r;
		}else {
			beg = i*step - r;   end = (i + 1)*step;   start = r;
		}
		HANDLE_ERROR(cudaMemcpyAsync(devData[i][0], data[0]+beg*Col, (end-beg)*Col*sizeof(uchar), cudaMemcpyHostToDevice, stream[i]));
		HANDLE_ERROR(cudaMemcpyAsync(devData[i][1], data[1]+beg*Col, (end-beg)*Col*sizeof(uchar), cudaMemcpyHostToDevice, stream[i]));
		getDepthMap_stream<<<dim3(4, end-beg), dim3(128, 1), 0, stream[i] >>>(devData[i][0], devData[i][1], devData[i][2], Col, Row);
		HANDLE_ERROR(cudaMemcpyAsync(data[2]+i*step*Col, devData[i][2]+start*Col, (end-beg-start)*Col*sizeof(uchar), cudaMemcpyDeviceToHost, stream[i]));
	}
	//
	for (int i = 0; i < streamSize; i++) {
		HANDLE_ERROR(cudaStreamSynchronize(stream[i]));
	}
	fillImage(data[2], imD);
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapseTime, start, stop));
	imshow("stereo", imD);
	imwrite("imD.png", imD);
	printf("time = %lfms\n", elapseTime);
	waitKey(0);
	return 0;
}