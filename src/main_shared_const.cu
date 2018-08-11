#include "stereo.cuh"


int main_shared_const()
{
	float elapseTime;
	cudaEvent_t  start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	Mat imL = imread("imr.png", IMREAD_GRAYSCALE);//◊ÛÕº
	Mat imR = imread("iml.png", IMREAD_GRAYSCALE);//”“Õº
	Mat imD = Mat::zeros(imL.size(), imL.type());// ”≤ÓÕº
	printf("%d %d\n", imL.cols, imL.rows);
	if (imL.empty() || imR.empty()) {
		perror("load image\n");
		return -1;
	}
	uchar *data[3], *devData[3];
	int Col = imL.cols, Row = imL.rows, dataSize = imL.cols*imL.rows;
	HANDLE_ERROR(cudaEventRecord(start, 0));
	//ƒ⁄¥Ê∑÷≈‰
	for (int i = 0; i < 3; i++) {
		data[i] = (uchar*)malloc(dataSize * sizeof(uchar));
		HANDLE_ERROR(cudaMalloc((void**)&devData[i], dataSize * sizeof(uchar)));
	}
	//Host to Device
	fillData(imL, data[0]);
	fillData(imR, data[1]);

	//for constant memory
	HANDLE_ERROR(cudaMemcpy(devData[1], data[1], sizeof(uchar)*dataSize, cudaMemcpyHostToDevice));
	int div = 3, r = WinSize / 2;
	int step = Row / div;
	for (int i = 0; i < div; i++) {
		int beg, end, start;
		if (i == 0) {
			beg = 0;	end = step + r;    start = 0;
		}
		else if (i == div - 1) {
			beg = i*step - r;   end = Row;   start = r;
		}
		else {
			beg = i*step - r;   end = (i + 1)*step + r;   start = r;
		}
		HANDLE_ERROR(cudaMemcpyToSymbol(imLcst, data[0] + beg*Col, (end - beg)*Col));
		getDepthMap_constant << <end - beg, 450 >> >(devData[1], devData[2], Col, (end - beg));
		HANDLE_ERROR(cudaMemcpy(data[2] + (i*step)*Col, devData[2] + start*Col, (end - beg - r)*Col, cudaMemcpyDeviceToHost));
	}
	//Device to Host
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