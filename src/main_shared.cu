#include "stereo.cuh"


int main(int argc, char **argv)
{
	if (argc != 3) {
		printf("Parameter: imgleft, imgright\n");
		return 0;
	}
	float elapseTime;
	cudaEvent_t  start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	Mat imL = imread("imr.png", IMREAD_GRAYSCALE);//��ͼ
	Mat imR = imread("iml.png", IMREAD_GRAYSCALE);//��ͼ
	Mat imD = Mat::zeros(imL.size(), imL.type());//�Ӳ�ͼ
	printf("%d %d\n", imL.cols, imL.rows);
	if (imL.empty() || imR.empty()) {
		perror("load image\n");
		return -1;
	}
	uchar *data[3], *devData[3];
	int dataSize = imL.cols*imL.rows;
	HANDLE_ERROR(cudaEventRecord(start, 0));
	//�ڴ����
	for (int i = 0; i < 3; i++) {
		data[i] = (uchar*)malloc(dataSize * sizeof(uchar));
		HANDLE_ERROR(cudaMalloc((void**)&devData[i], dataSize * sizeof(uchar)));
	}
	//Host to Device
	fillData(imL, data[0]);
	fillData(imR, data[1]);
	for (int i = 0; i < 2; i++) {
		HANDLE_ERROR(cudaMemcpy(devData[i], data[i], sizeof(uchar)*dataSize, cudaMemcpyHostToDevice));
	}
	//kernel
	getDepthMap_shared <<<375, 450>>> (devData[0], devData[1], devData[2], imL.cols, imL.rows);
	
	//Device to Host
	HANDLE_ERROR(cudaMemcpy(data[2], devData[2], sizeof(uchar)*dataSize, cudaMemcpyDeviceToHost));
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