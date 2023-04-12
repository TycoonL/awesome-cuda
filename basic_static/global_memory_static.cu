#include "cuda_runtime.h"   // 包含 CUDA 运行时库的头文件
#include "device_launch_parameters.h"   // 包含 CUDA 设备启动和执行参数的头文件
#include "iostream"

//定义设备静态全局变量
__device__ int cuda_x = 1;
__device__ int cuda_y[2];

__global__
void kernel()
{
	cuda_y[0] += cuda_x;
	cuda_y[1] += cuda_x;
	printf("cuda_y1=%d,cuda_y2=%d\n", cuda_y[0], cuda_y[1]);
}

int main()
{

	int host[2] = {10, 20};
	//cudaMemcpy(cuda_y,host, sizeof(int) * 2,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(cuda_y, host, sizeof(int) * 2);//
	kernel << <1, 1 >> > ();
	cudaDeviceSynchronize();
	cudaMemcpyFromSymbol(host, cuda_y, sizeof(int) * 2);
	printf("host1=%d,host2=%d\n", host[0], host[1]);

	while (1);   // 程序结束前保持运行，以便查看程序的输出
	return 0;
}