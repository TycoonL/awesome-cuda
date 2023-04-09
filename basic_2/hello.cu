#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "cuda.h"

//核函数
__global__ void hello_from_device()
{
	int bid_x = blockIdx.x;
	int bid_y = blockIdx.y;
	int tid = threadIdx.x;
	int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	printf("hello world from block (%d,%d) and thread %d\n", bid_x,bid_y, tid,threadID);
}


int main(void)
{
	dim3 gridsize(2, 2);
	hello_from_device<<<gridsize,3 >>>();
	cudaDeviceSynchronize();//同步主机与设备
	return 0;
}
