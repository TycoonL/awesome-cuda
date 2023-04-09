#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "cuda.h"

//ºËº¯Êý
__global__ void hello_from_device()
{
	int bid_x = blockIdx.x;
	int bid_y = blockIdx.y;
	int tid = threadIdx.x;
	//int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	printf("hello world from block (%d,%d) and thread %d\n", bid_x,bid_y, tid);
}


int main(void)
{
	dim3 gridsize(2, 2);
	hello_from_device<<<gridsize,3 >>>();
	cudaDeviceSynchronize();//Í¬²½Ö÷»úÓëÉè±¸
	return 0;
}
