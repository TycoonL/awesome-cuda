#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"

//打印GPU设备信息
void print_GPU_device_info()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
		std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
		std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
		std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
		std::cout << "每个Block的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
		std::cout << "每个Block的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
		std::cout << "每个Block中可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
		std::cout << "======================================================" << std::endl;

	}
}


int main()
{
	print_GPU_device_info();

	while (1);
	return 0;
}
