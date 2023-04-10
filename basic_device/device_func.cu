#include "cuda_runtime.h"   // 包含 CUDA 运行时库的头文件
#include "device_launch_parameters.h"   // 包含 CUDA 设备启动和执行参数的头文件
#include "iostream"


const double a = 1.23;   // 常量 a
const double b = 4.56;   // 常量 b
const double res = 5.79;   // 预期结果
const double err = 10e-5;   // 允许误差
int N = 10;   // 数组大小

//返回值写法
__device__
double add_1(double x, double y)
{
	return (x + y);
}
//指针写法
__device__
double add_2(double x, double y,double *z)
{
	*z=x + y;
}
//引用写法
__device__
double add_3(double x, double y, double &z)
{
	z = x + y;
}
__global__
void add(double *x, double *y, double *r, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;   // 线程索引计算公式
	if (i >= N) return;
	//r[i] = add_1(x[i] , y[i]);   // 计算结果存储在 r 数组中
	add_2(x[i], y[i],&r[i]);
	//add_3(x[i], y[i], r[i]);
	printf("threadId:%d,%f\n", i, r[i]);   // 打印每个线程的结果
}



void check(double *r, int N);   // 检查结果是否正确的函数声明


int main()
{
	int M = sizeof(double)*N;   // 数组在内存中所占用的字节数
	double* x = (double*)malloc(M);   // 分配内存空间
	double* y = (double*)malloc(M);
	double* r = (double*)malloc(M);
	for (int i = 0;i < N;i++)
	{
		x[i] = a;   // 初始化数组 x
		y[i] = b;   // 初始化数组 y
	}

	double *cuda_x, *cuda_y, *cuda_r;   // 在设备中存储数据的指针变量
	cudaMalloc((void **)&cuda_x, M);   // 分配设备内存
	cudaMalloc((void **)&cuda_y, M);
	cudaMalloc((void **)&cuda_r, M);
	cudaMemcpy(cuda_x, x, M, cudaMemcpyHostToDevice);   // 将主机内存中的数据拷贝到设备内存中
	cudaMemcpy(cuda_y, y, M, cudaMemcpyHostToDevice);

	int blocksize = 128;   // 线程块大小
	int gridsize = (N - 1) / 128 + 1;   // 线程块数量
	add << <gridsize, blocksize >> > (cuda_x, cuda_y, cuda_r,N);   // 调用 CUDA 核函数加速计算

	cudaMemcpy(r, cuda_r, M, cudaMemcpyDeviceToHost);   // 将设备内存中的数据拷贝到主机内存中

	check(r, N);   // 检查结果是否正确

	free(x);   // 释放主机内存
	free(y);
	free(r);
	cudaFree(cuda_x);   // 释放设备内存
	cudaFree(cuda_y);
	cudaFree(cuda_r);

	while (1);   // 程序结束前保持运行，以便查看程序的输出
	return 0;
}
void check(double *r, int N)
{
	for (int i = 0;i < N;i++)
	{
		if (abs(r[i] - res) > err)
		{
			printf("error result!");
			break;
		}
	}
}