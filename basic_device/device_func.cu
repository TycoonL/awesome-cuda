#include "cuda_runtime.h"   // ���� CUDA ����ʱ���ͷ�ļ�
#include "device_launch_parameters.h"   // ���� CUDA �豸������ִ�в�����ͷ�ļ�
#include "iostream"


const double a = 1.23;   // ���� a
const double b = 4.56;   // ���� b
const double res = 5.79;   // Ԥ�ڽ��
const double err = 10e-5;   // �������
int N = 10;   // �����С

//����ֵд��
__device__
double add_1(double x, double y)
{
	return (x + y);
}
//ָ��д��
__device__
double add_2(double x, double y,double *z)
{
	*z=x + y;
}
//����д��
__device__
double add_3(double x, double y, double &z)
{
	z = x + y;
}
__global__
void add(double *x, double *y, double *r, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;   // �߳��������㹫ʽ
	if (i >= N) return;
	//r[i] = add_1(x[i] , y[i]);   // �������洢�� r ������
	add_2(x[i], y[i],&r[i]);
	//add_3(x[i], y[i], r[i]);
	printf("threadId:%d,%f\n", i, r[i]);   // ��ӡÿ���̵߳Ľ��
}



void check(double *r, int N);   // ������Ƿ���ȷ�ĺ�������


int main()
{
	int M = sizeof(double)*N;   // �������ڴ�����ռ�õ��ֽ���
	double* x = (double*)malloc(M);   // �����ڴ�ռ�
	double* y = (double*)malloc(M);
	double* r = (double*)malloc(M);
	for (int i = 0;i < N;i++)
	{
		x[i] = a;   // ��ʼ������ x
		y[i] = b;   // ��ʼ������ y
	}

	double *cuda_x, *cuda_y, *cuda_r;   // ���豸�д洢���ݵ�ָ�����
	cudaMalloc((void **)&cuda_x, M);   // �����豸�ڴ�
	cudaMalloc((void **)&cuda_y, M);
	cudaMalloc((void **)&cuda_r, M);
	cudaMemcpy(cuda_x, x, M, cudaMemcpyHostToDevice);   // �������ڴ��е����ݿ������豸�ڴ���
	cudaMemcpy(cuda_y, y, M, cudaMemcpyHostToDevice);

	int blocksize = 128;   // �߳̿��С
	int gridsize = (N - 1) / 128 + 1;   // �߳̿�����
	add << <gridsize, blocksize >> > (cuda_x, cuda_y, cuda_r,N);   // ���� CUDA �˺������ټ���

	cudaMemcpy(r, cuda_r, M, cudaMemcpyDeviceToHost);   // ���豸�ڴ��е����ݿ����������ڴ���

	check(r, N);   // ������Ƿ���ȷ

	free(x);   // �ͷ������ڴ�
	free(y);
	free(r);
	cudaFree(cuda_x);   // �ͷ��豸�ڴ�
	cudaFree(cuda_y);
	cudaFree(cuda_r);

	while (1);   // �������ǰ�������У��Ա�鿴��������
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