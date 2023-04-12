#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void transpose(float* in, float* out, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		int index_in = y * width + x;
		int index_out = x * height + y;
		//out[index_in] = __ldg(&in[index_out]);//Execution time: 0.019456 ms
		//out[index_out] = __ldg(&in[index_in]);//Execution time: 0.021504 ms
		//out[index_in] = in[index_out];//out合并写入，in非合并访问  Execution time: 0.019456 ms
		out[index_out] = in[index_in];//in合并访问，out非合并写入  Execution time: 0.021504 ms
	}
}

int main() {
	int width =64;
	int height = 64;

	int size = width * height * sizeof(float);
	float* in = (float*)malloc(size);
	float* out = (float*)malloc(size);

	for (int i = 0; i < width * height; i++) {
		in[i] = (float)i;
	}

	float *d_in, *d_out;
	cudaMalloc(&d_in, size);
	cudaMalloc(&d_out, size);

	cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block_size(32, 32);
	dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
	cudaEventRecord(start);
	transpose << <grid_size, block_size >> > (d_in, d_out, width, height);
	cudaEventRecord(stop);

	cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Execution time: %f ms\n", milliseconds);

	cudaFree(d_in);
	cudaFree(d_out);
	free(in);
	free(out);

	return 0;
}
