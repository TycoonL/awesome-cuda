前言：在学习CUDA编程前有必要先了解GPU的结构，例如SM、SP、memory等

例程地址：[https://github.com/TycoonL/awesome-cuda.git/](https://github.com/TycoonL/awesome-cuda.git/)

### **CUDA程序的基本框架：**

```c++
头文件包含
常量（宏）定义
C++自定义函数和CUDA核函数的定义和声明
int main(void)
{
	分配主机和设备内存
	初始化主机中的数据
	将某些数据复制到设备中
	调用核函数怎在设备中运算
	将某些数据从设备中复制到主机
	释放内存
}
```



#### 1.basic_1：用核函数查看device的参数

```c++
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
```

<img src=".\picture\basic_1.JPG" alt="basic_1" style="zoom: 67%;" />

#### 2.basic_2：使用线程索引

一个**核函数**可以指派**多个线程**，而这些线程的组织结构由执行配置**<<<grid_size,block_size>>>**决定

- 每个线程在核函数中都有一个**唯一的身份标识**

- **grid_size→gridDim(数据类型：dim3 （x，y，z）); block_size→blockDim; 0<=blockIdx<gridDim; 0<=threadIdx<blockDim**

- 在一维情况下：
  $$
  threadID=blockIdx.x\times blockDim.x+threadIdx.x
  $$

```c++
//核函数 
__global__ void hello_from_device()
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	printf("hello world from block %d and thread %d and threadId %d\n", bid, tid,threadID);
}
```

  ```C++
  hello_from_device<<< 2,3 >>>()的结果：
      hello world from block 1 and thread 0 and threadId 3
      hello world from block 1 and thread 1 and threadId 4
      hello world from block 1 and thread 2 and threadId 5
      hello world from block 0 and thread 0 and threadId 0
      hello world from block 0 and thread 1 and threadId 1
      hello world from block 0 and thread 2 and threadId 2
  ```

  上面是一维的情况，下面介绍用**结构体dim3**来定义“多维”的网络和线程块：

```c++
	#dim3 _size(x,y,z)
	dim3 grid_size(2,2)  //等价于 (2,2,1)
	dim3 block_size(3,2)
```

<img src=".\picture\basic_2_1.JPG" alt="basic_2_1" style="zoom: 67%;" />

对应的一维指标为：

```c++
	int tid=threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x
	int bid=blockIdx.z*gridDim.x*gridDim.y+blockIdx.y*gridDim.x+blockIdx.x
```

**注意：**

- 网格大小在x,y,z三个方向上要分别小于$2^{31}-1$,65535,65535
- 块线程大小在x,y,z三个方向上要分别小于1024，1024，64
- 另外，块线程在三个维度上的乘积还要小于1024，即：**一个块最多有1024线程**

#### 3.basic_add

这个例程是一个基础的CUDA程序结构，使用"单指令-多线程"的方式编写代码，通过分配内存、调用核函数、数据传递，实现了并行计算加法。核函数：

```c++
__global__
void add(double *x, double *y, double *r)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;   // 线程索引计算公式
	r[i] = x[i] + y[i];   // 计算结果存储在 r 数组中
	printf("threadId:%d,%f\n", i, r[i]);   // 打印每个线程的结果
}
```

```
add<<<1, 128>>>(cuda_x,cuda_y,cuda_r)结果：
threadId:0,5.790000
threadId:1,5.790000
threadId:2,5.790000
threadId:3,5.790000
threadId:4,5.790000
threadId:5,5.790000
threadId:6,5.790000
threadId:7,5.790000
threadId:8,5.790000
threadId:9,5.790000
threadId:10,0.000000
threadId:11,0.000000
threadId:12,0.000000
threadId:13,0.000000...
注意：例程中N为10，而块线程块大小为128，剩余的118个线程也会运行
```

为了防止线程大于数组元素个数的非法内存，需要加入if语句：

```c++
__global__
void add(double *x, double *y, double *r, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;   // 线程索引计算公式
	if (i >= N) return;
	r[i] = x[i] + y[i];   // 计算结果存储在 r 数组中
	printf("threadId:%d,%f\n", i, r[i]);   // 打印每个线程的结果
}
```

##### 数据传递cudaMemcpy(dst,src,count,kind)

- dst:目标地址
- src：源地址
- count：复制数据的字节数
- kind：传递方向
  - cudaMemcpyHostToDevice
  - cudaMemcpyDeviceToHost
  - cudaMemcpyHostToHost
  - cudaMemcpyDeviceToDevice
  - cudaMemcpyDefault：自动判断传递的方向

#### 4.basic_device

函数执行空间标识符：

- \_\_global\_\_：修饰的函数被称为**核函数**，由主机调用，设备执行
- \_\_device\_\_：修饰的函数被称为**设备函数**，由核函数和其它设备函数调用，设备执行
- \_\_host\_\_：修饰的函数就是主机端普通的函数，一般省略

注意：可以同时用\_\_device\_\_和\_\_host\_\_修饰一个函数，但其他的组合不行

该例程将例程3的加法核函数加入设备函数写法：

```c++
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
```

