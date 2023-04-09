前言：在学习CUDA编程前有必要先了解GPU的结构，例如SM、SP、memory等

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

![basic_1](.\picture\basic_1.JPG)

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