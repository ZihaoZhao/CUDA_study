#include <cuda_runtime.h>
#include <stdio.h>
#include "cudastart.h"
int recursiveReduce(int *data, int const size)
{
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
	int const stride = size / 2;
	if (size % 2 == 1)
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	}
	else
	{
		for (int i = 0; i < stride; i++)
		{
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce(data, stride);
}



__global__ void reduceNeighbored(int * g_idata,int * g_odata,unsigned int n) 
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the 
	int *idata = g_idata + blockIdx.x*blockDim.x;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceNeighboredLess(int * g_idata,int *g_odata,unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx > n)
		return;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		//convert tid into local array index
		int index = 2 * stride *tid;
		if (index < blockDim.x)
		{
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx >= n)
		return;
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride >0; stride >>=1)
	{
		
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}

int main(int argc,char** argv)
{
	initDevice(0);
	
	//initialization

	int size = 1 << 24;
	printf("	with array size %d  ", size);

	//execution configuration
	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);   //从命令行输入设置block大小
	}
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int * tmp = (int*)malloc(bytes);

	//initialize the array
	initialData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double timeStart, timeElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	//cpu reduction 对照组
	int cpu_sum = 0;
	timeStart = cpuSecond();
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	timeElaps = 1000*(cpuSecond() - timeStart);

	printf("cpu sum:%d \n", cpu_sum);
	printf("cpu reduction elapsed %lf ms cpu_sum: %d\n", timeElaps, cpu_sum);


	//kernel 1 reduceNeighbored

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceNeighbored <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceNeighbored elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);
    
    //kernel 2 reduceNeighboredless
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceNeighboredLess <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceNeighboredless elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);

    //kernel 3 reduceInterleaved
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceInterleaved <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceInterleaved elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x, block.x);
    
	// free host memory

	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	//reset device
	cudaDeviceReset();

	//check the results
	if (gpu_sum == cpu_sum)
	{
		printf("Test success!\n");
	}
	return EXIT_SUCCESS;
}
