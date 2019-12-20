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


__global__ void reduceUnroll2(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*2;
	if(idx+blockDim.x<n)
	{
		g_idata[idx]+=g_idata[idx+blockDim.x];

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
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

__global__ void reduceUnroll4(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*4+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*4;
	if(idx+blockDim.x<n)
	{
		g_idata[idx]+=g_idata[idx+blockDim.x];
		g_idata[idx]+=g_idata[idx+blockDim.x*2];
		g_idata[idx]+=g_idata[idx+blockDim.x*3];
	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
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

__global__ void reduceUnroll8(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*8;
	if(idx+blockDim.x<n)
	{
		g_idata[idx]+=g_idata[idx+blockDim.x];
		g_idata[idx]+=g_idata[idx+blockDim.x*2];
		g_idata[idx]+=g_idata[idx+blockDim.x*3];
		g_idata[idx]+=g_idata[idx+blockDim.x*4];
		g_idata[idx]+=g_idata[idx+blockDim.x*5];
		g_idata[idx]+=g_idata[idx+blockDim.x*6];
		g_idata[idx]+=g_idata[idx+blockDim.x*7];

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
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


__global__ void reduceUnrollWarp8(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*8;
	//unrolling 8;
	if(idx+7 * blockDim.x<n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		int a5=g_idata[idx+4*blockDim.x];
		int a6=g_idata[idx+5*blockDim.x];
		int a7=g_idata[idx+6*blockDim.x];
		int a8=g_idata[idx+7*blockDim.x];
		g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>32; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}

__global__ void reduceCompleteUnrollWarp8(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*8;
	if(idx+7 * blockDim.x<n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		int a5=g_idata[idx+4*blockDim.x];
		int a6=g_idata[idx+5*blockDim.x];
		int a7=g_idata[idx+6*blockDim.x];
		int a8=g_idata[idx+7*blockDim.x];
		g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;

	}
	__syncthreads();
	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		idata[tid]+=idata[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		idata[tid]+=idata[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		idata[tid]+=idata[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		idata[tid]+=idata[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}

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
    
    //kernel 4 reduceUnroll2
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceUnroll2 <<<grid.x/2, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/2; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll2 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x/2, block.x);
    
    
    //kernel 5 reduceUnroll4
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceUnroll4 <<<grid.x/4, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/4; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll4 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x/4, block.x);
    
    
    //kernel 6 reduceUnroll8
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceUnroll8 <<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnroll8 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x/8, block.x);
    
    
    //kernel 7 reduceUnrollWarp8
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceUnrollWarp8 <<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceUnrollWarp8 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x/8, block.x);
    
    
    //kernel 8 reduceCompleteUnrollWarp8
	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	timeStart = cpuSecond();
	reduceCompleteUnrollWarp8 <<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];	
    timeElaps = 1000*(cpuSecond() - timeStart);

	printf("gpu sum:%d \n", gpu_sum);
	printf("gpu reduceCompleteUnrollWarp8 elapsed %lf ms     <<<grid %d block %d>>>\n",
		timeElaps, grid.x/8, block.x);    



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
