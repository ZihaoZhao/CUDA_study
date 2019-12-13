#include <cuda_runtime.h>
#include <stdio.h>
#include "cudastart.h"


//CPU对照组，用于对比加速比
void sumMatrix2DonCPU(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    float* a = MatA;
    float* b = MatB;
    float* c = MatC;
    for(int j=0; j<ny; j++)
    {
        for(int i=0; i<nx; i++)
        {
          c[i] = a[i]+b[i];
        }
        c += nx;
        b += nx;
        a += nx;
    }
}

//核函数，每一个线程计算矩阵中的一个元素。
__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*ny;
    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx]+MatB[idx];
    }
}

//主函数
int main(int argc,char** argv)
{
    //设备初始化
    printf("strating...\n");
    initDevice(0);

    //输入二维矩阵，4096*4096，单精度浮点型。
    int nx = 1<<12;
    int ny = 1<<12;
    int nBytes = nx*ny*sizeof(float);

    //Malloc，开辟主机内存
    float* A_host = (float*)malloc(nBytes);
    float* B_host = (float*)malloc(nBytes);
    float* C_host = (float*)malloc(nBytes);
    float* C_from_gpu = (float*)malloc(nBytes);
    initialData(A_host, nx*ny);
    initialData(B_host, nx*ny);

    //cudaMalloc，开辟设备内存
    float* A_dev = NULL;
    float* B_dev = NULL;
    float* C_dev = NULL;
    CHECK(cudaMalloc((void**)&A_dev, nBytes));
    CHECK(cudaMalloc((void**)&B_dev, nBytes));
    CHECK(cudaMalloc((void**)&C_dev, nBytes));

    //输入数据从主机内存拷贝到设备内存
    CHECK(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice));

    //二维线程块，32×32
    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((nx-1)/block.x+1, (ny-1)/block.y+1);

    //测试GPU执行时间
    double gpuStart = cpuSecond();
    //将核函数放在线程网格中执行
    sumMatrix<<<grid,block>>>(A_dev, B_dev, C_dev, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double gpuTime = cpuSecond() - gpuStart;
    printf("GPU Execution Time: %f sec\n", gpuTime);

    //在CPU上完成相同的任务
    cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost);
    double cpuStart=cpuSecond();
    sumMatrix2DonCPU(A_host, B_host, C_host, nx, ny);
    double cpuTime = cpuSecond() - cpuStart;
    printf("CPU Execution Time: %f sec\n", cpuTime);

    //检查GPU与CPU计算结果是否相同
    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, nx*ny);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    cudaDeviceReset();
    return 0;
}
