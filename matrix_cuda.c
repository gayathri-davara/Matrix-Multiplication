#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4000

__global__ void matMulKernel(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
        {
            sum += A[row * n + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

int main()
{
    size_t size = N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("CUDA Matrix Multiplication Completed\n");
    printf("Matrix Size: %d x %d\n", N, N);
    printf("Block Size: %d x %d\n", block.x, block.y);
    printf("Grid Size: %d x %d\n", grid.x, grid.y);
    printf("C[0][0] = %.2f\n", h_C[0]);
    printf("Kernel time: %.6f seconds\n", milliseconds / 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}