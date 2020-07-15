#ifndef _UNIFIED_AND_SHARED_H
#define _UNIFIED_AND_SHARED_H

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

__global__
void unified_and_shared_analyser(  const int   *d_cscRowIdx,
                                   const int    nnz,
                                         int   *s_in_degree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id < nnz)
    {
        atomicAdd(&s_in_degree[d_cscRowIdx[global_id]], 1);
    }

}

__global__
void unified_and_shared_executor(  const int*        __restrict__ d_cscColPtr,
                                   const int*        __restrict__ d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                   const int                      n,
                                   const int                      displs,
                                   const int                      e_displs,
                                   const int                      substitution,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     s_in_degree,
                                         VALUE_TYPE*              s_left_sum)
{    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_x_id = global_id / WARP_SIZE;
    if(global_x_id >= n) return;

    volatile __shared__ int x_in_degree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE x_left_sum[WARP_PER_BLOCK];

    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int starting_x = blockIdx.x * WARP_PER_BLOCK + displs;

    const int pos = d_cscColPtr[global_x_id];
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos - e_displs];

    if(threadIdx.x < WARP_PER_BLOCK)
    {
        x_in_degree[threadIdx.x] = 1;
        x_left_sum[threadIdx.x] = 0;
    }
    __syncthreads();
    
    clock_t start;
    do
    {   
        start = clock();
        __syncwarp();
    }
    while(x_in_degree[local_warp_id] != s_in_degree[global_x_id + displs]);

    VALUE_TYPE xi = s_left_sum[global_x_id + displs] + x_left_sum[local_warp_id];
    xi = (d_b[global_x_id] - xi) * coef;

    const int start_ptr = d_cscColPtr[global_x_id] + 1;
    const int stop_ptr  = d_cscColPtr[global_x_id + 1];

    for(int j = start_ptr + lane_id; j < stop_ptr; j += WARP_SIZE)
    {
        const int row = d_cscRowIdx[j - e_displs];
        const bool cond = row < n + displs && row < starting_x + WARP_PER_BLOCK;

        if(cond)
        {   
            const int index = row - starting_x;
            atomicAdd((VALUE_TYPE *)&x_left_sum[index], xi * d_cscVal[j - e_displs]);
            __threadfence_block();
            atomicAdd((int *)&x_in_degree[index], 1);
            
        } else {
            atomicAdd(&s_left_sum[row], xi * d_cscVal[j - e_displs]);
            __threadfence();
            atomicSub(&s_in_degree[row], 1);
        }
    }

    if(!lane_id) d_x[global_x_id] = xi;
}

int unified_and_shared  (const int           *cscColPtrTR,
                         const int           *cscRowIdxTR,
                         const VALUE_TYPE    *cscValTR,
                         const int            m,
                         const int            n,
                         const int            nnzTR,
                         const int            substitution,
                         const int            partition,
                               VALUE_TYPE    *x,
                         const VALUE_TYPE    *b,
                         const VALUE_TYPE    *x_ref,
                         const int            ngpu)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }
    
    // calculate which GPU gets which cols
    int colCounts[ngpu], colDispls[ngpu];
    colDispls[0] = 0; colCounts[0] = 0;
    if(partition == COL_PARTITION)
        equal_col_partition(ngpu, n, colCounts, colDispls);
    else if(partition == NNZ_PARTITION)
        equal_nnz_partition(ngpu, n, nnzTR, cscColPtrTR, colCounts, colDispls);
    else {
        printf("Unknown partition of matrix\n");
        return EXIT_FAILURE;
    }
        
    // calculate which GPU gets how many nnz
    int eCounts[ngpu], eDispls[ngpu];
    for(int i = 0; i < ngpu; i++)
    {
        eCounts[i] = cscColPtrTR[colDispls[i] + colCounts[i]] - cscColPtrTR[colDispls[i]];
        eDispls[i] = cscColPtrTR[colDispls[i]];
    }

    // initialize intermediate arrays
    VALUE_TYPE* s_left_sum;
    int *s_in_degree;
    CHECK_CUDA( cudaMallocManaged((void**)&s_left_sum, n * sizeof(VALUE_TYPE)) )
    CHECK_CUDA( cudaMallocManaged((void**)&s_in_degree, n * sizeof(int)) )
    CHECK_CUDA( cudaMemset(s_left_sum, 0, n * sizeof(VALUE_TYPE)) )
    CHECK_CUDA( cudaMemset(s_in_degree, 0, n * sizeof(int)) )

    // Define partial device arrays for matrix L and vectors X and B
    int *d_cscColPtrTR[ngpu];
    int *d_cscRowIdxTR[ngpu];
    VALUE_TYPE *d_cscValTR[ngpu];
    VALUE_TYPE *d_b[ngpu];
    VALUE_TYPE *d_x[ngpu];

    // Allocate partial device arrays for matrix L on devices
    for(int i = 0; i < ngpu; i++)
    {
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaMalloc((void **)&d_cscColPtrTR[i], (colCounts[i] + 1) * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void **)&d_cscRowIdxTR[i], eCounts[i]  * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void **)&d_cscValTR[i],    eCounts[i]  * sizeof(VALUE_TYPE)) )
    }

    // Distribute values of L from host to devices
    for(int i = 0; i < ngpu; i++)
    {   
        int pos = colDispls[i], cols = colCounts[i], e_pos = eDispls[i], elms = eCounts[i];
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMemcpy(d_cscColPtrTR[i], cscColPtrTR + pos, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(d_cscRowIdxTR[i], cscRowIdxTR + e_pos, elms * sizeof(int), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(d_cscValTR[i],  cscValTR + e_pos, elms * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice) )
    }

    // Allocate and distribute vector B on devices
    for(int i = 0; i < ngpu; i++)
    {   
        int pos = colDispls[i], cols = colCounts[i];
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMalloc((void **)&d_b[i], cols * sizeof(VALUE_TYPE)) )
        CHECK_CUDA( cudaMemcpy(d_b[i], b + pos, cols * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice) )
    }

    // Allocate and set vector X on devices
    for(int i = 0; i < ngpu; i++)
    {   
        int cols = colCounts[i];
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMalloc((void **)&d_x[i], cols * sizeof(VALUE_TYPE)) )
        CHECK_CUDA( cudaMemset(d_x[i], 0, cols * sizeof(VALUE_TYPE)) )
    }

    // For safety, wait until all allocations are done.
    for(int i = 0; i < ngpu; i++)
    {
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaDeviceSynchronize() )
    }

    //  - cuda zercopu SpTRSV analysis start!
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    int num_threads = 128;
    int num_blocks;

    for(int k = 0; k < BENCH_REPEAT; k++)
    {
        CHECK_CUDA( cudaMemset(s_in_degree, 0, n * sizeof(int)) )
        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            num_blocks = ceil((double)eCounts[i] / num_threads);
            unified_and_shared_analyser<<< num_blocks, num_threads >>>
                                        (d_cscRowIdxTR[i], eCounts[i], s_in_degree);
        }

        // For safety, wait until are devices are done with analysis.
        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            CHECK_CUDA( cudaDeviceSynchronize() )
        }
    }

    gettimeofday(&t2, NULL);
    double time_cuda_analysis = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_analysis /= BENCH_REPEAT;

    printf("unified+shared SpTRSV analysis on L used %4.2f ms\n", time_cuda_analysis);

    //  - cuda syncfree SpTRSV solve start!
    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    double time_cuda_solve = 0;

    // For benchmarking, backup intermediate in_degree array
    int *s_in_degree_backup;
    CHECK_CUDA( cudaMallocManaged((void**)&s_in_degree_backup, n * sizeof(int)) )
    memcpy(s_in_degree_backup, s_in_degree, n * sizeof(int));

    // SOLVE KERNEL
    for(int k = 0; k < 100; k++)
    {
        CHECK_CUDA( cudaMemset(s_left_sum, 0, n * sizeof(VALUE_TYPE)) );
        memcpy(s_in_degree, s_in_degree_backup, n * sizeof(int));

        //reset device arrays for the iteration.
        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            CHECK_CUDA( cudaMemset(d_x[i], 0, colCounts[i] * sizeof(VALUE_TYPE)) )
        }

        gettimeofday(&t1, NULL);

        for(int i = 0; i < ngpu; i++)
        {
            cudaSetDevice(i);
            num_blocks = ceil((double)colCounts[i] / ((double)num_threads/WARP_SIZE) );
            unified_and_shared_executor<<< num_blocks, num_threads >>>
                                (d_cscColPtrTR[i], d_cscRowIdxTR[i], d_cscValTR[i],
                                    colCounts[i], colDispls[i], eDispls[i], substitution, 
                                        d_b[i], d_x[i], s_in_degree, s_left_sum);
        }

        for(int i = 0; i < ngpu; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        
        gettimeofday(&t2, NULL);
        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_solve /= 100;
    double flop = 2*(double)nnzTR;

    printf("unified+shared SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, flop/(1e6*time_cuda_solve));

    // Gather partial device x vectors on host
    for(int i = 0; i < ngpu; i++)
    {
        int pos = colDispls[i], cols = colCounts[i];
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMemcpy(x + pos, d_x[i], cols * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost) )
    }

    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("unified+shared SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("unified+shared SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    for(int i = 0; i < ngpu; i++)
    {   
        cudaSetDevice(i);
        cudaFree(d_cscColPtrTR[i]);
        cudaFree(d_cscRowIdxTR[i]);
        cudaFree(d_cscValTR[i]);
        cudaFree(d_b[i]);
        cudaFree(d_x[i]);
    }

    cudaFree(s_in_degree);
    cudaFree(s_left_sum);
    cudaFree(s_in_degree_backup);

    return 0;
}

#endif