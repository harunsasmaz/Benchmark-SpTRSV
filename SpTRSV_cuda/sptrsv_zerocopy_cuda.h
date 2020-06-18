#ifndef _SPTRSV_ZEROCOPY_CUDA_
#define _SPTRSV_ZEROCOPY_CUDA_

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
void sptrsv_zerocopy_cuda_analyser(const int   *d_cscRowIdx,
                                   const int    m,
                                   const int    nnz,
                                   const int    displs,
                                         int   *s_in_degree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&s_in_degree[d_cscRowIdx[global_id]], 1);
    }

}

__global__
void sptrsv_zerocopy_cuda_executor(const int* __restrict__        d_cscColPtr,
                                   const int* __restrict__        d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int*                     d_in_degree,
                                         VALUE_TYPE*              d_left_sum,
                                   const int                      n,
                                   const int                      displs,
                                   const int                      e_displs,
                                   const int                      substitution,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     s_in_degree,
                                         VALUE_TYPE*              s_left_sum)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_x_id = global_id / WARP_SIZE;
    if (local_x_id >= n) return;
 
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int starting_x = displs;
    starting_x = substitution == SUBSTITUTION_FORWARD ? 
                  starting_x : n - 1 + starting_x;
    
    int global_x_id = local_x_id;
    global_x_id = substitution == SUBSTITUTION_FORWARD ? 
                    global_x_id + starting_x : starting_x - global_x_id;

    clock_t start;
    do {
        start = clock();
        __syncwarp();
    }
    while (s_in_degree[global_x_id] != d_in_degree[local_x_id] + 1);
    
    const int index = substitution == SUBSTITUTION_FORWARD ? local_x_id : n - 1 - local_x_id;
    const int pos = substitution == SUBSTITUTION_FORWARD ?
                    d_cscColPtr[index] : d_cscColPtr[index + 1] - 1;
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos - e_displs];
    VALUE_TYPE xi = d_left_sum[local_x_id] + s_left_sum[global_x_id];
    xi = (d_b[index] - xi) * coef;

    const int start_ptr = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[index] + 1 : d_cscColPtr[index];
    const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ? 
                          d_cscColPtr[index + 1] : d_cscColPtr[index + 1] - 1;

    for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)
    {   
        const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);
        const int rowIdx = d_cscRowIdx[j - e_displs];
        const bool cond = substitution == SUBSTITUTION_FORWARD ? 
                    (rowIdx < starting_x + n) : (rowIdx > starting_x - n);

        if (cond) {
            const int pos = substitution == SUBSTITUTION_FORWARD ? 
                            rowIdx - starting_x : starting_x - rowIdx;

            atomicAdd((VALUE_TYPE*)&d_left_sum[pos], xi * d_cscVal[j - e_displs]);
            __threadfence();
            atomicAdd((int*)&d_in_degree[pos], 1);
        }
        else {
            atomicAdd((VALUE_TYPE*)&s_left_sum[rowIdx], xi * d_cscVal[j - e_displs]);
            __threadfence();
            atomicSub((int*)&s_in_degree[rowIdx], 1);
        }
    }

    if (!lane_id) d_x[index] = xi;
}

int sptrsv_zerocopy_cuda(const int           *cscColPtrTR,
                         const int           *cscRowIdxTR,
                         const VALUE_TYPE    *cscValTR,
                         const int            m,
                         const int            n,
                         const int            nnzTR,
                         const int            substitution,
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
    // almost even col distribution
    int colCounts[ngpu], colDispls[ngpu];
    int colCount = round((double)n / ngpu);
    for(int i = 0; i < ngpu - 1; i++)
    {
        colCounts[i] = colCount;
        colDispls[i] = i * colCount;
    }
    colCounts[ngpu - 1] = n - (ngpu - 1) * colCount;
    colDispls[ngpu - 1] = (ngpu - 1) * colCount;

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

    //  Define device in_degree and left_sum arrays for each device
    int *d_in_degree[ngpu];
    VALUE_TYPE *d_left_sum[ngpu];

    // Allocate device in_degree and left_sum arrays for each device
    for(int i = 0; i < ngpu; i++)
    {
        int cols = colCounts[i];
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMalloc((void **)&d_in_degree[i], cols * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void **)&d_left_sum[i], cols * sizeof(VALUE_TYPE)) )
        CHECK_CUDA( cudaMemset(d_left_sum[i], 0, cols * sizeof(VALUE_TYPE)) )
        CHECK_CUDA( cudaMemset(d_in_degree[i], 0, cols * sizeof(int)) )
    }

    // For safety, wait until all allocations are done.
    for(int i = 0; i < ngpu; i++)
    {
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaDeviceSynchronize() )
    }

    //  - cuda zercopu SpTRSV analysis start!
    printf(" - cuda zerocopy SpTRSV analysis start!\n");

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
            sptrsv_zerocopy_cuda_analyser<<< num_blocks, num_threads >>>
                                        (d_cscRowIdxTR[i], colCounts[i], eCounts[i], eDispls[i], s_in_degree);
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

    printf("cuda zerocopy SpTRSV analysis on L used %4.2f ms\n", time_cuda_analysis);

    //  - cuda syncfree SpTRSV solve start!
    printf(" - cuda zerocopy SpTRSV solve start!\n");

    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    double time_cuda_solve = 0;

    // For benchmarking, backup intermediate in degree array
    int *s_in_degree_backup;
    CHECK_CUDA( cudaMallocManaged((void**)&s_in_degree_backup, n * sizeof(int)) )
    memcpy(s_in_degree_backup, s_in_degree, n * sizeof(int));

    // SOLVE KERNEL
    for(int k = 0; k < BENCH_REPEAT; k++)
    {
        CHECK_CUDA( cudaMemset(s_left_sum, 0, n * sizeof(VALUE_TYPE)) );
        memcpy(s_in_degree, s_in_degree_backup, n * sizeof(int));

        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            CHECK_CUDA( cudaMemset(d_x[i], 0, colCounts[i] * sizeof(VALUE_TYPE)) )
            CHECK_CUDA( cudaMemset(d_in_degree[i], 0, colCounts[i] * sizeof(int)) )
            CHECK_CUDA( cudaMemset(d_left_sum[i], 0, colCounts[i] * sizeof(VALUE_TYPE)) )
        }

        gettimeofday(&t1, NULL);

        for(int i = 0; i < ngpu; i++)
        {
            cudaSetDevice(i);
            num_blocks = ceil((double)colCounts[i] / ((double)num_threads/WARP_SIZE) );
            sptrsv_zerocopy_cuda_executor<<< num_blocks, num_threads >>>
                                (d_cscColPtrTR[i], d_cscRowIdxTR[i], d_cscValTR[i],
                                    d_in_degree[i], d_left_sum[i], colCounts[i], colDispls[i], 
                                        eDispls[i], substitution, d_b[i], d_x[i], s_in_degree, s_left_sum);
        }

        for(int i = 0; i < ngpu; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        
        gettimeofday(&t2, NULL);
        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_solve /= BENCH_REPEAT;
    double flop = 2*(double)nnzTR;

    printf("cuda zerocopy SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
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
        printf("cuda zerocopy SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("cuda zerocopy SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    for(int i = 0; i < ngpu; i++)
    {   
        cudaSetDevice(i);
        cudaFree(d_in_degree[i]);
        cudaFree(d_left_sum[i]);
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