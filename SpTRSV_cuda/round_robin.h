#ifndef _ROUND_ROBIN_H
#define _ROUND_ROBIN_H

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

int sum_elms(int elms[], int n)
{   
    int sum = 0;
    for(int i = 0; i < n; ++i)
    {
        sum += elms[i];
    }
    return sum;
}

__global__
void printk(double* ptr, double* qtr, int n)
{
    for(int i = 0; i < n; ++i)
    {
        printf("%d: %d -- %d\n", i, ptr[i], qtr[i]);
    }
}

__global__
void round_robin_analyser(         const int   *d_cscRowIdx,
                                   const int    nnz,
                                         int   *s_in_degree)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x; //get_global_id(0);
    if (global_id < nnz)
    {
        atomicAdd(&s_in_degree[d_cscRowIdx[global_id]], 1);
    }

}

__global__
void round_robin_executor(         const int*        __restrict__ d_cscColPtr,
                                   const int*        __restrict__ d_cscRowIdx,
                                   const VALUE_TYPE* __restrict__ d_cscVal,
                                         int                      n,
                                         int*                     c_displs,
                                         int*                     e_displs,
                                         int*                     d_displs,
                                   const VALUE_TYPE* __restrict__ d_b,
                                         VALUE_TYPE*              d_x,
                                         int*                     s_in_degree,
                                         VALUE_TYPE*              s_left_sum)
{   
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_x_id = global_id / WARP_SIZE;
    if(local_x_id >= n) return;

    const int global_x_id = local_x_id + local_x_id / WARP_PER_BLOCK;

    volatile __shared__ int x_in_degree[WARP_PER_BLOCK];
    volatile __shared__ VALUE_TYPE x_left_sum[WARP_PER_BLOCK];

    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    int starting_x = c_displs[blockIdx.x];

    const int pos = d_cscColPtr[global_x_id] - e_displs[blockIdx.x] + d_displs[blockIdx.x];
    const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];

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
    while(x_in_degree[local_warp_id] != s_in_degree[local_warp_id + starting_x]);

    VALUE_TYPE xi = s_left_sum[local_warp_id + starting_x] + x_left_sum[local_warp_id];
    xi = (d_b[local_x_id] - xi) * coef;

    const int start_ptr = d_cscColPtr[global_x_id] + 1;
    const int stop_ptr  = d_cscColPtr[global_x_id + 1];

    for(int j = start_ptr + lane_id; j < stop_ptr; j += WARP_SIZE)
    {
        const int row = d_cscRowIdx[j - e_displs[blockIdx.x] + d_displs[blockIdx.x]];
        const bool cond = row < starting_x + WARP_PER_BLOCK;

        if(cond)
        {   
            const int index = row - starting_x;
            atomicAdd((VALUE_TYPE *)&x_left_sum[index], xi * d_cscVal[j - e_displs[blockIdx.x] + d_displs[blockIdx.x]]);
            __threadfence_block();
            atomicAdd((int *)&x_in_degree[index], 1);
        } else {
            atomicAdd(&s_left_sum[row], xi * d_cscVal[j - e_displs[blockIdx.x] + d_displs[blockIdx.x]]);
            __threadfence();
            atomicSub(&s_in_degree[row], 1);
        }
    }

    if(!lane_id) d_x[local_x_id] = xi;
}

int round_robin(         const int           *cscColPtrTR,
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

    int rr_times = ceil((double)n/WARP_PER_BLOCK);          // the number of rounds
    int loop_count = rr_times / ngpu;                       // the number of complete loops for each device
    int remainder = rr_times % ngpu;                        // how many gpu has an additional loop
    int last_round = n - (rr_times - 1) * WARP_PER_BLOCK;   // last round gets the remaining cols.
    
    // Allocate displacement holder pointers
    int colCounts[ngpu], *colDispls[ngpu];
    int eCounts[ngpu][loop_count + 1], *eDispls[ngpu];
    for(int i = 0; i < ngpu; ++i)
    {
        colDispls[i] = (int*)malloc((loop_count + 1) * sizeof(int));
        eDispls[i] = (int*)malloc((loop_count + 1) * sizeof(int));
    }

    // calculate column count and its displacements of each device
    int col = WARP_PER_BLOCK, dev_id = 0, loop = 0;
    for(int i = 0; i < rr_times; ++i)
    {
        if(i % ngpu == 0 && i > 0) loop++;
        if(i == rr_times - 1) col = last_round;
        dev_id = i % ngpu;
        if(loop == 0) colCounts[dev_id] = 0;
        colCounts[dev_id] += col;
        colDispls[dev_id][loop] = i * WARP_PER_BLOCK;
    }

    // calculate elm count and its displacements of each device
    col = WARP_PER_BLOCK; dev_id = 0; loop = 0;
    for(int i = 0; i < rr_times; ++i)
    {
        if(i % ngpu == 0 && i > 0) loop++;
        if(i == rr_times - 1) col = last_round;
        dev_id = i % ngpu;
        eCounts[dev_id][loop] = cscColPtrTR[colDispls[dev_id][loop] + col] - cscColPtrTR[colDispls[dev_id][loop]];
        eDispls[dev_id][loop] = cscColPtrTR[colDispls[dev_id][loop]];
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

    // Allocate partial device arrays for matrix L and vector b for devices
    for(int i = 0; i < ngpu; i++)
    {   
        int offset = remainder == i + 1 ? loop_count + 1 : loop_count;
        int elms = sum_elms(eCounts[i], offset);

        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMalloc((void **)&d_cscColPtrTR[i], (colCounts[i] + offset) * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void **)&d_cscRowIdxTR[i], elms * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void **)&d_cscValTR[i],    elms * sizeof(VALUE_TYPE)) )
        CHECK_CUDA( cudaMalloc((void **)&d_b[i],           colCounts[i] * sizeof(VALUE_TYPE)) )
    }

    // Distribute values of matrix L and vector B from host to devices
    int total_e[ngpu], e_count, e_start;
    col = WARP_PER_BLOCK; dev_id = 0; loop = 0;
    int e_pos[ngpu][loop_count + 1]; int col_pos[ngpu];
    for(int i = 0; i < rr_times; ++i)
    {
        if(i % ngpu == 0 && i > 0) loop++;
        if(i == rr_times - 1) col = last_round;

        dev_id = i % ngpu; 
        e_count = eCounts[dev_id][loop]; e_start = eDispls[dev_id][loop];

        if(loop == 0) 
        {
            e_pos[dev_id][loop] = 0; 
            col_pos[dev_id] = 0;
            total_e[dev_id] = 0;
        }

        CHECK_CUDA( cudaSetDevice(dev_id) )

        CHECK_CUDA( cudaMemcpy(d_cscRowIdxTR[dev_id] + total_e[dev_id], cscRowIdxTR + e_start, 
                        e_count * sizeof(int), cudaMemcpyHostToDevice) )
        
        CHECK_CUDA( cudaMemcpy(d_cscValTR[dev_id] + total_e[dev_id], cscValTR + e_start, 
                        e_count * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice) )
        
        CHECK_CUDA( cudaMemcpy(d_cscColPtrTR[dev_id] + col_pos[dev_id], cscColPtrTR + colDispls[dev_id][loop], 
                        (col + 1) * sizeof(int), cudaMemcpyHostToDevice) )

        CHECK_CUDA( cudaMemcpy(d_b[dev_id] + col_pos[dev_id] - loop, b + colDispls[dev_id][loop], 
                        col * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice) )

        if(loop > 0) e_pos[dev_id][loop] = eCounts[dev_id][loop - 1] + e_pos[dev_id][loop - 1];
        col_pos[dev_id] += col + 1;
        total_e[dev_id] += e_count;
    }

    // Allocate and set vector X on devices
    for(int i = 0; i < ngpu; i++)
    {   
        int cols = colCounts[i];
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMalloc((void **)&d_x[i], cols * sizeof(VALUE_TYPE)) )
        CHECK_CUDA( cudaMemset(d_x[i], 0, cols * sizeof(VALUE_TYPE)) )
    }

    // Allocate and copy col and elm displacement arrays for each device.
    int* d_colDispls[ngpu], *d_eDispls[ngpu], *d_ePos[ngpu];
    for(int i = 0; i < ngpu; ++i)
    {
        int offset = remainder == i + 1 ? loop_count + 1 : loop_count;
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaMalloc((void**)&d_colDispls[i], offset * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void**)&d_eDispls[i], offset * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void**)&d_ePos[i], offset * sizeof(int)) )
        CHECK_CUDA( cudaMemcpy(d_colDispls[i], colDispls[i], offset * sizeof(int), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(d_eDispls[i], eDispls[i], offset * sizeof(int), cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(d_ePos[i], e_pos[i], offset * sizeof(int), cudaMemcpyHostToDevice) )
    }

    // For safety, wait until all allocations are done.
    for(int i = 0; i < ngpu; i++)
    {
        CHECK_CUDA( cudaSetDevice(i) )
        CHECK_CUDA( cudaDeviceSynchronize() )
    }

    // ANALYSIS KERNEL
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    int num_threads = 128;
    int num_blocks;
    double time_cuda_analysis = 0.0;

    int elms[ngpu], offset = 0;
    for(int i = 0; i < ngpu; ++i)
    {
        offset = remainder == i + 1 ? loop_count + 1 : loop_count;
        elms[i] = sum_elms(eCounts[i], offset);
    }

    for(int k = 0; k < BENCH_REPEAT; k++)
    {
        CHECK_CUDA( cudaMemset(s_in_degree, 0, n * sizeof(int)) )

        gettimeofday(&t1, NULL);

        for(int i = 0; i < ngpu; i++)
        {   
            CHECK_CUDA( cudaSetDevice(i) )
            num_blocks = ceil((double)elms[i] / num_threads);
            round_robin_analyser<<< num_blocks, num_threads >>>
                                        (d_cscRowIdxTR[i], elms[i], s_in_degree);
        }

        // For safety, wait until are devices are done with analysis.
        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            CHECK_CUDA( cudaDeviceSynchronize() )
        }

        gettimeofday(&t2, NULL);
        time_cuda_analysis += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    time_cuda_analysis /= BENCH_REPEAT;
    printf("round robin SpTRSV analysis on L used %4.2f ms\n", time_cuda_analysis);

    //  - round robin SpTRSV solve start!
    num_threads = WARP_PER_BLOCK * WARP_SIZE;
    double time_cuda_solve = 0;

    // For benchmarking, backup intermediate in_degree array
    int *s_in_degree_backup;
    CHECK_CUDA( cudaMallocManaged((void**)&s_in_degree_backup, n * sizeof(int)) )
    memcpy(s_in_degree_backup, s_in_degree, n * sizeof(int));

    // SOLVE KERNEL
    for(int k = 0; k < 1; k++)
    {
        CHECK_CUDA( cudaMemset(s_left_sum, 0, n * sizeof(VALUE_TYPE)) );
        memcpy(s_in_degree, s_in_degree_backup, n * sizeof(int));

        gettimeofday(&t1, NULL);

        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            num_blocks = remainder == i + 1 ? loop_count + 1 : loop_count;
            round_robin_executor<<< num_blocks, num_threads >>>
                                (d_cscColPtrTR[i], d_cscRowIdxTR[i], d_cscValTR[i],
                                    colCounts[i], d_colDispls[i], d_eDispls[i], d_ePos[i],
                                        d_b[i], d_x[i], s_in_degree, s_left_sum);
        }

        for(int i = 0; i < ngpu; i++)
        {
            CHECK_CUDA( cudaSetDevice(i) )
            CHECK_CUDA( cudaDeviceSynchronize() )
        }
        
        gettimeofday(&t2, NULL);
        time_cuda_solve += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }

    //time_cuda_solve /= BENCH_REPEAT;
    double flop = 2*(double)nnzTR;

    printf("round robin SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_cuda_solve, flop/(1e6*time_cuda_solve));

    // Gather partial device x vectors on host
    int pos;
    col = WARP_PER_BLOCK; dev_id = 0; loop = 0;
    for(int i = 0; i < rr_times; ++i)
    {
        if(i % ngpu == 0 && i > 0) loop++;
        if(i == rr_times - 1) col = last_round;
        dev_id = i % ngpu;
        pos = WARP_PER_BLOCK * loop;
        CHECK_CUDA( cudaSetDevice(dev_id) )
        CHECK_CUDA( cudaMemcpy(x + colDispls[dev_id][loop], d_x[dev_id] + pos, 
                        col * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost) )
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
        printf("round robin SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("round robin SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    for(int i = 0; i < ngpu; i++)
    {   
        cudaSetDevice(i);
        cudaFree(d_colDispls[i]);
        cudaFree(d_eDispls[i]);
        cudaFree(d_ePos[i]);
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