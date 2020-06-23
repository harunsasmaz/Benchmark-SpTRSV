# Benchmark-SpTRSV

## Changes:

  - sptrsv_zerocopy_cuda.h: related unified memory algorithm.
  - unified_and_shared.h: uses shared memory not global memory
  - unified_and_shared_2.h: uses shared memory and global memory together.
  - utils.h: matrix partition, equal col and nnz, added.
  - main.cu: new command line parameter <-p> defines partition, either equal column or equal nnz.

## Note

  - new newly added three .h files are the same except executor kernel.
 
## Compile and Run

  - make
  - ./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx <matrix_file> 
    - <-p> 0 for equal column, 1 for equal nnz
