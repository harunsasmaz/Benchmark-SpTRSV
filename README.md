# Benchmark-SpTRSV

## Changes:

  - sptrsv_zerocopy_cuda.h: related unified memory algorithms.
  - main.cu: new command line parameter <-p> defines partition, either equal column or equal nnz.
 
## Compile and Run

  - make
  - ./sptrsv -d 0 -rhs 1 -forward -p 0 -mtx <matrix> 
    - <-p> 0 for equal column, 1 for equal nnz
