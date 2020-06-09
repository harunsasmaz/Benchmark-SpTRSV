# Benchmark_SpTRSM_using_CSC
Fast Synchronization-Free Algorithms for Parallel Sparse Triangular Solves with Multiple Right-Hand Sides (SpTRSM)
<br><hr>
<h3>Introduction</h3>

This is the source code of a paper entitled "Fast Synchronization-Free Algorithms for Parallel Sparse Triangular Solves with Multiple Right-Hand Sides", Concurrency and Computation: Practice and Experience, 2017, by Weifeng Liu, Ang Li, Jonathan D. Hogg, Iain S. Duff, and Brian Vinter. [[PDF](http://www.nbi.dk/~weifeng/papers/sptrsm_liu_ccpe.pdf)] [[DOI](https://doi.org/10.1002/cpe.4244)].

The code supports both forward and backward substitution and multiple right-hand sides.

Please contact [Weifeng Liu](http://www.nbi.dk/~weifeng/) for reporting any issues in the code.

Update (14 Feb. 2020, cuda): A problem about deadlock on CUDA 10 has been fixed.

Update (13 Feb. 2017, cuda): A problem about caching has been fixed for Tesla P100. Thanks to Hartwig Anzt for identifying the probem and Ang Li for fixing it!

<br><hr>
<h3>nVidia GPU (CUDA) version</h3>

- Execution

1. Set CUDA path in the Makefile,
2. Run ``make``,
3. Run ``./sptrsv -d 0 -rhs 2 -forward -mtx example.mtx``. Here changable parameters `0` and `2` refer to device id and the number of right-hand sides, respectively. When `-rhs` is set to `1`, the operation is SpTRSV, otherwise SpTRSM. The `-forward` (for solving lower triangular part of the input .mtx matrix) can be replaced by `-backward` (for solving its upper triangular part).

- Tested environments

1. nVidia GeForce Titan X (Pascal) GPU in a host with CUDA v8.0 and CentOS 7.2 64-bit Linux installed.
2. nVidia GeForce GTX 1080 GPU in a host with CUDA v8.0 and CentOS 7.2 64-bit Linux installed.
3. nVidia Geforce GT 650m GPU in a host with CUDA v7.5 and Mac OS X 10.9.2 installed.

- Data type

1. The code supports both double precision and single precision SpTRSV and SpTRSM. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision. (Note that for CUDA devices older than Pascal and CUDA SDKs older v8.0, lines 16-31 of file `utils.h` should be uncommented for double precision support.)

<br><hr>
<h3>AMD GPU (OpenCL 2.0) version</h3>

- Execution

1. Set OpenCL path in the Makefile,
2. Run ``make``,
3. Run ``./sptrsv -d 0 -rhs 2 -forward -mtx example.mtx``. Here changable parameters `0` and `2` refer to device id and the number of right-hand sides, respectively. When `-rhs` is set to `1`, the operation is SpTRSV, otherwise SpTRSM. The `-forward` (for solving lower triangular part of the input .mtx matrix) can be replaced by `-backward` (for solving its upper triangular part).

- Tested environments (Note that an OpenCL 2.0 device is required for running the code)

1. AMD Radeon Fury X GPU in a host with AMD APP SDK 3.0 and Ubuntu 15.04 64-bit Linux installed.
2. AMD Radeon 290X GPU in a host with AMD APP SDK 3.0 and Ubuntu 15.04 64-bit Linux installed.

- Data type

1. The code supports both double precision and single precision SpTRSV and SpTRSM. Use ``make VALUE_TYPE=double`` for double precision or ``make VALUE_TYPE=float`` for single precision. 
# Benchmark-SpTRSV
