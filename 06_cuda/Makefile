00_hello: 00_hello.cu
	nvcc 00_hello.cu
01_thread: 01_thread.cu
	nvcc 01_thread.cu
02_block: 02_block.cu
	nvcc 02_block.cu
03_if: 03_if.cu
	nvcc 03_if.cu
04_atomic: 04_atomic.cu
	nvcc 04_atomic.cu
05_reduction: 05_reduction.cu
	nvcc 05_reduction.cu
06_shared: 06_shared.cu
	nvcc 06_shared.cu
07_warp: 07_warp.cu
	nvcc 07_warp.cu
08_scan: 08_scan.cu
	nvcc 08_scan.cu
09_cooperative: 09_cooperative.cu
	nvcc 09_cooperative.cu -arch=sm_60 -rdc=true
10_mpi: 10_mpi.cu
	nvcc 10_mpi.cu -lmpi
11_hip: 11_hip.cpp
	hipcc 11_hip.cpp
12_bucket_sort: 12_bucket_sort.cpp
	g++ 12_bucket_sort.cpp
