00_base: 00_base.cpp
	g++ 00_base.cpp -O3
01_loop_order: 01_loop_order.cpp
	g++ 01_loop_order.cpp -O3
02_openmp: 02_openmp.cpp
	g++ 02_openmp.cpp -O3 -fopenmp
03_blocking: 03_blocking.cpp
	g++ 03_blocking.cpp -O3 -fopenmp
04_simd_blocking: 04_simd_blocking.cpp
	g++ 04_simd_blocking.cpp -O3 -fopenmp
05_sgemm: 05_sgemm.cpp
	g++ 05_sgemm.cpp -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5
06_base: 06_base.cu
	nvcc 06_base.cu -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp"
07_block: 07_block.cu
	nvcc 07_block.cu -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp"
08_grid: 08_grid.cu
	nvcc 08_grid.cu -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp"
09_shared: 09_shared.cu
	nvcc 09_shared.cu -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp"
10_sgemm: 10_sgemm.cu
	nvcc 10_sgemm.cu -O3 -Xcompiler "-O3 -fopenmp" -lcublas
11_cutlass: 11_cutlass.cu
	nvcc 11_cutlass.cu -arch=sm_60 -O3 -Xcompiler "-O3 -fopenmp"
