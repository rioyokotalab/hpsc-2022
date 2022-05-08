00_base: 00_base.cpp
	g++ 00_base.cpp -O3
01_loop_order: 01_loop_order.cpp
	g++ 01_loop_order.cpp -O3
02_openmp: 02_openmp.cpp
	g++ 02_openmp.cpp -O3 -fopenmp
03_blocking: 03_blocking.cpp
	g++ 03_blocking.cpp -O3 -fopenmp
04_simd_blocking: 04_simd_blocking.cpp
	g++ 04_simd_blocking.cpp -O3 -fopenmp -march=native
05_sgemm: 05_sgemm.cpp
	g++ 05_sgemm.cpp -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5
06_base: 06_base.cu
	nvcc 06_base.cu -Xcompiler "-O3 -fopenmp"
07_block: 07_block.cu
	nvcc 07_block.cu -Xcompiler "-O3 -fopenmp"
08_grid: 08_grid.cu
	nvcc 08_grid.cu -Xcompiler "-O3 -fopenmp"
09_shared: 09_shared.cu
	nvcc 09_shared.cu -Xcompiler "-O3 -fopenmp"
10_sgemm: 10_sgemm.cu
	nvcc 10_sgemm.cu -Xcompiler "-O3 -fopenmp" -lcublas
11_vector_4x2: 11_vector_4x2.cu
	nvcc 11_vector_4x2.cu -Xcompiler "-O3 -fopenmp" -lcublas
12_warp: 12_warp.cu
	nvcc 12_warp.cu -Xcompiler "-O3 -fopenmp" -lcublas
13_align: 13_align.cu
	nvcc 13_align.cu -Xcompiler "-O3 -fopenmp" -lcublas
14_vector: 14_vector.cu
	nvcc 14_vector.cu -Xcompiler "-O3 -fopenmp" -lcublas
