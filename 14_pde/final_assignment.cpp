//
//  main.cpp
//  final Assignment
//
//  Created by ZHANG HAOKE (22M31261) on 2022/06/07.
//

#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdio>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <immintrin.h>
using namespace std;

// For using CUDA function, first we should change the resource's format from ".cpp" into ".cu"
// Second we need to define the CUDA function. In the final assignment, I want to use CUDA functions to calculate the "for" loop in "Zeros" class.

// __global__ void calculate(int *m_strides, int *m_dims, int stride){
//	 int i = blockIdx.x * blockDim.x + threadIdx.x;
//	 m_strides[i] = stride;
//	 stride *= m_dims[i];
// }

template<typename T>
vector<float> linspace(T start_in, T end_in, int num_in)
{

  vector<float> linspaced;

  double start = static_cast<float>(start_in);
  double end = static_cast<float>(end_in);
  double num = static_cast<float>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1)
    {
      linspaced.push_back(start);
      return linspaced;
    }

  float delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
} // The operation for the same utilization like the numpy.linspace function in Python by using template form of C++

template <typename IntType>
vector<IntType> range(IntType start, IntType stop, IntType step)
{
  if (step == IntType(0))
  {
    throw invalid_argument("step for range must be non-zero");
  }

  vector<IntType> result;
  IntType i = start;
  while ((step > 0) ? (i < stop) : (i > stop))
  {
    result.push_back(i);
    i += step;
  }

  return result;
}

template <typename IntType>
vector<IntType> range(IntType start, IntType stop)
{
  return range(start, stop, IntType(1));
}

template <typename IntType>
vector<IntType> range(IntType stop)
{
  return range(IntType(0), stop, IntType(1));
}  // The operation for the same utilization like the numpy.range function in Python by using template form of C++

class Zeros {
    vector<size_t> m_dims, m_strides;
    unique_ptr<float[]> m_buf;

    public:
        Zeros(vector<size_t> dims):
            m_dims{move(dims)}
        {
            m_strides.resize(m_dims.size());
            size_t stride = 1;
            
            // The operation for using SMID to calculate vectors.
            // for (int i = m_dims.size() - 1; i >= 0; -- i) {
            //      __m256 dimsvec = _mm256_load_ps(m_dims[i]);
            //      __m256 mstridesvec = _mm256_load_ps(m_strides[i]);
            //      __m256 stridesvec = _mm256_load_ps(stride);
            //      _mm256_store_ps(m_strides[i], stridesvec);
            //      __m256 mulvec = _mm256_mul_ps(stridesvec, dimsvec);
            //      _mm256_store_ps(stride, stridesvec);
            // }
          
            // To use CUDA function defined at the beginning:
            // int range = m_dims.size();
            // int *temp_strides, *temp_dims;
            // cudaMallocManaged(&temp_strides, range*sizeof(int));
            // cudaMallocManaged(&temp_dims, range*sizeof(int));
            // for(int i = 0; i < range; i++){
            //     temp_strides[i] = m_strides[i];
            //     temp_dims[i] = m_dims[i];    
            // }
            // calculate<<<range, n/range>>>(temp_strides, temp_dims, stride);
            // cudaDeviceSynchronize();
            // cudaFree(temp_strides);
            // cudaFree(temp_dims);
          
            for (int i = m_dims.size() - 1; i >= 0; -- i) {
                m_strides[i] = stride;
                stride *= m_dims[i];
            }
            m_buf.reset(new float[stride]);
        }

    
        float& operator[] (initializer_list<size_t> idx) {
            size_t offset = 0;
            auto stride = m_strides.begin();
            
            #pragma acc parallel loop reduction(+:offset)
            for (int i = 0; i <  m_dims.size(); i++) {
                offset += i * *stride;
                ++stride;
            } // The parallel operation for using OpenACC
            return m_buf[offset];
        }
}; // The operation for the same utilization like the numpy.zeros function in Python by using Class form of C++

struct Body {
  float x, y;
};

int main(int argc, char** argv) {
    // insert code here...
    int nx = 41;
    int ny = 41;
    unsigned long n = 41;
    int nt = 500;
    int nit = 50;
    float dx = 2 / (nx - 1);
    float dy = 2 / (ny - 1);
    float dt = 0.01;
    int rho = 1;
    float nu = 0.02;
    Zeros X({n, n}), Y({n, n});
  
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Body jbody[n/size];
    
    auto x = linspace(0, 2, nx);
    auto y = linspace(0, 2, ny);
  
    for(int i=0; i<n/size; i++) {
      jbody[i].x = x[i];
      jbody[i].y = y[i];
  }
    int recv_from = (rank + 1) % size;
    int send_to = (rank - 1 + size) % size;
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
  
    for(int irank=0; irank<size; irank++) {
    MPI_Win win;
    MPI_Win_create(jbody, N/size*sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Put(jbody, N/size, MPI_BODY, send_to, 0, N/size, MPI_BODY, win);
    MPI_Win_fence(0, win);
    for(int i=0; i<N/size; i++) {
        unsigned long li = static_cast<int>(i);
      for(int j=0; j<N/size; j++) {
          unsigned long lj = static_cast<int>(j);
          X[{li, lj}] = jbody[j].x;
          Y[{lj, li}] = jbody[j].y;
      }
    }
  }   // The operation for the same utilization like the numpy.meshgrid function in Python by using MPI.
    
    Zeros u({n, n});
    Zeros v({n, n});
    Zeros p({n, n});
    Zeros b({n, n});
    
    for (int n : range(nt)) {
        #pragma omp parallel for private(i) private(j)
        for (int j = 1; j < ny; j++) {
            unsigned long lj = static_cast<int>(j);
            for (int i = 1; i < nx; i++) {
                unsigned long li = static_cast<int>(i);
                b[{lj, li}] = rho * (1 / dt * ((u[{lj, li+1}] - u[{lj, li-1}]) / (2 * dx) + (v[{lj+1, li}] - v[{lj-1, li}]) / (2 * dy)) -
                            pow(((u[{lj, li+1}] - u[{lj, li-1}]) / (2 * dx)), 2) - 2 * ((u[{lj+1, li}] - u[{lj-1, li}]) / (2 * dy) * (v[{lj, li+1}] - v[{lj, li-1}]) / (2 * dx)) - pow(((v[{lj+1, li}] - v[{lj-1, li}]) / (2 * dy)), 2));
            }
        }
    }
    for (int it : range(nit)) {
        Zeros&& pn = move(p);
        #pragma omp parallel for private(i) private(j)
        for (int j = 1; j != ny; ++j) {
            unsigned long lj = static_cast<int>(j);
            for (int i = 1; i != nx; ++i) {
                unsigned long li = static_cast<int>(i);
                p[{lj, li}] = (pow(dy, 2) * (pn[{lj, li+1}] + pn[{lj, li-1}]) +
                             pow(dx, 2) * (pn[{lj+1, li}] + pn[{lj-1, li}]) -
                             b[{lj, li}] * pow(dx, 2) * pow(dy, 2))
                / (2 * (pow(dx, 2) + pow(dy, 2)));
            }
        }
    }
    #pragma omp parallel for private(i)
    for (unsigned long i = 0; i < n; i++) {
        p[{n-1, i}] = p[{n-2, i}];
        p[{i, 0}] = p[{i, 1}];
        p[{0, i}] = p[{1, i}];
        p[{i, n-1}] = 0;
    }
  
    Zeros&& un = move(u);
    Zeros&& vn = move(v);
    
    #pragma omp parallel for private(i) private(j)
    for (unsigned long j = 1; j != n; ++j) {
        for (unsigned long i = 1; i != n; ++i) {
            u[{j, i}] = un[{j, i}] - un[{j, i}] * dt / dx * (un[{j, i}] - un[{j, i - 1}])
            - un[{j, i}] * dt / dy * (un[{j, i}] - un[{j - 1, i}])
            - dt / (2 * rho * dx) * (p[{j, i+1}] - p[{j, i-1}])
            + nu * dt / pow(dx, 2) * (un[{j, i+1}] - 2 * un[{j, i}] + un[{j, i-1}])
            + nu * dt / pow(dy, 2) * (un[{j+1, i}] - 2 * un[{j, i}] + un[{j-1, i}]);
            
            v[{j, i}] = vn[{j, i}] - vn[{j, i}] * dt / dx * (vn[{j, i}] - vn[{j, i - 1}])
            - vn[{j, i}] * dt / dy * (vn[{j, i}] - vn[{j - 1, i}])
            - dt / (2 * rho * dx) * (p[{j+1, i}] - p[{j-1, i}])
            + nu * dt / pow(dx, 2) * (vn[{j, i+1}] - 2 * vn[{j, i}] + vn[{j, i-1}])
            + nu * dt / pow(dy, 2) * (vn[{j+1, i}] - 2 * vn[{j, i}] + vn[{j-1, i}]);
        }
    }
    #pragma omp parallel for private(i)
    for (int i = 0; i < n; i++) {
        u[{i, 0}] = 0;
        u[{0, i}] = 0;
        u[{n-1, i}] = 0;
        u[{i, n-1}] = 1;
        v[{i, 0}] = 0;
        v[{i, n-1}] = 0;
        v[{0, i}] = 0;
        v[{n-1, i}] = 0;
    }
    return 0;
}
