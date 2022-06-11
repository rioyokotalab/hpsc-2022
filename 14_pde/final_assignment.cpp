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
    vector<int> m_dims, m_strides;
    unique_ptr<float[]> m_buf;

    public:
        Zeros(vector<int> dims):
            m_dims{move(dims)}
        {
            m_strides.resize(m_dims.size());
            int stride = 1;
            
            // The operation for using SMID to calculate vectors.
            // for (int i = m_dims.size() - 1; i >= 0; -- i) {
          //      __m256 dimsvec = _mm256_load_ps(m_dims[i]);
          //      __m256 mstridesvec = _mm256_load_ps(m_strides[i]);
          //      __m256 stridesvec = _mm256_load_ps(stride);
          //      _mm256_store_ps(m_strides[i], stridesvec);
          //      __m256 mulvec = _mm256_mul_ps(stridesvec, dimsvec);
          //      _mm256_store_ps(stride, stridesvec);
            // }
          
            for (int i = m_dims.size() - 1; i >= 0; -- i) {
                m_strides[i] = stride;
                stride *= m_dims[i];
            }
            m_buf.reset(new float[stride]);
        }

    Zeros(Zeros&& z);
    Zeros& operator=(Zeros&& z);
    
    Zeros(const Zeros& z);
    Zeros& operator=(const Zeros& z);
    
        float& operator[] (initializer_list<int> idx) {
            size_t offset = 0;
            auto stride = m_strides.begin();
            
            #pragma acc parallel loop reduction(+:offset)
            for (int i = 0; i < size; i++) {
                offset += i * *stride;
                ++ stride;
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
    const int n = 41;
    int nt = 500;
    int nit = 50;
    float dx = 2 / (nx - 1);
    float dy = 2 / (ny - 1);
    float dt = 0.01;
    int rho = 1;
    float nu = 0.02;
    Zeros X({nx, ny}), Y({nx, ny});
  
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
      for(int j=0; j<N/size; j++) {
          X[{i, j}] = jbody[j].x;
          Y[{j, i}] = jbody[j].y;
      }
    }
  }   // The operation for the same utilization like the numpy.meshgrid function in Python by using MPI.
    
    Zeros u({ny, nx});
    Zeros v({ny, nx});
    Zeros p({ny, nx});
    Zeros b({ny, nx});
    
    for (int n : range(nt)) {
        for (int j = 1; j < ny; j++) {
            for (int i = 1; i < nx; i++) {
                b[{j, i}] = rho * (1 / dt *
                                   ((u[{j, i+1}] - u[{j, i-1}]) / (2 * dx) + (v[{j+1, i}] - v[{j-1, i}]) / (2 * dy)) -
                                   pow(((u[{j, i+1}] - u[{j, i-1}]) / (2 * dx)), 2) - 2 * ((u[{j+1, i}] - u[{j-1, i}]) / (2 * dy) *\
                                                                                      (v[{j, i+1}] - v[{j, i-1}]) / (2 * dx)) - pow(((v[{j+1, i}] - v[{j-1, i}]) / (2 * dy)), 2));
            }
        }
    }
    for (int it : range(nit)) {
        Zeros pn({ny, nx});
        pn = p;
        for (int j = 1; j < ny; j++) {
            for (int i = 1; i < nx; i++) {
                p[{j, i}] = (pow(dy, 2) * (pn[{j, i+1}] + pn[{j, i-1}]) +
                             pow(dx, 2) * (pn[{j+1, i}] + pn[{j-1, i}]) -
                             b[{j, i}] * pow(dx, 2) * pow(dy, 2))
                / (2 * (pow(dx, 2) + pow(dy, 2)));
            }
        }
    }
    #pragma omp parallel for private(i)
    for (int i = 0; i < n; i++) {
        p[{n-1, i}] = p[{n-2, i}];
        p[{i, 0}] = p[{i, 1}];
        p[{0, i}] = p[{1, i}];
        p[{i, n-1}] = 0;
    }
    Zeros un({ny, nx}), vn({ny, nx});
    un = u;
    vn = v;
    
    for (int j = 1; j < ny; j++) {
        for (int i = 1; i < nx; i++) {
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
