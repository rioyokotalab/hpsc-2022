#include <iostream>         //Student ID: 20M28152
#include <vector>           //Name: Mallela Nikihil Rao
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "matplotlibcpp.h"  //Header files to run Matplotlib from C++

namespace plt = matplotlibcpp;
using namespace std;

//Constant Variables in Global memory
const int ny = 41;
const int nx = 41;
const int nt = 500;
const int nit = 50;
const float dx = 2.0 / (nx-1);
const float dy = 2.0 / (ny-1);
const float dt = 0.01;
const int rho = 1;
const float nu = 0.02;

//Device Function to calculate b[i][j]
__global__ void b_uv(float b[ny][nx], float u[ny][nx], float v[ny][nx]) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ny-1 && j < nx-1) {
     if (i != 0 && j != 0 ) {
        b[j][i] = rho * (1 / dt *
                        ((u[j][i+1] - u[j][i-1]) / (2*dx) + (v[j+1][i] - v[j-1][i]) / (2*dy)) -
                         pow(((u[j][i+1] - u[j][i-1]) / (2*dx)),2) - 2 * ((u[j+1][i] - u[j-1][i]) / (2*dy) *
                         (v[j][i+1] - v[j][i-1]) / (2*dx)) - pow(((v[j+1][i] - v[j-1][i]) / (2*dy)),2));
     }
  }
  __syncthreads();

}

//Device function to calculate p[i][j]
__global__ void p_pn(float b[ny][nx], float p[ny][nx], float pn[ny][nx]) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ny-1 && j < nx-1) {
     if (i != 0 && j != 0 ) {

        memcpy (pn, p, ny*nx*sizeof(float));
        p[j][i] = (pow(dy,2) * (pn[j][i+1] + pn[j][i-1]) +
                   pow(dx,2) * (pn[j+1][i] + pn[j-1][i]) -
                   b[j][i] * pow(dx,2) * pow(dy,2)) / (2 * (pow(dx,2) + pow(dy,2)));
        __syncthreads();
     }

     //Boundary Conditions
     p[i][nx-1] = p[i][nx-2];
     p[0][i] = p[1][i];
     p[i][0] = p[i][1];
     p[nx-1][i] = 0;

  }
  __syncthreads();
}

//Device function to calculate u[i][j] and v[i][j]
__global__ void uv_n(float u[ny][nx], float v[ny][nx], float un[ny][nx], float vn[ny][nx], float p[ny][nx]) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < ny-1 && j < nx-1) {
     if (i != 0 && j != 0 ) {

         memcpy (un, u, ny*nx*sizeof(float));
         memcpy (vn, v, ny*nx*sizeof(float));
         u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                            - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                            - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                            + nu * dt / pow(dx,2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                            + nu * dt / pow(dy,2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
         v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                            - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                            - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                            + nu * dt / pow(dx,2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                            + nu * dt / pow(dy,2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
         __syncthreads();
     }

     //Boundary Conditions
     u[0][i] = 0;
     u[i][0] = 0;
     u[i][nx-1] = 0;
     u[nx-1][i] = 1;
     v[0][i] = 0;
     v[nx-1][i] = 0;
     v[i][0] = 0;
    
     __syncthreads();
  }
}

//Function that replicates numpy.linespace on C++
template <typename T>
vector<T> linspace(T a, T b, T  N) {
   vector<T> linespaced;
   T h = (b - a) / (N-1);
   vector<T> xs(N);

typename vector<T>::iterator x;
   T val;
   for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
       *x = val;
   }
   return xs;
}

//Host Function to initialize arrays to zeros
void Zeros_init(float M[ny][nx]) {
   for (int i = 0; i <ny; i++) {
       for (int j=0; j<nx; j++) {
           M[i][j] = 0;
       }
   }
}

//Host Main
int main() {
  int n = 0;
  int N = 50;
  size_t bytes = ny*nx*sizeof(float); //size to allocate for unified memory

  //Vector of pressure, p for plotting contour plot
  vector<vector<float>> pvec (ny , vector<float> (nx, 0));

  //Allocation of variables to unified memory
  float* u; cudaMallocManaged(&u, bytes);
  float* v; cudaMallocManaged(&v, bytes);
  float* p; cudaMallocManaged(&p, bytes);
  float* b; cudaMallocManaged(&b, bytes);
  float* pn; cudaMallocManaged(&pn, bytes);
  float* un; cudaMallocManaged(&un, bytes);
  float* vn; cudaMallocManaged(&vn, bytes);

  //Defining 2D arrays
  float u_vals[ny][nx], v_vals[ny][nx];
  float p_vals[ny][nx], b_vals[ny][nx];
  float pn_vals[nx][nx], un_vals[ny][nx], vn_vals[ny][nx];


  //Initialization of 2D arrays to zeros
  Zeros_init(u_vals);
  Zeros_init(v_vals);
  Zeros_init(p_vals);
  Zeros_init(b_vals);
  Zeros_init(pn_vals);
  Zeros_init(un_vals);
  Zeros_init(vn_vals);

  //Copying
  copy(&u_vals[0][0], &u_vals[0][0] + ny*nx, u);
  copy(&v_vals[0][0], &v_vals[0][0] + ny*nx, v);
  copy(&p_vals[0][0], &p_vals[0][0] + ny*nx, p);
  copy(&b_vals[0][0], &b_vals[0][0] + ny*nx, b);
  copy(&pn_vals[0][0], &pn_vals[0][0] + ny*nx, pn);
  copy(&un_vals[0][0], &un_vals[0][0] + ny*nx, un);
  copy(&vn_vals[0][0], &vn_vals[0][0] + ny*nx, vn);


  //Two Dimenional allocation of threads
  dim3 threadsPerBlock(N,  N);
  dim3 numBlocks(1, 1);

  //Meshgrid for contour plot: Start
  vector<float> y(ny);
  vector<float> x(nx);
  x = linspace<float>(0, 2, nx);
  y = linspace<float>(0, 2, ny);
  //X and Y grid vectors
  vector<vector<float>> X( ny , vector<float> (nx, 0));
  vector<vector<float>> Y( ny , vector<float> (nx, 0));
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++ j) {
      X[i][j] = float(*(x.data() + j));
      Y[i][j] = float(*(y.data() + i));

    }
  }
  //Meshgrid for contour plot: End


  //Pointers to variables to pass in GPU call
  float (*U)[ny] = reinterpret_cast<float (*)[ny]>(u);
  float (*V)[ny] = reinterpret_cast<float (*)[ny]>(v);
  float (*P)[ny] = reinterpret_cast<float (*)[ny]>(p);
  float (*B)[ny] = reinterpret_cast<float (*)[ny]>(b);
  float (*PN)[ny] = reinterpret_cast<float (*)[ny]>(pn);
  float (*UN)[ny] = reinterpret_cast<float (*)[ny]>(un);
  float (*VN)[ny] = reinterpret_cast<float (*)[ny]>(vn);


  //Calling Device kernel
  while (n < nt) {

      //Calculates b[i][j]
      b_uv<<<numBlocks, threadsPerBlock>>>(B, U, V);
      cudaDeviceSynchronize();

      //Calculates p[i][j]
      for (int it = 0; it < nit; it++) {
          p_pn<<<numBlocks, threadsPerBlock>>>(B, P, PN);
          cudaDeviceSynchronize();
      }
      `
      //Calculates u[i][j] and v[i][j]
      uv_n<<<numBlocks, threadsPerBlock>>>(U, V, UN, VN, P);
      cudaDeviceSynchronize();

      //Converts p array to p vector to plot using matplotlibcpp.h
      for (int i = 0; i < ny; i++) {
          for (int j = 0; j < nx; j++) {
              pvec[i][j] = p[i][j];
          }
      }

      //Execute Matplotlibcpp.h
      plt::clf();
      //Arguments to plt::contour "must" be vectors
      plt::contour(X, Y, pvec);
//     plt::quiver(X, Y, u, v);   //Unable to use due to error in function call
      plt::pause(0.01);

      n++;
  }
  plt::show();
}


                            
