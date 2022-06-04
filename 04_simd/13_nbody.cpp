#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

//Student ID 20M28152

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], xnew[N], ynew[N];;
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = xnew[i] = ynew[i] = 0;
  }
  for(int i=0; i<N; i++) {
        __m256 J = _mm256_load_ps(count);
        __m256 I = _mm256_set1_ps(count[i]);
        __m256 J = _mm256_set_ps(0, 1, 2, 3, 4, 5, 6, 7);
        __m256 I = _mm256_set1_ps(i);
    
        __m256 xi = _mm256_set1_ps(x[i]);
        __m256 xj = _mm256_load_ps(x);
        __m256 rx = _mm256_sub_ps(xi, xj);
    
        __m256 yi = _mm256_set1_ps(y[i]);
        __m256 yj = _mm256_load_ps(y);
        __m256 ry = _mm256_sub_ps(yi, yj);
    
        __m256 mj = _mm256_load_ps(m);

        __m256 fx_vec = _mm256_setzero_ps();
        __m256 fy_vec = _mm256_setzero_ps();
    
        __m256 rx2 = _mm256_mul_ps(rx, rx);
        __m256 ry2 = _mm256_mul_ps(ry, ry);
        __m256 r_vec = _mm256_rsqrt_ps(_mm256_add_ps(rx2, ry2));
        __m256 r = _mm256_mul_ps(_mm256_mul_ps(r_vec, r_vec), r_vec);
        
        __m256 dx = _mm256_div_ps(_mm256_mul_ps(rx, mj), r);
        __m256 dy = _mm256_div_ps(_mm256_mul_ps(ry, mj), r);
    
        __m256 fx_new = _mm256_sub_ps(fx_vec, dx);
        __m256 fy_new = _mm256_sub_ps(fy_vec, dy);
        
        __m256 mask = _mm256_cmp_ps(I, J, _CMP_NEQ_OQ);
        fx_vec = _mm256_blendv_ps(fx, fx_new, mask);
        fy_vec = _mm256_blendv_ps(fy, fy_new, mask);
    
        _mm256_store_ps(xnew, fx_vec);
        _mm256_store_ps(ynew, fy_vec);


    for(int j=0; j<N; j++) {
/*      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }*/
        fx[i] += xnew[j];
        fy[i] += ynew[j];
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
