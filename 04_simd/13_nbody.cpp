#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], count[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
        __m256 J = _mm256_load_ps(count);
        __m256 I = _mm256_set1_ps(count[i]);
        __m256 xivec = _mm256_set1_ps(x[i]);
        __m256 xjvec = _mm256_load_ps(x);
        __m256 rxvec = _mm256_sub_ps(xivec, xjvec);
        __m256 yivec = _mm256_set1_ps(y[i]);
        __m256 yjvec = _mm256_load_ps(y);
        __m256 ryvec = _mm256_sub_ps(yivec, yjvec);
        
        __m256 fxvec = _mm256_set1_ps(fx);
        __m256 fyvec = _mm256_set1_ps(fy);
        __m256 mjvec = _mm256_load_ps(m);
        
        __m256 rxp2 = _mm256_mul_ps(rxvec, rxvec);
        __m256 ryp2 = _mm256_mul_ps(ryvec, ryvec);
        __m256 rvec = _mm256_rsqrt_ps(_mm256_add_ps(rxp2, ryp2));
        __m256 rp3 = _mm256_mul_ps(_mm256_mul_ps(rvec, rvec), rvec);
        
        __m256 tempx = _mm256_div_ps(_mm256_mul_ps(rxvec, mjvec), rp3);
        __m256 tempy = _mm256_div_ps(_mm256_mul_ps(ryvec, mjvec), rp3);
        __m256 fxnew = _mm256_sub_ps(fxvec, tempx);
        __m256 fynew = _mm256_sub_ps(fyvec, tempy);
        
        __m256 mask = _mm256_cmp_ps(I, J, _CMP_NEQ_OQ);
        fxvec = _mm256_blendv_ps(fxvec, fxnew, mask);
        fyvec = _mm256_blendv_ps(fyvec, fynew, mask);
        _mm256_store_ps(fx, fxnew);
        _mm256_store_ps(fy, fynew);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
