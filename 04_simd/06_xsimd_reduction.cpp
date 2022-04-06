#include <cstdio>
#include <cassert>
#include "xsimd/xsimd.hpp"

int main() {
  const int N = 8;
  xsimd::batch<float, N> a;
  for(int i=0; i<N; i++)
    a[i] = 1;
  float b = xsimd::hadd(a);
  printf("%g\n",b);
}
