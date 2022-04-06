#include <cstdio>
#include <openacc.h>

int main() {
  int a = 0;
  int b[1] = {0};
#pragma acc parallel loop //private(a)
  for(int i=0; i<8; i++) {
    a = __pgi_vectoridx();
    b[0] = __pgi_vectoridx();
  }
  printf("%d %d\n",a,b[0]);
}
