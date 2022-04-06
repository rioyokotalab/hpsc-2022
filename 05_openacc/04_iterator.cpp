#include <cstdio>
#include <vector>
#include <openacc.h>

int main() {
  std::vector<int> a(8);
  std::vector<int>::iterator it;
#pragma acc parallel loop
  for(it=a.begin(); it<a.end(); it++) {
    *it = __pgi_vectoridx();
    printf("%d\n",*it);
  }
}
