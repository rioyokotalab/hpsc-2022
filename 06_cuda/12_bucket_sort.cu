#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void loop(int *key, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int bucket[];
  __syncthreads();
  atomicAdd(&bucket[key[i]], 1);
  __syncthreads();
  key[i] = 0;
  __syncthreads();
  for (int j=1; j<range; j++) {
    offset[j] = offset[j-1] + bucket[j-1];
    if(i>=offset[j]) key[i]++;
  }
  
}


int main() {
  int n = 50;
  int range = 5;
  int *key, *offset;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&offset,  range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  /*
  std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  } */
  
  loop<<<1,n,range*sizeof(int)>>>(key, offset, range);
  cudaDeviceSynchronize();
  
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(offset);
}
