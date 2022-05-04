// Student ID: 22M31261
// Name: ZHANG HAOKE
#include <cstdio>
#include <cstdlib>

__global__ void addOne(int *bucket, int *key){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = key[i];
	bucket[j] += 1;
}

int main() {
  int n = 50;
  int range = 5;
  int *bucket;
  int *key;
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&key, n*sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  
  addOne<<<n/range, range>>>(bucket, key);
  cudaDeviceSynchronize();

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(bucket);
  cudaFree(key);
}
