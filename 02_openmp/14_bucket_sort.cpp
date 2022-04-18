#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
    int n = 50;
    int range = 5;
    std::vector<int> key(n);
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    std::vector<int> bucket(range, 0);
    for (int i = 0; i < n; i++)
        bucket[key[i]]++;
    std::vector<int> offset(range, 0);
    std::vector<int> buffer(range, 0);
#pragma omp parallel for
    for (int j = 1; j < range; j++)
    {
        offset[j] = bucket[j - 1];
    }
#pragma omp parallel
    for (int i = 1; i < range; i <<= 1) 
    {
#pragma omp for
        for (int j = 0; j < range; j++)
        {
            buffer[j] = offset[j];
        }
#pragma omp for
        for (int j = i; j < range; j++)
        {
            offset[j] += buffer[j - i];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < range; i++) {
        int j = offset[i];
        for (; bucket[i] > 0; bucket[i]--) {
            key[j++] = i;
        }
    }

    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");
}
