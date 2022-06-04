#include <cassert>
#include <cstdio>
#include <chrono>
#include <vector>
#include "hdf5.h"
using namespace std;

int main (int argc, char** argv) {
  hsize_t dim[2] = {2, 2};
  int mpisize, mpirank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  assert(mpisize == dim[0]*dim[1]);
  hid_t plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t file = H5Fopen("data.h5", H5F_ACC_RDONLY, plist);
  hid_t dataset = H5Dopen(file, "dataset", H5P_DEFAULT);
  hid_t globalspace = H5Dget_space(dataset);
  int ndim = H5Sget_simple_extent_ndims(globalspace);
  hsize_t N[ndim];
  H5Sget_simple_extent_dims(globalspace, N, NULL);
  hsize_t NX = N[0], NY = N[1];
  hsize_t Nlocal[2] = {NX/dim[0], NY/dim[1]};
  hsize_t Nlocal2[2] = {NX/dim[0]/2, NY/dim[1]};
  hsize_t offset[2] = {mpirank / dim[0], mpirank % dim[0]};
  hsize_t offset2[2] = {mpirank / dim[0], mpirank % dim[0]};
  for(int i=0; i<2; i++){
    offset[i] *= Nlocal[i];
    offset2[i] *= Nlocal2[i];
  }
  hsize_t count[2] = {1,1};
  hsize_t stride[2] = {1,1};
  hid_t localspace = H5Screate_simple(2, Nlocal, NULL);
  hid_t localspace_half = H5Screate_simple(2, Nlocal2, NULL);
  H5Sselect_hyperslab(globalspace, H5S_SELECT_SET, offset, stride, count, Nlocal);
  H5Sselect_hyperslab(globalspace, H5S_SELECT_SET, offset2, stride, count, Nlocal2);
  H5Pclose(plist);
  vector<int> buffer2(Nlocal2[0]*Nlocal2[1]);
  vector<int> buffer;
  for(int i = 0; i < 4; i++) buffer.insert(buffer.end(), buffer2.begin(), buffer2.end());
  plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist, H5FD_MPIO_COLLECTIVE);
  auto tic = chrono::steady_clock::now();
  H5Dread(dataset, H5T_NATIVE_INT, localspace, globalspace, plist, &buffer[0]);
  auto toc = chrono::steady_clock::now();
  H5Dclose(dataset);
  H5Sclose(localspace);
  H5Sclose(globalspace);
  H5Fclose(file);
  H5Pclose(plist);
  double time = chrono::duration<double>(toc - tic).count();
  int sum = 0;
  for (int i=0; i<Nlocal[0]*Nlocal[1]; i++)
    sum += buffer[i];
  printf("sum=%d\n",sum);
  printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
  MPI_Finalize();
}
