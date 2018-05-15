#include <cstdio>
#include <cassert>
#include <cfloat>
#include "cuda_helper.h"
#include "max_distance.h"

// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz4jyn0BBEW
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_max_distance_updateOutput_predictionsTargets(const float* d_input, const float* d_target, float* d_loss) {
  //const int batch_size = blockDim.x;
  const int n_points = gridDim.x;

  const int b = threadIdx.x;
  const int n1 = blockIdx.x;

  float min_distance = FLT_MAX;
  for (int n2 = 0; n2 < n_points; n2++) {
    float distance = 0;
    for (int d = 0; d < 3; d++) {
      distance += (d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d])
        * (d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d]);
    }

    if (distance < min_distance) {
      min_distance = distance;
    }
  }

  atomicMax(d_loss, min_distance);
  //printf("%f %f\n", *d_loss, min_distance);
}

__global__ void kernel_max_distance_updateOutput_targetsPredictions(const float* d_input, const float* d_target, float* d_loss) {
  //const int batch_size = blockDim.x;
  const int n_points = gridDim.x;

  const int b = threadIdx.x;
  const int n2 = blockIdx.x;

  float min_distance = FLT_MAX;
  for (int n1 = 0; n1 < n_points; n1++) {
    float distance = 0;
    for (int d = 0; d < 3; d++) {
      distance += (d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d])
        * (d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d]);
    }

    if (distance < min_distance) {
      min_distance = distance;
    }
  }

  atomicMax(d_loss, min_distance);
  //printf("%f %f\n", *d_loss, min_distance);
}

float max_distance_updateOutput(const int batch_size, const int n_points, const float* d_input, const float* d_target) {
  dim3 grid(n_points, 1, 1);
  dim3 block(batch_size, 1, 1);


  float loss = 0;
  float* d_loss = NULL;
  float overall_loss = 0;

  checkCudaErrors(cudaMalloc(&d_loss, sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_loss, &loss, sizeof(float), cudaMemcpyHostToDevice));

  kernel_max_distance_updateOutput_predictionsTargets<<<grid, block>>>(d_input, d_target, d_loss);
  cudaDeviceSynchronize();

  // http://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host
  checkCudaErrors(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  overall_loss += loss;

  kernel_max_distance_updateOutput_targetsPredictions<<<grid, block>>>(d_input, d_target, d_loss);
  cudaDeviceSynchronize();

  // http://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host
  checkCudaErrors(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  overall_loss += loss;

  //checkCudaErrors(cudaMemcpyFromSymbol(&loss, "d_loss", sizeof(float), 0, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_loss));

  return overall_loss;
}