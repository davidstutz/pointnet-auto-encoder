#include <cstdio>
#include <cassert>
#include <cfloat>
#include "cuda_helper.h"
#include "chamfer_distance.h"

__global__ void kernel_chamfer_distance_updateOutput_initializeIndices(int* d_indices) {
  //const int batch_size = blockDim.x;
  const int n_points = gridDim.x;

  const int b = threadIdx.x;
  const int n1 = blockIdx.x;

  d_indices[(b*n_points + n1)*2 + 0] = -1;
  d_indices[(b*n_points + n1)*2 + 1] = -1;
}

__global__ void kernel_chamfer_distance_updateOutput_predictionsTargets(const float* d_input, const float* d_target, int* d_indices, float* d_loss) {
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
      d_indices[(b*n_points + n1)*2 + 0] = n2;
    }
  }

  //*d_loss += min_distance;
  atomicAdd(d_loss, min_distance);
  //printf("%f %f\n", *d_loss, min_distance);
}

__global__ void kernel_chamfer_distance_updateOutput_targetsPredictions(const float* d_input, const float* d_target, int* d_indices, float* d_loss) {
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
      d_indices[(b*n_points + n1)*2 + 1] = n2;
    }
  }

  //*d_loss += min_distance;
  atomicAdd(d_loss, min_distance);
  //printf("%f %f\n", *d_loss, min_distance);
}

float chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* d_input, const float* d_target, int* d_indices, bool size_average) {
  dim3 grid(n_points, 1, 1);
  dim3 block(batch_size, 1, 1);

  kernel_chamfer_distance_updateOutput_initializeIndices<<<grid, block>>>(d_indices);
  cudaDeviceSynchronize();

  float loss = 0;
  float* d_loss = NULL;

  checkCudaErrors(cudaMalloc(&d_loss, sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_loss, &loss, sizeof(float), cudaMemcpyHostToDevice));

  kernel_chamfer_distance_updateOutput_predictionsTargets<<<grid, block>>>(d_input, d_target, d_indices, d_loss);
  //cudaDeviceSynchronize();

  kernel_chamfer_distance_updateOutput_targetsPredictions<<<grid, block>>>(d_input, d_target, d_indices, d_loss);
  cudaDeviceSynchronize();

  // http://stackoverflow.com/questions/34041372/access-cuda-global-device-variable-from-host
  checkCudaErrors(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaMemcpyFromSymbol(&loss, "d_loss", sizeof(float), 0, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_loss));

  if (size_average) {
    loss /= 2*batch_size*n_points;
  }

  return 0.5f*loss;
}

__global__ void kernel_chamfer_distance_updateGradInput(const float* d_input, const float* d_target, const int* d_indices, float* d_grad_input, bool size_average) {
  const int batch_size = blockDim.x;
  const int n_points = gridDim.x;

  const int b = threadIdx.x;
  const int n1 = blockIdx.x;

  int n2 = d_indices[(b*n_points + n1)*2 + 0];
  assert(n2 >= 0 && n2 < n_points);

  for (int d = 0; d < 3; d++) {
    d_grad_input[(b*n_points + n1)*3 + d] = d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d];
  }

  n2 = d_indices[(b*n_points + n1)*2 + 1];
  //assert(n2 >= 0 && n2 < n_points);

  // Note that n1 might not have been assigned to an n2 in the second round.
  if (n2 >= 0) {
    for (int d = 0; d < 3; d++) {
      d_grad_input[(b*n_points + n1)*3 + d] += d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d];
    }
  }

  if (size_average) {
    for (int d = 0; d < 3; d++) {
      d_grad_input[(b*n_points + n1)*3 + d] /= 2*batch_size*n_points;
    }
  }
}

void chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* d_input, const float* d_target, const int* d_indices, float* d_grad_input, bool size_average) {
  dim3 grid(n_points, 1, 1);
  dim3 block(batch_size, 1, 1);

  kernel_chamfer_distance_updateGradInput<<<grid, block>>>(d_input, d_target, d_indices, d_grad_input, size_average);
}