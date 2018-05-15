#include <cstdio>
#include <cassert>
#include <cfloat>
#include <cmath>
#include "cuda_helper.h"
#include "smooth_l1_chamfer_distance.h"

__global__ void kernel_smooth_l1_chamfer_distance_updateOutput_initializeIndices(int* d_indices, int indices_size) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i >= indices_size) {
    return;
  }

  d_indices[i] = -1;
}

__global__ void kernel_smooth_l1_chamfer_distance_updateOutput_computeDistances(const float* d_input, const float* d_target, float* d_distances, int n_points) {
  const float EPSILON = 1e-8;

  int b = blockIdx.z;
  int n1 = threadIdx.x + blockDim.x*blockIdx.x;
  int n2 = threadIdx.y + blockDim.y*blockIdx.y;

  if (n1 >= n_points || n2 >= n_points) {
    return;
  }

  for (int d = 0; d < 3; d++) {
    float difference = d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d];
    d_distances[(b*n_points + n1)*n_points + n2] += sqrt(difference*difference + EPSILON);
  }
}

__global__ void kernel_smooth_l1_chamfer_distance_updateOutput_computeLoss(float* d_distances, int* d_indices, float* d_loss, int n_points) {
  int mode = threadIdx.y;
  int b = blockIdx.y;

  if (mode) {
    int n1 = threadIdx.x + blockDim.x*blockIdx.x;

    if (n1 >= n_points) {
      return;
    }

    float min_distance = FLT_MAX;

    for (int n2 = 0; n2 < n_points; n2++) {
      float distance = d_distances[(b*n_points + n1)*n_points + n2];
      //printf("%d %d %d %d %f\n", mode, b, n1, n2, distance);
      if (distance < min_distance) {
        min_distance = distance;
        d_indices[(b*n_points + n1)*2 + 0] = n2;
      }
    }

    atomicAdd(d_loss, min_distance);
  }
  else {
    int n2 = threadIdx.x + blockDim.x*blockIdx.x;

    if (n2 >= n_points) {
      return;
    }

    float min_distance = FLT_MAX;

    for (int n1 = 0; n1 < n_points; n1++) {
      float distance = d_distances[(b*n_points + n1)*n_points + n2];
      //printf("%d %d %d %d %f\n", mode, b, n1, n2, distance);
      if (distance < min_distance) {
        min_distance = distance;
        d_indices[(b*n_points + n1)*2 + 1] = n2;
      }
    }

    atomicAdd(d_loss, min_distance);
  }
}

float smooth_l1_chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* d_input, const float* d_target, int* d_indices, bool size_average) {

  const int indices_size = 2*batch_size*n_points;
  const int max_threads = 1024; // Square-root should be integer (for 1024 -> 32).

  int blocks = ceil((float) indices_size / (float) max_threads);
  int threads = max_threads;

  kernel_smooth_l1_chamfer_distance_updateOutput_initializeIndices<<<blocks, threads>>>(d_indices, indices_size);
  cudaDeviceSynchronize();

  float loss = 0;
  float* d_loss = NULL;

  checkCudaErrors(cudaMalloc((void**) &d_loss, sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_loss, &loss, sizeof(float), cudaMemcpyHostToDevice));

  float* d_distances = NULL;

  checkCudaErrors(cudaMalloc((void**) &d_distances, batch_size*n_points*n_points*sizeof(float)));
  checkCudaErrors(cudaMemset(d_distances, 0, batch_size*n_points*n_points*sizeof(float)));

  threads = sqrt(max_threads);
  blocks = ceil((float) n_points / (float) threads);

  dim3 grid(blocks, blocks, batch_size);
  dim3 block(threads, threads);

  kernel_smooth_l1_chamfer_distance_updateOutput_computeDistances<<<grid, block>>>(d_input, d_target, d_distances, n_points);

  threads = max_threads/2;
  grid = dim3(ceil((float) n_points / (float) threads), batch_size);
  block = dim3(threads, 2);

  kernel_smooth_l1_chamfer_distance_updateOutput_computeLoss<<<grid, block>>>(d_distances, d_indices, d_loss, n_points);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_loss));

  if (size_average) {
    loss /= 2*batch_size*n_points;
  }

  checkCudaErrors(cudaFree(d_distances));

  return 0.5f*loss;
}

__global__ void kernel_smooth_l1_chamfer_distance_updateGradInput(const float* d_input, const float* d_target, const int* d_indices, float* d_grad_input, bool size_average) {
  const float EPSILON = 1e-8;

  const int batch_size = blockDim.x;
  const int n_points = gridDim.x;

  const int b = threadIdx.x;
  const int n1 = blockIdx.x;

  int n2 = d_indices[(b*n_points + n1)*2 + 0];
  assert(n2 >= 0 && n2 < n_points);

  for (int d = 0; d < 3; d++) {
    float difference = d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d];
    d_grad_input[(b*n_points + n1)*3 + d] = difference/sqrt(difference*difference + EPSILON);
  }

  n2 = d_indices[(b*n_points + n1)*2 + 1];
  //assert(n2 >= 0 && n2 < n_points);

  // Note that n1 might not have been assigned to an n2 in the second round.
  if (n2 >= 0) {
    for (int d = 0; d < 3; d++) {
      float difference = d_input[(b*n_points + n1)*3 + d] - d_target[(b*n_points + n2)*3 + d];
      d_grad_input[(b*n_points + n1)*3 + d] += difference/sqrt(difference*difference + EPSILON);
    }
  }

  if (size_average) {
    for (int d = 0; d < 3; d++) {
      d_grad_input[(b*n_points + n1)*3 + d] /= 2*batch_size*n_points;
    }
  }
}

void smooth_l1_chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* d_input, const float* d_target, const int* d_indices, float* d_grad_input, bool size_average) {
  dim3 grid(n_points, 1, 1);
  dim3 block(batch_size, 1, 1);

  kernel_smooth_l1_chamfer_distance_updateGradInput<<<grid, block>>>(d_input, d_target, d_indices, d_grad_input, size_average);
}