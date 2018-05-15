#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include "chamfer_distance.h"
#include "cuda_helper.h"

void test_updateOutput() {
  int n_points = 3;
  int batch_size = 2;
  float* input = new float[n_points*batch_size*3];
  float* target = new float[n_points*batch_size*3];

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < n_points; n++) {
      input[(b*n_points + n)*3 + 0] = 0;
      input[(b*n_points + n)*3 + 1] = 0;
      input[(b*n_points + n)*3 + 2] = 0;
      input[(b*n_points + n)*3 + n] = 1;
      //printf("%d %d %f %f %f\n", b, n, input[(b*n_points + n)*3 + 0], input[(b*n_points + n)*3 + 1],
      //  input[(b*n_points + n)*3 + 2]);

      target[(b*n_points + (n_points - n - 1))*3 + 0] = 0;
      target[(b*n_points + (n_points - n - 1))*3 + 1] = 0;
      target[(b*n_points + (n_points - n - 1))*3 + 2] = 0;
      target[(b*n_points + (n_points - n - 1))*3 + n] = 1.1;
      //printf("%d %d %f %f %f\n", b, n_points - n - 1, target[(b*n_points + (n_points - n - 1))*3 + 0],
      //  target[(b*n_points + (n_points - n - 1))*3 + 1], target[(b*n_points + (n_points - n - 1))*3 + 2]);
    }
  }

  int* indices = new int[n_points*batch_size*2];

  float* d_input = NULL;
  float* d_target = NULL;
  int* d_indices = NULL;

  unsigned int data_size = n_points*batch_size*3*sizeof(float);
  unsigned int indices_size = n_points*batch_size*2*sizeof(int);

  checkCudaErrors(cudaMalloc((void **) &d_input, data_size));
  checkCudaErrors(cudaMalloc((void **) &d_target, data_size));
  checkCudaErrors(cudaMalloc((void **) &d_indices, indices_size));

  checkCudaErrors(cudaMemcpy(d_input, input, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_target, target, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_indices, indices, indices_size, cudaMemcpyHostToDevice));

  float loss = chamfer_distance_updateOutput(batch_size, n_points, d_input, d_target, d_indices, false);

  checkCudaErrors(cudaMemcpy(indices, d_indices, indices_size, cudaMemcpyDeviceToHost));

  printf("%f\n", loss);
  assert(fabs(loss - 0.06f) < 1e-6);

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < n_points; n++) {
      //printf("%d %d %d\n", b, n, indices[n]);
      assert(indices[(b*n_points + n)*2 + 0] == (n_points - n - 1));
      assert(indices[(b*n_points + n)*2 + 1] == (n_points - n - 1));
    }
  }

  delete[] input;
  delete[] target;
  delete[] indices;

  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_target));
  checkCudaErrors(cudaFree(d_indices));
}

void test_updateGradInput() {
  int n_points = 3;
  int batch_size = 2;
  float* input = new float[n_points*batch_size*3];
  float* target = new float[n_points*batch_size*3];
  float* grad_input = new float[n_points*batch_size*3];
  int* indices = new int[batch_size*n_points*2];

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < n_points; n++) {
      input[(b*n_points + n)*3 + 0] = 0;
      input[(b*n_points + n)*3 + 1] = 0;
      input[(b*n_points + n)*3 + 2] = 0;
      input[(b*n_points + n)*3 + n] = 1;
      //printf("%d %d %f %f %f\n", b, n, input[(b*n_points + n)*3 + 0], input[(b*n_points + n)*3 + 1],
      //  input[(b*n_points + n)*3 + 2]);

      target[(b*n_points + (n_points - n - 1))*3 + 0] = 0;
      target[(b*n_points + (n_points - n - 1))*3 + 1] = 0;
      target[(b*n_points + (n_points - n - 1))*3 + 2] = 0;
      target[(b*n_points + (n_points - n - 1))*3 + n] = 1.1;
      //printf("%d %d %f %f %f\n", b, n_points - n - 1, target[(b*n_points + (n_points - n - 1))*3 + 0],
      //  target[(b*n_points + (n_points - n - 1))*3 + 1], target[(b*n_points + (n_points - n - 1))*3 + 2]);

      indices[(b*n_points + n)*2 + 0] = (n_points - n - 1);
      indices[(b*n_points + n)*2 + 1] = (n_points - n - 1);
    }
  }

  float* d_input = NULL;
  float* d_target = NULL;
  float* d_grad_input = NULL;
  int* d_indices = NULL;

  unsigned int data_size = n_points*batch_size*3*sizeof(float);
  unsigned int indices_size = n_points*batch_size*2*sizeof(int);

  checkCudaErrors(cudaMalloc((void **) &d_input, data_size));
  checkCudaErrors(cudaMalloc((void **) &d_target, data_size));
  checkCudaErrors(cudaMalloc((void **) &d_grad_input, data_size));
  checkCudaErrors(cudaMalloc((void **) &d_indices, indices_size));

  checkCudaErrors(cudaMemcpy(d_input, input, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_target, target, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_grad_input, grad_input, data_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_indices, indices, indices_size, cudaMemcpyHostToDevice));

  chamfer_distance_updateGradInput(batch_size, n_points, d_input, d_target, d_indices, d_grad_input, false);

  checkCudaErrors(cudaMemcpy(grad_input, d_grad_input, data_size, cudaMemcpyDeviceToHost));

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < n_points; n++) {
      assert(fabs(grad_input[(b*n_points + n)*3 + n] + 0.2) < 1e-6);
      //printf("%f \n", grad_input[(b*n_points + n)*3 + n]);
    }
  }

  delete[] input;
  delete[] target;
  delete[] indices;
  delete[] grad_input;

  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_target));
  checkCudaErrors(cudaFree(d_indices));
  checkCudaErrors(cudaFree(d_grad_input));
}

int main(int argc, char** argv) {
  test_updateOutput();
  printf("test_updateOutput complete");
  test_updateGradInput();
  printf("test_updateOutput complete");
}