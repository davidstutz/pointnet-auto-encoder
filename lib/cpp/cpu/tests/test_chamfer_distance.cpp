#include <cstdio>
#include <cmath>
#include <cassert>
#include "chamfer_distance.h"

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

  int* indices = new int[batch_size*n_points*2];
  float loss = chamfer_distance_updateOutput(batch_size, n_points, input, target, indices, false);

  printf("%f\n", loss);
  assert(fabs(loss - 0.06f) < 1e-6);

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < n_points; n++) {
      printf("%d %d %d\n", b, n, indices[n]);
      assert(indices[(b*n_points + n)*2 + 0] == (n_points - n - 1));
      assert(indices[(b*n_points + n)*2 + 1] == (n_points - n - 1));
    }
  }

  delete[] input;
  delete[] target;
  delete[] indices;
}

void test_updateGradInput() {
  int n_points = 3;
  int batch_size = 2;

  float* input = new float[n_points*batch_size*3];
  float* target = new float[n_points*batch_size*3];
  int* indices = new int[n_points*batch_size*2];

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

  float* grad_input = new float[batch_size*n_points*3];
  chamfer_distance_updateGradInput(batch_size, n_points, input, target, indices, grad_input, false);

  for (int b = 0; b < batch_size; b++) {
    for (int n = 0; n < n_points; n++) {
      assert(fabs(grad_input[(b*n_points + n)*3 + n] + 0.2) < 1e-6);
    }
  }

  delete[] input;
  delete[] target;
  delete[] indices;
  delete[] grad_input;
}

int main(int argc, char** argv) {
  test_updateOutput();
  test_updateGradInput();
}