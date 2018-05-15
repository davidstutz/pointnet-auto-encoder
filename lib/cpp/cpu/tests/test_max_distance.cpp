#include <cstdio>
#include <cmath>
#include <cassert>
#include "max_distance.h"

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

  float loss = max_distance_updateOutput(batch_size, n_points, input, target);

  printf("%f\n", loss);
  assert(fabs(loss - 0.02f) < 1e-6);

  delete[] input;
  delete[] target;
}

int main(int argc, char** argv) {
  test_updateOutput();
}