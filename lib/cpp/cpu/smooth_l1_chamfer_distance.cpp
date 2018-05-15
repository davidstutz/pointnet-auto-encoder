#include <cstdio>
#include <cassert>
#include <cfloat>
#include <cmath>
#include "smooth_l1_chamfer_distance.h"

#define EPSILON 1e-8

float smooth_l1_chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average) {
  float chamfer_distance = 0;

  for (int i = 0; i < batch_size*n_points*2; i++) {
    indices[i] = -1;
  }

  // Matching predicted points against targets.
  for (int b = 0; b < batch_size; b++) {
    // Loop over predicted points in input.
    for (int n1 = 0; n1 < n_points; n1++) {
      float min_distance = FLT_MAX;

      // Loop over target points.
      for (int n2 = 0; n2 < n_points; n2++) {
        float distance = 0;
        for (int d = 0; d < 3; d++) {
          float difference = input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d];
          distance += sqrt(difference*difference + EPSILON);
        }

        if (distance < min_distance) {
          min_distance = distance;
          indices[(b*n_points + n1)*2 + 0] = n2;
        }
      }

      chamfer_distance += min_distance;
    }
  }

  // Matching targets against predicted points.
  for (int b = 0; b < batch_size; b++) {
    for (int n2 = 0; n2 < n_points; n2++) {
      float min_distance = FLT_MAX;

      for (int n1 = 0; n1 < n_points; n1++) {
        float distance = 0;
        for (int d = 0; d < 3; d++) {
          float difference = input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d];
          distance += sqrt(difference*difference + EPSILON);
        }

        if (distance < min_distance) {
          min_distance = distance;
          indices[(b*n_points + n1)*2 + 1] = n2;
        }
      }

      chamfer_distance += min_distance;
    }
  }

  if (size_average) {
    chamfer_distance /= 2*batch_size*n_points;
  }

  return 0.5f*chamfer_distance;
}

void smooth_l1_chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* input, const float* target, const int* indices, float* grad_input, bool size_average) {
  for (int b = 0; b < batch_size; b++) {

    // Loop over predicted points in input.
    for (int n1 = 0; n1 < n_points; n1++) {

      // Target from matching predictions against targets.
      int n2 = indices[(b*n_points + n1)*2 + 0];
      assert(n2 >= 0 && n2 < n_points);

      for (int d = 0; d < 3; d++) {
        float difference = input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d];
        grad_input[(b*n_points + n1)*3 + d] = difference/sqrt(difference*difference + EPSILON);
      }

      // Target from matching targets against predictions.
      n2 = indices[(b*n_points + n1)*2 + 1];
      //assert(n2 >= 0 && n2 < n_points);

      if (n2 >= 0) {
        for (int d = 0; d < 3; d++) {
          float difference = input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d];
          grad_input[(b*n_points + n1)*3 + d] += difference/sqrt(difference*difference + EPSILON);
        }
      }

      if (size_average) {
        for (int d = 0; d < 3; d++) {
          grad_input[(b*n_points + n1)*3 + d] /= 2*batch_size*n_points;
        }
      }
    }
  }
}