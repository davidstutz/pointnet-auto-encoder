#include <cstdio>
#include <cassert>
#include <cfloat>
#include "max_distance.h"

float max_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target) {
  float loss = 0;
  float max_distance = 0;

  // Matching predicted points against targets.
  for (int b = 0; b < batch_size; b++) {
    // Loop over predicted points in input.
    for (int n1 = 0; n1 < n_points; n1++) {
      float min_distance = FLT_MAX;

      // Loop over target points.
      for (int n2 = 0; n2 < n_points; n2++) {
        float distance = 0;
        for (int d = 0; d < 3; d++) {
          distance += (input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d])
            * (input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d]);
        }

        if (distance < min_distance) {
          min_distance = distance;
        }
      }

      if (min_distance > max_distance) {
        max_distance = min_distance;
      }
    }
  }

  loss += max_distance;
  max_distance = 0;

  // Matching targets against predicted points.
  for (int b = 0; b < batch_size; b++) {
    for (int n2 = 0; n2 < n_points; n2++) {
      float min_distance = FLT_MAX;

      for (int n1 = 0; n1 < n_points; n1++) {
        float distance = 0;
        for (int d = 0; d < 3; d++) {
          distance += (input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d])
            * (input[(b*n_points + n1)*3 + d] - target[(b*n_points + n2)*3 + d]);
        }

        if (distance < min_distance) {
          min_distance = distance;
        }
      }

      if (min_distance > max_distance) {
        max_distance = min_distance;
      }
    }
  }

  loss += max_distance;
  return loss;
}