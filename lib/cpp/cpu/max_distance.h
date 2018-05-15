#ifndef CPU_MAX_DISTANCE
#define CPU_MAX_DISTANCE

extern "C" {
  float max_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target);
}

#endif