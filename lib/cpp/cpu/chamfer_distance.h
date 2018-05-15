#ifndef CPU_CHAMFER_DISTANCE
#define CPU_CHAMFER_DISTANCE

extern "C" {
  float chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
  void chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* input, const float* target, const int* indices, float* grad_input, bool size_average);
}

#endif