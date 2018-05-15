#ifndef GPU_FAST_CHAMFER_DISTANCE
#define GPU_FAST_CHAMFER_DISTANCE

extern "C" {
  float fast_chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
}

#endif