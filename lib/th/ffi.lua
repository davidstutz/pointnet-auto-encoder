-- Include C modules.

require('os')
local ffi = require('ffi')

-- Will contain all C modules later ...
lib.cpu = {}
lib.gpu = {}

ffi.cdef[[
float chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
void chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* input, const float* target, const int* indices, float* grad_input, bool size_average);
float max_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target);
float smooth_l1_chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
void smooth_l1_chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* input, const float* target, const int* indices, float* grad_input, bool size_average);
]]

local function scriptPath()
  local str = debug.getinfo(2, "S").source:sub(2)
  return str:match("(.*/)")
end

local libname = scriptPath() .. '../cpp/cpu/build/libcpu.so'
local found = pcall(function () lib.cpu = ffi.load(libname) end)

if found then
  print('[Lib] found ' .. libname)
else
  print('[Info] could not find CPU module, tried ' .. libname)
  print('[Info] will continue without CPU module')
  lib.gpu = false
  --os.exit()
end

if cutorch then
  ffi.cdef[[
  float chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
  void chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* input, const float* target, const int* indices, float* grad_input, bool size_average);
  float fast_chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
  float max_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target);
  float smooth_l1_chamfer_distance_updateOutput(const int batch_size, const int n_points, const float* input, const float* target, int* indices, bool size_average);
  void smooth_l1_chamfer_distance_updateGradInput(const int batch_size, const int n_points, const float* input, const float* target, const int* indices, float* grad_input, bool size_average);
  ]]

  local libname = scriptPath() .. '../cpp/gpu/build/libgpu.so'
  local found = pcall(function () lib.gpu = ffi.load(libname) end)

  if found then
    print('[Lib] found ' .. libname)
  else
    print('[Info] could not find GPU module, tried ' .. libname)
    print('[Info] will continue without GPU module')
    lib.gpu = false
    --os.exit()
  end
end