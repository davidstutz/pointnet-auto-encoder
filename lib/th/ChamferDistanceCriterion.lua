require('torch')
require('nn')

--- @class ChamferDistanceCriterion
local ChamferDistanceCriterion, ChamferDistanceCriterionParent = torch.class('nn.ChamferDistanceCriterion', 'nn.Criterion')

--- Initialize.
function ChamferDistanceCriterion:__init()
  self.sizeAverage = false
  self.indices = nil
end

--- Compute forward pass.
-- @param input inputs
-- @param target targets
-- @param output
function ChamferDistanceCriterion:updateOutput(input, target)
  assert(input:dim() == target:dim())
  assert(input:size(1) == target:size(1))
  assert(input:size(2) == 1)
  assert(input:size(3) == target:size(3))
  assert(input:size(4) == target:size(4))

  local batchSize = input:size(1)
  local nPoints = input:size(3)

  if input:type() == 'torch.FloatTensor' then
    assert(lib.cpu)
    self.indices = torch.IntTensor(batchSize, nPoints, 2)
    self.output = lib.cpu.chamfer_distance_updateOutput(batchSize, nPoints, input:data(), target:data(), self.indices:data(), self.sizeAverage)
  elseif input:type() == 'torch.CudaTensor' then
    assert(lib.gpu)
    self.indices = torch.CudaIntTensor(batchSize, nPoints, 2)
    self.output = lib.gpu.fast_chamfer_distance_updateOutput(batchSize, nPoints, input:data(), target:data(), self.indices:data(), self.sizeAverage)
  else
    assert(false)
  end

  return self.output
end

--- Compute the backward pass.
-- @param input inputs
-- @param target targets
-- @return gradients with respect to input
function ChamferDistanceCriterion:updateGradInput(input, target)
  assert(self.indices ~= nil)
  assert(input:dim() == target:dim())
  assert(input:size(1) == target:size(1))
  assert(input:size(2) == 1)
  assert(input:size(3) == target:size(3))
  assert(input:size(4) == target:size(4))

  self.gradInput = input:clone()
  local batchSize = input:size(1)
  local nPoints = input:size(3)

  if input:type() == 'torch.FloatTensor' then
    assert(lib.cpu)
    lib.cpu.chamfer_distance_updateGradInput(batchSize, nPoints, input:data(), target:data(), self.indices:data(), self.gradInput:data(), self.sizeAverage)
  elseif input:type() == 'torch.CudaTensor' then
    assert(lib.gpu)
    lib.gpu.chamfer_distance_updateGradInput(batchSize, nPoints, input:data(), target:data(), self.indices:data(), self.gradInput:data(), self.sizeAverage)
  else
    assert(false)
  end

  return self.gradInput
end