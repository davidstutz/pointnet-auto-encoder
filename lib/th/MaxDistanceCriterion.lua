require('torch')
require('nn')

--- @class MaxDistanceCriterion
local MaxDistanceCriterion, MaxDistanceCriterionParent = torch.class('nn.MaxDistanceCriterion', 'nn.Criterion')

--- Initialize.
function MaxDistanceCriterion:__init()

end

--- Compute forward pass.
-- @param input inputs
-- @param target targets
-- @param output
function MaxDistanceCriterion:updateOutput(input, target)
  assert(input:dim() == target:dim())
  assert(input:size(1) == target:size(1))
  assert(input:size(2) == 1)
  assert(input:size(3) == target:size(3))
  assert(input:size(4) == target:size(4))

  local batchSize = input:size(1)
  local nPoints = input:size(3)

  if input:type() == 'torch.FloatTensor' then
    assert(lib.cpu)
    self.output = lib.cpu.maxdistance_updateOutput(batchSize, nPoints, input:data(), target:data())
  elseif input:type() == 'torch.CudaTensor' then
    assert(lib.gpu)
    self.output = lib.gpu.max_distance_updateOutput(batchSize, nPoints, input:data(), target:data())
  else
    assert(false)
  end

  return self.output
end

--- Compute the backward pass.
-- @param input inputs
-- @param target targets
-- @return gradients with respect to input
function MaxDistanceCriterion:updateGradInput(input, target)
  assert(false)
end