require('torch')
require('nn')

--- @class PrintDimensions
local PrintDimensions, PrintDimensionsParent = torch.class('nn.PrintDimensions', 'nn.Module')

--- Initialize.
function PrintDimensions:__init()
  -- Nothing ...
end

--- Print dimensions of last layer.
-- @param input output of last layer
-- @return unchanged output of last layer
function PrintDimensions:updateOutput(input)
  self.output = input
  print(#self.output)
  return self.output
end

--- Print the gradients of the next layer.
-- @param input original input of last layer
-- @param gradOutput gradients of next layer
-- @return unchanged gradients of next layer
function PrintDimensions:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  print(#self.gradInput)
  return self.gradInput
end