require('torch')
require('nn')
require('os')

--- @class CheckNaN
local CheckNaN, CheckNaNParent = torch.class('nn.CheckNaN', 'nn.Module')

--- Initialize.
function CheckNaN:__init()
  -- Nothing ...
end

--- Print dimensions of last layer.
-- @param input output of last layer
-- @return unchanged output of last layer
function CheckNaN:updateOutput(input)
  self.output = input

  if torch.any(input:ne(input)) then
    print('NaN value detected (forward)')
    print(input:size())
    os.exit(1)
  end

  return self.output
end

--- Print the gradients of the next layer.
-- @param input original input of last layer
-- @param gradOutput gradients of next layer
-- @return unchanged gradients of next layer
function CheckNaN:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput

  if torch.any(gradOutput:ne(gradOutput)) then
    print('NaN value detected (backward)')
    print(gradOutput:size())
    os.exit(1)
  end

  return self.gradInput
end