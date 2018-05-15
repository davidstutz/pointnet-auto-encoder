-- Train an auto-encoder using config.json.

require('torch')
require('nn')
require('nnx')
require('optim')
require('hdf5')
require('cunn')
require('cunnx')
require('lfs')

package.path = package.path .. ";" .. lfs.currentdir() .. '/?/th/init.lua'
lib = require('lib')

--- Append the tensor tensor to the tensor acc which may initially be nil.
local function appendTensor(acc, tensor, dim)
  local dim = dim or 1
  if acc == nil then
    acc = tensor:float()
  else
    acc = torch.cat(acc, tensor:float(), dim)
  end

  return acc
end

inputFile = '/BS/dstutz/work/data/3d/training_prior_points_10000_5000_32x32x32_easy.h5'
valInputFile = '/BS/dstutz/work/data/3d/validation_points_1000_5000_32x32x32_easy.h5'

inputs = lib.utils.readHDF5(inputFile)
print('[Training] read ' .. inputFile)
valInputs = lib.utils.readHDF5(valInputFile)
print('[Training] read ' .. valInputFile)

--inputs = inputs + 0.5
--valInputs = valInputs + 0.5

shuffle = torch.randperm(inputs:size(2))
shuffle = shuffle:narrow(1, 1, 1000)
shuffle = shuffle:long()

inputs = inputs:index(2, shuffle)
valInputs = valInputs:index(2, shuffle)

-- Check dimensions.
N = inputs:size(1)
nPoints = inputs:size(2)
print('[Training] using ' .. nPoints .. ' points')

inputs = nn.utils.addSingletonDimension(inputs, 2)
valInputs = nn.utils.addSingletonDimension(valInputs, 2)

outputs = inputs:clone()
valOutputs = valInputs:clone()

--- This is a model for testing which allows the network, at least in theory, to learn
-- the identity mapping without any bottleneck
-- @return model
local function model1()
  local model = nn.Sequential()
  model:add(nn.Identity())

  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.SpatialConvolution(1, 128, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialConvolution(128, 128, 3, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialConvolution(128, 256, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(256))

  model:add(nn.SpatialConvolution(256, 4, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.SpatialConvolution(4, 256, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(256))

  model:add(nn.SpatialConvolution(256, 128, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialFullConvolution(128, 128, 3, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialConvolution(128, 1, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  return model
end

--- This is a bottleneck model where average pooling is used to compute a 256-dimensional bottleneck.
-- @return model
local function model2()
  local model = nn.Sequential()
  model:add(nn.Identity())

  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.SpatialConvolution(1, 128, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialConvolution(128, 256, 3, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(256))

  model:add(nn.SpatialConvolution(256, 256, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(256))

  model:add(nn.SpatialAveragePooling(1, 1000, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  --model:add(nn.SpatialConvolution(256, 256, 1, 1, 1, 1, 0, 0))
  --if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.SpatialFullConvolution(256, 256, 1, 1000, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(256))

  model:add(nn.SpatialConvolution(256, 128, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialFullConvolution(128, 128, 3, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  model:add(nn.ReLU(true))
  model:add(nn.SpatialBatchNormalization(128))

  model:add(nn.SpatialConvolution(128, 1, 1, 1, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end

  return model
end

--- This is the general model where encoder and decoder can be adapted and the
-- bottleneck is computed using a linear layer.
-- @return model
local function model3()
  local model = nn.Sequential()
  local autoEncoderConfig = lib.pointAutoEncoder.config
  autoEncoderConfig.encoder.features = {64, 128, 256, 512}
  autoEncoderConfig.encoder.transfers = {true, true, true, true}
  autoEncoderConfig.encoder.normalizations = {true, true, true, true}
  autoEncoderConfig.encoder.transfer = nn.ReLU

  autoEncoderConfig.decoder.features = {512, 256, 128, 64}
  autoEncoderConfig.decoder.transfers = {true, true, true, true}
  autoEncoderConfig.decoder.normalizations = {true, true, true, true}
  autoEncoderConfig.decoder.transfer = nn.ReLU

  autoEncoderConfig.inputNumber = nPoints
  autoEncoderConfig.outputNumber = nPoints
  autoEncoderConfig.code = 10

  local model, context = lib.pointAutoEncoder.autoEncoder(model, autoEncoderConfig)
  return model
end

model = model2()
model = model:cuda()
print(model)

-- Criterion.
criterion = nn.SmoothL1ChamferDistanceCriterion()
criterion.sizeAverage = false
criterion = criterion:cuda()

errCriterion = nn.MaxDistanceCriterion()
errCriterion = errCriterion:cuda()

-- Learning hyperparameters.
batchSize = 32
learningRate = 0.05
momentum = 0.5
weightDecay = 0.0001
lossIterations = 10
testIterations = 500
decayIterations = 100

minimumLearningRate = 0.000000001
decayLearningRate = 0.75
decayMomentum = 1.05
maximumMomentum = 0.95

parameters, gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()

-- Smoothed statistics.
epochs = 20
iterations = epochs*math.floor(N/batchSize)
protocol = torch.Tensor(iterations, 2)

for t = 1, iterations do

  -- Sample a random batch from the dataset.
  local shuffle = torch.randperm(N)
  shuffle = shuffle:narrow(1, 1, batchSize)
  shuffle = shuffle:long()

  local input = inputs:index(1, shuffle)
  local output = outputs:index(1, shuffle)

  -- Appyl a random permutation on inputs and outputs
  -- to enforce invariance to the order of points in input and output.
  for b = 1, input:size(1) do
    local shuffle = torch.randperm(input:size(3)):long()
    input[b] = input[b]:index(2, shuffle)
    --shuffle = torch.randperm(input:size(3)):long()
    output[b] = output[b]:index(2, shuffle)
  end

  input = input:cuda()
  output = output:cuda()

  --- Definition of the objective on the current mini-batch.
  -- This will be the objective fed to the optimization algorithm.
  -- @param x input parameters
  -- @return object value, gradients
  local feval = function(x)

    -- Get new parameters.
    if x ~= parameters then
      parameters:copy(x)
    end

    -- Reset gradients
    gradParameters:zero()

    -- Evaluate function on mini-batch.
    local pred = model:forward(input)
    local f = criterion:forward(pred, output)
    local d = errCriterion:forward(pred, output)

    protocol[t][1] = f
    protocol[t][2] = d

    -- Estimate df/dW.
    local df_do = criterion:backward(pred, input)
    model:backward(input, df_do)

    -- Weight decay:
    if weightDecay > 0 then
       f = f + weightDecay * torch.norm(parameters,2)^2/2
       gradParameters:add(parameters:clone():mul(weightDecay))
    end

    -- return f and df/dX
    return f, gradParameters
  end

  adamState = adamState or {
    learningRate = learningRate,
    momentum = momentum,
    learningRateDecay = 0 -- will be done manually below
  }

  -- Returns the new parameters and the objective evaluated
  -- before the update.
  --p, f = optim.adam(feval, parameters, adamState)
  p, f = optim.adam(feval, parameters, adamState)

  -- Report a smoothed loss instead of batch loss.
  if t%lossIterations == 0 then
    local smoothedLoss = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 1, 1))
    local smoothedDistance = torch.mean(protocol:narrow(1, t - lossIterations + 1, lossIterations):narrow(2, 2, 1))
    print('[Training] ' .. t .. ': ' .. smoothedLoss .. ' | ' .. smoothedDistance)
  end

  -- Validate on validation set.
  if t%testIterations == 0 then

    local valBatchSize = batchSize
    local valNumBatches = math.floor(valInputs:size(1)/valBatchSize)

    local valLoss = 0
    local valErr = 0
    local accValPreds = nil

    for b = 0, valNumBatches - 1 do
      local input = valInputs:narrow(1, b*valBatchSize + 1, math.min((b + 1)*valBatchSize - b*valBatchSize, valInputs:size(1) - b*valBatchSize))
      input = input:cuda()

      local output = valOutputs:narrow(1, b*valBatchSize + 1, math.min((b + 1)*valBatchSize - b*valBatchSize, valOutputs:size(1) - b*valBatchSize))
      output = output:cuda()

      local valPreds = model:forward(input)
      accValPreds = appendTensor(accValPreds, valPreds)

      valLoss = valLoss + criterion:forward(valPreds, output)
      valErr = valErr + errCriterion:forward(valPreds, output)
    end

    print('[Training] ' .. t .. ': validation loss ' .. valLoss/valNumBatches)
    print('[Training] ' .. t .. ': max error ' .. valErr/valNumBatches)

    predFile = t .. '.h5'
    lib.utils.writeHDF5(predFile, accValPreds)
    print('[Training] wrote ' .. predFile)
  end

  -- Decay learning rate.
  if t%decayIterations == 0 then
    learningRate = math.max(minimumLearningRate, learningRate*decayLearningRate)
    momentum = math.min(maximumMomentum, momentum*decayMomentum)

    print('[Training] ' .. t .. ': learning rate ' .. learningRate)
    print('[Training] ' .. t .. ': momentum ' .. momentum)
  end
end

torch.save('model.dat', model)
print('[Training] snapshot model.dat')