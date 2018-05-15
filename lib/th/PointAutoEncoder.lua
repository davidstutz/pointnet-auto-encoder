-- Implementation of simple convolutional encoder/decoder achitecture with
-- variable number of channels, layers and kernel sizes.

require('nn')
require('cunn')
require('nnx')
require('cunnx')

local models = {}

--- Default options for the auto-encoder, encoder and decoder models.
models.config = {
  encoder = {
    features = nil, -- equivalent to channesl for convolutional auto encoders
                    -- the enumber of features per point per layer
    transfers = nil,
    normalizations = nil,
    transfer = nn.ReLU,
  },
  decoder = {
    features = nil, -- equivalent to channesl for convolutional auto encoders
                    -- the enumber of features per point per layer
    transfers = nil,
    normalizations = nil,
    transfer = nn.ReLU,
  },
  code = 0,
  outputNumber = 0, -- number of predicted points
  inputNumber = 0, -- number of input points
  printDimensions = false, -- whether to print dimensions after each layer
  checkNaN = false, -- whether to check for NaN values after each layer
}

--- Simple encoder structure as also explained by models.autoEncoder.
-- @param model model to add encoder to
-- @param config configuration as illustrated in models.autoEncoderConfig
-- @return model
function models.encoder(model, config)
  assert(config.encoder)
  assert(config.encoder.features)
  assert(#config.encoder.features > 1)
  assert(config.encoder.transfers == nil or #config.encoder.transfers == #config.encoder.features)
  assert(config.encoder.normalizations == nil or #config.encoder.normalizations == #config.encoder.features)
  assert(config.encoder.transfer)
  assert(config.inputNumber > 0)
  assert(config.code > 0)

  local features = config.encoder.features
  local transfer = config.encoder.transfer
  local transfers = config.encoder.transfers
  local normalizations = config.encoder.normalizations
  local inputNumber = config.inputNumber
  local printDimensions = config.printDimensions
  local checkNaN = config.checkNaN
  local code = config.code

  for i = 1, #features do

    -- First layer needs to reduce the 3 dimensions of the points.
    if i == 1 then
      model:add(nn.SpatialConvolution(1, features[i], 3, 1, 1, 1, 0, 0))
    else
      model:add(nn.SpatialConvolution(features[i - 1], features[i], 1, 1, 1, 1, 0, 0))
    end

    if printDimensions then model:add(nn.PrintDimensions()) end
    if checkNaN then model:add(nn.CheckNaN()) end

    if normalizations and normalizations[i] then
      model:add(nn.SpatialBatchNormalization(features[i]))
      if printDimensions then model:add(nn.PrintDimensions()) end
      if checkNaN then model:add(nn.CheckNaN()) end
    end

    if transfers and transfers[i] then
      model:add(transfer(true))
      if printDimensions then model:add(nn.PrintDimensions()) end
      if checkNaN then model:add(nn.CheckNaN()) end
    end
  end

  -- TODO replace by custom, number independent layer!
  model:add(nn.SpatialAveragePooling(1, inputNumber, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end
  if checkNaN then model:add(nn.CheckNaN()) end

  model:add(nn.View(features[#features]))
  model:add(nn.Linear(features[#features], code))
  -- No checks ...

  return model, {}
end

--- Simple decoder structure as also explained by models.autoEncoder.
-- @param model model to add decoder to
-- @param config configuration as illustrated in models.autoEncoderConfig
-- @return model
function models.decoder(model, config)
  assert(config.decoder)
  assert(config.decoder.features)
  assert(#config.decoder.features > 1)
  assert(config.decoder.transfers == nil or #config.decoder.transfers == #config.decoder.features)
  assert(config.decoder.normalizations == nil or #config.decoder.normalizations == #config.decoder.features)
  assert(config.decoder.transfer)
  assert(config.outputNumber > 0)

  local features = config.decoder.features
  local transfer = config.decoder.transfer
  local transfers = config.decoder.transfers
  local normalizations = config.decoder.normalizations
  local outputNumber = config.outputNumber
  local code = config.code
  local printDimensions = config.printDimensions
  local checkNaN = config.checkNaN

  model:add(nn.Linear(code, code))
  if printDimensions then model:add(nn.PrintDimensions()) end
  if checkNaN then model:add(nn.CheckNaN()) end

  model:add(nn.View(code, 1, 1))
  model:add(nn.SpatialFullConvolution(code, features[1], 1, outputNumber, 1, 1, 0, 0))
  if printDimensions then model:add(nn.PrintDimensions()) end
  if checkNaN then model:add(nn.CheckNaN()) end

  if normalizations and normalizations[1] then
    model:add(nn.SpatialBatchNormalization(features[1]))
    if printDimensions then model:add(nn.PrintDimensions()) end
    if checkNaN then model:add(nn.CheckNaN()) end
  end

  if transfers and transfers[1] then
    model:add(transfer(true))
    if printDimensions then model:add(nn.PrintDimensions()) end
    if checkNaN then model:add(nn.CheckNaN()) end
  end

  for i = 2, #features do
    if i == 2 then
      model:add(nn.SpatialFullConvolution(features[i - 1], features[i], 3, 1, 1, 1, 0, 0))
    else
      model:add(nn.SpatialConvolution(features[i - 1], features[i], 1, 1, 1, 1, 0, 0))
    end

    if printDimensions then model:add(nn.PrintDimensions()) end
    if checkNaN then model:add(nn.CheckNaN()) end

    if normalizations and normalizations[i] then
      model:add(nn.SpatialBatchNormalization(features[i]))
      if printDimensions then model:add(nn.PrintDimensions()) end
      if checkNaN then model:add(nn.CheckNaN()) end
    end

    if transfers and transfers[i] then
      model:add(transfer(true))
      if printDimensions then model:add(nn.PrintDimensions()) end
      if checkNaN then model:add(nn.CheckNaN()) end
    end
  end

  model:add(nn.SpatialConvolution(features[#features], 1, 1, 1, 1, 1, 0, 0))
  -- No checks ...

  return model, {}
end

--- Sets up a decoder/encoder architecture with the given code dimensionality,
-- number of channels for each layer and the corresponding kernel sizes.
-- @param model model to add encoder and decoder to
-- @param config configuration as illustrated in models.autoEncoderConfig
-- @return model
function models.autoEncoder(model, config)
  local model = model or nn.Sequential()

  local context = {}
  local encoder = nn.Sequential()
  encoder, context = models.encoder(encoder, config)

  local decoder = nn.Sequential()
  decoder, _ = models.decoder(decoder, config)

  model:add(encoder)
  model:add(decoder)

  context['encoder'] = encoder
  context['decoder'] = decoder
  return model, context
end

lib.pointAutoEncoder = models