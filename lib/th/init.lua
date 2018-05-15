-- Allow to require files from this directory ...
--require('lfs')
--package.path = package.path .. ";" .. lfs.currentdir() .. '/lib/th/?.lua'
--print(package.path)
lib = {}

-- Include CPU/GPU modules first.
include('ffi.lua')
include('Utils.lua')
include('CheckNaN.lua')
include('PrintDimensions.lua')
include('MaxDistanceCriterion.lua')
include('ChamferDistanceCriterion.lua')
include('SmoothL1ChamferDistanceCriterion.lua')
include('PointAutoEncoder.lua')

return lib