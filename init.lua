require 'nn'
require 'mkltorch'
mklnn= require 'mklnn.env'
require('mklnn.ffi')

local C = mklnn.C
local ffi = require 'ffi'

local wrapper = function(dataType,f,...)
   local funcName = 'MKLNN_'..dataType..f
   print("funcName = ", funcName)
   return C[funcName](...)
end
mklnn.wrapper = wrapper


local typeMap = {

   ['torch.FloatTensor']   = 'Float',
   ['torch.DoubleTensor']  = 'Double',
   ['torch.LongTensor']    = 'Long',
   ['torch.MKLFloatTensor']   = 'Float',
   ['torch.MKLDoubleTensor']  = 'Double',
   ['torch.MKLLongTensor']    = 'Long',

}
local getType = function(tensor)
   local tensorType = tensor:type()
   return typeMap[tensorType]
end
mklnn.getType = getType

require('mklnn.SpatialConvolution')
require('mklnn.test')
require('mklnn.Threshold')
require('mklnn.ReLU')
require('mklnn.SpatialMaxPooling')
require('mklnn.BatchNormalization')
require('mklnn.SpatialBatchNormalization')
require('mklnn.SpatialCrossMapLRN')
require('mklnn.I2U')

mklnn.convert = require('mklnn.Convert')
return mklnn
