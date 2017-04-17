require 'nn'
mklnn= require 'mklnn.env'
require('mklnn.ffi')

local C = mklnn.C
local ffi = require 'ffi'

local wrapper = function(f,...)
   print('call functoin ',f,' in wrapper')
   return C[f](...)
end
mklnn.wrapper = wrapper


require('mklnn.SpatialConvolution')
require('mklnn.test')
return mklnn
