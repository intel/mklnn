--[[
   This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy

   This implementation is useful for inputs NOT coming from convolution layers.
   For convolution layers, use nn.SpatialBatchNormalization.

   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
        standard-deviation(x)
   where gamma and beta are learnable parameters.

   The learning of gamma and beta is optional.

   Usage:
   with    learnable parameters: nn.BatchNormalization(N [,eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(N [,eps] [,momentum], false)

   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5

   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentum of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--


local BN,parent = torch.class('mklnn.BatchNormalization', 'nn.Module')
local THNN = require 'nn.THNN'

local wrapper = mklnn.wrapper
local getType = mklnn.getType

BN.__version = 2

-- expected dimension of input
BN.nDim = 2

function BN:__init(nOutput, eps, momentum, affine)
   parent.__init(self)
   assert(nOutput and type(nOutput) == 'number',
          'Missing argument #1: dimensionality of input. ')
   assert(nOutput ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nOutput,  eps, momentum, false) ')
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = true
   end
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   self.running_mean = torch.zeros(nOutput)
   self.running_var = torch.ones(nOutput)

   --self:setEngine(1)


   if self.affine then
      self.weight = torch.Tensor(nOutput)
      self.bias = torch.Tensor(nOutput)
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
      self:reset()
   end
end

function BN:reset()
   if self.weight then
      self.weight:uniform()
   end
   if self.bias then
      self.bias:zero()
   end
   self.running_mean:zero()
   self.running_var:fill(1)
end

function BN:checkInputDim(input)
   --[[assert(input:tensor():dim() == self.nDim, string.format(
      'only mini-batch supported (%dD tensor), got %dD tensor instead',
      self.nDim, input:tensor():dim()))
   assert(input:size(2) == self.running_mean:nElement(), string.format(
      'got %d-feature tensor, expected %d',
      input:size(2), self.running_mean:nElement()))
   ]]--
end

local function makeContiguous(self, input, gradOutput)
   --[[
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput then
      if not gradOutput:isContiguous() then
         self._gradOutput = self._gradOutput or gradOutput.new()
         self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
         gradOutput = self._gradOutput
      end
   end
   ]]--
   return input, gradOutput
end

function BN:updateOutput(input)
   self:checkInputDim(input)

   if self.dnnPrimitives then
      self.mkldnnInitOk = 1
   else
      self.mkldnnInitOk = 0
   end
   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(15)


   --input = makeContiguous(self, input)
   self.output = self.output:mkl()
   --self.output:resizeAs(input)
   wrapper(getType(input),'BatchNormalization_updateOutput',
      input:cdata(),
      self.output:cdata(),
      THNN.optionalTensor(self.weight),
      THNN.optionalTensor(self.bias),
      self.running_mean:cdata(),
      self.running_var:cdata(),
      self.train,
      self.momentum,
      self.eps,
      self.dnnPrimitives:cdata(),self.mkldnnInitOk)
   return self.output
end

local function backward(self, input, gradOutput, scale, gradInput, gradWeight, gradBias)
   self:checkInputDim(input)
   self:checkInputDim(gradOutput)
   input, gradOutput = makeContiguous(self, input, gradOutput)
   self.gradInput = self.gradInput:mkl()
   scale = scale or 1
   --if gradInput then
   --   gradInput:resizeAs(gradOutput)
   --end
   
   if gradInput then
      wrapper(getType(input),'BatchNormalization_backward',
         input:cdata(),
         gradOutput:cdata(),
         --THNN.optionalTensor(gradInput),
         self.gradInput:cdata(),
         THNN.optionalTensor(gradWeight),
         THNN.optionalTensor(gradBias),
         THNN.optionalTensor(self.weight),
         self.running_mean:cdata(),
         self.running_var:cdata(),
         self.train,
         scale,
         self.eps,
         self.dnnPrimitives:cdata(),self.mkldnnInitOk)
   end
   return self.gradInput
end

function BN:backward(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, self.gradInput, self.gradWeight, self.gradBias)
end

function BN:updateGradInput(input, gradOutput)
   return backward(self, input, gradOutput, 1, self.gradInput)
end

function BN:accGradParameters(input, gradOutput, scale)
   return backward(self, input, gradOutput, scale, nil, self.gradWeight, self.gradBias)
end

function BN:read(file, version)
   parent.read(self, file)
   if version < 2 then
      if self.running_std then
         self.running_var = self.running_std:pow(-2):add(-self.eps)
         self.running_std = nil
      end
   end
end

function BN:clearState()
   -- first 5 buffers are not present in the current implementation,
   -- but we keep them for cleaning old saved models
   nn.utils.clear(self, {
      'buffer',
      'buffer2',
      'centered',
      'std',
      'normalized',
      '_input',
      '_gradOutput',
   })
   return parent.clearState(self)
end
