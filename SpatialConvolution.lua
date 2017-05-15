local SpatialConvolution, parent = torch.class('mklnn.SpatialConvolution', 'nn.Module')

local wrapper = mklnn.wrapper
local getType = mklnn.getType
function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH,group)
   parent.__init(self)
   
   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW

   self.group = group or 1
   self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW/self.group)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW/self.group)
   self.gradBias = torch.Tensor(nOutputPlane)



   self:reset()
end

function SpatialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end)  
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
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

function SpatialConvolution:updateOutput(input)
   if self.dnnPrimitives then
      self.mkldnnInitOk = 1
   else
      self.mkldnnInitOk = 0
   end
   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(30)

   self.output = self.output:mkl()
   self.gradInput = self.gradInput:mkl()

   self.finput = torch.FloatTensor()
   self.fgradInput = torch.FloatTensor()
   if self.padding then
      self.padW = self.padding
      self.padH = self.padding
      self.padding = nil
   end
   input = makeContiguous(self, input)
   wrapper(getType(input),'SpatialConvolution_forward',
      input:cdata(),
      self.output:cdata(),
      self.weight:cdata(),
      self.bias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dnnPrimitives:cdata(),self.mkldnnInitOk,
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,self.group
   )
   return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      input, gradOutput = makeContiguous(self, input, gradOutput)
      wrapper(getType(input),'SpatialConvolution_bwdData',
         input:cdata(),
         gradOutput:cdata(),
         self.gradInput:cdata(),
         self.weight:cdata(),
         self.bias:cdata(),
         self.finput:cdata(),
         self.fgradInput:cdata(),
         self.dnnPrimitives:cdata(),self.mkldnnInitOk,
         self.kW, self.kH,
         self.dW, self.dH,
         self.padW, self.padH,self.group
         )
   return self.gradInput
   end
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input, gradOutput = makeContiguous(self, input, gradOutput)
   wrapper(getType(input),'SpatialConvolution_bwdFilter',
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      self.gradBias:cdata(),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.dnnPrimitives:cdata(),self.mkldnnInitOk,
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale,self.group
   )
end

function SpatialConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   self.fgradInput = self.fgradInput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function SpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
   return s .. ')'
end

function SpatialConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
   return parent.clearState(self)
end
