local LRN, parent = torch.class('mklnn.SpatialCrossMapLRN', 'nn.Module')
local ffi = require 'ffi'

local wrapper = mklnn.wrapper
local getType = mklnn.getType

function LRN:__init(size, alpha, beta, k)
   parent.__init(self)
   self.size = size or 5
   self.alpha = alpha or 1e-4
   self.beta = beta or 0.75
   self.k = k or 1.0
   assert(self.size >= 1 and self.size <= 16, "size has to be between 1 and 16")
   assert(self.k >= 1e-5, "k has to be greater than 1e-5")
   assert(self.beta >= 0.01, "Beta has to be > 0.01")

end

function LRN:updateOutput(input)
   if self.K then self.k, self.K = self.K, nil end
   if self.dnnPrimitives then
      self.mkldnnInitOk = 1 
   else
      self.mkldnnInitOk = 0 
   end 
   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(30)
   self.output = self.output:mkl()
   self.gradInput = self.gradInput:mkl()
   --self.output:resizeAs(input)
   wrapper(getType(input),'CrossChannelLRN_updateOutput',
      input:cdata(),
      self.output:cdata(),
      self.size,
      self.alpha,
      self.beta,
      self.k,
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk
      )
   return self.output
end

function LRN:updateGradInput(input, gradOutput)
   if not self.gradInput then return end

   --self.gradInput:resizeAs(input)
   --assert(gradOutput:dim() == 3 or gradOutput:dim() == 4);
   --if not gradOutput:isContiguous() then
   --   self._gradOutput = self._gradOutput or gradOutput.new()
   --   self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
   --   gradOutput = self._gradOutput
   --end
   wrapper(getType(input),'CrossChannelLRN_backward',
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.size,
      self.alpha,
      self.beta,
      self.k,
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk
      )
   return self.gradInput
end

function LRN:write(f)
   --self:clearDesc()
   local var = {}
   for k,v in pairs(self) do
      var[k] = v
   end
   f:writeObject(var)
end

--[[
function LRN:clearState()
   self:clearDesc()
   nn.utils.clear(self, '_gradOutput')
   return nn.Module.clearState(self)
end
]]--
