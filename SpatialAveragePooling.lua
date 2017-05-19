local SpatialAveragePooling, parent = torch.class('mklnn.SpatialAveragePooling', 'nn.Module')

local wrapper = mklnn.wrapper
local getType = mklnn.getType
function SpatialAveragePooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH
   self.dW = dW or 1
   self.dH = dH or 1
   self.padW = padW or 0
   self.padH = padH or 0
   self.ceil_mode = false
   self.count_include_pad = true
   self.divide = true

end

function SpatialAveragePooling:ceil()
   self.ceil_mode = true
   return self
end

function SpatialAveragePooling:floor()
   self.ceil_mode = false
   return self
end

function SpatialAveragePooling:setCountIncludePad()
   self.count_include_pad = true
   return self
end

function SpatialAveragePooling:setCountExcludePad()
   self.count_include_pad = false
   return self
end

local function backwardCompatible(self)
   if self.ceil_mode == nil then
      self.ceil_mode = false
      self.count_include_pad = true
      self.padH = 0
      self.padW = 0
   end
end

function SpatialAveragePooling:updateOutput(input)
   if self.dnnPrimitives then
      self.mkldnnInitOk = 1 
   else
      self.mkldnnInitOk = 0 
   end 
   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(16)
   self.output = self.output:mkl()
   self.gradInput = self.gradInput:mkl()
   backwardCompatible(self)

   wrapper(getType(input),'SpatialAveragePooling_updateOutput',
      input:cdata(),
      self.output:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      self.ceil_mode,
      self.count_include_pad,
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk
   )

   -- for backward compatibility with saved models
   -- which are not supposed to have "divide" field
   if not self.divide then
     self.output:mul(self.kW*self.kH)
   end
   return self.output
end

function SpatialAveragePooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      wrapper(getType(input),'SpatialAveragePooling_updateGradInput',
	 input:cdata(),
	 gradOutput:cdata(),
	 self.gradInput:cdata(),
	 self.kW, self.kH,
	 self.dW, self.dH,
	 self.padW, self.padH,
	 self.ceil_mode,
	 self.count_include_pad,
	 self.dnnPrimitives:cdata(),
         self.mkldnnInitOk
      )
      -- for backward compatibility
      if not self.divide then
         self.gradInput:mul(self.kW*self.kH)
      end

      return self.gradInput
   end
end

function SpatialAveragePooling:__tostring__()
   local s = string.format('%s(%d,%d,%d,%d', torch.type(self),
                            self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'
   return s 
end
