local SpatialMaxPooling, parent = torch.class('mklnn.SpatialMaxPooling', 'nn.Module')

local wrapper = mklnn.wrapper
local getType = mklnn.getType

function SpatialMaxPooling:__init(kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.padW = padW or 0
   self.padH = padH or 0

   self:setEngine(1)


   self.ceil_mode = false
   self.indices = torch.Tensor()
end

function SpatialMaxPooling:ceil()
  self.ceil_mode = true
  return self
end

function SpatialMaxPooling:floor()
  self.ceil_mode = false
  return self
end

function SpatialMaxPooling:updateOutput(input)
   --[[
   if self.timerEnable then
	startTime = sys.clock()
   end
   ]]--
   self:updateForLoadSnapshot()
   if self.initStep == 0 then
   	self.initStep = 1
      self.dnnPrimitives = torch.LongTensor(16)
   else
	self.mkldnnInitOk = 1
   end
   self.indices = self.indices or input.new()
   -- backward compatibility
   self.ceil_mode = self.ceil_mode or false
   self.padW = self.padW or 0
   self.padH = self.padH or 0
   --[[
   if self.compare  then
	   input.THNN.SpatialMaxPooling_updateOutput(
	   --wrapper(getType(input),'SpatialMaxPooling_updateOutput'
	      input:cdata(),
	      self.output:cdata(),
	      self.indices:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode
	   )
	   tmpOut = torch.Tensor(self.output:size())
	   --input.THNN.SpatialMaxPooling_MKLDNN_updateOutput(
	   wrapper(getType(input),'SpatialMaxPooling_updateOutput',
	      input:cdata(),
	      tmpOut:cdata(),
	      self.indices:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	       self.ceil_mode,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	   )
	   outSize = tonumber(tmpOut:cdata().size[0]*tmpOut:cdata().size[1]*tmpOut:cdata().size[2]*tmpOut:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.output:cdata(), outSize,4)
   else
   ]]
	   --input.THNN.SpatialMaxPooling_MKLDNN_updateOutput(
	   wrapper(getType(input),'SpatialMaxPooling_updateOutput',
	      input:cdata(),
	      self.output:cdata(),
	      self.indices:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	       self.ceil_mode,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	   )
   --end
   --[[
   if self.timerEnable then
        print("mkldnn SpatialMaxPooling forward time = ,",self.timeForward," backward time =",self.timeBackward)
        sys.maxpoolingTime_forward = sys.maxpoolingTime_forward + self.timeForward 
        sys.maxpoolingTime_backward = sys.maxpoolingTime_backward + self.timeBackward
        self.timeForward = sys.clock() - startTime
        self.cnt = self.cnt + 1
   end]]--
   return self.output
end

function SpatialMaxPooling:updateGradInput(input, gradOutput)
   --[[if self.timerEnable then
	startTime = sys.clock()
   end--]]
   --[[
   if self.compare  then

	input.THNN.SpatialMaxPooling_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      self.gradInput:cdata(),
	      self.indices:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode
	   )
	outSize = tonumber(self.gradInput:cdata().size[0] *self.gradInput:cdata().size[1] *self.gradInput:cdata().size[2] *self.gradInput:cdata().size[3])
	tmpOut = torch.Tensor(outSize)
	input.THNN.SpatialMaxPooling_MKLDNN_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      tmpOut:cdata(),
	      self.indices:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode,
	      self.dnnPrimitives:cdata(),self.mkldnnInitOk
	   )
	      input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.gradInput:cdata(), outSize,5)

   else]]--

	--input.THNN.SpatialMaxPooling_MKLDNN_updateGradInput(
	wrapper(getType(input),'SpatialMaxPooling_updateGradInput',
	      input:cdata(),
	      gradOutput:cdata(),
	      self.gradInput:cdata(),
	      self.indices:cdata(),
	      self.kW, self.kH,
	      self.dW, self.dH,
	      self.padW, self.padH,
	      self.ceil_mode,
	      self.dnnPrimitives:cdata(),self.mkldnnInitOk
	   )
   --[[
   if self.timerEnable then
	self.timeBackward = sys.clock() - startTime
   end
   ]]--
   return self.gradInput
end

-- for backward compat
function SpatialMaxPooling:empty()
   self:clearState()
end

function SpatialMaxPooling:__tostring__()
   local s =  string.format('%s(%d,%d,%d,%d', torch.type(self),
                            self.kW, self.kH, self.dW, self.dH)
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
      s = s .. ',' .. self.padW .. ','.. self.padH
   end
   s = s .. ')'

   return s
end

function SpatialMaxPooling:clearState()
   if self.indices then
      self.indices:set()
   end
   return parent.clearState(self)
end
