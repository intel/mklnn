local LRN, parent = torch.class('mklnn.LRN', 'nn.Module')
local ffi = require 'ffi'

local wrapper = mklnn.wrapper
local getType = mklnn.getType

function LRN:__init(size, alpha, beta, k)
   parent.__init(self)
   self.size = size or 5
   self.alpha = alpha or 1e-4
   self.beta = beta or 0.75
   self.k = k or 1.0
   self:setEngine(1)
   assert(self.size >= 1 and self.size <= 16, "size has to be between 1 and 16")
   assert(self.k >= 1e-5, "k has to be greater than 1e-5")
   assert(self.beta >= 0.01, "Beta has to be > 0.01")
end




function LRN:updateOutput(input)
   if self.K then self.k, self.K = self.K, nil end
   --[[if self.timerEnable then
   	startTime = sys.clock()
   end]]--
   self:updateForLoadSnapshot()
   if self.initStep == 0 then
   	self.initStep = 1
   	self.dnnPrimitives = torch.LongTensor(9)     
   else
	self.mkldnnInitOk = 1
   end


   self.output:resizeAs(input)
   --[[
   if self.compare then
	
  	     self.scale = self.scale or input.new()

	     local isBatch = true
	     if input:dim() == 3 then
	       input = nn.utils.addSingletonDimension(input)
	       isBatch = false
	     end

	     local batchSize   = input:size(1)
	     local channels    = input:size(2) 
	     local inputHeight = input:size(3) 
	     local inputWidth  = input:size(4) 

	     self.output:resizeAs(input)
	     self.scale:resizeAs(input)

	     -- use output storage as temporary buffer
	     local inputSquare = self.output
	     inputSquare:pow(input, 2)
	       
	     local prePad = (self.size - 1)/2 + 1
	     local prePadCrop = prePad > channels and channels or prePad

	     local scaleFirst = self.scale:select(2,1)
	     scaleFirst:zero()
	     -- compute first feature map normalization
	     for c = 1, prePadCrop do
	       scaleFirst:add(inputSquare:select(2, c))
	     end

	     -- reuse computations for next feature maps normalization
	     -- by adding the next feature map and removing the previous
	     for c = 2, channels do
	       local scalePrevious = self.scale:select(2, c -1)
	       local scaleCurrent  = self.scale:select(2, c)
	       scaleCurrent:copy(scalePrevious)
	       if c < channels - prePad + 2 then
		 local squareNext   = inputSquare:select(2, c + prePad - 1)
		 scaleCurrent:add(1, squareNext)
	       end
	       if c > prePad  then
		 local squarePrevious = inputSquare:select(2, c - prePad )
		 scaleCurrent:add(-1, squarePrevious)
	       end
	     end

	     self.scale:mul(self.alpha/self.size):add(self.k)

	     self.output:pow(self.scale,-self.beta)
	     self.output:cmul(input)

	     if not isBatch then
	       self.output = self.output[1]
	     end

	

	   tmpOut = torch.Tensor(self.output:size())
	   input.THNN.CrossChannelLRN_MKLDNN_updateOutput(
	      input:cdata(),
	      tmpOut:cdata(),
	      self.size,
	      self.alpha,
	      self.beta,
	      self.k,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	      )
           outSize = tonumber(tmpOut:cdata().size[0]*tmpOut:cdata().size[1]*tmpOut:cdata().size[2]*tmpOut:cdata().size[3])
           input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.output:cdata(), outSize,12)
   else
   ]]--
	  -- input.THNN.CrossChannelLRN_MKLDNN_updateOutput(
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
   --end
   --[[
   if self.timerEnable then
	print("LRN  forward time =         ",self.timeForward," backward time =",self.timeBackward)
	sys.lrnTime_forward = sys.lrnTime_forward + self.timeForward
	sys.lrnTime_backward = sys.lrnTime_backward + self.timeBackward
	self.timeForward =  (sys.clock() - startTime)
	self.timeBackward = 0
	self.cnt = self.cnt + 1
   end]]--
   
   return self.output
end

function LRN:updateGradInput(input, gradOutput)
   --[[
   if self.timerEnable then
	startTime = sys.clock()
   end]]--
   if not self.gradInput then return end
   self.gradInput:resizeAs(input)

   assert(gradOutput:dim() == 3 or gradOutput:dim() == 4);
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   --[[
   if self.compare then

     local isBatch = true
     if input:dim() == 3 then
       input = nn.utils.addSingletonDimension(input)
       gradOutput = nn.utils.addSingletonDimension(gradOutput)
       self.output = nn.utils.addSingletonDimension(self.output)
       isBatch = false
     end

     local batchSize   = input:size(1)
     local channels    = input:size(2) 
     local inputHeight = input:size(3) 
     local inputWidth  = input:size(4) 

     self.paddedRatio = self.paddedRatio or input.new()
     self.accumRatio = self.accumRatio or input.new()
     self.paddedRatio:resize(channels + self.size - 1, inputHeight, inputWidth)
     self.accumRatio:resize(inputHeight,inputWidth)

     local cacheRatioValue = 2*self.alpha*self.beta/self.size
     local inversePrePad = self.size - (self.size - 1) / 2

     self.gradInput:resizeAs(input)
     self.gradInput:pow(self.scale,-self.beta):cmul(gradOutput)

     self.paddedRatio:zero()
     local paddedRatioCenter = self.paddedRatio:narrow(1, inversePrePad, channels)
     for n = 1, batchSize do
       paddedRatioCenter:cmul(gradOutput[n],self.output[n])
       paddedRatioCenter:cdiv(self.scale[n])
       self.accumRatio:sum(self.paddedRatio:narrow(1,1,self.size-1), 1)
       for c = 1, channels do
	 self.accumRatio:add(self.paddedRatio[c+self.size-1])
	 self.gradInput[n][c]:addcmul(-cacheRatioValue, input[n][c], self.accumRatio)
	 self.accumRatio:add(-1, self.paddedRatio[c])
       end
     end

     if not isBatch then
       self.gradInput = self.gradInput[1]
       self.output = self.output[1]
     end



	   tmpGradInput = torch.Tensor(self.gradInput:size())

   input.THNN.CrossChannelLRN_MKLDNN_backward(
      input:cdata(),
      gradOutput:cdata(),
      tmpGradInput:cdata(),
      self.size,
      self.alpha,
      self.beta,
      self.k,
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk
      )


	   outSize = tonumber(tmpGradInput:cdata().size[0]*tmpGradInput:cdata().size[1]*tmpGradInput:cdata().size[2]*tmpGradInput:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpGradInput:cdata(), self.gradInput:cdata(), outSize,13)
   else]]--
   --input.THNN.CrossChannelLRN_MKLDNN_backward(
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
   --end
   --[[
   if self.timerEnable then
	self.timeBackward = (sys.clock() - startTime)
   end]]--

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
