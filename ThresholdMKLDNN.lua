local Threshold, parent = torch.class('mklnn.ThresholdMKLDNN','nn.Module')

local wrapper = mklnn.wrapper
function Threshold:__init(th,v,ip)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.Threshold(threshold, value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
  -- self:setEngine(1)

   self:validateParameters()
end

function Threshold:updateOutput(input)

   -- if self.timerEnable then
   --	startTime = sys.clock()
   --end
   self:updateForLoadSnapshot()
   --[[
   if self.initStep == 0 then
   	self.initStep = 1
        self.dnnPrimitives = torch.LongTensor(11)
   else
	self.mkldnnInitOk = 1
   end
   ]]

   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(11)
   self.mkldnnInitOk = 0

   self:validateParameters()
   wrapper('Threshold_MKLDNN_updateOutput',
           input:cdata(),
           self.output:cdata(),
           self.threshold,
           self.val,
           self.inplace,
           self.dnnPrimitives:cdata(),
           self.mkldnnInitOk
          ) 
   --[[
   if self.compare  then
	   input.THNN.Threshold_updateOutput(
	      input:cdata(),
	      self.output:cdata(),
	      self.threshold,
	      self.val,
	      self.inplace
	   )
	   tmpOut = torch.Tensor(self.output:size())
	   input.THNN.Threshold_MKLDNN_updateOutput(
	      input:cdata(),
	      tmpOut:cdata(),
	      self.threshold,
	      self.val,
	      self.inplace,
	      self.dnnPrimitives:cdata(),
	      self.mkldnnInitOk
	   )
	   outSize = tonumber(tmpOut:cdata().size[0]*tmpOut:cdata().size[1]*tmpOut:cdata().size[2]*tmpOut:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpOut:cdata(), self.output:cdata(), outSize,6)
   else

   input.THNN.Threshold_MKLDNN_updateOutput(
      input:cdata(),
      self.output:cdata(),
      self.threshold,
      self.val,
      self.inplace,
      self.dnnPrimitives:cdata(),
      self.mkldnnInitOk
   )
   end
   
   if self.timerEnable then
        print("mkldnn Threshold forward time = ,",self.timeForward," backward time =",self.timeBackward)
        sys.reluTime_forward = sys.reluTime_forward + self.timeForward
        sys.reluTime_backward = sys.reluTime_backward + self.timeBackward
        self.timeForward = sys.clock() - startTime
        self.cnt = self.cnt + 1
   end
   ]]--
   return self.output
end

function Threshold:updateGradInput(input, gradOutput)
   --if self.timerEnable then
   --	startTime = sys.clock()
   --end
   self:validateParameters()
   wrapper('Threshold_MKLDNN_updateGradInput',
              input:cdata(),
              gradOutput:cdata(),
              self.gradInput:cdata(),
              self.threshold,
              self.inplace,
              self.dnnPrimitives:cdata(),self.mkldnnInitOk
          )
   --[[
   if self.compare then
	   input.THNN.Threshold_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      self.gradInput:cdata(),
	      self.threshold,
	      self.inplace
	   )
	   tmpGradInput = torch.Tensor(self.gradInput:size())
	   input.THNN.Threshold_MKLDNN_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      tmpGradInput:cdata(),
	      self.threshold,
	      self.inplace,
	      self.dnnPrimitives:cdata(),self.mkldnnInitOk
	   )
	   --print("mkldnn Threshold backward compare")
	   outSize = tonumber(tmpGradInput:cdata().size[0]*tmpGradInput:cdata().size[1]*tmpGradInput:cdata().size[2]*tmpGradInput:cdata().size[3])
	   input.THNN.SpatialConvolutionMM_compare(tmpGradInput:cdata(), self.gradInput:cdata(), outSize,7)
   else
	   input.THNN.Threshold_MKLDNN_updateGradInput(
	      input:cdata(),
	      gradOutput:cdata(),
	      self.gradInput:cdata(),
	      self.threshold,
	      self.inplace,
	      self.dnnPrimitives:cdata(),self.mkldnnInitOk
	   )
   end
   ]]
   --if self.timerEnable then
   --	self.timeBackward = sys.clock() - startTime
   --end
   return self.gradInput
end

function Threshold:validateParameters()
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   if self.inplace then
      if self.val > self.threshold then
         error('in-place processing requires value (' .. self.val ..
                  ') not exceed threshold (' .. self.threshold .. ')')
      end
   end
end
