local Threshold, parent = torch.class('mklnn.Threshold','nn.Module')

local wrapper = mklnn.wrapper
local getType = mklnn.getType
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

   self:validateParameters()
end

function Threshold:updateOutput(input)
   if self.dnnPrimitives then
      self.mkldnnInitOk = 1
   else
      self.mkldnnInitOk = 0
   end
   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(11)
   self.output = self.output:mkl()
   self:validateParameters()
   wrapper(getType(input),'Threshold_updateOutput',
           input:cdata(),
           self.output:cdata(),
           self.threshold,
           self.val,
           self.inplace,
           self.dnnPrimitives:cdata(),
           self.mkldnnInitOk
          ) 
   return self.output
end

function Threshold:updateGradInput(input, gradOutput)
   self:validateParameters()
   self.gradInput = self.gradInput:mkl()
   wrapper(getType(input),'Threshold_updateGradInput',
              input:cdata(),
              gradOutput:cdata(),
              self.gradInput:cdata(),
              self.threshold,
              self.inplace,
              self.dnnPrimitives:cdata(),self.mkldnnInitOk
          )
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
