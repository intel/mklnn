local Concat, parent = torch.class('mklnn.Concat', 'nn.Container')

local wrapper = mklnn.wrapper
local getType = mklnn.getType

function Concat:__init(dimension)
   parent.__init(self)
   self.outputSize = torch.LongStorage()
   self.dimension = dimension

end

function Concat:updateOutput(input)
   self.outputSize = self.outputSize or torch.LongStorage()
   if self.dnnPrimitives then
      self.mkldnnInitOk = 1 
   else
      self.mkldnnInitOk = 0 
   end 

   self.dnnPrimitives = self.dnnPrimitives or torch.LongTensor(20)
   self.outputArray = self.outputArray or torch.LongTensor(10)
   self.gradOutputArray = self.gradOutputArray or torch.LongTensor(10)

   local iterStartTime
   local iterForward
   local forwardTime = 0
   local outs = {}
   local outputTable = {}
   for i=1,#self.modules do
      local currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', input)
      outs[i] = currentOutput
      outputTable = currentOutput:cdata()
      wrapper(getType(currentOutput),
             'Concat_setupLongTensor',
              self.outputArray:cdata(),
              currentOutput:cdata(),
              i)
      if i == 1 then
         self.outputSize:resize(currentOutput:dim()):copy(currentOutput:size())
      else
         self.outputSize[self.dimension] = self.outputSize[self.dimension] + currentOutput:size(self.dimension)
      end

   end
   


   self.output = self.output:mkl()
   self.output:resize(self.outputSize)
   wrapper(getType(self.output),
          'Concat_updateOutput',
          self.outputArray:cdata(),
          self.output:cdata(),
          tonumber(#self.modules),
          self.dnnPrimitives:cdata(),
          self.mkldnnInitOk
          )

   return self.output
end

function Concat:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput:mkl()
   self.gradInput:resizeAs(input)
   local gradOutputBuffer = {}
   for i,module in ipairs(self.modules) do
      local gradOutputPart = torch.FloatTensor():mkl()
      gradOutputPart:resizeAs(module.output)
      gradOutputBuffer[i] = gradOutputPart
      wrapper(getType(gradOutputPart),
             'Concat_setupLongTensor',
              self.gradOutputArray:cdata(),
              gradOutputPart:cdata(),
              i)
   end
   wrapper(getType(gradOutput),
          'Concat_backward_split',
          self.gradOutputArray:cdata(),
          gradOutput:cdata(),
          tonumber(#self.modules),
          self.dnnPrimitives:cdata(),
          self.mkldnnInitOk
          )

   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      gradOutputPart = gradOutputBuffer[i]
      local currentGradInput = self:rethrowErrors(module, i, 'updateGradInput', input, gradOutputPart)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end
   end
   return self.gradInput
end

function Concat:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      local gradOutputPart = torch.FloatTensor():mkl()
      gradOutputPart:resizeAs(module.output)
      --local gradOutputPart = gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension))
      self:rethrowErrors(module, i, 'accGradParameters',
          input,
          gradOutputPart,
          scale)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:backward(input, gradOutput, scale)
   self.gradInput = self.gradInput:mkl()
   self.gradInput:resizeAs(input)
   local gradOutputBuffer = {}
   for i,module in ipairs(self.modules) do
      local gradOutputPart = torch.FloatTensor():mkl()
      gradOutputPart:resizeAs(module.output)
      gradOutputBuffer[i] = gradOutputPart
      wrapper(getType(gradOutputPart),
             'Concat_setupLongTensor',
              self.gradOutputArray:cdata(),
              gradOutputPart:cdata(),
              i)
   end
   wrapper(getType(gradOutput),
          'Concat_backward_split',
          self.gradOutputArray:cdata(),
          gradOutput:cdata(),
          tonumber(#self.modules),
          self.dnnPrimitives:cdata(),
          self.mkldnnInitOk
          )
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      gradOutputPart = gradOutputBuffer[i]
      local currentGradInput = self:rethrowErrors(module, i, 'backward', input, gradOutputPart, scale)
      if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
         if i==1 then
            self.gradInput:copy(currentGradInput)
         else
            self.gradInput:add(currentGradInput)
         end
      end

   end
   return self.gradInput
end

function Concat:accUpdateGradParameters(input, gradOutput, lr)
   local offset = 1
   for i,module in ipairs(self.modules) do
      local currentOutput = module.output
      self:rethrowErrors(module, i, 'accUpdateGradParameters',
          input,
          gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
          lr)
      offset = offset + currentOutput:size(self.dimension)
   end
end

function Concat:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == #self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end
