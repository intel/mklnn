local I2U, parent = torch.class('mklnn.I2U','nn.Module')


function I2U:__init()
   parent.__init(self)
end

function I2U:updateOutput(input)
   if input:type() == 'torch.MKLFloatTensor' then
      self.output = input:th()
      return self.output
   else
      print("Warning: I2U op forward, input is not torch.MKLFloatTensor")
      return input
   end
end

function I2U:updateGradInput(input, gradOutput)
   if gradOutput:type() == 'torch.FloatTensor' then
      self.gradInput = gradOutput:mkl()
      return self.gradInput
   else
      print("Warning: I2U op backward, gradOutput is not torch.FloatTensor")
      return gradOutput
   end

end
