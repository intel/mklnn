local U2I, parent = torch.class('mklnn.U2I','nn.Module')


function U2I:__init()
   parent.__init(self)
end

function U2I:updateOutput(input)
   if input:type() == 'torch.FloatTensor' then
      self.output = input:clone():mkl()
      return self.output
   else
      print("Warning: U2I op forward, input is not torch.FloatTensor")
      return input
   end
end

function U2I:updateGradInput(input, gradOutput)


   if gradOutput:type() == 'torch.MKLFloatTensor' then
      self.gradInput = gradOutput:th()
      return self.gradInput
   else
      print("Warning: U2I op backward, gradOutput is not torch.MKLFloatTensor")
      return gradOutput
   end

end
