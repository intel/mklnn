local I2U, parent = torch.class('mklnn.I2U','nn.Module')


function I2U:__init()
   parent.__init(self)
end

function I2U:updateOutput(input)
   self.output = input:th()

   return self.output
end

function I2U:updateGradInput(input, gradOutput)

   self.gradInput = gradOutput:mkl()

   return self.gradInput
end
