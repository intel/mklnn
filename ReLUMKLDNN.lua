local ReLU, Parent = torch.class('nn.ReLUMKLDNN', 'nn.ThresholdMKLDNN')

function ReLU:__init(p)
   Parent.__init(self,0,0,p)
end
