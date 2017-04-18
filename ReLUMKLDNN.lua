local ReLU, Parent = torch.class('mklnn.ReLUMKLDNN', 'mklnn.ThresholdMKLDNN')

function ReLU:__init(p)
   Parent.__init(self,0,0,p)
end
