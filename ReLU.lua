local ReLU, Parent = torch.class('mklnn.ReLU', 'mklnn.Threshold')

function ReLU:__init(p)
   Parent.__init(self,0,0,p)
end
