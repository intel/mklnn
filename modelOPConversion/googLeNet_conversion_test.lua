
require 'paths'

paths.dofile('googLeNet_model.lua')  -- corresponding file must provide a function named as createModel() to create your expected MKLDNN  model
paths.dofile('model2mkl.lua')

torch.setdefaulttensortype('torch.FloatTensor')

local dnn_model = createModel()     -- build the model
local ord_model = modelAdvancedTransform(dnn_model, 0)

print('-------------------ord_model---------------------')
print(dnn_model)
print('-------------------dnn_model---------------------')
print(ord_model)

