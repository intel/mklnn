
local mklOP2thOP = {}
mklOP2thOP['mklnn.SpatialConvolution']    = nn.SpatialConvolution
mklOP2thOP['nn.SpatialConvolution']       = nn.SpatialConvolution
mklOP2thOP['mklnn.SpatialMaxPooling']     = nn.SpatialMaxPooling
mklOP2thOP['nn.SpatialMaxPooling']        = nn.SpatialMaxPooling
mklOP2thOP['mklnn.SpatialAveragePooling'] = nn.SpatialAveragePooling
mklOP2thOP['nn.SpatialAveragePooling']    = nn.SpatialAveragePooling
mklOP2thOP['mklnn.SpatialCrossMapLRN']    = nn.SpatialCrossMapLRN
mklOP2thOP['nn.SpatialCrossMapLRN']       = nn.SpatialCrossMapLRN
mklOP2thOP['mklnn.ReLU']                  = nn.ReLU
mklOP2thOP['nn.ReLU']                     = nn.ReLU
mklOP2thOP['mklnn.Concat']                = nn.Concat
mklOP2thOP['nn.Concat']                   = nn.Concat


local thOP2mklOP = {}
thOP2mklOP['nn.SpatialConvolution']       = mklnn.SpatialConvolution
thOP2mklOP['mklnn.SpatialConvolution']    = mklnn.SpatialConvolution
thOP2mklOP['nn.SpatialMaxPooling']        = mklnn.SpatialMaxPooling
thOP2mklOP['mklnn.SpatialMaxPooling']     = mklnn.SpatialMaxPooling
thOP2mklOP['nn.SpatialAveragePooling']    = mklnn.SpatialAveragePooling
thOP2mklOP['mklnn.SpatialAveragePooling'] = mklnn.SpatialAveragePooling
thOP2mklOP['nn.SpatialCrossMapLRN']       = mklnn.SpatialCrossMapLRN
thOP2mklOP['mklnn.SpatialCrossMapLRN']    = mklnn.SpatialCrossMapLRN
thOP2mklOP['nn.ReLU']                     = mklnn.ReLU
thOP2mklOP['mklnn.ReLU']                  = mklnn.ReLU
thOP2mklOP['nn.Concat']                   = mklnn.Concat
thOP2mklOP['mklnn.Concat']                = mklnn.Concat

--[[
NOTE:
the model won't convert to the other version when OPs of source model are same with the refered OPs you specify 
src_model:  model to be convert to the other version
th2mkl:    when th2mkl==0, the thinary OP will convert to mkldnn OP
                when th2mkl!=0, the mkldnn OP will convert to thinary OP
]]--


local convert = function(src_model, th2mkl)
  
  local cvtOp = th2mkl or 'mkl'
  if ('mkl' == th2mkl) then
    cvtOp = thOP2mklOP
  elseif('nn' == th2mkl) then
    cvtOp = mklOP2thOP
  else
    print("wrong type")
    return nil
  end
  model_flag, model = convertAdvancedModel(src_model, cvtOp, false)    -- false: regular true: mklnn
end

function convertAdvancedModel(src_module, cvtOP, prevOPFlag)
  local dst_module
  local module_type = torch.type(src_module)
  -- prevOPFlag = 0  -- 0:regular op 1:mklnn op
  --print(module_type)
  if(module_type == 'nn.Sequential') then
    dst_module = nn.Sequential()
    for i = 1, #src_module do
      local src_layer = src_module:get(i)
      local name = src_layer.name
      -- print(name)
      local layer_type = torch.type(src_layer)
      print(layer_type)
      if(string.find(layer_type, 'SpatialConvolution')) then       
        --print('SC')
        local nInputPlane,nOutputPlane = src_layer.nInputPlane, src_layer.nOutputPlane
        local kW,kH = src_layer.kW, src_layer.kH
        local dW,dH = src_layer.dW, src_layer.dH
        local padW,padH = src_layer.padW, src_layer.padH
        local dst_layer = cvtOP[layer_type](nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
        dst_layer.weight:copy(src_layer.weight)
        dst_layer.bias:copy(src_layer.bias)
        dst_module:add(dst_layer)
        prevOPFlag = true
         
      elseif(string.find(layer_type, 'SpatialMaxPooling')) then
        --print('SMP')
        local kW,kH = src_layer.kW, src_layer.kH
        local dW,dH = src_layer.dW, src_layer.dH
        local padW,padH = src_layer.padW, src_layer.padH
        local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH):ceil()
        dst_module:add(dst_layer)
        prevOPFlag = true
     
      elseif(string.find(layer_type, 'SpatialAveragePooling')) then
        --print('SAP')
        local kW,kH = src_layer.kW, src_layer.kH
        local dW,dH = src_layer.dW, src_layer.dH
        local padW,padH = src_layer.padW, src_layer.padH
        local dst_layer = cvtOP[layer_type](kW, kH, dW, dH, padW, padH)
        dst_module:add(dst_layer)
        prevOPFlag = true
        
      elseif(string.find(layer_type, 'SpatialCrossMapLRN')) then
        --print('LRN')
        local size = src_layer.size
        local alpha, beta = src_layer.alpha, src_layer.bata
        local k = src_layer.k
        local dst_layer = cvtOP[layer_type](size, alpha, beta, k)
        dst_module:add(dst_layer)
        prevOPFlag = true
        
      elseif(string.find(layer_type, 'ReLU')) then
        --print('ReLU')
        local dst_layer = src_layer
        if prevOPFlag then
          local ip = src_layer.inplace
          dst_layer = cvtOP[layer_type](ip)
          prevOPFlag = true
        end
          dst_module:add(dst_layer)        
       
      elseif(string.find(layer_type, 'Concat') or string.find(layer_type, 'Sequential')) then 
        local sub_module = convertAdvancedModel(src_layer, cvtOP, prevOPFlag)
        dst_module:add(sub_module)
        prevOPFlag = true
        
      else
        if prevOPFlag then
          print('----------need convertion before using this op    ' .. layer_type)
          local convert_layer = mklnn.I2U()
          dst_module:add(convert_layer)
        end
        --local new_layer = src_layer:clone()
        dst_module:add(src_layer)
        prevOPFlag = false
      end
    end
  elseif(string.find(module_type, 'Concat')) then
    local dimension = src_module.dimension
    
    local last_op_flag = nil
    local cat_op_flag = nil
    local op_flag_table = {}
    local sub_module_table = {}
    local add_op = false
    for j = 1, src_module:size() do 
      local dnn = src_module:get(j)
      local sub_module_flag, sub_module = convertAdvancedModel(dnn, cvtOP, prevOPFlag)
      if (nil == last_op_flag) then
        last_op_flag = sub_module_flag
        cat_op_flag = sub_module_flag
      elseif (last_op_flag != sub_module_flag)
        cat_op_flag = false                  -- true; regular 
        add_op_flag = false
      end
      table.insert(op_flag_table, sub_module_flag)
      table.insert(sub_module_table, sub_module)    
    end 
    
    if cat_op_flag then
      dst_module = src_module
    else
      dst_module = mklnn.Concat(dimension)
    end
    
    for j = 1, src_module:size() do
      local sub_module = sub_module_table[j]
      if add_op_flag and op_flag_table[j] then
        local convert_layer = mklnn.I2U()
        sub_module:add(convert_layer)         
      end 
      dst_module:add(sub_module)
    end
  end
  --print(dst_module)
  return prevOPFlag, dst_module 
end

return convert
