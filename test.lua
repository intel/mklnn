-- you can easily test specific uni：ts like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"
--require 'nn'
--require 'mklnn'
local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local dnnInputMin = {3, 3, 4, 4}
local dnnTensorNrm = {256, 96, 227, 227}
local testInputMin = dnnInputMin

local PRINT_EN = 0

local mklnntest = torch.TestSuite()

local function equal(t1, t2, msg)
   if (torch.type(t1) == "table") then
      for k, v in pairs(t2) do
         equal(t1[k], t2[k], msg)
      end
   else
      mytester:eq(t1, t2, 0.00001, msg)
   end
end

function mklnntest.SpatialConvolutionMKLDNN_g1()
   -- batch
   local from = math.random(1,5)
   local to = math.random(1,5)
   local ki = math.random(1,5)
   --local kj = math.random(1,5)
   local kj = ki
   local si = math.random(1,4)
   --local sj = math.random(1,4)
   local sj = si
   local batch = math.random(2,5)
   local outi = math.random(4,8)
   --local outj = math.random(4,8)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   
   local input = torch.randn(batch,from,inj,ini):float()
   local gradOutput = torch.randn(batch,to,outj,outi):float()
   local input_clone = input:clone():float()
   local gradOutput_clone = gradOutput:clone():float()
   
   local oriModule = nn.SpatialConvolution(from, to, ki, kj, si, sj):float()
   local dnnModule = mklnn.SpatialConvolution(from, to, ki, kj, si, sj):float()
   dnnModule.weight:copy(oriModule.weight)
   dnnModule.bias:copy(oriModule.bias)
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   local dnnprimitives = torch.LongTensor(3)
   dnnOutput.THNN.MKLDNN_ConvertLayoutBackToNCHW(dnnOutput:cdata(), dnnprimitives:cdata(),0,0)
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'SpatialConvolutionMKLNN g1 output')
   
   if (PRINT_EN == 1) then 
      print("SpatialConvolution g1 MKLNN >>>>>>>>")
      local flatInput = torch.Tensor(input:nElement()):copy(input)
      local flatOriOutput = torch.Tensor(oriOutput:nElement()):copy(oriOutput)
      local flatDnnOutput = torch.Tensor(dnnOutput:nElement()):copy(dnnOutput)
      local diff = flatDnnOutput-flatOriOutput
      print('SpatialConvolution input')
      print(flatInput)
      print('SpatialConvolution oriOutput') 
      print(flatOriOutput)
      print('SpatialConvolution mklnnOutput')
      print(flatDnnOutput)
      print('SpatialConvolution diff')
      print(diff)    
   end  
   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
   mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'SpatialConvolution g1 gradInput')
   if (PRINT_EN == 1) then 
      print("SpatialConvolution g1 MKLNN <<<<<<<<")
      local flatGradOutput = torch.Tensor(gradOutput:nElement()):copy(gradOutput)
      local flatOriGradInput = torch.Tensor(oriGradInput:nElement()):copy(oriGradInput)
      local flatDnnGradInput = torch.Tensor(dnnGradInput:nElement()):copy(dnnGradInput)
      local diff = flatDnnGradInput-flatOriGradInput
      print('SpatialConvolution gradOutput')
      print(flatGradOutput)
      print('SpatialConvolution oriGradInput')
      print(flatOriGradInput)
      print('SpatialConvolution dnnGradInput')
      print(flatDnnGradInput)
      print('SpatialConvolution diff')
      print( diff)   
   end 
end

function mklnntest.ReLU()
   local batch = math.random(2,5)
   local from = math.random(1,5)
   local outi = math.random(5,9)
   local outj = outi
   local input = torch.randn(batch, from, outi, outj):float()
   local gradOutput = torch.randn(batch, from, outi, outj):float()
   local input_clone = input:clone():float()
   local gradOutput_clone = gradOutput:clone():float()
   local oriModule = nn.ReLU():float()
   local dnnModule = mklnn.ReLU():float()
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'ReLUMKLDNN output')
   if (PRINT_EN == 1) then 
     print("ReLU MKLDNN >>>>>>>>")
     local flatInput = torch.Tensor(input:nElement()):copy(input)
     local flatOriOutput = torch.Tensor(oriOutput:nElement()):copy(oriOutput)
     local flatDnnOutput = torch.Tensor(dnnOutput:nElement()):copy(dnnOutput)
     local diff = flatDnnOutput-flatOriOutput
     print('ReLU input')
     print(flatInput)
     print('ReLU oriOutput') 
     print(flatOriOutput)
     print('ReLU dnnOutput')
     print(flatDnnOutput)
     print('ReLU diff')
     print(diff)    
   end
   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
   mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'ReLUMKLDNN gradInput')
   if (PRINT_EN == 1) then 
      print("ReLU MKLDNN <<<<<<<<")
      local flatGradOutput = torch.Tensor(gradOutput:nElement()):copy(gradOutput)
      local flatOriGradInput = torch.Tensor(oriGradInput:nElement()):copy(oriGradInput)
      local flatDnnGradInput = torch.Tensor(dnnGradInput:nElement()):copy(dnnGradInput)
      local diff = flatDnnGradInput-flatOriGradInput
      print('ReLU gradOutput')
      print(flatGradOutput)
      print('ReLU oriGradInput')
      print(flatOriGradInput)
      print('ReLU dnnGradInput')
      print(flatDnnGradInput)
      print('ReLU diff')
      print( diff)  
   end  
end
 
function mklnntest.SpatialConvolutionMKLDNN_g2()

  -- batch
   local batch = math.random(2,5)
   local group = math.random(2,5)
   local partFrom = math.random(1,3)
   local from = partFrom*group
   local partTo = math.random(1,3)
   local to = partTo*group
   local ki = math.random(1,2)*2+1
   local kj = ki
   local si = math.random(1,4)
   local sj = si
   
   local ini = math.random(4,8)
   local inj = ini


   local input = torch.randn(batch, from, inj, ini):float()
   

   local dnnModule = mklnn.SpatialConvolution(from, to, ki, kj, si, sj, 1, 1, group):float()
   local weights = torch.randn(dnnModule.weight:size())
   local bias = torch.randn(dnnModule.bias:size())
   dnnModule.weight:copy(weights)
   dnnModule.bias:copy(bias)
   
   local dnnOutput = dnnModule:forward(input)

   local dnnprimitives = torch.LongTensor(2)
   dnnOutput.THNN.MKLDNN_ConvertLayoutBackToNCHW(dnnOutput:cdata(), dnnprimitives:cdata(),0,0)
   
   local gradOutput = torch.randn(dnnOutput:size()):float()
   
   local dnnGradInput = dnnModule:backward(input, gradOutput)
   dnnOutput.THNN.MKLDNN_ConvertLayoutBackToNCHW(dnnGradInput:cdata(), dnnprimitives:cdata(),0,0)

   local oriWeightT = {}
   local oriBiasT = {}
   local oriInputT = {}
   local oriGradOutputT = {}
   local convModuleT ={}
   local oriOutputT = {}
   local oriGradInputT = {}
   
   local oriOutput = torch.Tensor(dnnOutput:size()):float()
   local oriGradInput = torch.Tensor(dnnGradInput:size()):float()
   
   for i = 1,group,1 do
        local rsOut = 1+(i-1)*partTo
        local reOut = i*partTo
        local rsIn = 1+(i-1)*partFrom
        local reIn = i*partFrom
    oriWeightT[i] = weights[{{rsOut,reOut},{}}]:clone()
    oriBiasT[i] = bias[{{rsOut,reOut}}]:clone()
    oriInputT[i] = input[{{},{rsIn,reIn},{},{}}]:clone()
    oriGradOutputT[i] = gradOutput[{{}, {rsOut,reOut}, {}, {}}] 
    convModuleT[i] = nn.SpatialConvolution(partFrom, partTo, ki, kj, si, sj, 1, 1):float()
    --print(convModuleT[i])
    --print(oriInputT[i]:size())
    --print(oriGradOutputT[i]:size())
    convModuleT[i].weight:copy(oriWeightT[i])
    convModuleT[i].bias:copy(oriBiasT[i])
    oriOutputT[i] = convModuleT[i]:forward(oriInputT[i])
    --print(oriOutputT[i]:size())
    oriGradInputT[i] = convModuleT[i]:backward(oriInputT[i], oriGradOutputT[i])
    oriOutput[{{},{rsOut,reOut},{},{}}] = oriOutputT[i]:clone()
    oriGradInput[{{},{rsIn,reIn},{},{}}] = oriGradInputT[i]:clone()
  end
    
    mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'SpatialConvolutionMKLNN g2 output')
   
   if (PRINT_EN == 1) then 
      print("SpatialConvolution g2 MKLDNN >>>>>>>>")
      local flatInput = torch.Tensor(input:nElement()):copy(input)
      local flatOriOutput = torch.Tensor(oriOutput:nElement()):copy(oriOutput)
      local flatDnnOutput = torch.Tensor(dnnOutput:nElement()):copy(dnnOutput)
      local diff = flatDnnOutput-flatOriOutput
      print('SpatialConvolution input')
      print(flatInput)
      print('SpatialConvolution oriOutput') 
      print(flatOriOutput)
      print('SpatialConvolution mklnnOutput')
      print(flatDnnOutput)
      print('SpatialConvolution diff')
      print(diff)    
   end
   mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'SpatialConvolutionMKLDNN g2 gradInput')
   if (PRINT_EN == 1) then 
      print("SpatialConvolution g2 MKLDNN <<<<<<<<")
      local flatGradOutput = torch.Tensor(gradOutput:nElement()):copy(gradOutput)
      local flatOriGradInput = torch.Tensor(oriGradInput:nElement()):copy(oriGradInput)
      local flatDnnGradInput = torch.Tensor(dnnGradInput:nElement()):copy(dnnGradInput)
      local diff = flatDnnGradInput-flatOriGradInput
      print('SpatialConvolution gradOutput')
      print(flatGradOutput)
      print('SpatialConvolution oriGradInput')
      print(flatOriGradInput)
      print('SpatialConvolution mklnnGradInput')
      print(flatDnnGradInput)
      print('SpatialConvolution diff')
      print( diff)   
   end  
end


mytester:add(mklnntest)
jac = nn.Jacobian
sjac = nn.SparseJacobian

function mklnn.test(tests,seed)  
   -- Limit number of threads since everything is small
   local nThreads = torch.getnumthreads()
   torch.setnumthreads(1)
   -- randomize stuff
   local seed = seed or (1e5 * torch.tic())
   print('Seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   mytester:run(tests)
   torch.setnumthreads(nThreads)
   return mytester 
end
