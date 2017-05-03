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
   local input_clone = input:clone():float():mkl()
   local gradOutput_clone = gradOutput:clone():float()
   
   local oriModule = nn.SpatialConvolution(from, to, ki, kj, si, sj):float()
   local dnnModule = mklnn.SpatialConvolution(from, to, ki, kj, si, sj):float()
   dnnModule.weight:copy(oriModule.weight)
   dnnModule.bias:copy(oriModule.bias)
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   --local dnnprimitives = torch.LongTensor(3)
   --dnnOutput.THNN.MKLDNN_ConvertLayoutBackToNCHW(dnnOutput:cdata(), dnnprimitives:cdata(),0,0)
   dnnOutput = dnnOutput:th()
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'SpatialConvolutionMKLNN g1 output')
--[[   
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
]]--
end


function mklnntest.ReLU()
   local batch = math.random(2,5)
   local from = math.random(1,5)
   local outi = math.random(5,9)
   local outj = outi
   local input = torch.randn(batch, from, outi, outj):float()
   local gradOutput = torch.randn(batch, from, outi, outj):float()
   local input_clone = input:clone():float():mkl()--add
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
--[[ 
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

function mklnntest.SpatialMaxPooling()
  for _,ceil_mode in pairs({true,false}) do
    local from = math.random(1,5)
    local ki = math.random(1,4)
    --local kj = math.random(1,4)
    local kj = ki
    local si = math.random(1,3)
    --local sj = math.random(1,3)
    local sj  = si
    local outi = math.random(4,5)
    --local outj = math.random(4,5)
    local outj = outi
    local padW = math.min(math.random(0,1),math.floor(ki/2))
    --local padH =  math.min(math.random(0,1),math.floor(kj/2))
    local padH = padW
    local ini = (outi-1)*si+ki-2*padW
    local inj = (outj-1)*sj+kj-2*padH
    -- batch
    local nbatch = math.random(2,5)
    local input = torch.rand(nbatch,from,inj,ini):float() 
    local gradOutput = torch.rand(nbatch,from,outj,outi):float() 
		    
    local oriModule = nn.SpatialMaxPooling(ki,kj,si,sj,padW,padH):float()
    local dnnModule = mklnn.SpatialMaxPooling(ki,kj,si,sj,padW,padH):float()
		   
    if ceil_mode then 
      oriModule:ceil() 
      dnnModule:ceil()
    else 
      oriModule:floor() 
      dnnModule:floor()
    end
	  
    local input_clone = input:clone():float()
    local gradOutput_clone = gradOutput:clone():float()

    local oriOutput = oriModule:forward(input)
    local dnnOutput = dnnModule:forward(input_clone)
    mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'SpatialMaxPoolingMKLDNN output')

    if (PRINT_EN == 1) then 
      print("SpatialMaxPooling MKLDNN >>>>>>>>")
      local flatInput = torch.Tensor(input:nElement()):copy(input)
      local flatOriOutput = torch.Tensor(oriOutput:nElement()):copy(oriOutput)
      local flatDnnOutput = torch.Tensor(dnnOutput:nElement()):copy(dnnOutput)
      local diff = flatDnnOutput-flatOriOutput
      print('SpatialMaxPooling input')
      print(flatInput)
      print('SpatialMaxPooling oriOutput') 
      print(flatOriOutput)
      print('SpatialMaxPooling dnnOutput')
      print(flatDnnOutput)
      print('SpatialMaxPooling diff')
      print(diff)    
    end  
    local oriGradInput = oriModule:backward(input, gradOutput)
    local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
    mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'SpatialMaxPoolingMKLDNN gradInput')
    if (PRINT_EN == 1) then 
      print("SpatialMaxPooling MKLDNN <<<<<<<<")
      local flatGradOutput = torch.Tensor(gradOutput:nElement()):copy(gradOutput)
      local flatOriGradInput = torch.Tensor(oriGradInput:nElement()):copy(oriGradInput)
      local flatDnnGradInput = torch.Tensor(dnnGradInput:nElement()):copy(dnnGradInput)
      local diff = flatDnnGradInput-flatOriGradInput
      print('SpatialMaxPooling gradOutput')
      print(flatGradOutput)
      print('SpatialMaxPooling oriGradInput')
      print(flatOriGradInput)
      print('SpatialMaxPooling dnnGradInput')
      print(flatDnnGradInput)
      print('SpatialMaxPooling diff')
      print(diff)   
    end  
  end
end

function mklnntest.SpatialBatchNormalization()
   local planes = torch.random(1,6)
   local size = { torch.random(2, 6), planes }
   local hw = torch.random(1,6) + 10
   for i=1,2 do
      table.insert(size, hw)
   end
   local input = torch.zeros(table.unpack(size)):uniform():float()
   local input_clone = input:clone():float()

   for _,affine_mode in pairs({true,false}) do

      local mode_string = affine_mode and 'affine true' or 'affile false'
      local oriModule = nn.SpatialBatchNormalization(planes, 1e-5, 0.1, affine_mode):float()
      local dnnModule = mklnn.SpatialBatchNormalization(planes, 1e-5, 0.1, affine_mode):float()

      if affine_mode then
         dnnModule.weight:copy(oriModule.weight)
         dnnModule.bias:copy(oriModule.bias)
      end
      local oriOutput = oriModule:forward(input)
      local dnnOutput = dnnModule:forward(input_clone)

      local dnnprimitives = torch.LongTensor(3)

      dnnOutput.THNN.MKLDNN_ConvertLayoutBackToNCHW(dnnOutput:cdata(), dnnprimitives:cdata(),0,0)

      mode_string = mode_string .. '  SpatialBatchNormalizationMKLDNN output'
      mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, mode_string)
            if (PRINT_EN == 1) then
                print("SpatialBatchNormalization MKLDNN >>>>>>>>")
                local flatInput = torch.Tensor(input:nElement()):copy(input)
                local flatOriOutput = torch.Tensor(oriOutput:nElement()):copy(oriOutput)
                local flatDnnOutput = torch.Tensor(dnnOutput:nElement()):copy(dnnOutput)
                local diff = flatDnnOutput-flatOriOutput
                print('SpatialBatchNormalization input')
                print(flatInput)
                print('SpatialBatchNormalization oriOutput')
                print(flatOriOutput)
                print('SpatialBatchNormalization dnnOutput')
                print(flatDnnOutput)
                print('SpatialBatchNormalization diff')
                print(diff)
      end
      local gradOutput = oriOutput:clone():uniform(0,1)  --use original OP to aquire the size of output
      local gradOutput_clone = gradOutput:clone()
      local oriGradInput = oriModule:backward(input, gradOutput)
      local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
      dnnGradInput.THNN.MKLDNN_ConvertLayoutBackToNCHW(dnnGradInput:cdata(), dnnprimitives:cdata(),0,0)
      mode_string = mode_string .. '  SpatialBatchNormalizationMKLDNN gradInput'
      mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001,  mode_string)
	  if (PRINT_EN == 1) then
			print("SpatialBatchNormalization MKLDNN <<<<<<<<")
			local flatGradOutput = torch.Tensor(gradOutput:nElement()):copy(gradOutput)
			local flatOriGradInput = torch.Tensor(oriGradInput:nElement()):copy(oriGradInput)
			local flatDnnGradInput = torch.Tensor(dnnGradInput:nElement()):copy(dnnGradInput)
			local diff = flatDnnGradInput-flatOriGradInput
			print('SpatialBatchNormalization gradOutput')
			print(flatGradOutput)
			print('SpatialBatchNormalization oriGradInput')
			print(flatOriGradInput)
			print('SpatialBatchNormalization dnnGradInput')
			print(flatDnnGradInput)
			print('SpatialBatchNormalization diff')
			print( diff)
      end
   end
end

function mklnntest.SpatialCrossMapLRN()
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)
   
   local oriModule = nn.SpatialCrossMapLRN(size, alpha, beta, k):float()
   local dnnModule = mklnn.LRN(size, alpha, beta, k):float()
   local batchSize = math.random(1,5)
   local from = math.random(3,8)
   local input = torch.rand(batchSize,from, inputSize, inputSize):float()
   local input_clone = input:clone():float()
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'SpatialCrossMapLRNMKLDNN output')
   
   if (PRINT_EN == 1) then 
      print("SpatialCrossMapLRN MKLDNN >>>>>>>>")
      local flatInput = torch.Tensor(input:nElement()):copy(input)
      local flatOriOutput = torch.Tensor(oriOutput:nElement()):copy(oriOutput)
      local flatDnnOutput = torch.Tensor(dnnOutput:nElement()):copy(dnnOutput)
      local diff = flatDnnOutput-flatOriOutput
      print('SpatialCrossMapLRN input')
      print(flatInput)
      print('SpatialCrossMapLRN oriOutput') 
      print(flatOriOutput)
      print('SpatialCrossMapLRN dnnOutput')
      print(flatDnnOutput)
      print('SpatialCrossMapLRN diff')
      print(diff)    
   end 
   local gradOutput = oriOutput:clone():uniform(0,1)  --use original OP to aquire the size of output 
   local gradOutput_clone = gradOutput:clone()
   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
   mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'SpatialCrossMapLRNMKLDNN gradInput')
   if (PRINT_EN == 1) then 
      print("SpatialCrossMapLRN MKLDNN <<<<<<<<")
      local flatGradOutput = torch.Tensor(gradOutput:nElement()):copy(gradOutput)
      local flatOriGradInput = torch.Tensor(oriGradInput:nElement()):copy(oriGradInput)
      local flatDnnGradInput = torch.Tensor(dnnGradInput:nElement()):copy(dnnGradInput)
      local diff = flatDnnGradInput-flatOriGradInput
      print('SpatialCrossMapLRN gradOutput')
      print(flatGradOutput)
      print('SpatialCrossMapLRN oriGradInput')
      print(flatOriGradInput)
      print('SpatialCrossMapLRN dnnGradInput')
      print(flatDnnGradInput)
      print('SpatialCrossMapLRN diff')
      print( diff)   
   end  
end

]]--

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
