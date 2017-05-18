-- you can easily test specific uni：ts like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"
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


function mklnntest.SpatialConvolution_g1()
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
   local gradOutput_clone = gradOutput:clone():float():mkl()
   
   local oriModule = nn.SpatialConvolution(from, to, ki, kj, si, sj):float()
   local dnnModule = mklnn.SpatialConvolution(from, to, ki, kj, si, sj):float()
   dnnModule.weight:copy(oriModule.weight)
   dnnModule.bias:copy(oriModule.bias)
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   dnnOutput = dnnOutput:th()
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'mklnn.SpatialConvolution g1 output')
   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
   mytester:assertTensorEq(oriGradInput, dnnGradInput:th(), 0.00001, 'mklnn.SpatialConvolution g1 gradInput')

end

function mklnntest.ReLU()
   local batch = math.random(2,5)
   local from = math.random(1,5)
   local outi = math.random(5,9)
   local outj = outi
   local input = torch.randn(batch, from, outi, outj):float()
   local gradOutput = torch.randn(batch, from, outi, outj):float()
   local input_clone = input:clone():float():mkl()--add
   local gradOutput_clone = gradOutput:clone():float():mkl()--add
   local oriModule = nn.ReLU():float()
   local dnnModule = mklnn.ReLU():float()
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   mytester:assertTensorEq(oriOutput, dnnOutput:th(), 0.00001, 'mklnn.ReLU output')
   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
   mytester:assertTensorEq(oriGradInput, dnnGradInput:th(), 0.00001, 'mklnn.ReLU gradInput')
end


function mklnntest.SpatialConvolutionMKLDNN_g2()
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
   
   local dnnOutput = dnnModule:forward(input:mkl()):th()
   local gradOutput = torch.randn(dnnOutput:size()):float()
   local dnnGradInput = dnnModule:backward(input:mkl(), gradOutput:mkl()):th()

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
      convModuleT[i].weight:copy(oriWeightT[i])
      convModuleT[i].bias:copy(oriBiasT[i])
      oriOutputT[i] = convModuleT[i]:forward(oriInputT[i])
      oriGradInputT[i] = convModuleT[i]:backward(oriInputT[i], oriGradOutputT[i])
      oriOutput[{{},{rsOut,reOut},{},{}}] = oriOutputT[i]:clone()
      oriGradInput[{{},{rsIn,reIn},{},{}}] = oriGradInputT[i]:clone()
   end
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'mklnn.SpatialConvolution g2 output')
   mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'mklnn.SpatialConvolution g2 gradInput')

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
    local input_clone = input:clone():float():mkl()
    local gradOutput_clone = gradOutput:clone():float():mkl()

    local oriOutput = oriModule:forward(input)
    local dnnOutput = dnnModule:forward(input_clone)
    dnnOutput = dnnOutput:th()
    mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'mklnn.SpatialMaxPooling output')

    local oriGradInput = oriModule:backward(input, gradOutput)
    local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone):th()
    mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'mklnn.SpatialMaxPooling gradInput')
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
   local input_clone = input:clone():float():mkl()

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

      mode_string = mode_string .. '  mklnn.SpatialBatchNormalization output'
      mytester:assertTensorEq(oriOutput, dnnOutput:th(), 0.00001, mode_string)
      local gradOutput = oriOutput:clone():uniform(0,1)  --use original OP to aquire the size of output
      local gradOutput_clone = gradOutput:clone():mkl()
      local oriGradInput = oriModule:backward(input, gradOutput)
      local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone):th()
      mode_string = mode_string .. '  mklnn.SpatialBatchNormalization gradInput'
      mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001,  mode_string)
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
   local dnnModule = mklnn.SpatialCrossMapLRN(size, alpha, beta, k):float()
   local batchSize = math.random(1,5)
   local from = math.random(3,8)
   local input = torch.rand(batchSize,from, inputSize, inputSize):float()
   local input_clone = input:clone():float():mkl()
   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone):th()
   mytester:assertTensorEq(oriOutput, dnnOutput, 0.00001, 'mklnn.SpatialCrossMapLRN output')
   local gradOutput = oriOutput:clone():uniform(0,1)  --use original OP to aquire the size of output 
   local gradOutput_clone = gradOutput:clone():mkl()
   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone):th()
   mytester:assertTensorEq(oriGradInput, dnnGradInput, 0.00001, 'mklnn.SpatialCrossMapLRN gradInput')
end

function mklnntest.Concat()
     -- batch
   local from = math.random(2,5)
   local inc = math.random(2,4)
   local to = from+inc
   local ki = math.random(1,5)

   local kj = ki 
   local si = math.random(1,4)
   local sj = si 
   local batch = math.random(2,5)

   local ini = math.random(3,7)*2+1
   local num_modules = math.random(2, 5)
   local inj = ini

   local input = torch.randn(batch, from, ini, ini):float()
   local input_clone = input:clone():mkl()
   
   local convs = {} 
   local convs_clone = {} 
   for i = 1,num_modules do
      convs[i] = nn.SpatialConvolution(from, to, ki, kj, si, sj):float()
      clone_tmp = mklnn.SpatialConvolution(from, to, ki, kj, si, sj):float()
      clone_tmp.weight:copy(convs[i].weight)
      clone_tmp.bias:copy(convs[i].bias)
      convs_clone[i] = clone_tmp
      inc = math.random(2,4)
      to = to + inc
   end  

   local dnnModule = mklnn.Concat(2):float()
   local oriModule = nn.Concat(2):float()

   for _,module in ipairs(convs) do
      oriModule:add(module)
   end  

   for _,module in ipairs(convs_clone) do
      dnnModule:add(module)
   end  

   local oriOutput = oriModule:forward(input)
   local dnnOutput = dnnModule:forward(input_clone)
   mytester:assertTensorEq(oriOutput, dnnOutput:th(), 0.00001, 'mklnn.Concat forward err')

   local gradOutput = torch.randn(oriOutput:size()):float()
   local gradOutput_clone = gradOutput:clone():mkl()

   local oriGradInput = oriModule:backward(input, gradOutput)
   local dnnGradInput = dnnModule:backward(input_clone, gradOutput_clone)
   
   mytester:assertTensorEq(oriGradInput, dnnGradInput:th(), 0.00001, 'mklnn.Concat backward err (gradInput)')
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
