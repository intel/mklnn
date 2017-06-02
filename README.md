**mklnn**
===========

Torch-7 FFI binding and C warpper for Intel MKLDNN library, and MKLDNN library is designed by Intel to accelerate Deep Neural Network(DNN) computation on CPU, in particular Intel® Xeon processors (HSW, BDW, Xeon Phi), which is competitive to cuDNN library.

Modules are API compatible with their [`nn`](https://github.com/torch/nn) equivalents. Fully unit-tested against `nn` implementations.
Conversion between `nn` and `mklnn` is available through `mklnn.convert` function.

#### Dependency and installation

* Install torch with this [instructions](http://torch.ch/docs/getting-started.html)
* MKLML library auto-download and setting(see this [link](https://github.com/xhzhao/EnvCheck))
* Install mkltorch (luarocks install mkltorch)
* Install mklnn (luarocks install mklnn)

#### Performance

Convnet Benchmark performance from this [link](https://github.com/xhzhao/convnet-benchmarks/tree/mklnn) 
* distro: The Out-Of-Box Torch is installed from [distro](https://github.com/torch/distro) with openblas
* distro+mklnn: mklml version
* distro+cudnn: cudnn version

|  Inference      |    distro     |   distro+mklnn  | distro+cudnn |
|:-------------:|:-----------------:|:---------------:|:---------------:|
| alexnet      |:-----------------:|:---------------:|:---------------:|
| overfeat     |:-----------------:|:---------------:|:---------------:|
| vgg_a        |:-----------------:|:---------------:|:---------------:|
| googlenet    |:-----------------:|:---------------:|:---------------:|

#### Modules

```lua
require 'mklnn'  -- will automatically require mkltorch

The following OP are supported in this package:

-- All inputs have to be 3D or 4D(batch-mode)
mklnn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW = 1], [dH = 1], [padW = 0], [padH = 0], [groups = 1])
mklnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH)
mklnn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
mklnn.SpatialBatchNormalization(nFeature, eps, momentum, affine)
mklnn.SpatialCrossMapLRN(size, alpha, beta, k)
mklnn.Concat(dimension)
mklnn.ReLU([inplace=false])

-- Two layout conversion op
mklnn.U2I()  -- convert the user layout(default NCHW) to internal layout(required by MKLDNN library)
mklnn.I2U()  -- convert the internel layout to user layout

-- Op in plan, and this list will increase
mklnn.SpatialFullConvolution()
```

#### Conversion between mklnn and nn

Conversion is done by `mklnn.convert` function which takes a network and backend arguments('mkl' or 'nn') and goes over
network modules recursively substituting equivalents. 

```lua
require 'nn'
require 'mklnn'
net = nn.Sequential()
net:add(nn.SpatialConvolution(3,96,11,11,4,4))
net:add(nn.ReLU())
mklnet = mklnn.convert(net, 'mkl')
print(mklnet)
```
will result in:
```lua
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> output]
  (1): mklnn.U2I
  (2): mklnn.SpatialConvolution(3 -> 96, 11x11, 4,4)
  (3): mklnn.ReLU
  (4): mklnn.I2U
}
```

Get another demo from this [link](https://github.com/xhzhao/convnet-benchmarks/tree/mklnn) to perform an Convnet benchmark test.
