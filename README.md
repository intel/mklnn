mklnn
===========

Torch-7 FFI binding and C warpper for Intel MKLDNN library 

Modules are API compatible their [`nn`](https://github.com/torch/nn) equivalents. Fully unit-tested against `nn` implementations.
Conversion between `nn` and `mklnn` is available through `mklnn.convert` function.

##### Dependency and installation

* Install torch with this [instructions](http://torch.ch/docs/getting-started.html)
* MKLML library auto-download and setting(see this [link](https://github.com/xhzhao/EnvCheck))
* Install mkltorch (luarocks install mkltorch)
* Install mklnn (luarocks install mklnn)

#### Supported OPs

The following OP are supported in this package:
* mklnn.SpatialConvolution
* mklnn.ReLU
* mklnn.SpatialMaxPooling
* mklnn.SpatialAveragePooling
* mklnn.SpatialBatchNormalization
* mklnn.SpatialCrossMapLRN
* mklnn.Concat

#### Conversion between mklnn and nn

Conversion is done by `mklnn.convert` function which takes a network and backend arguments('mkl' or 'nn') and goes over
network modules recursively substituting equivalents. 

Get an very easy demo from this [link](https://github.com/xhzhao/convnet-benchmarks/tree/mklnn) to perform an Convnet benchmark test.
