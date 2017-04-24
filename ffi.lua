local ffi = require 'ffi'


local cdefs = [[


void MKLNN_RealSpatialConvolution_forward(
  THRealTensor *input,
  THRealTensor *output,
  THRealTensor *weight,
  THRealTensor *bias,
  THRealTensor *finput,
  THRealTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group);


void MKLNN_RealSpatialConvolution_bwdData(
  THRealTensor *input,
  THRealTensor *gradOutput,
  THRealTensor *gradInput,
  THRealTensor *weight,
  THRealTensor *bias,
  THRealTensor *finput,
  THRealTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group);

void MKLNN_RealSpatialConvolution_bwdFilter(
  THRealTensor *input,
  THRealTensor *gradOutput,
  THRealTensor *gradWeight,
  THRealTensor *gradBias,
  THRealTensor *finput,
  THRealTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  float scale,
  int group);

void MKLNN_RealThreshold_updateGradInput(
  THRealTensor *input,
  THRealTensor *gradOutput,
  THRealTensor *gradInput,
  float threshold,
  bool inplace,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealThreshold_updateOutput(
  THRealTensor *input,
  THRealTensor *output,
  float threshold,
  float val,
  bool inplace,
  THLongTensor *primitives,
  int initOk);

static void MKLNN_RealSpatialMaxPooling_init_forward(
  THLongTensor *primitives,
  int N,
  int inC,
  int inH,
  int inW,
  int kH,
  int kW,
  int dH,
  int dW,
  int padH,
  int padW,
  int outC,
  int outH,
  int outW);

static void MKLNN_RealSpatialMaxPooling_init_backward(
  THLongTensor *primitives,
  int N,
  int inC,
  int inH,
  int inW,
  int kH,
  int kW,
  int dH,
  int dW,
  int padH,
  int padW,
  int outC,
  int outH,
  int outW);

void MKLNN_RealSpatialMaxPooling_updateOutput(
  THRealTensor *input,
  THRealTensor *output,
  THRealTensor *indices,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealSpatialMaxPooling_updateGradInput(
  THRealTensor *input,
  THRealTensor *gradOutput,
  THRealTensor *gradInput,
  THRealTensor *indices,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode,
  THLongTensor *primitives,
  int initOk);

static void MKLNN_RealBatchNormalization_init_forward(
  THLongTensor *primitives,
  int N,
  int inC,
  int inH,
  int inW,
  double eps);

static void MKLNN_RealBatchNormalization_init_backward(
  THLongTensor *primitives,
  int N,
  int outC,
  int outH,
  int outW,
  double eps);

void MKLNN_BatchNormalization_updateOutput(
   THRealTensor *input, 
   THRealTensor *output,
   THRealTensor *weight, 
   THRealTensor *bias,
   THRealTensor *running_mean, 
   THRealTensor *running_var,
   THRealTensor *save_mean, 
   THRealTensor *save_std,
   bool train, 
   double momentum, 
   double eps,
   THLongTensor *primitives,
   int initOk);


]]


local Real2real = {
   Long='long',
   Float='float',
   Double='double'
}


for Real, real in pairs(Real2real) do
   local type_cdefs=cdefs:gsub('Real', Real):gsub('real', real)
   ffi.cdef(type_cdefs)
end


local MKLENGINE_PATH = package.searchpath('libmklnn', package.cpath)
mklnn.C = ffi.load(MKLENGINE_PATH)
