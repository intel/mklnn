local ffi = require 'ffi'


local cdefs = [[

void MKLNN_RealSpatialConvolution_forward(
  THMKLRealTensor *input,
  THMKLRealTensor *output,
  THRealTensor *weight,
  THRealTensor *bias,
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
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput,
  THRealTensor *weight,
  THRealTensor *bias,
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
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THRealTensor *gradWeight,
  THRealTensor *gradBias,
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
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput,
  float threshold,
  bool inplace,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealThreshold_updateOutput(
  THMKLRealTensor *input,
  THMKLRealTensor *output,
  float threshold,
  float val,
  bool inplace,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealSpatialMaxPooling_updateOutput(
  THMKLRealTensor *input,
  THMKLRealTensor *output,
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
  THMKLRealTensor *input,
  THMKLRealTensor *gradOutput,
  THMKLRealTensor *gradInput,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealSpatialAveragePooling_updateOutput(
          THMKLRealTensor *input,
          THMKLRealTensor *output,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad,
          THLongTensor *primitives,
          int initOk);

void MKLNN_RealSpatialAveragePooling_updateGradInput(
          THMKLRealTensor *input,
          THMKLRealTensor *gradOutput,
          THMKLRealTensor *gradInput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool ceil_mode,
          bool count_include_pad,
          THLongTensor *primitives,
          int initOk);

void MKLNN_RealBatchNormalization_updateOutput(
   THMKLRealTensor *input, 
   THMKLRealTensor *output,
   THRealTensor *weight, 
   THRealTensor *bias,
   THRealTensor *running_mean, 
   THRealTensor *running_var,
   bool train, 
   double momentum, 
   double eps,
   THLongTensor *primitives,
   int initOk);

void MKLNN_RealBatchNormalization_backward(
  THMKLRealTensor *input, 
  THMKLRealTensor *gradOutput, 
  THMKLRealTensor *gradInput,
  THRealTensor *gradWeight, 
  THRealTensor *gradBias, 
  THRealTensor *weight,
  THRealTensor *running_mean, 
  THRealTensor *running_var,
  bool train, 
  double scale, 
  double eps,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealCrossChannelLRN_updateOutput(
  THMKLRealTensor *input, 
  THMKLRealTensor *output,
  int size, 
  float alpha, 
  float beta, 
  float k,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealCrossChannelLRN_backward(
  THMKLRealTensor *input, 
  THMKLRealTensor *gradOutput, 
  THMKLRealTensor *gradInput,
  int size, 
  float alpha, 
  float beta, 
  float k,
  THLongTensor *primitives,
  int initOk);

void MKLNN_RealConcat_setupLongTensor(
          THLongTensor * array,
          THMKLRealTensor *input,
          int  index);

void MKLNN_RealConcat_updateOutput(
          THLongTensor *inputarray,
          THMKLRealTensor *output,
          int  moduleNum,
          THLongTensor *primitives,
          int initOk);

void MKLNN_RealConcat_backward_split(
          THLongTensor *gradarray,
          THMKLRealTensor *gradOutput,
          int  moduleNum,
          THLongTensor *primitives,
          int initOk);
]]

local Real2real = {
   Float='float',
   Double='double'
}


for Real, real in pairs(Real2real) do
   local type_cdefs=cdefs:gsub('Real', Real):gsub('real', real)
   ffi.cdef(type_cdefs)
end


local MKLENGINE_PATH = package.searchpath('libmklnn', package.cpath)
mklnn.C = ffi.load(MKLENGINE_PATH)
