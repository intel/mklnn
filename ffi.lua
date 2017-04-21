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
