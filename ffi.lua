local ffi = require 'ffi'


ffi.cdef[[

typedef struct THMklFloatStorage
{
  float *data;
  long size;
  int refcount;
  char flag;
  //THAllocator *allocator;
  void *allocatorContext;
} THMklFloatStorage;



typedef struct THMklFloatTensor
{
  long *size;
  long *stride;
  int nDimension;
  THMklFloatStorage *storage;
  long storageOffset;
  int refcount;

  char flag;
  long mkldnnLayout;
} THMklFloatTensor;


void THNN_FloatMKLDNN_ConvertLayoutBackToNCHW(
  THFloatTensor * input,
  THLongTensor *primitives,
  int i,
  int initOk
);

void SpatialConvolution_forward(
  THFloatTensor *input,
  THFloatTensor *output,
  THFloatTensor *weight,
  THFloatTensor *bias,
  THFloatTensor *finput,
  THFloatTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group);

void SpatialConvolution_bwdData(
  THFloatTensor *input,
  THFloatTensor *gradOutput,
  THFloatTensor *gradInput,
  THFloatTensor *weight,
  THFloatTensor *bias,
  THFloatTensor *finput,
  THFloatTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group);

void SpatialConvolution_bwdFilter(
  THFloatTensor *input,
  THFloatTensor *gradOutput,
  THFloatTensor *gradWeight,
  THFloatTensor *gradBias,
  THFloatTensor *finput,
  THFloatTensor *fgradInput,
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

void Threshold_updateGradInput(
          THFloatTensor *input,
          THFloatTensor *gradOutput,
          THFloatTensor *gradInput,
          float threshold,
          bool inplace,
          THLongTensor *primitives,
          int initOk);

void Threshold_updateOutput(
          THFloatTensor *input,
          THFloatTensor *output,
          float threshold,
          float val,
          bool inplace,
          THLongTensor *primitives,
          int initOk);
]]


local MKLENGINE_PATH = package.searchpath('libmklnn', package.cpath)
mklnn.C = ffi.load(MKLENGINE_PATH)


