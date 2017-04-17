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

void SpatialConvolutionMM_MKLDNN_forward(
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


]]


local MKLENGINE_PATH = package.searchpath('libmklnn', package.cpath)
mklnn.C = ffi.load(MKLENGINE_PATH)


