
#include<stddef.h>
/*
* fileName: MKLDNN.h
* file description:
* 	integrate Torch + MKLDNN to accelerate CNN
* created by zhao xiaohui, intel@shanghai, email xiaohui.zhao@intel.com
*/

//#include <mkl.h>
//#include "/opt/intel/mkl/include/mkl.h"

#ifndef _TORCH_MKLDNN_H
#define _TORCH_MKLDNN_H

typedef struct _uniPrimitive_s* dnnPrimitive_t;
typedef struct _dnnLayout_s* dnnLayout_t;
typedef void* dnnPrimitiveAttributes_t;

#define DNN_MAX_DIMENSION       32
#define DNN_QUERY_MAX_LENGTH    128

typedef enum {
    E_SUCCESS                   =  0,
    E_INCORRECT_INPUT_PARAMETER = -1,
    E_UNEXPECTED_NULL_POINTER   = -2,
    E_MEMORY_ERROR              = -3,
    E_UNSUPPORTED_DIMENSION     = -4,
    E_UNIMPLEMENTED             = -127
} dnnError_t;

typedef enum {
    dnnAlgorithmConvolutionGemm  , // GEMM based convolution
    dnnAlgorithmConvolutionDirect, // Direct convolution
    dnnAlgorithmConvolutionFFT   , // FFT based convolution
    dnnAlgorithmPoolingMax       , // Maximum pooling
    dnnAlgorithmPoolingMin       , // Minimum pooling
    dnnAlgorithmPoolingAvg         // Average pooling
} dnnAlgorithm_t;

typedef enum {
    dnnResourceSrc            = 0,
    dnnResourceFrom           = 0,
    dnnResourceDst            = 1,
    dnnResourceTo             = 1,
    dnnResourceFilter         = 2,
    dnnResourceScaleShift     = 2,
    dnnResourceBias           = 3,
    dnnResourceDiffSrc        = 4,
    dnnResourceDiffFilter     = 5,
    dnnResourceDiffScaleShift = 5,
    dnnResourceDiffBias       = 6,
    dnnResourceDiffDst        = 7,
    dnnResourceWorkspace      = 8,
    dnnResourceMultipleSrc    = 16,
    dnnResourceMultipleDst    = 24,
    dnnResourceNumber         = 32
} dnnResourceType_t;

typedef enum {
    dnnBorderZeros          = 0x0,
    dnnBorderZerosAsymm     = 0x100,
    dnnBorderExtrapolation  = 0x3
} dnnBorder_t;


/*******************************************************************************
 * F32 section: single precision
 ******************************************************************************/

dnnError_t dnnLayoutCreate_F32(
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t dnnLayoutCreateFromPrimitive_F32(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);
size_t dnnLayoutGetMemorySize_F32(
        const dnnLayout_t layout);
int dnnLayoutCompare_F32(
        const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t dnnAllocateBuffer_F32(
        void **pPtr, dnnLayout_t layout);
dnnError_t dnnReleaseBuffer_F32(
        void *ptr);
dnnError_t dnnLayoutDelete_F32(
        dnnLayout_t layout);

dnnError_t dnnPrimitiveAttributesCreate_F32(
        dnnPrimitiveAttributes_t *attributes);
dnnError_t dnnPrimitiveAttributesDestroy_F32(
        dnnPrimitiveAttributes_t attributes);
dnnError_t dnnPrimitiveGetAttributes_F32(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);

dnnError_t dnnExecute_F32(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnExecuteAsync_F32(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnWaitFor_F32(
        dnnPrimitive_t primitive);
dnnError_t dnnDelete_F32(
        dnnPrimitive_t primitive);

dnnError_t dnnConversionCreate_F32(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
dnnError_t dnnConversionExecute_F32(
        dnnPrimitive_t conversion, void *from, void *to);



dnnError_t dnnSumCreate_F32(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, float *coefficients);
dnnError_t dnnConcatCreate_F32(
        dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
dnnError_t dnnSplitCreate_F32(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t dnnScaleCreate_F32(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float alpha);

dnnError_t dnnConvolutionCreateForward_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardData_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardFilter_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);

dnnError_t dnnGroupsConvolutionCreateForward_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardData_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);

dnnError_t dnnReLUCreateForward_F32(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t dnnReLUCreateBackward_F32(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope);

dnnError_t dnnLRNCreateForward_F32(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);
dnnError_t dnnLRNCreateBackward_F32(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);

dnnError_t dnnBatchNormalizationCreateForward_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);
dnnError_t dnnBatchNormalizationCreateBackwardData_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);

dnnError_t dnnPoolingCreateForward_F32(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnPoolingCreateBackward_F32(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);

dnnError_t dnnInnerProductCreateForward_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateForwardBias_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardData_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardFilter_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardBias_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]);


/*******************************************************************************
 * F64 section: double precision
 ******************************************************************************/

dnnError_t dnnLayoutCreate_F64 (
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t dnnLayoutCreateFromPrimitive_F64(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);
size_t dnnLayoutGetMemorySize_F64(
        const dnnLayout_t layout);
int dnnLayoutCompare_F64(
        const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t dnnAllocateBuffer_F64(
        void **pPtr, dnnLayout_t layout);
dnnError_t dnnReleaseBuffer_F64(
        void *ptr);
dnnError_t dnnLayoutDelete_F64(
        dnnLayout_t layout);

dnnError_t dnnPrimitiveAttributesCreate_F64(
        dnnPrimitiveAttributes_t *attributes);
dnnError_t dnnPrimitiveAttributesDestroy_F64(
        dnnPrimitiveAttributes_t attributes);
dnnError_t dnnPrimitiveGetAttributes_F64(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);

dnnError_t dnnExecute_F64(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnExecuteAsync_F64(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnWaitFor_F64(
        dnnPrimitive_t primitive);
dnnError_t dnnDelete_F64(
        dnnPrimitive_t primitive);

dnnError_t dnnConversionCreate_F64(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
dnnError_t dnnConversionExecute_F64(
        dnnPrimitive_t conversion, void *from, void *to);

dnnError_t dnnSumCreate_F64(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, double *coefficients);
dnnError_t dnnConcatCreate_F64(
         dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
dnnError_t dnnSplitCreate_F64(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t dnnScaleCreate_F64(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double alpha);

dnnError_t dnnConvolutionCreateForward_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateForwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardData_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardFilter_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);

dnnError_t dnnGroupsConvolutionCreateForward_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateForwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardData_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);

dnnError_t dnnReLUCreateForward_F64(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t dnnReLUCreateBackward_F64(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, double negativeSlope);

dnnError_t dnnLRNCreateForward_F64(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);
dnnError_t dnnLRNCreateBackward_F64(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);

dnnError_t dnnBatchNormalizationCreateForward_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);
dnnError_t dnnBatchNormalizationCreateBackwardData_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);

dnnError_t dnnPoolingCreateForward_F64(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnPoolingCreateBackward_F64(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);

dnnError_t dnnInnerProductCreateForward_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateForwardBias_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardData_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardFilter_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardBias_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]);



#include <sys/time.h>
/**
	MKLDNN init functions:
	SpatialConvolutionMM_MKLDNN_init()

	MKLDNN op run function:
	SpatialConvolutionMM_MKLDNN_forward()
	SpatialConvolutionMM_MKLDNN_bwdData()
	SpatialConvolutionMM_MKLDNN_bwdFilter()

	Primitives ara created in the init function and saved in the tensor.
	dnnPrimitives->storage->data[0]: forward
	dnnPrimitives->storage->data[1]: bwdData
	dnnPrimitives->storage->data[2]: bwdFilter
	...
*/
#define LOG_ENABLE 		0
#define CONVERSION_LOG		0
#define MKL_TIME		0
#define NEW_INTERFACE		1

#define dimension 		4


typedef enum {
    CONV_LAYOUT_INPUT			= 0,
    CONV_LAYOUT_OUTPUT			= 1,
    CONV_LAYOUT_FORWARD_OUTPUT		= 2,
    CONV_LAYOUT_BWDDATA_INPUT		= 3,
    CONV_LAYOUT_BWDFILT_OUTPUT		= 4,
    CONV_LAYOUT_FILTER			= 5,
    FORWARD_INDEX   			= 6,
    BWD_DATA_INDEX  			= 7,
    BWD_FILTER_INDEX  			= 8,
    BWD_BIAS_INDEX			= 9,
    CONVERT_FORWARD_INPUT 		= 10,
    CONVERT_FORWARD_FILTER        	= 11,
    CONVERT_FORWARD_BIAS     		= 12,
    CONVERT_FORWARD_OUTPUT   		= 13,
    CONVERT_BWDDATA_INPUT 		= 14,
    CONVERT_BWDDATA_FILTER        	= 15,
    CONVERT_BWDDATA_OUTPUT   		= 16,
    CONVERT_BWDFILTER_INPUT 		= 17,
    CONVERT_BWDFILTER_FILTER        	= 18,
    CONVERT_BWDFILTER_OUTPUT   		= 19,
    BUFFER_FORWARD_INPUT 		= 20,
    BUFFER_FORWARD_FILTER        	= 21,
    BUFFER_FORWARD_BIAS     		= 22,
    BUFFER_FORWARD_OUTPUT   		= 23,
    BUFFER_BWDDATA_INPUT 		= 24,
    BUFFER_BWDDATA_FILTER        	= 25,
    BUFFER_BWDDATA_OUTPUT   		= 26,
    BUFFER_BWDFILTER_INPUT 		= 27,
    BUFFER_BWDFILTER_FILTER        	= 28,
    BUFFER_BWDFILTER_OUTPUT   		= 29
} mkldnnConvolutionIndex_t;



typedef enum {
    POOLING_LAYOUT_INPUT		= 0,
    POOLING_LAYOUT_OUTPUT		= 1,
    POOLING_LAYOUT_FORWARD_OUTPUT	= 2,
    POOLING_LAYOUT_BACKWARD_INPUT	= 3,
    POOLING_FORWARD            		= 4,
    POOLING_BACKWARD           		= 5,
    CV_POOLING_FORWARD_INPUT   		= 6,
    CV_POOLING_FORWARD_OUTPUT  		= 7,
    CV_POOLING_BACKWARD_INPUT  		= 8,
    CV_POOLING_BACKWARD_OUTPUT 		= 9,
    BUFFER_POOLING_FORWARD_INPUT       	= 10,
    BUFFER_POOLING_FORWARD_OUTPUT      	= 11,
    BUFFER_POOLING_FORWARD_WORKSPACE   	= 12,
    BUFFER_POOLING_BACKWARD_INPUT       = 13,
    BUFFER_POOLING_BACKWARD_OUTPUT      = 14,
    BUFFER_POOLING_BACKWARD_WORKSPACE   = 15
} mkldnnPoolingIndex_t;


typedef enum {
    RELU_LAYOUT_INPUT			= 0,
    RELU_LAYOUT_OUTPUT			= 1,
    RELU_LAYOUT_FORWARD_OUTPUT		= 2,
    RELU_LAYOUT_BACKWARD_INPUT		= 3,
    RELU_FORWARD            		= 4,
    RELU_BACKWARD           		= 5,
    CV_RELU_BACKWARD_OUTPUT		= 6,
    BUFFER_RELU_FORWARD_INPUT		= 7,
    BUFFER_RELU_FORWARD_OUTPUT		= 8,
    BUFFER_RELU_BACKWARD_INPUT		= 9,
    BUFFER_RELU_BACKWARD_OUTPUT		= 10
} mkldnnReLUIndex_t;

typedef enum {
    BN_LAYOUT_INPUT			= 0,
    BN_LAYOUT_OUTPUT			= 1,
    BN_LAYOUT_FORWARD_OUTPUT		= 2,
    BN_LAYOUT_BACKWARD_INPUT		= 3,
    BN_FORWARD            		= 4,
    BN_BACKWARD           		= 5,
    BN_SCALESHIFT           		= 6,
    BUFFER_BN_FORWARD_WORKSPACE       	= 7,
    BUFFER_BN_FORWARD_SCALESHIFT      	= 8,
    BUFFER_BN_FORWARD_OUTPUT      	= 9,
    BUFFER_BN_BACKWARD_WORKSPACE       	= 10,
    BUFFER_BN_BACKWARD_SCALESHIFT      	= 11,
    CV_BN_BACKWARD_OUTPUT		= 12,
    BUFFER_BN_BACKWARD_OUTPUT		= 13,
    BUFFER_BN_BACKWARD_INPUT		= 14
} mkldnnBNIndex_t;

typedef enum {
    LRN_LAYOUT_INPUT			= 0,
    LRN_LAYOUT_OUTPUT			= 1,
    LRN_LAYOUT_FORWARD_OUTPUT		= 2,
    LRN_LAYOUT_BACKWARD_INPUT		= 3,
    LRN_FORWARD            		= 4,
    LRN_BACKWARD           		= 5,
    BUFFER_LRN_WORKSPACE       		= 6,
    CV_LRN_BACKWARD_OUTPUT		= 7,
    BUFFER_LRN_BACKWARD_OUTPUT		= 8
} mkldnnLRNIndex_t;

typedef enum {
    CONCAT_LAYOUT_INPUT			= 0,
    CONCAT_LAYOUT_OUTPUT		= 1,
    CONCAT_LAYOUT_FORWARD_OUTPUT	= 2,
    CONCAT_LAYOUT_BACKWARD_INPUT	= 3,
    CONCAT_FORWARD			= 4,
    CONCAT_BACKWARD			= 5
} mkldnnConcatIndex_t;


/*compare source define:
Convolution:1(forward),2(bwd data),3(bwd filter)
MaxPooling:4(forward),5(backward)
ReLU:6(forward),7(backward)
AvgPooling:8(forward),9(backward)
BatchNormalization:10(forward),11(backward)
*/

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        fprintf(stderr,"[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
    } \
} while(0)	

#endif
