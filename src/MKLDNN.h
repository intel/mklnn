#ifndef _TORCH_MKLDNN_H
#define _TORCH_MKLDNN_H

#define LOG_ENABLE 	     0
#define CONVERSION_LOG	 0
#define MKL_TIME         0
#define NEW_INTERFACE    1
#define dimension        4

typedef enum {
   CONV_LAYOUT_INPUT                = 0,
   CONV_LAYOUT_OUTPUT               = 1,
   CONV_LAYOUT_FORWARD_OUTPUT       = 2,
   CONV_LAYOUT_BWDDATA_INPUT        = 3,
   CONV_LAYOUT_BWDFILT_OUTPUT       = 4,
   CONV_LAYOUT_FILTER               = 5,
   FORWARD_INDEX                    = 6,
   BWD_DATA_INDEX                   = 7,
   BWD_FILTER_INDEX                 = 8,
   BWD_BIAS_INDEX                   = 9,
   CONVERT_FORWARD_INPUT            = 10,
   CONVERT_FORWARD_FILTER           = 11,
   CONVERT_FORWARD_BIAS             = 12,
   CONVERT_FORWARD_OUTPUT           = 13,
   CONVERT_BWDDATA_INPUT            = 14,
   CONVERT_BWDDATA_FILTER           = 15,
   CONVERT_BWDDATA_OUTPUT           = 16,
   CONVERT_BWDFILTER_INPUT          = 17,
   CONVERT_BWDFILTER_FILTER         = 18,
   CONVERT_BWDFILTER_OUTPUT         = 19,
   BUFFER_FORWARD_INPUT             = 20,
   BUFFER_FORWARD_FILTER            = 21,
   BUFFER_FORWARD_BIAS              = 22,
   BUFFER_FORWARD_OUTPUT            = 23,
   BUFFER_BWDDATA_INPUT             = 24,
   BUFFER_BWDDATA_FILTER            = 25,
   BUFFER_BWDDATA_OUTPUT            = 26,
   BUFFER_BWDFILTER_INPUT           = 27,
   BUFFER_BWDFILTER_FILTER          = 28,
   BUFFER_BWDFILTER_OUTPUT          = 29
} mkldnnConvolutionIndex_t;

typedef enum {
   POOLING_LAYOUT_INPUT             = 0,
   POOLING_LAYOUT_OUTPUT            = 1,
   POOLING_LAYOUT_FORWARD_OUTPUT    = 2,
   POOLING_LAYOUT_BACKWARD_INPUT    = 3,
   POOLING_FORWARD                  = 4,
   POOLING_BACKWARD                 = 5,
   CV_POOLING_FORWARD_INPUT         = 6,
   CV_POOLING_FORWARD_OUTPUT        = 7,
   CV_POOLING_BACKWARD_INPUT        = 8,
   CV_POOLING_BACKWARD_OUTPUT       = 9,
   BUFFER_POOLING_FORWARD_INPUT     = 10,
   BUFFER_POOLING_FORWARD_OUTPUT    = 11,
   BUFFER_POOLING_FORWARD_WORKSPACE = 12,
   BUFFER_POOLING_BACKWARD_INPUT    = 13,
   BUFFER_POOLING_BACKWARD_OUTPUT   = 14,
   BUFFER_POOLING_BACKWARD_WORKSPACE= 15
} mkldnnPoolingIndex_t;

typedef enum {
   RELU_LAYOUT_INPUT                = 0,
   RELU_LAYOUT_OUTPUT               = 1,
   RELU_LAYOUT_FORWARD_OUTPUT       = 2,
   RELU_LAYOUT_BACKWARD_INPUT       = 3,
   RELU_FORWARD                     = 4,
   RELU_BACKWARD                    = 5,
   CV_RELU_BACKWARD_OUTPUT	        = 6,
   BUFFER_RELU_FORWARD_INPUT        = 7,
   BUFFER_RELU_FORWARD_OUTPUT       = 8,
   BUFFER_RELU_BACKWARD_INPUT       = 9,
   BUFFER_RELU_BACKWARD_OUTPUT      = 10
} mkldnnReLUIndex_t;

typedef enum {
   BN_LAYOUT_INPUT                  = 0,
   BN_LAYOUT_OUTPUT	                = 1,
   BN_LAYOUT_FORWARD_OUTPUT         = 2,
   BN_LAYOUT_BACKWARD_INPUT         = 3,
   BN_FORWARD                       = 4,
   BN_BACKWARD                      = 5,
   BN_SCALESHIFT                    = 6,
   BUFFER_BN_FORWARD_WORKSPACE      = 7,
   BUFFER_BN_FORWARD_SCALESHIFT     = 8,
   BUFFER_BN_FORWARD_OUTPUT         = 9,
   BUFFER_BN_BACKWARD_WORKSPACE     = 10,
   BUFFER_BN_BACKWARD_SCALESHIFT    = 11,
   CV_BN_BACKWARD_OUTPUT            = 12,
   BUFFER_BN_BACKWARD_OUTPUT        = 13,
   BUFFER_BN_BACKWARD_INPUT         = 14
} mkldnnBNIndex_t;

typedef enum {
   LRN_LAYOUT_INPUT                 = 0,
   LRN_LAYOUT_OUTPUT                = 1,
   LRN_LAYOUT_FORWARD_OUTPUT        = 2,
   LRN_LAYOUT_BACKWARD_INPUT        = 3,
   LRN_FORWARD                      = 4,
   LRN_BACKWARD                     = 5,
   BUFFER_LRN_WORKSPACE             = 6,
   CV_LRN_BACKWARD_OUTPUT           = 7,
   BUFFER_LRN_BACKWARD_OUTPUT       = 8
} mkldnnLRNIndex_t;

typedef enum {
   CONCAT_LAYOUT_INPUT              = 0,
   CONCAT_LAYOUT_OUTPUT             = 1,
   CONCAT_LAYOUT_FORWARD_OUTPUT	    = 2,
   CONCAT_LAYOUT_BACKWARD_INPUT	    = 3,
   CONCAT_FORWARD                   = 4,
   CONCAT_BACKWARD                  = 5
} mkldnnConcatIndex_t;

#define CHECK_ERR(f, err) do { \
    (err) = (f); \
    if ((err) != E_SUCCESS) { \
        fprintf(stderr,"[%s:%d] err (%d)\n", __FILE__, __LINE__, err); \
    } \
} while(0)

#endif
