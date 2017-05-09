#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/LRN.c"
#else


#include "MKLDNN.h"


static void MKLNN_(CrossChannelLRN_init_forward)(
  THLongTensor *primitives,
  int N,
  int inC,
  int inH,
  int inW,
  int size, float alpha, float beta, float k)
{
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_init_forward start, N=%d,C=%d,H=%d,W=%d, size = %d, alpha = %.2f, beta = %.2f, k = %.2f \n", N,inC,inH,inW,size,alpha,beta,k);
#endif

  dnnError_t err;
  dnnPrimitive_t lrn_forward = NULL;
  dnnPrimitive_t lrn_backward = NULL;

  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };

  dnnLayout_t lt_user_input = NULL;

  if(primitives->storage->data[BN_LAYOUT_INPUT] == 0) {
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN LRN get input layout FAIL......\n");
#endif
  }
  else {
    lt_user_input = (dnnLayout_t)primitives->storage->data[BN_LAYOUT_INPUT];
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN LRN get input layout OK\n");
#endif
  }

  CHECK_ERR( dnnLRNCreateForward_F32(&lrn_forward,NULL,lt_user_input,size,alpha,beta,k), err );
  CHECK_ERR( dnnLRNCreateBackward_F32(&lrn_backward,NULL,lt_user_input,lt_user_input,size,alpha,beta,k), err );

  dnnLayout_t lt_lrn_workspace,lt_lrn_forward_output,lt_lrn_backward_input;;
  real * buffer_workspace = NULL;
  dnnLayoutCreateFromPrimitive_F32(&lt_lrn_workspace, lrn_forward, dnnResourceWorkspace);
  CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_workspace), lt_lrn_workspace), err );


  dnnLayoutCreateFromPrimitive_F32(&lt_lrn_forward_output, lrn_forward, dnnResourceDst);
  dnnLayoutCreateFromPrimitive_F32(&lt_lrn_backward_input, lrn_backward, dnnResourceDiffSrc);



  //save the dnnPrimitive to THTensor(long int array)
  primitives->storage->data[LRN_LAYOUT_FORWARD_OUTPUT] = (long long)lt_lrn_forward_output;
  primitives->storage->data[LRN_LAYOUT_BACKWARD_INPUT] = (long long)lt_lrn_backward_input;

  primitives->storage->data[LRN_FORWARD] = (long long)lrn_forward;
  primitives->storage->data[LRN_BACKWARD] = (long long)lrn_backward;
  primitives->storage->data[BUFFER_LRN_WORKSPACE] = (long long)buffer_workspace;
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_init_forward end.\n");
#endif
}

static void MKLNN_(CrossChannelLRN_init_backward)(
  THLongTensor *primitives,
  int N,
  int outC,
  int outH,
  int outW,
  int size, float alpha, float beta, float k)
{
  dnnError_t err;
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_init_backward start, N=%d,C=%d,H=%d,W=%d, size = %d, alpha = %.2f, beta = %.2f, k = %.2f \n", N,outC,outH,outW,size,alpha,beta,k);
#endif
  dnnPrimitive_t lrn_backward = (dnnPrimitive_t)primitives->storage->data[LRN_BACKWARD];
  size_t outputSize[dimension] = 	{outW,outH,outC,N};
  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

  dnnLayout_t lt_user_output,lt_lrn_backward_output=NULL;

  if(primitives->storage->data[LRN_LAYOUT_OUTPUT] == 0) {
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN LRN get output layout FAIL......\n");
#endif
  }
  else {
    lt_user_output = (dnnLayout_t)primitives->storage->data[LRN_LAYOUT_OUTPUT];
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN LRN get output layout OK\n");
#endif
  }

  dnnLayoutCreateFromPrimitive_F32(&lt_lrn_backward_output, lrn_backward, dnnResourceDiffDst);
  dnnPrimitive_t cv_backward_output = NULL;
  real * buffer_backward_output = NULL;
  //backward conversion init
  CHECK_ERR( MKLNN_(init_conversion)(&cv_backward_output, &buffer_backward_output, lt_lrn_backward_output, lt_user_output), err );

  //save the dnnPrimitive to THTensor(long int array)
  primitives->storage->data[CV_LRN_BACKWARD_OUTPUT] = (long long)cv_backward_output;
  primitives->storage->data[BUFFER_LRN_BACKWARD_OUTPUT] = (long long)buffer_backward_output;
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_init_backward end.\n");
#endif
}



void MKLNN_(CrossChannelLRN_updateOutput)(
  //THNNState *state,
  THMKLTensor *input, 
  THMKLTensor *output,
  int size, 
  float alpha, 
  float beta, 
  float k,
  THLongTensor *primitives,
  int initOk)
{
#if LOG_ENABLE
  fprintf(stderr, "BatchNormalization_MKLDNN_updateOutput start.\n");
#endif
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];
  TH_MKL_(resizeAs)(output,input);

  if(initOk == 0) {
    primitives->storage->data[LRN_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
    MKLNN_(CrossChannelLRN_init_forward)(primitives,N,inC,inH,inW,size,alpha,beta,k);
  }
  dnnPrimitive_t lrn_forward = (dnnPrimitive_t)primitives->storage->data[LRN_FORWARD];
  real * buffer_workspace = (real *)primitives->storage->data[BUFFER_LRN_WORKSPACE];


  void* LRN_res[dnnResourceNumber];
  LRN_res[dnnResourceSrc] = TH_MKL_(data)(input);
  LRN_res[dnnResourceDst] = TH_MKL_(data)(output);
  LRN_res[dnnResourceWorkspace] = buffer_workspace;

  CHECK_ERR( dnnExecute_F32(lrn_forward, (void*)LRN_res), err );
  output->mkldnnLayout = (long long)primitives->storage->data[LRN_LAYOUT_FORWARD_OUTPUT];
  //output->storageOffset = 0;
#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	LRN MKLDNN forward time = %.2f ms \n",duration1);
#endif

#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_updateOutput end.\n");
#endif
}


void MKLNN_(CrossChannelLRN_backward)(
  //THNNState *state,
  THMKLTensor *input, 
  THMKLTensor *gradOutput, 
  THMKLTensor *gradInput,
  int size, 
  float alpha, 
  float beta, 
  float k,
  THLongTensor *primitives,
  int initOk)
{
#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_backward start.\n");
#endif
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  dnnPrimitive_t bn_backward 		= (dnnPrimitive_t)primitives->storage->data[BN_BACKWARD];
  real * buffer_workspace 	= (real * )primitives->storage->data[BUFFER_LRN_WORKSPACE];

  if(initOk == 0) {
    int N = gradOutput->size[0];
    int outC = gradOutput->size[1];
    int outH = gradOutput->size[2];
    int outW = gradOutput->size[3];
    primitives->storage->data[LRN_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
    MKLNN_(CrossChannelLRN_init_backward)(primitives,N,outC,outH,outW,size,alpha,beta,k);
  }


  dnnPrimitive_t cv_backward_output = (dnnPrimitive_t) (primitives->storage->data[CV_LRN_BACKWARD_OUTPUT]);
  real * buffer_backward_output = (real *) (primitives->storage->data[BUFFER_LRN_BACKWARD_OUTPUT]);

  void* LRN_res[dnnResourceNumber];
  LRN_res[dnnResourceSrc] = THTensor_(data)(input);
  LRN_res[dnnResourceDiffDst] = THTensor_(data)(gradOutput);
  LRN_res[dnnResourceDiffSrc] = THTensor_(data)(gradInput);
  LRN_res[dnnResourceWorkspace] = buffer_workspace;

  if(cv_backward_output) {
#if CONVERSION_LOG
    fprintf(stderr, "	LRN backward output conversion... \n");
#endif
    LRN_res[dnnResourceDiffDst] = buffer_backward_output;
    CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, THTensor_(data)(gradOutput), LRN_res[dnnResourceDiffDst]), err );
  }

  CHECK_ERR( dnnExecute_F32(bn_backward, (void*)LRN_res), err );
  gradInput->mkldnnLayout = (long long)primitives->storage->data[LRN_LAYOUT_BACKWARD_INPUT];

#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	LRN MKLDNN backward time = %.2f ms \n",duration1);
#endif

#if LOG_ENABLE
  fprintf(stderr, "CrossChannelLRN_MKLDNN_backward end.\n");
#endif
}












#endif
