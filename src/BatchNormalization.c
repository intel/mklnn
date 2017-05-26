#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/BatchNormalization.c"
#else

//#include "MKLDNN.h"

static void MKLNN_(BatchNormalization_init_forward)(
  THLongTensor *primitives,
  int N,
  int inC,
  int inH,
  int inW,
  double eps)
{
  dnnError_t err;
  dnnPrimitive_t bn_forward = NULL;
  dnnPrimitive_t bn_backward = NULL;
  dnnPrimitive_t bn_bwd_scaleshift = NULL;

  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };

  dnnLayout_t lt_user_input = NULL;

  if(primitives->storage->data[BN_LAYOUT_INPUT] == 0) {
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN BN get input layout FAIL......\n");
#endif
  }
  else {
    lt_user_input = (dnnLayout_t)primitives->storage->data[BN_LAYOUT_INPUT];
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN BN get input layout OK\n");
#endif
  }

  CHECK_ERR( dnnBatchNormalizationCreateForward_F32(&bn_forward,NULL,lt_user_input,eps), err );
  CHECK_ERR( dnnBatchNormalizationCreateBackwardData_F32(&bn_backward,NULL,lt_user_input,eps), err );
  CHECK_ERR( dnnBatchNormalizationCreateBackwardScaleShift_F32(&bn_bwd_scaleshift,NULL,lt_user_input,eps), err );


  dnnLayout_t lt_bn_forward_workspace,lt_bn_forward_scaleshift,lt_bn_forward_output,lt_bn_backward_input;
  real * buffer_forward_workspace = NULL;
  real * buffer_forward_scaleshift = NULL;
  real * buffer_forward_output = NULL;
  real * buffer_backward_input = NULL;
  dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_workspace, bn_forward, dnnResourceWorkspace);
  dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_output, bn_forward, dnnResourceDst);
  dnnLayoutCreateFromPrimitive_F32(&lt_bn_forward_scaleshift, bn_forward, dnnResourceScaleShift);
  dnnLayoutCreateFromPrimitive_F32(&lt_bn_backward_input, bn_backward, dnnResourceDiffSrc);


  CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_workspace), lt_bn_forward_workspace), err );
  CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_scaleshift), lt_bn_forward_scaleshift), err );
  //CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_bn_forward_output), err );
  //CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backward_input), lt_bn_backward_input), err );

  int size1 = dnnLayoutGetMemorySize_F32(lt_bn_forward_output);
  int size2 = inW*inH*inC*N*4;
  if(size1 == size2) {
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN BN forward ouput layout match OK\n");
#endif
  }
  else {
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_bn_forward_output), err );
    fprintf(stderr,"MKLDNN BN forward ouput layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
  }

  size1 = dnnLayoutGetMemorySize_F32(lt_bn_backward_input);
  if(size1 == size2) {
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling bwddata input layout match OK\n");
#endif
  }
  else {
    CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backward_input), lt_bn_backward_input), err );
    fprintf(stderr,"MKLDNN MaxPooling bwddata input layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
  }

  //save the dnnPrimitive to THTensor(long int array)
  primitives->storage->data[BN_LAYOUT_FORWARD_OUTPUT] = (long long)lt_bn_forward_output;
  primitives->storage->data[BN_LAYOUT_BACKWARD_INPUT] = (long long)lt_bn_backward_input;

  primitives->storage->data[BN_FORWARD] = (long long)bn_forward;
  primitives->storage->data[BN_BACKWARD] = (long long)bn_backward;
  primitives->storage->data[BN_SCALESHIFT] = (long long)bn_bwd_scaleshift;
  primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE] = (long long)buffer_forward_workspace;
  primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT] = (long long)buffer_forward_scaleshift;
  primitives->storage->data[BUFFER_BN_FORWARD_OUTPUT] = (long long)buffer_forward_output;
  primitives->storage->data[BUFFER_BN_BACKWARD_INPUT] = (long long)buffer_backward_input;
  primitives->storage->data[BUFFER_BN_BACKWARD_WORKSPACE] = (long long)buffer_forward_workspace;
}

static void MKLNN_(BatchNormalization_init_backward)(
  THLongTensor *primitives,
  int N,
  int outC,
  int outH,
  int outW,
  double eps)
{
  dnnError_t err;

  dnnPrimitive_t bn_backward = (dnnPrimitive_t)primitives->storage->data[BN_BACKWARD];
  size_t outputSize[dimension] = {outW,outH,outC,N};
  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

  dnnLayout_t lt_user_output,lt_bn_backward_output=NULL;

  if(primitives->storage->data[BN_LAYOUT_OUTPUT] == 0) {
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN BN get output layout FAIL......\n");
#endif
  }
  else {
    lt_user_output = (dnnLayout_t)primitives->storage->data[BN_LAYOUT_OUTPUT];
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN BN get output layout OK\n");
#endif
  }

  dnnLayoutCreateFromPrimitive_F32(&lt_bn_backward_output, bn_backward, dnnResourceDiffDst);
  dnnPrimitive_t cv_backward_output = NULL;
  real * buffer_backward_output = NULL;
  //backward conversion init
  CHECK_ERR( MKLNN_(init_conversion)(&cv_backward_output, &buffer_backward_output, lt_bn_backward_output, lt_user_output), err );

  //save the dnnPrimitive to THTensor(long int array)
  primitives->storage->data[CV_BN_BACKWARD_OUTPUT] = (long long)cv_backward_output;
  primitives->storage->data[BUFFER_BN_BACKWARD_OUTPUT] = (long long)buffer_backward_output;
}

void MKLNN_(BatchNormalization_updateOutput)(
  //THNNState *state,
  THMKLTensor *input,
  THMKLTensor *output,
  THTensor *weight,
  THTensor *bias,
  THTensor *running_mean,
  THTensor *running_var,
  bool train,
  double momentum,
  double eps,
  THLongTensor *primitives,
  int initOk)
{ 
  //change
  //long nInput = THTensor_(size)(input, 1);
  long nInput = input->size[1];
  long f,n = THTensor_(nElement)(input->tensor) / nInput;
  TH_MKL_(resizeAs)(output,input);
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  if(initOk == 0) {
    primitives->storage->data[BN_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
    MKLNN_(BatchNormalization_init_forward)(primitives,N,inC,inH,inW,eps);
  }
  dnnPrimitive_t bn_forward = (dnnPrimitive_t)primitives->storage->data[BN_FORWARD];
  real * buffer_forward_workspace = (real *)primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE];
  real * buffer_forward_scaleshift = (real *)primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT];
  real * buffer_forward_output = (real *)primitives->storage->data[BUFFER_BN_FORWARD_OUTPUT];

  //release the original buffer, replace it with the internal buffer
  /*	if(output->mkldnnLayout == 0)
  {
  	int memSize = output->storage->size;
  	THStorage_(free)(output->storage);
  	output->storage = THStorage_(newWithData)(buffer_forward_output,memSize);
  }
  output->storage->data = buffer_forward_output;
  output->storageOffset = 0;
  */
  //fprintf(stderr, "BN MKLDNN, nInput = %d \n", nInput);
  int i = 0;
  for(; i < inC; i++) {
    buffer_forward_scaleshift[i] = weight ? THTensor_(get1d)(weight, i) : 1;
    buffer_forward_scaleshift[i+inC] = bias ? THTensor_(get1d)(bias, i) : 0;
  }
  gettimeofday(&mid,NULL);
  void* BatchNorm_res[dnnResourceNumber];
  BatchNorm_res[dnnResourceSrc] = TH_MKL_(data)(input);
  BatchNorm_res[dnnResourceDst] = TH_MKL_(data)(output);
  BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
  BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;
  CHECK_ERR( dnnExecute_F32(bn_forward, (void*)BatchNorm_res), err );
  output->mkldnnLayout = (long long)primitives->storage->data[BN_LAYOUT_FORWARD_OUTPUT];
  //output->storageOffset = 0;
#if LOG_ENABLE
  gettimeofday(&end,NULL);
  double duration1 = (mid.tv_sec - start.tv_sec) * 1000 + (double)(mid.tv_usec - start.tv_usec) /1000;
  double duration2 = (end.tv_sec - mid.tv_sec) * 1000 + (double)(end.tv_usec - mid.tv_usec) /1000;
  fprintf(stderr,"	BatchNorm MKLDNN forward time1 = %.2f ms, time2 = %.2f ms \n",duration1,duration2);
#endif
}

void MKLNN_(BatchNormalization_backward)(
  //THNNState *state,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *weight,
  THTensor *running_mean,
  THTensor *running_var,
  bool train,
  double scale,
  double eps,
  THLongTensor *primitives,
  int initOk)
{
  long nInput = THTensor_(size)(input->tensor, 1);
  long f,n = THTensor_(nElement)(input->tensor) / nInput;
  struct timeval start,mid,end;
  gettimeofday(&start,NULL);
  TH_MKL_(resizeAs)(gradInput,input);

  dnnError_t err;
  int inC = input->size[1];
  dnnPrimitive_t bn_backward 		= (dnnPrimitive_t)primitives->storage->data[BN_BACKWARD];
  dnnPrimitive_t bn_bwd_scaleshift 	= (dnnPrimitive_t)primitives->storage->data[BN_SCALESHIFT];
  real * buffer_forward_workspace 	= (real * )primitives->storage->data[BUFFER_BN_FORWARD_WORKSPACE];
  real * buffer_forward_scaleshift 	= (real * )primitives->storage->data[BUFFER_BN_FORWARD_SCALESHIFT];

  if(gradInput == 0) {
    void* BatchNormScaleshift_res[dnnResourceNumber];
    BatchNormScaleshift_res[dnnResourceSrc] = TH_MKL_(data)(input);
    BatchNormScaleshift_res[dnnResourceDiffDst] = TH_MKL_(data)(gradOutput);
    BatchNormScaleshift_res[dnnResourceDiffSrc] = TH_MKL_(data)(gradInput);
    BatchNormScaleshift_res[dnnResourceWorkspace] = buffer_forward_workspace;
    BatchNormScaleshift_res[dnnResourceScaleShift] = buffer_forward_scaleshift;
    CHECK_ERR( dnnExecute_F32(bn_bwd_scaleshift, (void*)BatchNormScaleshift_res), err );
    int i = 0;
    for(; i < inC; i++) {
      THTensor_(set1d)(gradWeight, i, buffer_forward_scaleshift[i]);
      THTensor_(set1d)(gradBias, i, buffer_forward_scaleshift[i+inC]);
    }
  }
  else {
    if(initOk == 0) {
      int N = gradOutput->size[0];
      int outC = gradOutput->size[1];
      int outH = gradOutput->size[2];
      int outW = gradOutput->size[3];

      primitives->storage->data[BN_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
      MKLNN_(BatchNormalization_init_backward)(primitives,N,outC,outH,outW,eps);
    }
    dnnPrimitive_t cv_backward_output = (dnnPrimitive_t) (primitives->storage->data[CV_BN_BACKWARD_OUTPUT]);

    real * buffer_backward_output = (real *) (primitives->storage->data[BUFFER_BN_BACKWARD_OUTPUT]);
    real * buffer_backward_input = (real *) (primitives->storage->data[BUFFER_BN_BACKWARD_INPUT]);
    /*
    if(gradInput->mkldnnLayout == 0)
    {
    	int memSize = gradInput->storage->size;
    	THStorage_(free)(gradInput->storage);
    	gradInput->storage = THStorage_(newWithData)(buffer_backward_input,memSize);
    }
    gradInput->storage->data = buffer_backward_input;
    gradInput->storageOffset = 0;
     */
    void* BatchNorm_res[dnnResourceNumber];
    BatchNorm_res[dnnResourceSrc] = TH_MKL_(data)(input);
    BatchNorm_res[dnnResourceDiffDst] = TH_MKL_(data)(gradOutput);
    BatchNorm_res[dnnResourceDiffSrc] = TH_MKL_(data)(gradInput);
    BatchNorm_res[dnnResourceWorkspace] = buffer_forward_workspace;
    BatchNorm_res[dnnResourceScaleShift] = buffer_forward_scaleshift;
    if(cv_backward_output) {
#if CONVERSION_LOG
      fprintf(stderr, "	BN backward output conversion... \n");
#endif
      BatchNorm_res[dnnResourceDiffDst] = buffer_backward_output;
      CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, TH_MKL_(data)(gradOutput), BatchNorm_res[dnnResourceDiffDst]), err );
    }
    gettimeofday(&mid,NULL);

    CHECK_ERR( dnnExecute_F32(bn_backward, (void*)BatchNorm_res), err );
    //fprintf(stderr, "bn_backward exec done");
    gradInput->mkldnnLayout = (long long)primitives->storage->data[BN_LAYOUT_BACKWARD_INPUT];
    //gradInput->storageOffset = 0;
  }
#if LOG_ENABLE
  gettimeofday(&end,NULL);
  double duration1 = (mid.tv_sec - start.tv_sec) * 1000 + (double)(mid.tv_usec - start.tv_usec) /1000;
  double duration2 = (end.tv_sec - mid.tv_sec) * 1000 + (double)(end.tv_usec - mid.tv_usec) /1000;
  fprintf(stderr,"        BatchNorm MKLDNN backward time1 = %.2f ms, time2 = %.2f ms \n",duration1,duration2);
#endif
}

#endif
