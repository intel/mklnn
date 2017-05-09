#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/SpatialMaxPooling.c"
#else

#include "MKLDNN.h"

static void MKLNN_(SpatialMaxPooling_init_forward)(
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
  int outW,
  int ceilmode)
{
#if LOG_ENABLE
  fprintf(stderr,"	SpatialMaxPooling_MKLDNN_init_forward start, N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d,ceilmode=%d\n",N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,ceilmode );
#endif
  dnnError_t err;

  int inputOffset[dimension - 2 ] = { 0, 0 };
  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
  size_t outputSize[dimension] = 	{outW,outH,outC,N};
  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

  size_t kernelSize[2] = { kH, kW };
  size_t kernelStride[2] = { dH, dW };


  real * resPool1[dnnResourceNumber] = {0};
  dnnLayout_t lt_user_input = NULL,lt_user_output=NULL;

  if(primitives->storage->data[POOLING_LAYOUT_INPUT] == 0) {
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling get input layout FAIL......\n");
#endif
  }
  else {
    lt_user_input = (dnnLayout_t)primitives->storage->data[POOLING_LAYOUT_INPUT];
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling get input layout OK\n");
#endif
  }
  CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
#if NEW_INTERFACE
  /*for new interface*/
  dnnPrimitiveAttributes_t attributes = NULL;
  CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif
  dnnPrimitive_t pool1 = NULL;
  dnnPrimitive_t pool_bwd = NULL;
#if NEW_INTERFACE
  if(ceilmode) {
    int pad[dimension] = 	{-padW,-padH};
    CHECK_ERR( dnnPoolingCreateForward_F32(&pool1, attributes, dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, pad, dnnBorderZeros), err );
    CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_bwd,attributes,dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, pad,dnnBorderZeros), err );
  }
  else {
    int pad[dimension] = 	{-padW,-padH,-padW,-padH};
    CHECK_ERR( dnnPoolingCreateForward_F32(&pool1, attributes, dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, pad, dnnBorderZerosAsymm), err );
    CHECK_ERR( dnnPoolingCreateBackward_F32(&pool_bwd,attributes,dnnAlgorithmPoolingMax,lt_user_input, kernelSize, kernelStride, pad,dnnBorderZerosAsymm), err );
  }
#endif
  dnnLayout_t lt_pool_forward_output = NULL,lt_pool_forward_input = NULL,lt_pool_forward_workspace = NULL;
  dnnPrimitive_t cv_forward_input = NULL,cv_forward_output = NULL;
  real * buffer_forward_input = NULL;
  real * buffer_forward_output = NULL;
  real * buffer_forward_workspace = NULL;

  CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_forward_input, pool1, dnnResourceSrc), err );
  CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_forward_output, pool1, dnnResourceDst), err );
  CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_forward_workspace, pool1, dnnResourceWorkspace), err );
  CHECK_ERR( dnnAllocateBuffer_F32((void**)&buffer_forward_workspace, lt_pool_forward_workspace), err );

#if CONVERSION_LOG
  int input_size = dnnLayoutGetMemorySize_F32(lt_pool_forward_input);
  int output_size = dnnLayoutGetMemorySize_F32(lt_pool_forward_output);
  int workspace_size = dnnLayoutGetMemorySize_F32(lt_pool_forward_workspace);
  fprintf(stderr, "MKLDNN MaxPooling workspace_size = %d, input_size =%d, output_size =%d \n",workspace_size,input_size,output_size);
  fprintf(stderr, "MKLDNN MaxPooling NCHW output_size =%d \n",N*outC*outH*outW);
#endif

  int size1 = dnnLayoutGetMemorySize_F32(lt_pool_forward_output);
  int size2 = outW*outH*outC*N*4;
  if(size1 == size2) {
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling forward ouput layout match OK\n");
#endif
  }
  else {
    if(!dnnLayoutCompare_F32(lt_user_output, lt_pool_forward_output)) {
      //fprintf(stderr, "cv_forward_output = 0x%x, lt_pool_forward_output = 0x%x, lt_user_output=0x%x \n",cv_forward_output,lt_pool_forward_output,lt_user_output);
      CHECK_ERR( dnnConversionCreate_F32(&cv_forward_output, lt_pool_forward_output, lt_user_output), err );
      CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_pool_forward_output), err );
    }
    fprintf(stderr,"MKLDNN MaxPooling forward ouput layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
  }
  //save the dnnPrimitive to THTensor(long int array)
  primitives->storage->data[POOLING_LAYOUT_FORWARD_OUTPUT] = (long long)lt_pool_forward_output;
  primitives->storage->data[POOLING_FORWARD] = (long long)pool1;
  primitives->storage->data[POOLING_BACKWARD] = (long long)pool_bwd;
  primitives->storage->data[CV_POOLING_FORWARD_INPUT] = (long long)cv_forward_input;
  primitives->storage->data[CV_POOLING_FORWARD_OUTPUT] = (long long)cv_forward_output;

  primitives->storage->data[BUFFER_POOLING_FORWARD_INPUT] = (long long)buffer_forward_input;
  primitives->storage->data[BUFFER_POOLING_FORWARD_OUTPUT] = (long long)buffer_forward_output;
  primitives->storage->data[BUFFER_POOLING_FORWARD_WORKSPACE] = (long long)buffer_forward_workspace;
  primitives->storage->data[BUFFER_POOLING_BACKWARD_WORKSPACE] = (long long)buffer_forward_workspace;

#if LOG_ENABLE
  fprintf(stderr,"	SpatialMaxPooling_MKLDNN_init_forward end.\n" );
#endif
}

static void MKLNN_(SpatialMaxPooling_init_backward)(
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
  int outW)
{
#if LOG_ENABLE
  fprintf(stderr,"	SpatialMaxPooling_MKLDNN_init_backward start, N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n",N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW );
#endif
  dnnError_t err;
  int inputOffset[dimension - 2 ] = { 0, 0 };
  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
  size_t outputSize[dimension] = 	{outW,outH,outC,N};
  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };

  size_t kernelSize[2] = { kH, kW };
  size_t kernelStride[2] = { dH, dW };
  int pad[dimension-2] = 	{-padW,-padH};

  real * resPool1[dnnResourceNumber] = {0};
  dnnLayout_t lt_user_input = NULL,lt_user_output=NULL;

  if(primitives->storage->data[POOLING_LAYOUT_OUTPUT] == 0) {
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling get output layout FAIL......\n");
#endif
  }
  else {
    lt_user_output = (dnnLayout_t)primitives->storage->data[POOLING_LAYOUT_OUTPUT];
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling get output layout OK\n");
#endif
  }
  CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );

#if NEW_INTERFACE
  /*for new interface*/
  dnnPrimitiveAttributes_t attributes = NULL;
  CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
#endif

  dnnPrimitive_t pool_bwd = (dnnPrimitive_t) (primitives->storage->data[POOLING_BACKWARD]);
  dnnLayout_t lt_pool_backward_output = NULL,lt_pool_backward_input = NULL,lt_pool_backward_workspace = NULL;
  dnnPrimitive_t cv_backward_input = NULL,cv_backward_output = NULL;
  real * buffer_backward_input = NULL;
  real * buffer_backward_output = NULL;
  real * buffer_backward_workspace = NULL;

  CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_backward_input, pool_bwd, dnnResourceDiffSrc), err );
  CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_pool_backward_output, pool_bwd, dnnResourceDiffDst), err );

  //backward conversion init
  CHECK_ERR( MKLNN_(init_conversion)(&cv_backward_output, &buffer_backward_output, lt_pool_backward_output, lt_user_output), err );

  int size1 = dnnLayoutGetMemorySize_F32(lt_pool_backward_input);
  int size2 = inW*inH*inC*N*4;
  if(size1 == size2) {
#if CONVERSION_LOG
    fprintf(stderr,"MKLDNN MaxPooling bwddata input layout match OK\n");
#endif
  }
  else {
    if(!dnnLayoutCompare_F32(lt_user_input, lt_pool_backward_input)) {
      CHECK_ERR( dnnConversionCreate_F32(&cv_backward_input, lt_pool_backward_input, lt_user_input), err );
      CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_backward_input), lt_pool_backward_input), err );
    }
    fprintf(stderr,"MKLDNN MaxPooling bwddata input layout match FAIL, size1 = %d, size2 = %d \n", size1, size2);
  }
  //save the dnnPrimitive to THTensor(long int array)
  primitives->storage->data[POOLING_LAYOUT_BACKWARD_INPUT] = (long long)lt_pool_backward_input;
  primitives->storage->data[CV_POOLING_BACKWARD_INPUT] = (long long)cv_backward_input;
  primitives->storage->data[CV_POOLING_BACKWARD_OUTPUT] = (long long)cv_backward_output;

  primitives->storage->data[BUFFER_POOLING_BACKWARD_INPUT] = (long long)buffer_backward_input;
  primitives->storage->data[BUFFER_POOLING_BACKWARD_OUTPUT] = (long long)buffer_backward_output;

#if LOG_ENABLE
  fprintf(stderr,"	SpatialMaxPooling_MKLDNN_init_backward end.\n" );
#endif
}

void MKLNN_(SpatialMaxPooling_updateOutput)(
  //THNNState *state,
  THMKLTensor *input,
  THMKLTensor *output,
  THMKLTensor *indices,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode,
  THLongTensor *primitives,
  int initOk)
{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  long nslices;
  long iheight;
  long iwidth;
  long oheight;
  long owidth;
  real *input_data;
  real *output_data;
  real *indices_data;
  //change
  printf("input->tensor->nDimension = %d ........\n",input->tensor->nDimension);
  THArgCheck(input->tensor->nDimension == 3 || input->tensor->nDimension == 4, 2, "3D or 4D (batch mode) tensor expected");
  //change
  if (input->tensor->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }
  THArgCheck(input->size[dimw] >= kW - padW && input->size[dimh] >= kH - padH, 2, "input image smaller than kernel size");

  THArgCheck(kW/2 >= padW && kH/2 >= padH, 2, "pad should be smaller than half of kernel size");

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  if (ceil_mode) {
    oheight = (long)(ceil((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(ceil((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }
  else {
    oheight = (long)(floor((float)(iheight - kH + 2*padH) / dH)) + 1;
    owidth  = (long)(floor((float)(iwidth  - kW + 2*padW) / dW)) + 1;
  }

  if (padW || padH) {
    // ensure that the last pooling starts inside the image
    if ((oheight - 1)*dH >= iheight + padH)
      --oheight;
    if ((owidth  - 1)*dW >= iwidth  + padW)
      --owidth;
  }

  /* get contiguous input */
  input = THTensor_(newContiguous)(input);
  //THTensor_(resize4d)(output, nbatch, nslices, oheight, owidth);
  TH_MKL_(resize4d)(output, nbatch, nslices, oheight, owidth);
  /* indices will contain the locations for each output point */
  //THTensor_(resize4d)(indices, nbatch, nslices, oheight, owidth);
  TH_MKL_(resize4d)(indices, nbatch, nslices, oheight, owidth);

  input_data = TH_MKL_(data)(input);
  output_data = TH_MKL_(data)(output);
  indices_data = TH_MKL_(data)(indices);

  /**************************************MKLDNN interface*****************************************/
  struct timeval start,end;
  gettimeofday(&start,NULL);
  dnnError_t err;

  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  int outC = output->size[1];
  int outH = output->size[2];
  int outW = output->size[3];

  if(initOk == 0) {
    primitives->storage->data[POOLING_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
    MKLNN_(SpatialMaxPooling_init_forward)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,ceil_mode);
  }

  dnnPrimitive_t cv_forward_input = NULL,cv_forward_output = NULL;
  real * buffer_forward_input = NULL;
  real * buffer_forward_output = NULL;
  real * buffer_forward_workspace = NULL;

  dnnPrimitive_t pool1 = (dnnPrimitive_t) (primitives->storage->data[POOLING_FORWARD]);
  cv_forward_input = (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_FORWARD_INPUT]);
  cv_forward_output = (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_FORWARD_OUTPUT]);
  buffer_forward_input	= (real *) (primitives->storage->data[BUFFER_POOLING_FORWARD_INPUT]);
  buffer_forward_output = (real *) (primitives->storage->data[BUFFER_POOLING_FORWARD_OUTPUT]);
  buffer_forward_workspace= (real *) (primitives->storage->data[BUFFER_POOLING_FORWARD_WORKSPACE]);

  real * resPool1[dnnResourceNumber] = {0};
  resPool1[dnnResourceSrc] = input_data;
  resPool1[dnnResourceDst] = output_data;
  resPool1[dnnResourceWorkspace] = buffer_forward_workspace;

  /*
  if(cv_forward_output){
  	resPool1[dnnResourceDst] = buffer_forward_output;

  }
  */
  CHECK_ERR( dnnExecute_F32(pool1, (void*)resPool1), err );
  /*
  if(cv_forward_output){
  	//fprintf(stderr,"	Pooling MKLDNN forward check1,output->mkldnnLayout = 0x%x\n",output->mkldnnLayout);
  	if(output->mkldnnLayout == 0)
  	{
  		int memSize = output->storage->size;
  		THStorage_(free)(output->storage);
  		output->storage = THStorage_(newWithData)(resPool1[dnnResourceDst],memSize);
  	}
  	output->storage->data = resPool1[dnnResourceDst];
  	output->storageOffset = 0;
  }
  */
  output->mkldnnLayout = (long long)primitives->storage->data[POOLING_LAYOUT_FORWARD_OUTPUT];
#if LOG_ENABLE || MKL_TIME
  gettimeofday(&end,NULL);
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	MaxPooling MKLDNN time forward = %.2f ms\n",duration );
#endif
  /* cleanup */
  TH_MKL_(free)(input);
}

void MKLNN_(SpatialMaxPooling_updateGradInput)(
  //THNNState *state,
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput,
  THMKLTensor *indices,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  bool ceil_mode,
  THLongTensor *primitives,
  int initOk)
{
  int dimw = 2;
  int dimh = 1;
  long nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  int oheight;
  int owidth;
  real *gradInput_data;
  real *gradOutput_data;
  real *indices_data;

  /* get contiguous gradOutput */
  gradOutput->tensor = THTensor_(newContiguous)(gradOutput->tensor);
  /* resize */
  TH_MKL_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput->tensor);
  //change
  if (input->tensor->nDimension == 4) {
    nbatch = input->size[0];
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = input->size[dimh-1];
  iheight = input->size[dimh];
  iwidth = input->size[dimw];
  oheight = gradOutput->size[dimh];
  owidth = gradOutput->size[dimw];

  /* get raw pointers */
  gradInput_data = TH_MKL_(data)(gradInput);
  gradOutput_data = TH_MKL_(data)(gradOutput);
  indices_data = TH_MKL_(data)(indices);
  //change
  /* backprop */
  if (input->tensor->nDimension == 3) {
    MKLNN_(SpatialMaxPooling_updateGradInput_frame)(
      gradInput_data,
      gradOutput_data,
      indices_data,
      nslices,
      iwidth,
      iheight,
      owidth,
      oheight,
      dW,
      dH);
  }
  else {
    /**************************************MKLDNN interface*****************************************/
    struct timeval start,end;
    gettimeofday(&start,NULL);
    dnnError_t err;

    dnnPrimitive_t pool_bwd = (dnnPrimitive_t) (primitives->storage->data[POOLING_BACKWARD]);
    dnnPrimitive_t cv_backward_input = NULL,cv_backward_output = NULL;
    real * buffer_backward_input = NULL;
    real * buffer_backward_output = NULL;
    real * buffer_backward_workspace = NULL;

    int N = input->size[0];
    int inC = input->size[1];
    int inH = input->size[2];
    int inW = input->size[3];

    int outC = gradOutput->size[1];
    int outH = gradOutput->size[2];
    int outW = gradOutput->size[3];

    if(initOk == 0) {
      primitives->storage->data[POOLING_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
      MKLNN_(SpatialMaxPooling_init_backward)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
    }
    cv_backward_input = (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_BACKWARD_INPUT]);
    cv_backward_output = (dnnPrimitive_t) (primitives->storage->data[CV_POOLING_BACKWARD_OUTPUT]);
    buffer_backward_input = (real *) (primitives->storage->data[BUFFER_POOLING_BACKWARD_INPUT]);
    buffer_backward_output = (real *) (primitives->storage->data[BUFFER_POOLING_BACKWARD_OUTPUT]);
    buffer_backward_workspace = (real *) (primitives->storage->data[BUFFER_POOLING_BACKWARD_WORKSPACE]);

    real * resPool1[dnnResourceNumber] = {0};
    resPool1[dnnResourceDiffSrc] = gradInput_data;
    resPool1[dnnResourceDiffDst] = gradOutput_data;
    resPool1[dnnResourceWorkspace] = buffer_backward_workspace;

    if(cv_backward_output) {
#if CONVERSION_LOG
      fprintf(stderr, "	Maxpooling backward output conversion\n");
#endif
      resPool1[dnnResourceDiffDst] = buffer_backward_output;
      CHECK_ERR( dnnConversionExecute_F32(cv_backward_output, gradOutput_data, resPool1[dnnResourceDiffDst]), err );
    }
    /*
    if(cv_backward_input){
       resPool1[dnnResourceDiffSrc] = buffer_backward_input;
    }

    */
    CHECK_ERR( dnnExecute_F32(pool_bwd, (void*)resPool1), err );

    /*	if(cv_backward_input){
    //fprintf(stderr, "	Maxpooling backward input conversion \n");
    if(gradInput->mkldnnLayout == 0)
    {
    	int memSize = gradInput->storage->size;
    	THStorage_(free)(gradInput->storage);
    	gradInput->storage = THStorage_(newWithData)(buffer_backward_input,memSize);
    }
    gradInput->storage->data = buffer_backward_input;
    gradInput->storageOffset = 0;
    }
         */
    gradInput->mkldnnLayout = (long long)primitives->storage->data[POOLING_LAYOUT_BACKWARD_INPUT];

#if LOG_ENABLE || MKL_TIME
    gettimeofday(&end,NULL);
    double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
    fprintf(stderr,"	MaxPooling MKLDNN time backward = %.2f ms\n",duration );
#endif
  }

  /* cleanup */
  TH_MKL_(free)(gradOutput);
}

#endif
