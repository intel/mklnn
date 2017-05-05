#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/SpatialConvolution.c"
#else

dnnError_t  MKLNN_(init_conversion)(
  dnnPrimitive_t *cv,
  real **ptr_out,
  dnnLayout_t lt_pr,
  dnnLayout_t lt_us)
{
  dnnError_t err;
  *ptr_out = NULL;
  if(sizeof(real) == sizeof(float)) {
    if(!dnnLayoutCompare_F32(lt_pr, lt_us)) {
      CHECK_ERR( dnnConversionCreate_F32(cv, lt_us, lt_pr), err );
      CHECK_ERR( dnnAllocateBuffer_F32((void**)ptr_out, lt_pr), err );
    }
    return E_SUCCESS;
  }
  else if(sizeof(real) == sizeof(double)) {
    if(!dnnLayoutCompare_F64(lt_pr, lt_us)) {
      CHECK_ERR( dnnConversionCreate_F64(cv, lt_us, lt_pr), err );
      CHECK_ERR( dnnAllocateBuffer_F64((void**)ptr_out, lt_pr), err );
    }
    return E_SUCCESS;
  }
}

static void MKLNN_(SpatialConvolution_init_forward)(
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
  int group)
{
  dnnError_t err;
#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_forward: start.");
  fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
  dnnPrimitive_t m_conv_forward = NULL;
  dnnPrimitive_t m_conv_bwd_data = NULL;
  dnnPrimitive_t m_conv_bwd_filter = NULL;
  dnnPrimitive_t m_conv_bwd_bias = NULL;

  int f_dimension = dimension + (group != 1);
  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t filterSize[5] = 	{kW,kH,inC/group,outC/group,group};
  size_t outputSize[dimension] = 	{outW,outH,outC,N};
  size_t stride[dimension-2] = 	{dW,dH};
  int pad[dimension-2] = 		{-padW,-padH};

  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
  size_t filterStrides[5] = { 1, kW, kH * kW, (inC/group) * kH * kW, (inC/group)*(outC/group) * kH * kW };

  size_t biasSize[1] = { outputSize[2] };
  size_t biasStrides[1] = { 1 };

  //user layouts
  dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
  //forward layouts
  dnnLayout_t lt_forward_conv_input, lt_forward_conv_filter, lt_forward_conv_bias, lt_forward_conv_output;

  //forward conversions and buffers
  dnnPrimitive_t cv_forward_input = NULL,cv_forward_filter = NULL,cv_forward_bias = NULL,cv_forward_output = NULL;
  real * buffer_forward_input =NULL;
  real *buffer_forward_filter=NULL;
  real *buffer_forward_bias=NULL;
  real * buffer_forward_output =NULL;

  dnnPrimitiveAttributes_t attributes = NULL;
  CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
  if(sizeof(real) == sizeof(float)) {
    if(primitives->storage->data[CONV_LAYOUT_INPUT] == 0) {
      CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );
#if CONVERSION_LOG
      printf("MKLDNN Convolution get input layout FAIL......\n");
#endif
    }
    else {
      lt_user_input = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_INPUT];
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution get input layout OK\n");
#endif
    }
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, f_dimension, filterSize, filterStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );

    CHECK_ERR(dnnGroupsConvolutionCreateForwardBias_F32(&m_conv_forward, attributes, dnnAlgorithmConvolutionDirect, group, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
    CHECK_ERR(dnnGroupsConvolutionCreateBackwardData_F32(&m_conv_bwd_data, attributes, dnnAlgorithmConvolutionDirect, group, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
    CHECK_ERR(dnnGroupsConvolutionCreateBackwardFilter_F32(&m_conv_bwd_filter, attributes, dnnAlgorithmConvolutionDirect, group, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
    CHECK_ERR(dnnGroupsConvolutionCreateBackwardBias_F32(&m_conv_bwd_bias, attributes, dnnAlgorithmConvolutionDirect, group, dimension, outputSize),err);

    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_input, m_conv_forward, dnnResourceSrc), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_filter, m_conv_forward, dnnResourceFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_bias, m_conv_forward, dnnResourceBias), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_output,m_conv_forward, dnnResourceDst), err );

    //init forward conversions:
    CHECK_ERR( MKLNN_(init_conversion)(&cv_forward_input, 	&buffer_forward_input, 	lt_forward_conv_input, 	lt_user_input), err );
    CHECK_ERR( MKLNN_(init_conversion)(&cv_forward_filter, 	&buffer_forward_filter, lt_forward_conv_filter, lt_user_filter), err );
    CHECK_ERR( MKLNN_(init_conversion)(&cv_forward_bias, 	&buffer_forward_bias, 	lt_forward_conv_bias, 	lt_user_bias), err );


    int size1 = dnnLayoutGetMemorySize_F32(lt_forward_conv_output);
    int size2 = dnnLayoutGetMemorySize_F32(lt_user_output);
    if(size1 == size2 && size2 == (outW*outH*outC*N*4)) {
#if CONVERSION_LOG
      printf("MKLDNN Convolution forward ouput layout match OK\n");
#endif
    }
    else {
      if(!dnnLayoutCompare_F32(lt_forward_conv_output, lt_user_output)) {
        CHECK_ERR( dnnConversionCreate_F32(&cv_forward_output, 	lt_forward_conv_output, lt_user_output), err );
      }
      CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_forward_output), lt_forward_conv_output), err );
      printf("MKLDNN Convolution forward output layout match FAIL: size1 = %d, size2 = %d, NCHW = %d \n",size1,size2,outW*outH*outC*N);
    }
  }
  else if(sizeof(real) == sizeof(double)) {
    CHECK_ERR(dnnConvolutionCreateForward_F64(&m_conv_forward,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
  }

  //save the dnnPrimitive to THTensor(long int array)
  //save the output layout to dnnPrimitive
  primitives->storage->data[CONV_LAYOUT_FORWARD_OUTPUT] = (long long)lt_forward_conv_output;
  primitives->storage->data[CONV_LAYOUT_INPUT]          = (long long)lt_forward_conv_input;
  primitives->storage->data[FORWARD_INDEX]              = (long long)m_conv_forward;
  primitives->storage->data[BWD_DATA_INDEX]             = (long long)m_conv_bwd_data;
  primitives->storage->data[BWD_FILTER_INDEX]           = (long long)m_conv_bwd_filter;
  primitives->storage->data[BWD_BIAS_INDEX]             = (long long)m_conv_bwd_bias;

  primitives->storage->data[CONVERT_FORWARD_INPUT]      = (long long)cv_forward_input;
  primitives->storage->data[CONVERT_FORWARD_FILTER]     = (long long)cv_forward_filter;
  primitives->storage->data[CONVERT_FORWARD_BIAS]       = (long long)cv_forward_bias;
  primitives->storage->data[CONVERT_FORWARD_OUTPUT]     = (long long)cv_forward_output;

  primitives->storage->data[BUFFER_FORWARD_INPUT]       = (long long)buffer_forward_input;
  primitives->storage->data[BUFFER_FORWARD_FILTER]      = (long long)buffer_forward_filter;
  primitives->storage->data[BUFFER_FORWARD_BIAS]        = (long long)buffer_forward_bias;
  primitives->storage->data[BUFFER_FORWARD_OUTPUT]      = (long long)buffer_forward_output;



#if LOG_ENABLE
  printf("cv_forward_input=%d,cv_forward_filter=%d,cv_forward_output=%d \n",cv_forward_input,cv_forward_filter,cv_forward_output);
  printf("SpatialConvolutionMM_MKLDNN_init_forward: end, sizeof(real)=%d\n",sizeof(real));
#endif

}


static void MKLNN_(SpatialConvolution_init_bwddata)(
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
  int group)
{
  dnnError_t err;
#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwddata: start.");
  fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
  dnnPrimitive_t m_conv_bwd_data = NULL;

  int f_dimension = dimension + (group != 1);
  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t filterSize[5] = 	{kW,kH,inC/group,outC/group,group};
  size_t outputSize[dimension] = 	{outW,outH,outC,N};
  size_t stride[dimension-2] = 	{dW,dH};
  int pad[dimension-2] = 		{-padW,-padH};

  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
  size_t filterStrides[5] = { 1, kW, kH * kW, (inC/group) * kH * kW, (inC/group)*(outC/group) * kH * kW };
  size_t biasSize[1] = { outputSize[2] };
  size_t biasStrides[1] = { 1 };

  //user layouts
  dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
  //backward data layouts
  dnnLayout_t lt_bwddata_conv_input, lt_bwddata_conv_filter,lt_bwddata_conv_output;
  //backward data conversions and buffers
  dnnPrimitive_t cv_bwddata_input = NULL,cv_bwddata_filter = NULL,cv_bwddata_output = NULL;
  real * buffer_bwddata_input = NULL;
  real * buffer_bwddata_filter = NULL;
  real * buffer_bwddata_output=NULL;
  dnnPrimitiveAttributes_t attributes = NULL;
  CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );

  if(sizeof(real) == sizeof(float)) {
    if(primitives->storage->data[CONV_LAYOUT_OUTPUT] == 0) {
      CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution get output layout FAIL......\n");
#endif
    }
    else {
      lt_user_output = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_OUTPUT];
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution get output layout OK\n");
#endif
    }
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, f_dimension, filterSize, filterStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );

    m_conv_bwd_data = (dnnPrimitive_t) (primitives->storage->data[BWD_DATA_INDEX]);
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_input, m_conv_bwd_data, dnnResourceDiffSrc), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_filter, m_conv_bwd_data, dnnResourceFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwddata_conv_output, m_conv_bwd_data, dnnResourceDiffDst), err );

    //get forward filter layout, convert from forward filter to bdwdata filter
    dnnPrimitive_t m_conv_forward = (dnnPrimitive_t)primitives->storage->data[FORWARD_INDEX];
    dnnLayout_t lt_forward_conv_filter = NULL;
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_forward_conv_filter, m_conv_forward, dnnResourceFilter), err );

    CHECK_ERR( MKLNN_(init_conversion)(&cv_bwddata_filter, 	&buffer_bwddata_filter, lt_bwddata_conv_filter, lt_forward_conv_filter), err );
    CHECK_ERR( MKLNN_(init_conversion)(&cv_bwddata_output, 	&buffer_bwddata_output, lt_bwddata_conv_output, lt_user_output), err );

    int size1 = dnnLayoutGetMemorySize_F32(lt_bwddata_conv_input);
    int size2 = dnnLayoutGetMemorySize_F32(lt_user_input);
    if(size1 == size2 && size2 == (inW*inH*inC*N*4)) {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwddata input layout match OK\n");
#endif
    }
    else {
      if(!dnnLayoutCompare_F32(lt_bwddata_conv_input, lt_user_input)) {
        CHECK_ERR( dnnConversionCreate_F32(&cv_bwddata_input, lt_bwddata_conv_input, lt_user_input), err );
      }
      CHECK_ERR( dnnAllocateBuffer_F32((void**)(&buffer_bwddata_input), lt_bwddata_conv_input), err );
    }

#if CONVERSION_LOG
    dnnLayout_t lt_conv_forward_output = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_FORWARD_OUTPUT];
    int check1 = dnnLayoutCompare_F32(lt_user_output, lt_bwddata_conv_output);
    int check2 = dnnLayoutCompare_F32(lt_user_output, lt_conv_forward_output);
    int check3 = dnnLayoutCompare_F32(lt_conv_forward_output, lt_bwddata_conv_output);
    int check4 = dnnLayoutCompare_F32(primitives->storage->data[CONV_LAYOUT_INPUT], lt_bwddata_conv_input);
    fprintf(stderr, "	MKLDNN Convolution backward data, check1=%d,check2=%d,check3=%d, check4=%d\n", check1,check2,check3,check4);
#endif
  }
  else if(sizeof(real) == sizeof(double)) {
    CHECK_ERR(dnnConvolutionCreateBackwardData_F64(&m_conv_bwd_data,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
  }

  //save the dnnPrimitive to THTensor(long int array)
  //save the output layout to dnnPrimitive
  primitives->storage->data[CONV_LAYOUT_BWDDATA_INPUT]  = (long long)lt_bwddata_conv_input;
  primitives->storage->data[CONVERT_BWDDATA_INPUT]      = (long long)cv_bwddata_input;
  primitives->storage->data[CONVERT_BWDDATA_FILTER]     = (long long)cv_bwddata_filter;
  primitives->storage->data[CONVERT_BWDDATA_OUTPUT]     = (long long)cv_bwddata_output;
  primitives->storage->data[BUFFER_BWDDATA_INPUT]       = (long long)buffer_bwddata_input;
  primitives->storage->data[BUFFER_BWDDATA_FILTER]      = (long long)buffer_bwddata_filter;
  primitives->storage->data[BUFFER_BWDDATA_OUTPUT]      = (long long)buffer_bwddata_output;
#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwddata: end, sizeof(real)=%d\n",sizeof(real));
#endif

}

static void MKLNN_(SpatialConvolution_init_bwdfilter)(
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
  int outW,int group)

{
  dnnError_t err;
#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwdfilter: start.");
  fprintf(stderr, "N=%d,inC=%d,inH=%d,inW=%d,kH=%d,kW=%d,dH=%d,dW=%d,padH=%d,padW=%d,outC=%d,outH=%d,outW=%d\n", N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW);
#endif
  dnnPrimitive_t m_conv_bwd_filter = NULL;

  int f_dimension = dimension + (group != 1);
  size_t inputSize[dimension] = 	{inW,inH,inC,N};
  size_t filterSize[5] = 	{kW,kH,inC/group,outC/group,group};
  size_t outputSize[dimension] = 	{outW,outH,outC,N};
  size_t stride[dimension-2] = 	{dW,dH};
  int pad[dimension-2] = 		{-padW,-padH};

  size_t outputStrides[dimension] = { 1, outW, outH * outW, outC * outH * outW };
  size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
  size_t filterStrides[5] = { 1, kW, kH * kW, (inC/group) * kH * kW, (inC/group)*(outC/group) * kH * kW };

  size_t biasSize[1] = { outputSize[2] };
  size_t biasStrides[1] = { 1 };

  //user layouts
  dnnLayout_t lt_user_input, lt_user_filter, lt_user_bias, lt_user_output;
  //backward filter layouts
  dnnLayout_t lt_bwdfilter_conv_input, lt_bwdfilter_conv_filter,lt_bwdfilter_conv_output;
  //backward filter conversions and buffers
  dnnPrimitive_t cv_bwdfilter_input = NULL,cv_bwdfilter_filter = NULL,cv_bwdfilter_output = NULL;
  real * buffer_bwdfilter_input = NULL;
  real * buffer_bwdfilter_filter = NULL;
  real * buffer_bwdfilter_output = NULL;

  dnnPrimitiveAttributes_t attributes = NULL;
  CHECK_ERR( dnnPrimitiveAttributesCreate_F32(&attributes), err );
  if(sizeof(real) == sizeof(float)) {
    //check the src and diffdst layout
    if(primitives->storage->data[CONV_LAYOUT_INPUT] == 0) {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter get input layout FAIL......\n");
#endif
      CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides), err );
    }
    else {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter get input layout OK\n");
#endif
      lt_user_input = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_INPUT];
    }
    if(primitives->storage->data[CONV_LAYOUT_OUTPUT] == 0) {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter get output layout FAIL......\n");
#endif
      CHECK_ERR( dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides), err );
    }
    else {

#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter get output layout OK\n");
#endif
      lt_user_output = (dnnLayout_t)primitives->storage->data[CONV_LAYOUT_OUTPUT];
    }
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_filter, f_dimension, filterSize, filterStrides), err );
    CHECK_ERR( dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides), err );

    m_conv_bwd_filter = (dnnPrimitive_t) (primitives->storage->data[BWD_FILTER_INDEX]);
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_input, m_conv_bwd_filter, dnnResourceSrc), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_filter, m_conv_bwd_filter, dnnResourceDiffFilter), err );
    CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_bwdfilter_conv_output, m_conv_bwd_filter, dnnResourceDiffDst), err );

    //init backward filter conversions:
    CHECK_ERR( MKLNN_(init_conversion)(&cv_bwdfilter_input, &buffer_bwdfilter_input, lt_bwdfilter_conv_input, lt_user_input), err );
    CHECK_ERR( MKLNN_(init_conversion)(&cv_bwdfilter_output, &buffer_bwdfilter_output, lt_bwdfilter_conv_output, lt_user_output), err );
  }
  else if(sizeof(real) == sizeof(double)) {
    CHECK_ERR(dnnConvolutionCreateBackwardFilter_F64(&m_conv_bwd_filter,attributes, dnnAlgorithmConvolutionDirect, dimension, inputSize, outputSize, filterSize,stride,pad,dnnBorderZeros),err);
  }

  //save the dnnPrimitive to THTensor(long int array)
  //save the output layout to dnnPrimitive
  primitives->storage->data[CONV_LAYOUT_BWDFILT_OUTPUT] = (long long)lt_bwdfilter_conv_filter;
  primitives->storage->data[CONVERT_BWDFILTER_INPUT]    = (long long)cv_bwdfilter_input;
  primitives->storage->data[CONVERT_BWDFILTER_FILTER]   = (long long)cv_bwdfilter_filter;
  primitives->storage->data[CONVERT_BWDFILTER_OUTPUT]   = (long long)cv_bwdfilter_output;
  primitives->storage->data[BUFFER_BWDFILTER_INPUT]     = (long long)buffer_bwdfilter_input;
  primitives->storage->data[BUFFER_BWDFILTER_FILTER]    = (long long)buffer_bwdfilter_filter;
  primitives->storage->data[BUFFER_BWDFILTER_OUTPUT]    = (long long)buffer_bwdfilter_output;
#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_init_bwdfilter: end, sizeof(real)=%d\n",sizeof(real));
#endif

}


void MKLNN_(SpatialConvolution_forward)(
  THMKLTensor *input,
  THMKLTensor *output,
  THTensor *weight,
  THTensor *bias,
  THTensor *finput,
  THTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  int group)
{
  struct timeval start,mid,convert1,convert2,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  dnnPrimitive_t m_conv_forward = NULL;

  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  int outC = weight->size[0];
  int outH = (inH + 2*padH - kH)/dH + 1;
  int outW = (inW + 2*padW - kW)/dW + 1;

  dnnPrimitive_t cv_forward_input = NULL,cv_forward_filter = NULL,cv_forward_bias = NULL,cv_forward_output = NULL;
  real * buffer_forward_input =NULL;
  real *buffer_forward_filter=NULL;
  real *buffer_forward_bias=NULL;
  real * buffer_forward_output =NULL;
  if(initOk == 0) {
    primitives->storage->data[CONV_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
    MKLNN_(SpatialConvolution_init_forward)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,group);
  }
  m_conv_forward 		= (dnnPrimitive_t)(primitives->storage->data[FORWARD_INDEX]);
  cv_forward_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_INPUT];
  cv_forward_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_FILTER];
  cv_forward_bias 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_BIAS];
  cv_forward_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_FORWARD_OUTPUT];
  buffer_forward_input 	= (real *)(primitives->storage->data[BUFFER_FORWARD_INPUT]);
  buffer_forward_filter 	= (real *)(primitives->storage->data[BUFFER_FORWARD_FILTER]);
  buffer_forward_bias 	= (real *)(primitives->storage->data[BUFFER_FORWARD_BIAS]);
  buffer_forward_output 	= (real *)(primitives->storage->data[BUFFER_FORWARD_OUTPUT]);

  TH_MKL_(resize4d)(output, N, outC, outH, outW);
#if LOG_ENABLE
  gettimeofday(&mid,NULL);
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward: start, m_conv_forward = 0x%x \n",m_conv_forward);
  fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);
  fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", output->size[0],output->size[1],output->size[2],output->size[3]);
  fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", weight->size[0],weight->size[1]);
  fprintf(stderr, "	bias->nDimension=%d,bias->size[0]=%d,bias->storage->data[0]=%.3f\n", bias->nDimension,bias->size[0],bias->storage->data[0]);
  fprintf(stderr, " cv_forward_input=0x%x,cv_forward_filter=0x%x,cv_forward_bias=0x%x,cv_forward_output=0x%x",cv_forward_input,cv_forward_filter,cv_forward_bias,cv_forward_output);
#endif
  long long i = 0;
  real * inPtr = TH_MKL_(data)(input);
  real * filterPtr = THTensor_(data)(weight);
  real * outPtr = TH_MKL_(data)(output);
  real * biasPtr = THTensor_(data)(bias);

  real * resConv[dnnResourceNumber]= {0};
  resConv[dnnResourceSrc] 	= inPtr;
  resConv[dnnResourceFilter] 	= filterPtr;
  resConv[dnnResourceBias] 	= biasPtr;
  resConv[dnnResourceDst] 	= outPtr;
  void *convert_resources[dnnResourceNumber];

  if(sizeof(real) == sizeof(float)) {
    if(cv_forward_input) {
      resConv[dnnResourceSrc] = buffer_forward_input;
      convert_resources[dnnResourceFrom] = inPtr;
      convert_resources[dnnResourceTo]   = buffer_forward_input;
      CHECK_ERR( dnnExecute_F32(cv_forward_input, convert_resources), err );
#if CONVERSION_LOG
      fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion input \n");
#endif
      //optimize for input conversion, save the new layout , to avoid conversion in backward filter
      //	input->storage->data = buffer_forward_input;
      //	input->storageOffset = 0;
      //	input->mkldnnLayout = primitives->storage->data[CONV_LAYOUT_INPUT];
    }

    if(cv_forward_filter && initOk == 0) {
      resConv[dnnResourceFilter] = buffer_forward_filter;
      convert_resources[dnnResourceFrom] = filterPtr;
      convert_resources[dnnResourceTo]   = buffer_forward_filter;
      CHECK_ERR( dnnExecute_F32(cv_forward_filter, convert_resources), err );
      memcpy(filterPtr, buffer_forward_filter, THTensor_(nElement)(weight)*4);
      //weight->storage->data = buffer_forward_filter;
      //weight->storageOffset = 0;
#if CONVERSION_LOG
      fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion filter \n");
#endif
    }

    if(cv_forward_bias) {
      resConv[dnnResourceBias] = buffer_forward_bias;
      convert_resources[dnnResourceFrom] = biasPtr;
      convert_resources[dnnResourceTo]   = buffer_forward_bias;
      CHECK_ERR( dnnExecute_F32(cv_forward_bias,convert_resources), err );
#if CONVERSION_LOG
      fprintf(stderr, "SpatialConvolutionMM_MKLDNN_forward conversion bias \n");
#endif
    }
    gettimeofday(&convert1,NULL);
    CHECK_ERR(dnnExecute_F32(m_conv_forward, (void**)resConv),err);
    gettimeofday(&convert2,NULL);
    output->mkldnnLayout = (long long)primitives->storage->data[CONV_LAYOUT_FORWARD_OUTPUT];
  }
  else if(sizeof(real) == sizeof(double)) {
    CHECK_ERR(dnnExecute_F64(m_conv_forward, (void**)resConv),err);
  }
#if LOG_ENABLE
  gettimeofday(&end,NULL);
  double duration1 = (mid.tv_sec - start.tv_sec) * 1000 + (double)(mid.tv_usec - start.tv_usec) /1000;
  double duration2 = (end.tv_sec - mid.tv_sec) * 1000 + (double)(end.tv_usec - mid.tv_usec) /1000;
  fprintf(stderr,"	forward MKLDNN time1 = %.2f ms, time2 = %.2f\nms",duration1,duration2);
  double convert_time1 = (convert1.tv_sec - mid.tv_sec) * 1000 + (double)(convert1.tv_usec - mid.tv_usec) /1000;
  double exec_time = (convert2.tv_sec - convert1.tv_sec) * 1000 + (double)(convert2.tv_usec - convert1.tv_usec) /1000;
  double convert_time2 = (end.tv_sec - convert2.tv_sec) * 1000 + (double)(end.tv_usec - convert2.tv_usec) /1000;
  fprintf(stderr,"	forward MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2f\nms",convert_time1,exec_time,convert_time2);
#endif

#if MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	Convolution MKLDNN  forward time1 = %.2f ms \n",duration1);
#endif
}


void MKLNN_(SpatialConvolution_bwdData)(
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THMKLTensor *gradInput,
  THTensor *weight,
  THTensor *bias,
  THTensor *finput,
  THTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,int group)
{
  struct timeval start,mid1,mid2,mid3,convert1,convert2,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  dnnPrimitive_t m_conv_bwdData =NULL;

  dnnPrimitive_t cv_bwddata_input = NULL,cv_bwddata_filter = NULL,cv_bwddata_output = NULL;
  real * buffer_bwddata_input = NULL;
  real * buffer_bwddata_filter = NULL;
  real * buffer_bwddata_output=NULL;
  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  int outC = gradOutput->size[1];
  int outH = gradOutput->size[2];
  int outW = gradOutput->size[3];
  if(initOk == 0) {
    primitives->storage->data[CONV_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
    MKLNN_(SpatialConvolution_init_bwddata)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,group);
  }
  gettimeofday(&mid1,NULL);
  m_conv_bwdData = (dnnPrimitive_t) (primitives->storage->data[BWD_DATA_INDEX]);
  cv_bwddata_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_INPUT];
  cv_bwddata_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_FILTER];
  cv_bwddata_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDDATA_OUTPUT];
  buffer_bwddata_input 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_INPUT]);
  buffer_bwddata_filter 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_FILTER]);
  buffer_bwddata_output 	= (real *)(primitives->storage->data[BUFFER_BWDDATA_OUTPUT]);
  real * buffer_forward_filter 	= (real *)(primitives->storage->data[BUFFER_FORWARD_FILTER]);

  TH_MKL_(resizeAs)(gradInput, input);
  gettimeofday(&mid2,NULL);
#if LOG_ENABLE
  gettimeofday(&mid3,NULL);
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_bwdData: start. \n");
  fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);
  fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", gradOutput->size[0],gradOutput->size[1],gradOutput->size[2],gradOutput->size[3]);
  fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", weight->size[0],weight->size[1]);
#endif
  real * inPtr = TH_MKL_(data)(gradInput);
  real * filterPtr = THTensor_(data)(weight);
  real * outPtr = TH_MKL_(data)(gradOutput);
  real * resConv[dnnResourceNumber]= {0};
  resConv[dnnResourceDiffSrc] = inPtr;
  resConv[dnnResourceFilter] = filterPtr;
  resConv[dnnResourceDiffDst] = outPtr;

  void *convert_resources[dnnResourceNumber];
  if(sizeof(real) == sizeof(float)) {
    if(cv_bwddata_output) {
      resConv[dnnResourceDiffDst] = buffer_bwddata_output;
      convert_resources[dnnResourceFrom] = outPtr;
      convert_resources[dnnResourceTo]   = buffer_bwddata_output;
      CHECK_ERR( dnnExecute_F32(cv_bwddata_output,convert_resources), err );
    }
    if(cv_bwddata_filter) {
      real * buffer_forward_filter 	= (real *)(primitives->storage->data[BUFFER_FORWARD_FILTER]);
      resConv[dnnResourceFilter] = buffer_bwddata_filter;
      convert_resources[dnnResourceFrom] = buffer_forward_filter;
      convert_resources[dnnResourceTo]   = buffer_bwddata_filter;
      CHECK_ERR( dnnExecute_F32(cv_bwddata_filter, convert_resources), err );
    }
    if(cv_bwddata_input) {
      resConv[dnnResourceDiffSrc] = buffer_bwddata_input;
    }
    gettimeofday(&convert1,NULL);
    CHECK_ERR(dnnExecute_F32(m_conv_bwdData, (void**)resConv),err);
    gettimeofday(&convert2,NULL);

    if(cv_bwddata_input) {
      //TH_MKL_(setMKLdata)(buffer_bwddata_input);
    }
    gradInput->mkldnnLayout = (long long)primitives->storage->data[CONV_LAYOUT_BWDDATA_INPUT];
  }
  else if(sizeof(real) == sizeof(double)) {
    CHECK_ERR(dnnExecute_F64(m_conv_bwdData, (void**)resConv),err);
  }
  gettimeofday(&end,NULL);
#if LOG_ENABLE
  double time1 = (mid1.tv_sec - start.tv_sec) * 1000 + (double)(mid1.tv_usec - start.tv_usec) /1000;
  double time2 = (mid2.tv_sec - mid1.tv_sec) * 1000 + (double)(mid2.tv_usec - mid1.tv_usec) /1000;
  double time3 = (mid3.tv_sec - mid2.tv_sec) * 1000 + (double)(mid3.tv_usec - mid2.tv_usec) /1000;
  fprintf(stderr,"	bwdData MKLDNN mid1 = %.2f ms, mid2 = %.2f ms, mid3 = %.2f\n",time1,time2,time3) ;
  double duration1 = (mid3.tv_sec - start.tv_sec) * 1000 + (double)(mid3.tv_usec - start.tv_usec) /1000;
  double duration2 = (end.tv_sec - mid3.tv_sec) * 1000 + (double)(end.tv_usec - mid3.tv_usec) /1000;
  fprintf(stderr,"	bwdData MKLDNN time1 = %.2f ms, time2 = %.2f ms\n",duration1,duration2 );
  double convert_time1 = (convert1.tv_sec - mid3.tv_sec) * 1000 + (double)(convert1.tv_usec - mid3.tv_usec) /1000;
  double exec_time = (convert2.tv_sec - convert1.tv_sec) * 1000 + (double)(convert2.tv_usec - convert1.tv_usec) /1000;
  double convert_time2 = (end.tv_sec - convert2.tv_sec) * 1000 + (double)(end.tv_usec - convert2.tv_usec) /1000;
  fprintf(stderr,"        bwddata MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2fms \n",convert_time1,exec_time,convert_time2);
#endif

#if MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	Convolution MKLDNN  bwddata time1 = %.2f ms \n",duration1);
#endif
}

void MKLNN_(SpatialConvolution_bwdFilter)(
  THMKLTensor *input,
  THMKLTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *finput,
  THTensor *fgradInput,
  THLongTensor *primitives,
  int initOk,
  int kW,
  int kH,
  int dW,
  int dH,
  int padW,
  int padH,
  real scale,
  int group)
{
  struct timeval start,mid,convert1,convert2,end;
  gettimeofday(&start,NULL);
  dnnError_t err;
  dnnPrimitive_t m_conv_bwdFilter =NULL, m_conv_bwdBias = NULL;
  dnnPrimitive_t cv_bwdfilter_input = NULL,cv_bwdfilter_filter = NULL,cv_bwdfilter_output = NULL;
  real * buffer_bwdfilter_input = NULL;
  real * buffer_bwdfilter_filter = NULL;
  real * buffer_bwdfilter_output = NULL;

  int N = input->size[0];
  int inC = input->size[1];
  int inH = input->size[2];
  int inW = input->size[3];

  int outC = gradOutput->size[1];
  int outH = gradOutput->size[2];
  int outW = gradOutput->size[3];
  if(initOk == 0) {
    primitives->storage->data[CONV_LAYOUT_INPUT] = (long long)input->mkldnnLayout;
    primitives->storage->data[CONV_LAYOUT_OUTPUT] = (long long)gradOutput->mkldnnLayout;
    MKLNN_(SpatialConvolution_init_bwdfilter)(primitives,N,inC,inH,inW,kH,kW,dH,dW,padH,padW,outC,outH,outW,group);
  }

  m_conv_bwdFilter = (dnnPrimitive_t) (primitives->storage->data[BWD_FILTER_INDEX]);
  m_conv_bwdBias = (dnnPrimitive_t) (primitives->storage->data[BWD_BIAS_INDEX]);
  cv_bwdfilter_input 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_INPUT];
  cv_bwdfilter_filter 	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_FILTER];
  cv_bwdfilter_output  	= (dnnPrimitive_t)primitives->storage->data[CONVERT_BWDFILTER_OUTPUT];
  buffer_bwdfilter_input 	= (real *)(primitives->storage->data[BUFFER_BWDFILTER_INPUT]);
  buffer_bwdfilter_filter = (real *)(primitives->storage->data[BUFFER_BWDFILTER_FILTER]);
  buffer_bwdfilter_output = (real *)(primitives->storage->data[BUFFER_BWDFILTER_OUTPUT]);

#if LOG_ENABLE
  fprintf(stderr, "SpatialConvolutionMM_MKLDNN_bwdFilter: start. \n");
  //fprintf(stderr, "	input->nDimension = %d, finput->nDimension = %d, gradWeight->nDimension = %d\n", input->nDimension,finput->nDimension,gradWeight->nDimension);
  fprintf(stderr, "	input->size[0]=%d,input->size[1]=%d,input->size[2]=%d,input->size[3]=%d \n", input->size[0],input->size[1],input->size[2],input->size[3]);
  fprintf(stderr, "	output->size[0]=%d,output->size[1]=%d,output->size[2]=%d,output->size[3]=%d \n", gradOutput->size[0],gradOutput->size[1],gradOutput->size[2],gradOutput->size[3]);
  fprintf(stderr, "	weight->size[0]=%d,weight->size[1]=%d\n", gradWeight->size[0],gradWeight->size[1]);
#endif

  real * inPtr = TH_MKL_(data)(input);
  real * filterPtr = THTensor_(data)(gradWeight);
  real * outPtr = TH_MKL_(data)(gradOutput);
  real * biasPtr = THTensor_(data)(gradBias);

  real * resConv[dnnResourceNumber]= {0};
  resConv[dnnResourceSrc] = inPtr;
  resConv[dnnResourceDiffFilter] = filterPtr;
  resConv[dnnResourceDiffDst] = outPtr;
  resConv[dnnResourceDiffBias] = biasPtr;
  void *convert_resources[dnnResourceNumber];
  real * resBias[dnnResourceNumber]= {0};
  resBias[dnnResourceDiffDst] = outPtr;
  resBias[dnnResourceDiffBias] = biasPtr;

  if(sizeof(real) == sizeof(float)) {
    if(cv_bwdfilter_input) {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter input conversion\n");
#endif
      resConv[dnnResourceSrc] = buffer_bwdfilter_input;
      convert_resources[dnnResourceFrom] = inPtr;
      convert_resources[dnnResourceTo]   = buffer_bwdfilter_input;
      CHECK_ERR( dnnExecute_F32(cv_bwdfilter_input, convert_resources), err );
    }
    if(cv_bwdfilter_output) {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter output conversion\n");
#endif
      resConv[dnnResourceDiffDst] = buffer_bwdfilter_output;
      convert_resources[dnnResourceFrom] = outPtr;
      convert_resources[dnnResourceTo]   = buffer_bwdfilter_output;
      CHECK_ERR( dnnExecute_F32(cv_bwdfilter_output, convert_resources), err );
    }
    if(cv_bwdfilter_filter) {
      resConv[dnnResourceDiffFilter] = buffer_bwdfilter_filter;
    }
    gettimeofday(&convert1,NULL);
    CHECK_ERR(dnnExecute_F32(m_conv_bwdFilter, (void**)resConv),err);
    gettimeofday(&convert2,NULL);
    if(cv_bwdfilter_filter) {
#if CONVERSION_LOG
      fprintf(stderr,"MKLDNN Convolution bwdfilter filter conversion\n");
#endif
      convert_resources[dnnResourceFrom] = buffer_bwdfilter_filter;
      convert_resources[dnnResourceTo]   = filterPtr;
      CHECK_ERR( dnnExecute_F32(cv_bwdfilter_filter, convert_resources), err );
    }
    CHECK_ERR(dnnExecute_F32(m_conv_bwdBias, (void**)resBias),err);

  }
  else if(sizeof(real) == sizeof(double)) {
    CHECK_ERR(dnnExecute_F64(m_conv_bwdFilter, (void**)resConv),err);
  }
#if LOG_ENABLE
  gettimeofday(&end,NULL);
  double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	bwdFilter MKLDNN time = %.2f ms\n",duration );
  double convert_time1 = (convert1.tv_sec - start.tv_sec) * 1000 + (double)(convert1.tv_usec - start.tv_usec) /1000;
  double exec_time = (convert2.tv_sec - convert1.tv_sec) * 1000 + (double)(convert2.tv_usec - convert1.tv_usec) /1000;
  double convert_time2 = (end.tv_sec - convert2.tv_sec) * 1000 + (double)(end.tv_usec - convert2.tv_usec) /1000;
  fprintf(stderr,"        bwdfilter MKLDNN convert_time1 = %.2f ms, exec_time = %.2f, convert_time2=%.2f ms\n",convert_time1,exec_time,convert_time2);
#endif

#if MKL_TIME
  gettimeofday(&end,NULL);
  double duration1 = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
  fprintf(stderr,"	Convolution MKLDNN  bwdfilter time1 = %.2f ms \n",duration1);
#endif
}
#endif
