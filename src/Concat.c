
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "src/Concat.c"
#else

static void MKLNN_(Concat_init_forward)(
          THLongTensor *inputarray,
          THMKLTensor *output,
          int  moduleNum,
          THLongTensor *primitives
	  )
{
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_init_forward start\n");
#endif
	dnnError_t err;
	dnnPrimitive_t m_concat_forward = NULL;
	THMKLTensor * input = NULL;
	long inputPtr = 0;
	dnnLayout_t *layouts = malloc(moduleNum * sizeof(dnnLayout_t));
	for(int i=0; i < moduleNum; i++)
	{
		inputPtr = inputarray->storage->data[i];
		input = (THMKLTensor *)inputPtr;
		if(input->mkldnnLayout == 0)
		{
#if CONVERSION_LOG
			fprintf(stderr, "Concat MKLDNN forward get input layout fail, i = %d \n", i);
#endif
			//create NCHW layout here
			int N = input->size[0];
			int inC = input->size[1];
			int inH = input->size[2];
			int inW = input->size[3];
			dnnLayout_t lt_user_input = NULL;
			size_t inputSize[dimension] = 	{inW,inH,inC,N};
			size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
			CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
			layouts[i] = lt_user_input;
		}
		else
		{
#if CONVERSION_LOG
			int N = input->size[0];
			int inC = input->size[1];
			int inH = input->size[2];
			int inW = input->size[3];
			fprintf(stderr, "Concat MKLDNN forward get input layout OK, N = %d, inC = %d, inH = %d, inW = %d \n",N,inC,inH,inW);
#endif
			layouts[i] = (dnnLayout_t)input->mkldnnLayout;
		}
	}
	CHECK_ERR(dnnConcatCreate_F32(&m_concat_forward, NULL, moduleNum, layouts), err);

	dnnLayout_t lt_concat_forward_output = NULL;
	CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&lt_concat_forward_output, m_concat_forward, dnnResourceDst), err );


	primitives->storage->data[CONCAT_FORWARD] = (long)m_concat_forward;
	primitives->storage->data[CONCAT_LAYOUT_FORWARD_OUTPUT] = (long)lt_concat_forward_output;



#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_init_forward end. \n");
#endif
}

static void MKLNN_(Concat_init_backward)(
          THLongTensor *gradarray,
          THMKLTensor *gradOutput,
          int  moduleNum,
          THLongTensor *primitives)
{

#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_init_backward start. gradarray = 0x%x, gradOutput = 0x%d, moduleNum = %d,  primitives = 0x%x\n", gradarray, gradOutput, moduleNum, primitives);
#endif
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;

	dnnPrimitive_t concat_split = NULL;
	dnnLayout_t layout = NULL;

	if(gradOutput->mkldnnLayout == 0)
	{
#if CONVERSION_LOG
		fprintf(stderr, "Concat MKLDNN backward get input layout fail\n");
#endif
		//create NCHW layout here
		int N = gradOutput->size[0];
		int inC = gradOutput->size[1];
		int inH = gradOutput->size[2];
		int inW = gradOutput->size[3];
		dnnLayout_t lt_user_input = NULL;
		size_t inputSize[dimension] = 	{inW,inH,inC,N};
		size_t inputStrides[dimension] = { 1, inW, inH * inW, inC * inH * inW };
		CHECK_ERR( dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides) , err );
		layout = lt_user_input;
	}
	else
	{
#if CONVERSION_LOG
		int N = gradOutput->size[0];
		int inC = gradOutput->size[1];
		int inH = gradOutput->size[2];
		int inW = gradOutput->size[3];
		fprintf(stderr, "Concat MKLDNN backward get input layout OK, N = %d, inC = %d, inH = %d, inW = %d \n",N,inC,inH,inW);
#endif
		layout = (dnnLayout_t)gradOutput->mkldnnLayout;
	}

	THMKLTensor * grad = NULL;
	long gradPtr = 0;
	size_t split_channels[10]; 
	for(int i=0; i < moduleNum; i++)
	{
		gradPtr = gradarray->storage->data[i];
		grad = (THMKLTensor *)gradPtr;
		split_channels[i] = grad->size[1];
	}
	CHECK_ERR(dnnSplitCreate_F32(&concat_split, NULL, moduleNum, layout, split_channels), err);



	primitives->storage->data[CONCAT_BACKWARD] = (long)concat_split;
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_init_backward end. \n");
#endif

}

void MKLNN_(Concat_setupLongTensor)(
          THLongTensor * array,
          THMKLTensor *input,
          int  index)
{
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_setupLongTensor start, array = 0x%x, input = 0x%x, index = %d\n", array, input, index);
#endif
	array->storage->data[index-1] = (long )input;

}


/**
input: the long tensor , the tensor size = moduleNum, the data is the THTensor ptr which point to the real data
*/
void MKLNN_(Concat_updateOutput)(
          THLongTensor *inputarray,
          THMKLTensor *output,
          int  moduleNum,
          THLongTensor *primitives,
          int initOk)
{
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_updateOutput start. inputarray = 0x%x, output = 0x%d, moduleNum = %d,  primitives = 0x%x, initOk = %d \n", inputarray, output, moduleNum, primitives, initOk);
#endif
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;

	if(initOk == 0)
	{
		MKLNN_(Concat_init_forward)(inputarray, output, moduleNum, primitives);
	}

	dnnPrimitive_t m_concat_forward = (dnnPrimitive_t) (primitives->storage->data[CONCAT_FORWARD]);
	THMKLTensor * input = NULL;
	long inputPtr = 0;
	dnnLayout_t *layouts = NULL;
	void *concat_res[dnnResourceNumber];
	for(int i=0; i < moduleNum; i++)
	{
		inputPtr = inputarray->storage->data[i];
		input = (THMKLTensor *)inputPtr;
		concat_res[dnnResourceMultipleSrc + i] = TH_MKL_(data)(input);
	}
	concat_res[dnnResourceDst] = TH_MKL_(data)(output);
	CHECK_ERR( dnnExecute_F32(m_concat_forward, (void*)concat_res), err );
	
	output->mkldnnLayout = (long)primitives->storage->data[CONCAT_LAYOUT_FORWARD_OUTPUT];	
#if LOG_ENABLE || MKL_TIME
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Concat MKLDNN time forward = %.2f ms\n",duration );
#endif
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_updateOutput end. \n");
#endif

}

void MKLNN_(Concat_backward_split)(
          THLongTensor *gradarray,
          THMKLTensor *gradOutput,
          int  moduleNum,
          THLongTensor *primitives,
          int initOk)
{
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_backward_split start. gradarray = 0x%x, gradOutput = 0x%d, moduleNum = %d,  primitives = 0x%x, initOk = %d \n", gradarray, gradOutput, moduleNum, primitives, initOk);
#endif
	struct timeval start,end;
	gettimeofday(&start,NULL);
	dnnError_t err;
	dnnLayout_t layout = NULL;

	if(initOk == 0)
	{
		MKLNN_(Concat_init_backward)(gradarray, gradOutput, moduleNum, primitives);
	}
	dnnPrimitive_t concat_split = (dnnPrimitive_t)primitives->storage->data[CONCAT_BACKWARD];
	
	THMKLTensor * grad = NULL;
	long gradPtr = 0;
	void *split_res[dnnResourceNumber];
	split_res[dnnResourceSrc] = TH_MKL_(data)(gradOutput);
	for(int i=0; i < moduleNum; i++)
	{
		gradPtr = gradarray->storage->data[i];
		grad = (THMKLTensor *)gradPtr;
		split_res[dnnResourceMultipleDst + i] = TH_MKL_(data)(grad);
		//create layout from primitive, save the layout to tensor grad
		CHECK_ERR( dnnLayoutCreateFromPrimitive_F32(&layout, concat_split, dnnResourceMultipleDst + i) , err );
		grad->mkldnnLayout = (long)layout;
	}
	CHECK_ERR(dnnExecute_F32(concat_split, split_res), err);

#if LOG_ENABLE || MKL_TIME
	gettimeofday(&end,NULL);
	double duration = (end.tv_sec - start.tv_sec) * 1000 + (double)(end.tv_usec - start.tv_usec) /1000;
	fprintf(stderr,"	Concat MKLDNN time backward = %.2f ms\n",duration );
#endif
#if LOG_ENABLE
	fprintf(stderr, "Concat_MKLDNN_backward_split end. \n");
#endif

}



#endif
