#include <stdlib.h>
#include <string.h>

#include "luaT.h"
#include "THStorage.h"
#include "THTensor.h"
#include "src/MKLDNN.h"




#define THTensor            TH_CONCAT_3(TH,Real,Tensor)          
#define torch_mkl_(NAME)    TH_CONCAT_4(torch_MKL, Real, Tensor_, NAME)             
#define TH_MKL_(NAME)       TH_CONCAT_4(THMkl, Real, Tensor, NAME)                                      
#define torch_mkl_tensor    TH_CONCAT_STRING_4(torch., MKL, Real, Tensor)

#define THMklTensor         TH_CONCAT_3(THMkl, Real, Tensor)
#define MKLNN_(NAME)        TH_CONCAT_3(MKLNN_,Real, NAME)   


#include "src/SpatialConvolution.c"
#include "src/ThresholdMKLDNN.c"
#include "src/SpatialMaxPooling.c"
#include "src/BatchNormalization.c"
#include "src/LRN.c"
#include "generateAllTypes.h"


//#include "src/ThresholdMKLDNN.c"
//#include "generateAllTypes.h"

