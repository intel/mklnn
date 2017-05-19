#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "luaT.h"
#include "THStorage.h"
#include "THTensor.h"
#include "src/MKLDNN.h"
#include "MKLTensor.h"



#define THTensor            TH_CONCAT_3(TH,Real,Tensor)          
#define torch_mkl_(NAME)    TH_CONCAT_4(torch_MKL, Real, Tensor_, NAME)             
#define TH_MKL_(NAME)       TH_CONCAT_4(THMKL, Real, Tensor, NAME)                                      
#define torch_mkl_tensor    TH_CONCAT_STRING_4(torch., MKL, Real, Tensor)

#define THMKLTensor         TH_CONCAT_3(THMKL, Real, Tensor)
#define MKLNN_(NAME)        TH_CONCAT_3(MKLNN_,Real, NAME)   


#include "src/SpatialConvolution.c"
#include "generateAllTypes.h"
#include "src/ThresholdMKLDNN.c"
#include "generateAllTypes.h"
#include "src/SpatialMaxPooling.c"
#include "generateAllTypes.h"
#include "src/SpatialAveragePooling.c"
#include "generateAllTypes.h"
#include "src/BatchNormalization.c"
#include "generateAllTypes.h"
#include "src/SpatialCrossMapLRN.c"
#include "generateAllTypes.h"
#include "src/Concat.c"
#include "generateAllTypes.h"


/*
#include "generateAllTypes.h"
#include "src/SpatialMaxPooling.c"
#include "generateAllTypes.h"
#include "src/BatchNormalization.c"
#include "generateAllTypes.h"
#include "src/LRN.c"
#include "generateAllTypes.h"

*/
