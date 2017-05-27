cat mkl_types.h mkl_vsl_defines.h mkl_vsl_types.h mkl_vsl_functions.h mkl_dnn_types.h mkl_dnn_types.h mkl_dnn.h &> tmp.h
sed '/include/d' tmp.h > mkl_cat.h
