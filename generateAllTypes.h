#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define real float
#define accreal double
#define Real Float
#define BIT F32
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef BIT


#define real double
#define accreal double
#define Real Double
#define BIT F64
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef BIT
/*
#define real long
#define accreal long
#define Real Long
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef real
#undef accreal
#undef Real
*/
#undef TH_GENERIC_FILE

