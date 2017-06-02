/*******************************************************************************
* Copyright (c) 1999-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/*
! Content:
!      Intel(R) Math Kernel Library (Intel(R) MKL) types definition
!****************************************************************************/

#ifndef _MKL_TYPES_H_
#define _MKL_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* MKL Complex type for single precision */
#ifndef MKL_Complex8
typedef
struct _MKL_Complex8 {
    float real;
    float imag;
} MKL_Complex8;
#endif

/* MKL Complex type for double precision */
#ifndef MKL_Complex16
typedef
struct _MKL_Complex16 {
    double real;
    double imag;
} MKL_Complex16;
#endif

/* MKL Version type */
typedef
struct {
    int    MajorVersion;
    int    MinorVersion;
    int    UpdateVersion;
    char * ProductStatus;
    char * Build;
    char * Processor;
    char * Platform;
} MKLVersion;

/* MKL integer types for LP64 and ILP64 */
#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
    #define MKL_INT64 __int64
    #define MKL_UINT64 unsigned __int64
#else
    #define MKL_INT64 long long int
    #define MKL_UINT64 unsigned long long int
#endif

#ifdef MKL_ILP64

/* MKL ILP64 integer types */
#ifndef MKL_INT
    #define MKL_INT MKL_INT64
#endif
#ifndef MKL_UINT
    #define MKL_UINT MKL_UINT64
#endif
#define MKL_LONG MKL_INT64

#else

/* MKL LP64 integer types */
#ifndef MKL_INT
    #define MKL_INT int
#endif
#ifndef MKL_UINT
    #define MKL_UINT unsigned int
#endif
#define MKL_LONG long int

#endif

/* MKL threading stuff. MKL Domain names */
#define MKL_DOMAIN_ALL      0
#define MKL_DOMAIN_BLAS     1
#define MKL_DOMAIN_FFT      2
#define MKL_DOMAIN_VML      3
#define MKL_DOMAIN_PARDISO  4

/* MKL CBWR stuff */

/* options */
#define MKL_CBWR_BRANCH 1
#define MKL_CBWR_ALL   ~0

/* common settings */
#define MKL_CBWR_UNSET_ALL 0
#define MKL_CBWR_OFF       0

/* branch specific values */
#define MKL_CBWR_BRANCH_OFF     1
#define MKL_CBWR_AUTO           2
#define MKL_CBWR_COMPATIBLE     3
#define MKL_CBWR_SSE2           4
#define MKL_CBWR_SSSE3          6
#define MKL_CBWR_SSE4_1         7
#define MKL_CBWR_SSE4_2         8
#define MKL_CBWR_AVX            9
#define MKL_CBWR_AVX2          10
#define MKL_CBWR_AVX512_MIC    11
#define MKL_CBWR_AVX512        12

/* error codes */
#define MKL_CBWR_SUCCESS                   0
#define MKL_CBWR_ERR_INVALID_SETTINGS     -1
#define MKL_CBWR_ERR_INVALID_INPUT        -2
#define MKL_CBWR_ERR_UNSUPPORTED_BRANCH   -3
#define MKL_CBWR_ERR_UNKNOWN_BRANCH       -4
#define MKL_CBWR_ERR_MODE_CHANGE_FAILURE  -8

/* Obsolete */
#define MKL_CBWR_SSE3           5

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_TYPES_H_ */
/* file: mkl_vsl_defines.h */
/*******************************************************************************
* Copyright (c) 2006-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/*
//++
//  User-level macro definitions
//--
*/

#ifndef __MKL_VSL_DEFINES_H__
#define __MKL_VSL_DEFINES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/*
// "No error" status
*/
#define VSL_STATUS_OK                      0
#define VSL_ERROR_OK                       0

/*
// Common errors (-1..-999)
*/
#define VSL_ERROR_FEATURE_NOT_IMPLEMENTED  -1
#define VSL_ERROR_UNKNOWN                  -2
#define VSL_ERROR_BADARGS                  -3
#define VSL_ERROR_MEM_FAILURE              -4
#define VSL_ERROR_NULL_PTR                 -5
#define VSL_ERROR_CPU_NOT_SUPPORTED        -6


/*
// RNG errors (-1000..-1999)
*/
/* brng errors */
#define VSL_RNG_ERROR_INVALID_BRNG_INDEX        -1000
#define VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED      -1002
#define VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED     -1003
#define VSL_RNG_ERROR_BRNGS_INCOMPATIBLE        -1005
#define VSL_RNG_ERROR_BAD_STREAM                -1006
#define VSL_RNG_ERROR_BRNG_TABLE_FULL           -1007
#define VSL_RNG_ERROR_BAD_STREAM_STATE_SIZE     -1008
#define VSL_RNG_ERROR_BAD_WORD_SIZE             -1009
#define VSL_RNG_ERROR_BAD_NSEEDS                -1010
#define VSL_RNG_ERROR_BAD_NBITS                 -1011
#define VSL_RNG_ERROR_QRNG_PERIOD_ELAPSED       -1012
#define VSL_RNG_ERROR_LEAPFROG_NSTREAMS_TOO_BIG -1013
#define VSL_RNG_ERROR_BRNG_NOT_SUPPORTED        -1014

/* abstract stream related errors */
#define VSL_RNG_ERROR_BAD_UPDATE                -1120
#define VSL_RNG_ERROR_NO_NUMBERS                -1121
#define VSL_RNG_ERROR_INVALID_ABSTRACT_STREAM   -1122

/* non determenistic stream related errors */
#define VSL_RNG_ERROR_NONDETERM_NOT_SUPPORTED     -1130
#define VSL_RNG_ERROR_NONDETERM_NRETRIES_EXCEEDED -1131

/* ARS5 stream related errors */
#define VSL_RNG_ERROR_ARS5_NOT_SUPPORTED        -1140

/* read/write stream to file errors */
#define VSL_RNG_ERROR_FILE_CLOSE                -1100
#define VSL_RNG_ERROR_FILE_OPEN                 -1101
#define VSL_RNG_ERROR_FILE_WRITE                -1102
#define VSL_RNG_ERROR_FILE_READ                 -1103

#define VSL_RNG_ERROR_BAD_FILE_FORMAT           -1110
#define VSL_RNG_ERROR_UNSUPPORTED_FILE_VER      -1111

#define VSL_RNG_ERROR_BAD_MEM_FORMAT            -1200

/* Convolution/correlation errors */
#define VSL_CC_ERROR_NOT_IMPLEMENTED        (-2000)
#define VSL_CC_ERROR_ALLOCATION_FAILURE     (-2001)
#define VSL_CC_ERROR_BAD_DESCRIPTOR         (-2200)
#define VSL_CC_ERROR_SERVICE_FAILURE        (-2210)
#define VSL_CC_ERROR_EDIT_FAILURE           (-2211)
#define VSL_CC_ERROR_EDIT_PROHIBITED        (-2212)
#define VSL_CC_ERROR_COMMIT_FAILURE         (-2220)
#define VSL_CC_ERROR_COPY_FAILURE           (-2230)
#define VSL_CC_ERROR_DELETE_FAILURE         (-2240)
#define VSL_CC_ERROR_BAD_ARGUMENT           (-2300)
#define VSL_CC_ERROR_DIMS                   (-2301)
#define VSL_CC_ERROR_START                  (-2302)
#define VSL_CC_ERROR_DECIMATION             (-2303)
#define VSL_CC_ERROR_XSHAPE                 (-2311)
#define VSL_CC_ERROR_YSHAPE                 (-2312)
#define VSL_CC_ERROR_ZSHAPE                 (-2313)
#define VSL_CC_ERROR_XSTRIDE                (-2321)
#define VSL_CC_ERROR_YSTRIDE                (-2322)
#define VSL_CC_ERROR_ZSTRIDE                (-2323)
#define VSL_CC_ERROR_X                      (-2331)
#define VSL_CC_ERROR_Y                      (-2332)
#define VSL_CC_ERROR_Z                      (-2333)
#define VSL_CC_ERROR_JOB                    (-2100)
#define VSL_CC_ERROR_KIND                   (-2110)
#define VSL_CC_ERROR_MODE                   (-2120)
#define VSL_CC_ERROR_TYPE                   (-2130)
#define VSL_CC_ERROR_PRECISION              (-2400)
#define VSL_CC_ERROR_EXTERNAL_PRECISION     (-2141)
#define VSL_CC_ERROR_INTERNAL_PRECISION     (-2142)
#define VSL_CC_ERROR_METHOD                 (-2400)
#define VSL_CC_ERROR_OTHER                  (-2800)

/*
//++
// SUMMARY STATTISTICS ERROR/WARNING CODES
//--
*/

/*
// Warnings
*/
#define VSL_SS_NOT_FULL_RANK_MATRIX                   4028
#define VSL_SS_SEMIDEFINITE_COR                       4029
/*
// Errors (-4000..-4999)
*/
#define VSL_SS_ERROR_ALLOCATION_FAILURE              -4000
#define VSL_SS_ERROR_BAD_DIMEN                       -4001
#define VSL_SS_ERROR_BAD_OBSERV_N                    -4002
#define VSL_SS_ERROR_STORAGE_NOT_SUPPORTED           -4003
#define VSL_SS_ERROR_BAD_INDC_ADDR                   -4004
#define VSL_SS_ERROR_BAD_WEIGHTS                     -4005
#define VSL_SS_ERROR_BAD_MEAN_ADDR                   -4006
#define VSL_SS_ERROR_BAD_2R_MOM_ADDR                 -4007
#define VSL_SS_ERROR_BAD_3R_MOM_ADDR                 -4008
#define VSL_SS_ERROR_BAD_4R_MOM_ADDR                 -4009
#define VSL_SS_ERROR_BAD_2C_MOM_ADDR                 -4010
#define VSL_SS_ERROR_BAD_3C_MOM_ADDR                 -4011
#define VSL_SS_ERROR_BAD_4C_MOM_ADDR                 -4012
#define VSL_SS_ERROR_BAD_KURTOSIS_ADDR               -4013
#define VSL_SS_ERROR_BAD_SKEWNESS_ADDR               -4014
#define VSL_SS_ERROR_BAD_MIN_ADDR                    -4015
#define VSL_SS_ERROR_BAD_MAX_ADDR                    -4016
#define VSL_SS_ERROR_BAD_VARIATION_ADDR              -4017
#define VSL_SS_ERROR_BAD_COV_ADDR                    -4018
#define VSL_SS_ERROR_BAD_COR_ADDR                    -4019
#define VSL_SS_ERROR_BAD_ACCUM_WEIGHT_ADDR           -4020
#define VSL_SS_ERROR_BAD_QUANT_ORDER_ADDR            -4021
#define VSL_SS_ERROR_BAD_QUANT_ORDER                 -4022
#define VSL_SS_ERROR_BAD_QUANT_ADDR                  -4023
#define VSL_SS_ERROR_BAD_ORDER_STATS_ADDR            -4024
#define VSL_SS_ERROR_MOMORDER_NOT_SUPPORTED          -4025
#define VSL_SS_ERROR_ALL_OBSERVS_OUTLIERS            -4026
#define VSL_SS_ERROR_BAD_ROBUST_COV_ADDR             -4027
#define VSL_SS_ERROR_BAD_ROBUST_MEAN_ADDR            -4028
#define VSL_SS_ERROR_METHOD_NOT_SUPPORTED            -4029
#define VSL_SS_ERROR_BAD_GROUP_INDC_ADDR             -4030
#define VSL_SS_ERROR_NULL_TASK_DESCRIPTOR            -4031
#define VSL_SS_ERROR_BAD_OBSERV_ADDR                 -4032
#define VSL_SS_ERROR_SINGULAR_COV                    -4033
#define VSL_SS_ERROR_BAD_POOLED_COV_ADDR             -4034
#define VSL_SS_ERROR_BAD_POOLED_MEAN_ADDR            -4035
#define VSL_SS_ERROR_BAD_GROUP_COV_ADDR              -4036
#define VSL_SS_ERROR_BAD_GROUP_MEAN_ADDR             -4037
#define VSL_SS_ERROR_BAD_GROUP_INDC                  -4038
#define VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_ADDR        -4039
#define VSL_SS_ERROR_BAD_OUTLIERS_PARAMS_N_ADDR      -4040
#define VSL_SS_ERROR_BAD_OUTLIERS_WEIGHTS_ADDR       -4041
#define VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_ADDR      -4042
#define VSL_SS_ERROR_BAD_ROBUST_COV_PARAMS_N_ADDR    -4043
#define VSL_SS_ERROR_BAD_STORAGE_ADDR                -4044
#define VSL_SS_ERROR_BAD_PARTIAL_COV_IDX_ADDR        -4045
#define VSL_SS_ERROR_BAD_PARTIAL_COV_ADDR            -4046
#define VSL_SS_ERROR_BAD_PARTIAL_COR_ADDR            -4047
#define VSL_SS_ERROR_BAD_MI_PARAMS_ADDR              -4048
#define VSL_SS_ERROR_BAD_MI_PARAMS_N_ADDR            -4049
#define VSL_SS_ERROR_BAD_MI_BAD_PARAMS_N             -4050
#define VSL_SS_ERROR_BAD_MI_PARAMS                   -4051
#define VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_N_ADDR    -4052
#define VSL_SS_ERROR_BAD_MI_INIT_ESTIMATES_ADDR      -4053
#define VSL_SS_ERROR_BAD_MI_SIMUL_VALS_ADDR          -4054
#define VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N_ADDR        -4055
#define VSL_SS_ERROR_BAD_MI_ESTIMATES_N_ADDR         -4056
#define VSL_SS_ERROR_BAD_MI_ESTIMATES_ADDR           -4057
#define VSL_SS_ERROR_BAD_MI_SIMUL_VALS_N             -4058
#define VSL_SS_ERROR_BAD_MI_ESTIMATES_N              -4059
#define VSL_SS_ERROR_BAD_MI_OUTPUT_PARAMS            -4060
#define VSL_SS_ERROR_BAD_MI_PRIOR_N_ADDR             -4061
#define VSL_SS_ERROR_BAD_MI_PRIOR_ADDR               -4062
#define VSL_SS_ERROR_BAD_MI_MISSING_VALS_N           -4063
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N_ADDR  -4064
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_ADDR    -4065
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS_N       -4066
#define VSL_SS_ERROR_BAD_STREAM_QUANT_PARAMS         -4067
#define VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER_ADDR     -4068
#define VSL_SS_ERROR_BAD_STREAM_QUANT_ORDER          -4069
#define VSL_SS_ERROR_BAD_STREAM_QUANT_ADDR           -4070
#define VSL_SS_ERROR_BAD_PARAMTR_COR_ADDR            -4071
#define VSL_SS_ERROR_BAD_COR                         -4072
#define VSL_SS_ERROR_BAD_PARTIAL_COV_IDX             -4073
#define VSL_SS_ERROR_BAD_SUM_ADDR                    -4074
#define VSL_SS_ERROR_BAD_2R_SUM_ADDR                 -4075
#define VSL_SS_ERROR_BAD_3R_SUM_ADDR                 -4076
#define VSL_SS_ERROR_BAD_4R_SUM_ADDR                 -4077
#define VSL_SS_ERROR_BAD_2C_SUM_ADDR                 -4078
#define VSL_SS_ERROR_BAD_3C_SUM_ADDR                 -4079
#define VSL_SS_ERROR_BAD_4C_SUM_ADDR                 -4080
#define VSL_SS_ERROR_BAD_CP_ADDR                     -4081
#define VSL_SS_ERROR_BAD_MDAD_ADDR                   -4082
#define VSL_SS_ERROR_BAD_MNAD_ADDR                   -4083
#define VSL_SS_ERROR_BAD_SORTED_OBSERV_ADDR          -4084
#define VSL_SS_ERROR_INDICES_NOT_SUPPORTED           -4085


/*
// Internal errors caused by internal routines of the functions
*/
#define VSL_SS_ERROR_ROBCOV_INTERN_C1                -5000
#define VSL_SS_ERROR_PARTIALCOV_INTERN_C1            -5010
#define VSL_SS_ERROR_PARTIALCOV_INTERN_C2            -5011
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C1           -5021
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C2           -5022
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C3           -5023
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C4           -5024
#define VSL_SS_ERROR_MISSINGVALS_INTERN_C5           -5025
#define VSL_SS_ERROR_PARAMTRCOR_INTERN_C1            -5030
#define VSL_SS_ERROR_COVRANK_INTERNAL_ERROR_C1       -5040
#define VSL_SS_ERROR_INVCOV_INTERNAL_ERROR_C1        -5041
#define VSL_SS_ERROR_INVCOV_INTERNAL_ERROR_C2        -5042


/*
// CONV/CORR RELATED MACRO DEFINITIONS
*/
#define VSL_CONV_MODE_AUTO        0
#define VSL_CORR_MODE_AUTO        0
#define VSL_CONV_MODE_DIRECT      1
#define VSL_CORR_MODE_DIRECT      1
#define VSL_CONV_MODE_FFT         2
#define VSL_CORR_MODE_FFT         2
#define VSL_CONV_PRECISION_SINGLE 1
#define VSL_CORR_PRECISION_SINGLE 1
#define VSL_CONV_PRECISION_DOUBLE 2
#define VSL_CORR_PRECISION_DOUBLE 2

/*
//++
//  BASIC RANDOM NUMBER GENERATOR (BRNG) RELATED MACRO DEFINITIONS
//--
*/

/*
//  MAX NUMBER OF BRNGS CAN BE REGISTERED IN VSL
//  No more than VSL_MAX_REG_BRNGS basic generators can be registered in VSL
//  (including predefined basic generators).
//
//  Change this number to increase/decrease number of BRNGs can be registered.
*/
#define VSL_MAX_REG_BRNGS           512

/*
//  PREDEFINED BRNG NAMES
*/
#define VSL_BRNG_SHIFT      20
#define VSL_BRNG_INC        (1<<VSL_BRNG_SHIFT)

#define VSL_BRNG_MCG31          (VSL_BRNG_INC)
#define VSL_BRNG_R250           (VSL_BRNG_MCG31    +VSL_BRNG_INC)
#define VSL_BRNG_MRG32K3A       (VSL_BRNG_R250     +VSL_BRNG_INC)
#define VSL_BRNG_MCG59          (VSL_BRNG_MRG32K3A +VSL_BRNG_INC)
#define VSL_BRNG_WH             (VSL_BRNG_MCG59    +VSL_BRNG_INC)
#define VSL_BRNG_SOBOL          (VSL_BRNG_WH       +VSL_BRNG_INC)
#define VSL_BRNG_NIEDERR        (VSL_BRNG_SOBOL    +VSL_BRNG_INC)
#define VSL_BRNG_MT19937        (VSL_BRNG_NIEDERR  +VSL_BRNG_INC)
#define VSL_BRNG_MT2203         (VSL_BRNG_MT19937  +VSL_BRNG_INC)
#define VSL_BRNG_IABSTRACT      (VSL_BRNG_MT2203   +VSL_BRNG_INC)
#define VSL_BRNG_DABSTRACT      (VSL_BRNG_IABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_SABSTRACT      (VSL_BRNG_DABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_SFMT19937      (VSL_BRNG_SABSTRACT+VSL_BRNG_INC)
#define VSL_BRNG_NONDETERM      (VSL_BRNG_SFMT19937+VSL_BRNG_INC)
#define VSL_BRNG_ARS5           (VSL_BRNG_NONDETERM+VSL_BRNG_INC)
#define VSL_BRNG_PHILOX4X32X10  (VSL_BRNG_ARS5     +VSL_BRNG_INC)


/*
// PREDEFINED PARAMETERS FOR NON-DETERMNINISTIC RANDOM NUMBER
// GENERATOR
// The library provides an abstraction to the source of non-deterministic
// random numbers supported in HW. Current version of the library provides
// interface to RDRAND-based only, available in latest Intel CPU.
*/
#define VSL_BRNG_RDRAND  0x0
#define VSL_BRNG_NONDETERM_NRETRIES 10

/*
//  LEAPFROG METHOD FOR GRAY-CODE BASED QUASI-RANDOM NUMBER BASIC GENERATORS
//  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random number
//  basic generators. In contrast to pseudorandom number basic generators,
//  quasi-random ones take the dimension as initialization parameter.
//
//  Suppose that quasi-random number generator (QRNG) dimension is S. QRNG
//  sequence is a sequence of S-dimensional vectors:
//
//     x0=(x0[0],x0[1],...,x0[S-1]),x1=(x1[0],x1[1],...,x1[S-1]),...
//
//  VSL treats the output of any basic generator as 1-dimensional, however:
//
//     x0[0],x0[1],...,x0[S-1],x1[0],x1[1],...,x1[S-1],...
//
//  Because of nature of VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR QRNGs,
//  the only S-stride Leapfrog method is supported for them. In other words,
//  user can generate subsequences, which consist of fixed elements of
//  vectors x0,x1,... For example, if 0 element is fixed, the following
//  subsequence is generated:
//
//     x0[1],x1[1],x2[1],...
//
//  To use the s-stride Leapfrog method with given QRNG, user should call
//  vslLeapfrogStream function with parameter k equal to element to be fixed
//  (0<=k<S) and parameter nstreams equal to VSL_QRNG_LEAPFROG_COMPONENTS.
*/
#define VSL_QRNG_LEAPFROG_COMPONENTS    0x7fffffff

/*
//  USER-DEFINED PARAMETERS FOR QUASI-RANDOM NUMBER BASIC GENERATORS
//  VSL_BRNG_SOBOL and VSL_BRNG_NIEDERR are Gray-code based quasi-random
//  number basic generators. Default parameters of the generators
//  support generation of quasi-random number vectors of dimensions
//  S<=40 for SOBOL and S<=318 for NIEDERRITER. The library provides
//  opportunity to register user-defined initial values for the
//  generators and generate quasi-random vectors of desirable dimension.
//  There is also opportunity to register user-defined parameters for
//  default dimensions and obtain another sequence of quasi-random vectors.
//  Service function vslNewStreamEx is used to pass the parameters to
//  the library. Data are packed into array params, parameter of the routine.
//  First element of the array is used for dimension S, second element
//  contains indicator, VSL_USER_QRNG_INITIAL_VALUES, of user-defined
//  parameters for quasi-random number generators.
//  Macros VSL_USER_PRIMITIVE_POLYMS and VSL_USER_INIT_DIRECTION_NUMBERS
//  are used to describe which data are passed to SOBOL QRNG and
//  VSL_USER_IRRED_POLYMS - which data are passed to NIEDERRITER QRNG.
//  For example, to demonstrate that both primitive polynomials and initial
//  direction numbers are passed in SOBOL one should set third element of the
//  array params to  VSL_USER_PRIMITIVE_POLYMS | VSL_USER_DIRECTION_NUMBERS.
//  Macro VSL_QRNG_OVERRIDE_1ST_DIM_INIT is used to override default
//  initialization for the first dimension. Macro VSL_USER_DIRECTION_NUMBERS
//  is used when direction numbers calculated on the user side are passed
//  into the generators. More detailed description of interface for
//  registration of user-defined QRNG initial parameters can be found
//  in VslNotes.pdf.
*/
#define VSL_USER_QRNG_INITIAL_VALUES     0x1
#define VSL_USER_PRIMITIVE_POLYMS        0x1
#define VSL_USER_INIT_DIRECTION_NUMBERS  0x2
#define VSL_USER_IRRED_POLYMS            0x1
#define VSL_USER_DIRECTION_NUMBERS       0x4
#define VSL_QRNG_OVERRIDE_1ST_DIM_INIT   0x8


/*
//  INITIALIZATION METHODS FOR USER-DESIGNED BASIC RANDOM NUMBER GENERATORS.
//  Each BRNG must support at least VSL_INIT_METHOD_STANDARD initialization
//  method. In addition, VSL_INIT_METHOD_LEAPFROG and VSL_INIT_METHOD_SKIPAHEAD
//  initialization methods can be supported.
//
//  If VSL_INIT_METHOD_LEAPFROG is not supported then initialization routine
//  must return VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED error code.
//
//  If VSL_INIT_METHOD_SKIPAHEAD is not supported then initialization routine
//  must return VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED error code.
//
//  If there is no error during initialization, the initialization routine must
//  return VSL_ERROR_OK code.
*/
#define VSL_INIT_METHOD_STANDARD  0
#define VSL_INIT_METHOD_LEAPFROG  1
#define VSL_INIT_METHOD_SKIPAHEAD 2


/*
//++
//  ACCURACY FLAG FOR DISTRIBUTION GENERATORS
//  This flag defines mode of random number generation.
//  If accuracy mode is set distribution generators will produce
//  numbers lying exactly within definitional domain for all values
//  of distribution parameters. In this case slight performance
//  degradation is expected. By default accuracy mode is switched off
//  admitting random numbers to be out of the definitional domain for
//  specific values of distribution parameters.
//  This macro is used to form names for accuracy versions of
//  distribution number generators
//--
*/
#define VSL_RNG_METHOD_ACCURACY_FLAG (1<<30)

/*
//++
//  TRANSFORMATION METHOD NAMES FOR DISTRIBUTION RANDOM NUMBER GENERATORS
//  VSL interface allows more than one generation method in a distribution
//  transformation subroutine. Following macro definitions are used to
//  specify generation method for given distribution generator.
//
//  Method name macro is constructed as
//
//     VSL_RNG_METHOD_<Distribution>_<Method>
//
//  where
//
//     <Distribution> - probability distribution
//     <Method> - method name
//
//  VSL_RNG_METHOD_<Distribution>_<Method> should be used with
//  vsl<precision>Rng<Distribution> function only, where
//
//     <precision> - s (single) or d (double)
//     <Distribution> - probability distribution
//--
*/

/*
// Uniform
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORM_STD 0 /* vsl{s,d,i}RngUniform */

#define VSL_RNG_METHOD_UNIFORM_STD_ACCURATE \
  VSL_RNG_METHOD_UNIFORM_STD | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngUniform */

/*
// Uniform Bits
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORMBITS_STD 0 /* vsliRngUniformBits */

/*
// Uniform Bits 32
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORMBITS32_STD 0 /* vsliRngUniformBits32 */

/*
// Uniform Bits 64
//
// <Method>   <Short Description>
// STD        standard method. Currently there is only one method for this
//            distribution generator
*/
#define VSL_RNG_METHOD_UNIFORMBITS64_STD 0 /* vsliRngUniformBits64 */

/*
// Gaussian
//
// <Method>   <Short Description>
// BOXMULLER  generates normally distributed random number x thru the pair of
//            uniformly distributed numbers u1 and u2 according to the formula:
//
//               x=sqrt(-ln(u1))*sin(2*Pi*u2)
//
// BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
//            thru the pair of uniformly dustributed numbers u1 and u2
//            according to the formula
//
//               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
//               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
//
//            NOTE: implementation correctly works with odd vector lengths
//
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GAUSSIAN_BOXMULLER   0 /* vsl{d,s}RngGaussian */
#define VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2  1 /* vsl{d,s}RngGaussian */
#define VSL_RNG_METHOD_GAUSSIAN_ICDF        2 /* vsl{d,s}RngGaussian */

/*
// GaussianMV - multivariate (correlated) normal
// Multivariate (correlated) normal random number generator is based on
// uncorrelated Gaussian random number generator (see vslsRngGaussian and
// vsldRngGaussian functions):
//
// <Method>   <Short Description>
// BOXMULLER  generates normally distributed random number x thru the pair of
//            uniformly distributed numbers u1 and u2 according to the formula:
//
//               x=sqrt(-ln(u1))*sin(2*Pi*u2)
//
// BOXMULLER2 generates pair of normally distributed random numbers x1 and x2
//            thru the pair of uniformly dustributed numbers u1 and u2
//            according to the formula
//
//               x1=sqrt(-ln(u1))*sin(2*Pi*u2)
//               x2=sqrt(-ln(u1))*cos(2*Pi*u2)
//
//            NOTE: implementation correctly works with odd vector lengths
//
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER   0 /* vsl{d,s}RngGaussianMV */
#define VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2  1 /* vsl{d,s}RngGaussianMV */
#define VSL_RNG_METHOD_GAUSSIANMV_ICDF        2 /* vsl{d,s}RngGaussianMV */

/*
// Exponential
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_EXPONENTIAL_ICDF 0 /* vsl{d,s}RngExponential */

#define VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE \
   VSL_RNG_METHOD_EXPONENTIAL_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngExponential */

/*
// Laplace
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
//
// ICDF - inverse cumulative distribution function method:
//
//           x=+/-ln(u) with probability 1/2,
//
//        where
//
//           x - random number with Laplace distribution,
//           u - uniformly distributed random number
*/
#define VSL_RNG_METHOD_LAPLACE_ICDF 0 /* vsl{d,s}RngLaplace */

/*
// Weibull
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_WEIBULL_ICDF 0 /* vsl{d,s}RngWeibull */

#define VSL_RNG_METHOD_WEIBULL_ICDF_ACCURATE \
   VSL_RNG_METHOD_WEIBULL_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngWeibull */


/*
// Cauchy
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_CAUCHY_ICDF 0 /* vsl{d,s}RngCauchy */

/*
// Rayleigh
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_RAYLEIGH_ICDF 0 /* vsl{d,s}RngRayleigh */

#define VSL_RNG_METHOD_RAYLEIGH_ICDF_ACCURATE \
   VSL_RNG_METHOD_RAYLEIGH_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngRayleigh */

/*
// Lognormal
//
// <Method>   <Short Description>
// BOXMULLER2       Box-Muller 2 algorithm based method
*/
#define VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 0 /* vsl{d,s}RngLognormal */
#define VSL_RNG_METHOD_LOGNORMAL_ICDF 1       /* vsl{d,s}RngLognormal */

#define VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2_ACCURATE \
   VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2 | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngLognormal */

#define VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE \
   VSL_RNG_METHOD_LOGNORMAL_ICDF | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngLognormal */


/*
// Gumbel
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GUMBEL_ICDF 0 /* vsl{d,s}RngGumbel */

/*
// Gamma
//
// Comments:
// alpha>1             - algorithm of Marsaglia is used, nonlinear
//                       transformation of gaussian numbers based on
//                       acceptance/rejection method with squeezes;
// alpha>=0.6, alpha<1 - rejection from the Weibull distribution is used;
// alpha<0.6           - transformation of exponential power distribution
//                       (EPD) is used, EPD random numbers are generated
//                       by means of acceptance/rejection technique;
// alpha=1             - gamma distribution reduces to exponential
//                       distribution
*/
#define VSL_RNG_METHOD_GAMMA_GNORM 0 /* vsl{d,s}RngGamma */

#define VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE \
   VSL_RNG_METHOD_GAMMA_GNORM | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngGamma */


/*
// Beta
//
// Comments:
// CJA - stands for first letters of Cheng, Johnk, and Atkinson.
// Cheng    - for min(p,q) > 1 method of Cheng,
//            generation of beta random numbers of the second kind
//            based on acceptance/rejection technique and its
//            transformation to beta random numbers of the first kind;
// Johnk    - for max(p,q) < 1 methods of Johnk and Atkinson:
//            if q + K*p^2+C<=0, K=0.852..., C=-0.956...
//            algorithm of Johnk:
//            beta distributed random number is generated as
//            u1^(1/p) / (u1^(1/p)+u2^(1/q)), if u1^(1/p)+u2^(1/q)<=1;
//            otherwise switching algorithm of Atkinson: interval (0,1)
//            is divided into two domains (0,t) and (t,1), on each interval
//            acceptance/rejection technique with convenient majorizing
//            function is used;
// Atkinson - for min(p,q)<1, max(p,q)>1 switching algorithm of Atkinson
//            is used (with another point t, see short description above);
// ICDF     - inverse cumulative distribution function method according
//            to formulas x=1-u^(1/q) for p = 1, and x = u^(1/p) for q=1,
//            where x is beta distributed random number,
//            u - uniformly distributed random number.
//            for p=q=1 beta distribution reduces to uniform distribution.
//
*/
#define VSL_RNG_METHOD_BETA_CJA 0 /* vsl{d,s}RngBeta */

#define VSL_RNG_METHOD_BETA_CJA_ACCURATE \
   VSL_RNG_METHOD_BETA_CJA | VSL_RNG_METHOD_ACCURACY_FLAG
    /* accurate mode of vsl{d,s}RngBeta */

/*
// Bernoulli
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_BERNOULLI_ICDF 0 /* vsliRngBernoulli */

/*
// Geometric
//
// <Method>   <Short Description>
// ICDF       inverse cumulative distribution function method
*/
#define VSL_RNG_METHOD_GEOMETRIC_ICDF 0 /* vsliRngGeometric */

/*
// Binomial
//
// <Method>   <Short Description>
// BTPE       for ntrial*min(p,1-p)>30 acceptance/rejection method with
//            decomposition onto 4 regions:
//
//               * 2 parallelograms;
//               * triangle;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used
*/
#define VSL_RNG_METHOD_BINOMIAL_BTPE 0 /* vsliRngBinomial */

/*
// Hypergeometric
//
// <Method>   <Short Description>
// H2PE       if mode of distribution is large, acceptance/rejection method is
//            used with decomposition onto 3 regions:
//
//               * rectangular;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used
*/
#define VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE 0 /* vsliRngHypergeometric */

/*
// Poisson
//
// <Method>   <Short Description>
// PTPE       if lambda>=27, acceptance/rejection method is used with
//            decomposition onto 4 regions:
//
//               * 2 parallelograms;
//               * triangle;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used
//
// POISNORM   for lambda>=1 method is based on Poisson inverse CDF
//            approximation by Gaussian inverse CDF; for lambda<1
//            table lookup method is used.
*/
#define VSL_RNG_METHOD_POISSON_PTPE     0 /* vsliRngPoisson */
#define VSL_RNG_METHOD_POISSON_POISNORM 1 /* vsliRngPoisson */

/*
// Poisson
//
// <Method>   <Short Description>
// POISNORM   for lambda>=1 method is based on Poisson inverse CDF
//            approximation by Gaussian inverse CDF; for lambda<1
//            ICDF method is used.
*/
#define VSL_RNG_METHOD_POISSONV_POISNORM 0 /* vsliRngPoissonV */

/*
// Negbinomial
//
// <Method>   <Short Description>
// NBAR       if (a-1)*(1-p)/p>=100, acceptance/rejection method is used with
//            decomposition onto 5 regions:
//
//               * rectangular;
//               * 2 trapezoid;
//               * left exponential tail;
//               * right exponential tail.
//
//            othewise table lookup method is used.
*/
#define VSL_RNG_METHOD_NEGBINOMIAL_NBAR 0 /* vsliRngNegbinomial */

/*
//++
//  MATRIX STORAGE SCHEMES
//--
*/

/*
// Some multivariate random number generators, e.g. GaussianMV, operate
// with matrix parameters. To optimize matrix parameters usage VSL offers
// following matrix storage schemes. (See VSL documentation for more details).
//
// FULL     - whole matrix is stored
// PACKED   - lower/higher triangular matrix is packed in 1-dimensional array
// DIAGONAL - diagonal elements are packed in 1-dimensional array
*/
#define VSL_MATRIX_STORAGE_FULL     0
#define VSL_MATRIX_STORAGE_PACKED   1
#define VSL_MATRIX_STORAGE_DIAGONAL 2


/*
// SUMMARY STATISTICS (SS) RELATED MACRO DEFINITIONS
*/

/*
//++
//  MATRIX STORAGE SCHEMES
//--
*/
/*
// SS routines work with matrix parameters, e.g. matrix of observations,
// variance-covariance matrix. To optimize work with matrices the library
// provides the following storage matrix schemes.
*/
/*
// Matrix of observations:
// ROWS    - observations of the random vector are stored in raws, that
//           is, i-th row of the matrix of observations contains values
//           of i-th component of the random vector
// COLS    - observations of the random vector are stored in columns that
//           is, i-th column of the matrix of observations contains values
//           of i-th component of the random vector
*/
#define VSL_SS_MATRIX_STORAGE_ROWS     0x00010000
#define VSL_SS_MATRIX_STORAGE_COLS     0x00020000

/*
// Variance-covariance/correlation matrix:
// FULL     - whole matrix is stored
// L_PACKED - lower triangular matrix is stored as 1-dimensional array
// U_PACKED - upper triangular matrix is stored as 1-dimensional array
*/
#define VSL_SS_MATRIX_STORAGE_FULL            0x00000000
#define VSL_SS_MATRIX_STORAGE_L_PACKED        0x00000001
#define VSL_SS_MATRIX_STORAGE_U_PACKED        0x00000002


/*
//++
//  SUMMARY STATISTICS LIBRARY METHODS
//--
*/
/*
// SS routines provide computation of basic statistical estimates
// (central/raw moments up to 4th order, variance-covariance,
//  minimum, maximum, skewness/kurtosis) using the following methods
//  - FAST  - estimates are computed for price of one or two passes over
//            observations using highly optimized MKL routines
//  - 1PASS - estimate is computed for price of one pass of the observations
//  - FAST_USER_MEAN - estimates are computed for price of one or two passes
//            over observations given user defined mean for central moments,
//            covariance and correlation
//  - CP_TO_COVCOR - convert cross-product matrix to variance-covariance/
//            correlation matrix
//  - SUM_TO_MOM - convert raw/central sums to raw/central moments
//
*/
#define VSL_SS_METHOD_FAST                    0x00000001
#define VSL_SS_METHOD_1PASS                   0x00000002
#define VSL_SS_METHOD_FAST_USER_MEAN          0x00000100
#define VSL_SS_METHOD_CP_TO_COVCOR            0x00000200
#define VSL_SS_METHOD_SUM_TO_MOM              0x00000400

/*
// SS provides routine for parametrization of correlation matrix using
// SPECTRAL DECOMPOSITION (SD) method
*/
#define VSL_SS_METHOD_SD                      0x00000004

/*
// SS routine for robust estimation of variance-covariance matrix
// and mean supports Rocke algorithm, TBS-estimator
*/
#define VSL_SS_METHOD_TBS                     0x00000008

/*
//  SS routine for estimation of missing values
//  supports Multiple Imputation (MI) method
*/
#define VSL_SS_METHOD_MI                      0x00000010

/*
// SS provides routine for detection of outliers, BACON method
*/
#define VSL_SS_METHOD_BACON                   0x00000020

/*
// SS supports routine for estimation of quantiles for streaming data
// using the following methods:
// - ZW      - intermediate estimates of quantiles during processing
//             the next block are computed
// - ZW_FAST - intermediate estimates of quantiles during processing
//             the next block are not computed
*/
#define VSL_SS_METHOD_SQUANTS_ZW              0x00000040
#define VSL_SS_METHOD_SQUANTS_ZW_FAST         0x00000080


/*
// Input of BACON algorithm is set of 3 parameters:
// - Initialization method of the algorithm
// - Parameter alfa such that 1-alfa is percentile of Chi2 distribution
// - Stopping criterion
*/
/*
// Number of BACON algorithm parameters
*/
#define VSL_SS_BACON_PARAMS_N         3

/*
// SS implementation of BACON algorithm supports two initialization methods:
// - Mahalanobis distance based method
// - Median based method
*/
#define VSL_SS_METHOD_BACON_MAHALANOBIS_INIT  0x00000001
#define VSL_SS_METHOD_BACON_MEDIAN_INIT       0x00000002

/*
// SS routine for sorting data, RADIX method
*/
#define VSL_SS_METHOD_RADIX                   0x00100000

/*
// Input of TBS algorithm is set of 4 parameters:
// - Breakdown point
// - Asymptotic rejection probability
// - Stopping criterion
// - Maximum number of iterations
*/
/*
// Number of TBS algorithm parameters
*/
#define VSL_SS_TBS_PARAMS_N           4

/*
// Input of MI algorithm is set of 5 parameters:
// - Maximal number of iterations for EM algorithm
// - Maximal number of iterations for DA algorithm
// - Stopping criterion
// - Number of sets to impute
// - Total number of missing values in dataset
*/
/*
// Number of MI algorithm parameters
*/
#define VSL_SS_MI_PARAMS_SIZE         5

/*
// SS MI algorithm expects that missing values are
// marked with NANs
*/
#define VSL_SS_DNAN                    0xFFF8000000000000
#define VSL_SS_SNAN                    0xFFC00000

/*
// Input of ZW algorithm is 1 parameter:
// - accuracy of quantile estimation
*/
/*
// Number of ZW algorithm parameters
*/
#define VSL_SS_SQUANTS_ZW_PARAMS_N   1


/*
//++
// MACROS USED SS EDIT AND COMPUTE ROUTINES
//--
*/

/*
// SS EditTask routine is way to edit input and output parameters of the task,
// e.g., pointers to arrays which hold observations, weights of observations,
// arrays of mean estimates or covariance estimates.
// Macros below define parameters available for modification
*/
#define VSL_SS_ED_DIMEN                                 1
#define VSL_SS_ED_OBSERV_N                              2
#define VSL_SS_ED_OBSERV                                3
#define VSL_SS_ED_OBSERV_STORAGE                        4
#define VSL_SS_ED_INDC                                  5
#define VSL_SS_ED_WEIGHTS                               6
#define VSL_SS_ED_MEAN                                  7
#define VSL_SS_ED_2R_MOM                                8
#define VSL_SS_ED_3R_MOM                                9
#define VSL_SS_ED_4R_MOM                               10
#define VSL_SS_ED_2C_MOM                               11
#define VSL_SS_ED_3C_MOM                               12
#define VSL_SS_ED_4C_MOM                               13
#define VSL_SS_ED_SUM                                  67
#define VSL_SS_ED_2R_SUM                               68
#define VSL_SS_ED_3R_SUM                               69
#define VSL_SS_ED_4R_SUM                               70
#define VSL_SS_ED_2C_SUM                               71
#define VSL_SS_ED_3C_SUM                               72
#define VSL_SS_ED_4C_SUM                               73
#define VSL_SS_ED_KURTOSIS                             14
#define VSL_SS_ED_SKEWNESS                             15
#define VSL_SS_ED_MIN                                  16
#define VSL_SS_ED_MAX                                  17
#define VSL_SS_ED_VARIATION                            18
#define VSL_SS_ED_COV                                  19
#define VSL_SS_ED_COV_STORAGE                          20
#define VSL_SS_ED_COR                                  21
#define VSL_SS_ED_COR_STORAGE                          22
#define VSL_SS_ED_CP                                   74
#define VSL_SS_ED_CP_STORAGE                           75
#define VSL_SS_ED_ACCUM_WEIGHT                         23
#define VSL_SS_ED_QUANT_ORDER_N                        24
#define VSL_SS_ED_QUANT_ORDER                          25
#define VSL_SS_ED_QUANT_QUANTILES                      26
#define VSL_SS_ED_ORDER_STATS                          27
#define VSL_SS_ED_GROUP_INDC                           28
#define VSL_SS_ED_POOLED_COV_STORAGE                   29
#define VSL_SS_ED_POOLED_MEAN                          30
#define VSL_SS_ED_POOLED_COV                           31
#define VSL_SS_ED_GROUP_COV_INDC                       32
#define VSL_SS_ED_REQ_GROUP_INDC                       32
#define VSL_SS_ED_GROUP_MEAN                           33
#define VSL_SS_ED_GROUP_COV_STORAGE                    34
#define VSL_SS_ED_GROUP_COV                            35
#define VSL_SS_ED_ROBUST_COV_STORAGE                   36
#define VSL_SS_ED_ROBUST_COV_PARAMS_N                  37
#define VSL_SS_ED_ROBUST_COV_PARAMS                    38
#define VSL_SS_ED_ROBUST_MEAN                          39
#define VSL_SS_ED_ROBUST_COV                           40
#define VSL_SS_ED_OUTLIERS_PARAMS_N                    41
#define VSL_SS_ED_OUTLIERS_PARAMS                      42
#define VSL_SS_ED_OUTLIERS_WEIGHT                      43
#define VSL_SS_ED_ORDER_STATS_STORAGE                  44
#define VSL_SS_ED_PARTIAL_COV_IDX                      45
#define VSL_SS_ED_PARTIAL_COV                          46
#define VSL_SS_ED_PARTIAL_COV_STORAGE                  47
#define VSL_SS_ED_PARTIAL_COR                          48
#define VSL_SS_ED_PARTIAL_COR_STORAGE                  49
#define VSL_SS_ED_MI_PARAMS_N                          50
#define VSL_SS_ED_MI_PARAMS                            51
#define VSL_SS_ED_MI_INIT_ESTIMATES_N                  52
#define VSL_SS_ED_MI_INIT_ESTIMATES                    53
#define VSL_SS_ED_MI_SIMUL_VALS_N                      54
#define VSL_SS_ED_MI_SIMUL_VALS                        55
#define VSL_SS_ED_MI_ESTIMATES_N                       56
#define VSL_SS_ED_MI_ESTIMATES                         57
#define VSL_SS_ED_MI_PRIOR_N                           58
#define VSL_SS_ED_MI_PRIOR                             59
#define VSL_SS_ED_PARAMTR_COR                          60
#define VSL_SS_ED_PARAMTR_COR_STORAGE                  61
#define VSL_SS_ED_STREAM_QUANT_PARAMS_N                62
#define VSL_SS_ED_STREAM_QUANT_PARAMS                  63
#define VSL_SS_ED_STREAM_QUANT_ORDER_N                 64
#define VSL_SS_ED_STREAM_QUANT_ORDER                   65
#define VSL_SS_ED_STREAM_QUANT_QUANTILES               66
#define VSL_SS_ED_MDAD                                 76
#define VSL_SS_ED_MNAD                                 77
#define VSL_SS_ED_SORTED_OBSERV                        78
#define VSL_SS_ED_SORTED_OBSERV_STORAGE                79


/*
// SS Compute routine calculates estimates supported by the library
// Macros below define estimates to compute
*/
#define VSL_SS_MEAN                       0x0000000000000001
#define VSL_SS_2R_MOM                     0x0000000000000002
#define VSL_SS_3R_MOM                     0x0000000000000004
#define VSL_SS_4R_MOM                     0x0000000000000008
#define VSL_SS_2C_MOM                     0x0000000000000010
#define VSL_SS_3C_MOM                     0x0000000000000020
#define VSL_SS_4C_MOM                     0x0000000000000040
#define VSL_SS_SUM                        0x0000000002000000
#define VSL_SS_2R_SUM                     0x0000000004000000
#define VSL_SS_3R_SUM                     0x0000000008000000
#define VSL_SS_4R_SUM                     0x0000000010000000
#define VSL_SS_2C_SUM                     0x0000000020000000
#define VSL_SS_3C_SUM                     0x0000000040000000
#define VSL_SS_4C_SUM                     0x0000000080000000
#define VSL_SS_KURTOSIS                   0x0000000000000080
#define VSL_SS_SKEWNESS                   0x0000000000000100
#define VSL_SS_VARIATION                  0x0000000000000200
#define VSL_SS_MIN                        0x0000000000000400
#define VSL_SS_MAX                        0x0000000000000800
#define VSL_SS_COV                        0x0000000000001000
#define VSL_SS_COR                        0x0000000000002000
#define VSL_SS_CP                         0x0000000100000000
#define VSL_SS_POOLED_COV                 0x0000000000004000
#define VSL_SS_GROUP_COV                  0x0000000000008000
#define VSL_SS_POOLED_MEAN                0x0000000800000000
#define VSL_SS_GROUP_MEAN                 0x0000001000000000
#define VSL_SS_QUANTS                     0x0000000000010000
#define VSL_SS_ORDER_STATS                0x0000000000020000
#define VSL_SS_SORTED_OBSERV              0x0000008000000000
#define VSL_SS_ROBUST_COV                 0x0000000000040000
#define VSL_SS_OUTLIERS                   0x0000000000080000
#define VSL_SS_PARTIAL_COV                0x0000000000100000
#define VSL_SS_PARTIAL_COR                0x0000000000200000
#define VSL_SS_MISSING_VALS               0x0000000000400000
#define VSL_SS_PARAMTR_COR                0x0000000000800000
#define VSL_SS_STREAM_QUANTS              0x0000000001000000
#define VSL_SS_MDAD                       0x0000000200000000
#define VSL_SS_MNAD                       0x0000000400000000

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_DEFINES_H__ */
/* file: mkl_vsl_types.h */
/*******************************************************************************
* Copyright (c) 2006-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/*
//++
//  This file contains user-level type definitions.
//--
*/

#ifndef __MKL_VSL_TYPES_H__
#define __MKL_VSL_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/*
//++
//  TYPEDEFS
//--
*/

/*
//  POINTER TO STREAM STATE STRUCTURE
//  This is a void pointer to hide implementation details.
*/
typedef void* VSLStreamStatePtr;
typedef void* VSLConvTaskPtr;
typedef void* VSLCorrTaskPtr;
typedef void* VSLSSTaskPtr;

/*
//  POINTERS TO BASIC RANDOM NUMBER GENERATOR FUNCTIONS
//  Each BRNG must have following implementations:
//
//  * Stream initialization (InitStreamPtr)
//  * Integer-value recurrence implementation (iBRngPtr)
//  * Single precision implementation (sBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
//  * Double precision implementation (dBRngPtr) - for random number generation
//    uniformly distributed on the [a,b] interval
*/
typedef int (*InitStreamPtr)( int method, VSLStreamStatePtr stream, \
        int n, const unsigned int params[] );
typedef int (*sBRngPtr)( VSLStreamStatePtr stream, int n, float r[], \
        float a, float b );
typedef int (*dBRngPtr)( VSLStreamStatePtr stream, int n, double r[], \
        double a, double b );
typedef int (*iBRngPtr)( VSLStreamStatePtr stream, int n, unsigned int r[] );

/*********** Pointers to callback functions for abstract streams *************/
typedef int (*iUpdateFuncPtr)( VSLStreamStatePtr stream, int* n, \
     unsigned int ibuf[], int* nmin, int* nmax, int* idx );
typedef int (*dUpdateFuncPtr)( VSLStreamStatePtr stream, int* n,
     double dbuf[], int* nmin, int* nmax, int* idx );
typedef int (*sUpdateFuncPtr)( VSLStreamStatePtr stream, int* n, \
     float sbuf[], int* nmin, int* nmax, int* idx );


/*
//  BASIC RANDOM NUMBER GENERATOR PROPERTIES STRUCTURE
//  The structure describes the properties of given basic generator, e.g. size
//  of the stream state structure, pointers to function implementations, etc.
//
//  BRNG properties structure fields:
//  StreamStateSize - size of the stream state structure (in bytes)
//  WordSize        - size of base word (in bytes). Typically this is 4 bytes.
//  NSeeds          - number of words necessary to describe generator's state
//  NBits           - number of bits actually used in base word. For example,
//                    only 31 least significant bits are actually used in
//                    basic random number generator MCG31m1 with 4-byte base
//                    word. NBits field is useful while interpreting random
//                    words as a sequence of random bits.
//  IncludesZero    - FALSE if 0 cannot be generated in integer-valued
//                    implementation; TRUE if 0 can be potentially generated in
//                    integer-valued implementation.
//  InitStream      - pointer to stream state initialization function
//  sBRng           - pointer to single precision implementation
//  dBRng           - pointer to double precision implementation
//  iBRng           - pointer to integer-value implementation
*/
typedef struct _VSLBRngProperties {
    int StreamStateSize;       /* Stream state size (in bytes) */
    int NSeeds;                /* Number of seeds */
    int IncludesZero;          /* Zero flag */
    int WordSize;              /* Size (in bytes) of base word */
    int NBits;                 /* Number of actually used bits */
    InitStreamPtr InitStream;  /* Pointer to InitStream func */
    sBRngPtr sBRng;            /* Pointer to S func */
    dBRngPtr dBRng;            /* Pointer to D func */
    iBRngPtr iBRng;            /* Pointer to I func */
} VSLBRngProperties;

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_TYPES_H__ */
/* file: mkl_vsl_functions.h */
/*******************************************************************************
* Copyright (c) 2006-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/*
//++
//  User-level VSL function declarations
//--
*/

#ifndef __MKL_VSL_FUNCTIONS_H__
#define __MKL_VSL_FUNCTIONS_H__


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
//++
//  EXTERNAL API MACROS.
//  Used to construct VSL function declaration. Change them if you are going to
//  provide different API for VSL functions.
//--
*/

#if  !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)   extern rtype name    arg;
#endif

#if  !defined(_mkl_api)
#define _mkl_api(rtype,name,arg)   extern rtype name##_ arg;
#endif

#if  !defined(_MKL_API)
#define _MKL_API(rtype,name,arg)   extern rtype name##_ arg;
#endif

/*
//++
//  VSL CONTINUOUS DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Cauchy distribution */
_Mkl_Api(int,vdRngCauchy,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ))
_MKL_API(int,VDRNGCAUCHY,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_mkl_api(int,vdrngcauchy,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_Mkl_Api(int,vsRngCauchy,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGCAUCHY,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))
_mkl_api(int,vsrngcauchy,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))

/* Uniform distribution */
_Mkl_Api(int,vdRngUniform,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ))
_MKL_API(int,VDRNGUNIFORM,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_mkl_api(int,vdrnguniform,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_Mkl_Api(int,vsRngUniform,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGUNIFORM,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))
_mkl_api(int,vsrnguniform,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))

/* Gaussian distribution */
_Mkl_Api(int,vdRngGaussian,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ))
_MKL_API(int,VDRNGGAUSSIAN,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_mkl_api(int,vdrnggaussian,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_Mkl_Api(int,vsRngGaussian,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGGAUSSIAN,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))
_mkl_api(int,vsrnggaussian,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))

/* GaussianMV distribution */
_Mkl_Api(int,vdRngGaussianMV,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const MKL_INT  ,  const MKL_INT  , const double *, const double *))
_MKL_API(int,VDRNGGAUSSIANMV,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const MKL_INT *,  const MKL_INT *, const double *, const double *))
_mkl_api(int,vdrnggaussianmv,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const MKL_INT *,  const MKL_INT *, const double *, const double *))
_Mkl_Api(int,vsRngGaussianMV,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const MKL_INT  ,  const MKL_INT  , const float *,  const float * ))
_MKL_API(int,VSRNGGAUSSIANMV,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const MKL_INT *,  const MKL_INT *, const float *,  const float * ))
_mkl_api(int,vsrnggaussianmv,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const MKL_INT *,  const MKL_INT *, const float *,  const float * ))

/* Exponential distribution */
_Mkl_Api(int,vdRngExponential,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  double [], const double  , const double  ))
_MKL_API(int,VDRNGEXPONENTIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *))
_mkl_api(int,vdrngexponential,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *))
_Mkl_Api(int,vsRngExponential,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGEXPONENTIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ))
_mkl_api(int,vsrngexponential,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ))

/* Laplace distribution */
_Mkl_Api(int,vdRngLaplace,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ))
_MKL_API(int,VDRNGLAPLACE,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_mkl_api(int,vdrnglaplace,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_Mkl_Api(int,vsRngLaplace,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGLAPLACE,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))
_mkl_api(int,vsrnglaplace,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))

/* Weibull distribution */
_Mkl_Api(int,vdRngWeibull,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  ))
_MKL_API(int,VDRNGWEIBULL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *))
_mkl_api(int,vdrngweibull,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *))
_Mkl_Api(int,vsRngWeibull,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float   ))
_MKL_API(int,VSRNGWEIBULL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ))
_mkl_api(int,vsrngweibull,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ))

/* Rayleigh distribution */
_Mkl_Api(int,vdRngRayleigh,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  double [], const double  , const double  ))
_MKL_API(int,VDRNGRAYLEIGH,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *))
_mkl_api(int,vdrngrayleigh,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  double [], const double *, const double *))
_Mkl_Api(int,vsRngRayleigh,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  ,  float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGRAYLEIGH,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ))
_mkl_api(int,vsrngrayleigh,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *,  float [],  const float *,  const float * ))

/* Lognormal distribution */
_Mkl_Api(int,vdRngLognormal,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  , const double  ))
_MKL_API(int,VDRNGLOGNORMAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *))
_mkl_api(int,vdrnglognormal,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *))
_Mkl_Api(int,vsRngLognormal,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float  ,  const float   ))
_MKL_API(int,VSRNGLOGNORMAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ))
_mkl_api(int,vsrnglognormal,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ))

/* Gumbel distribution */
_Mkl_Api(int,vdRngGumbel,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  ))
_MKL_API(int,VDRNGGUMBEL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_mkl_api(int,vdrnggumbel,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *))
_Mkl_Api(int,vsRngGumbel,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float   ))
_MKL_API(int,VSRNGGUMBEL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))
_mkl_api(int,vsrnggumbel,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float * ))

/* Gamma distribution */
_Mkl_Api(int,vdRngGamma,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  ))
_MKL_API(int,VDRNGGAMMA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *))
_mkl_api(int,vdrnggamma,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *))
_Mkl_Api(int,vsRngGamma,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float   ))
_MKL_API(int,VSRNGGAMMA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ))
_mkl_api(int,vsrnggamma,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float * ))

/* Beta distribution */
_Mkl_Api(int,vdRngBeta,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , double [], const double  , const double  , const double  , const double  ))
_MKL_API(int,VDRNGBETA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *))
_mkl_api(int,vdrngbeta,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, double [], const double *, const double *, const double *, const double *))
_Mkl_Api(int,vsRngBeta,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , float [],  const float  ,  const float  ,  const float  ,  const float   ))
_MKL_API(int,VSRNGBETA,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ))
_mkl_api(int,vsrngbeta,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, float [],  const float *,  const float *,  const float *,  const float * ))

/*
//++
//  VSL DISCRETE DISTRIBUTION GENERATOR FUNCTION DECLARATIONS.
//--
*/
/* Bernoulli distribution */
_Mkl_Api(int,viRngBernoulli,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  ))
_MKL_API(int,VIRNGBERNOULLI,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *))
_mkl_api(int,virngbernoulli,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *))

/* Uniform distribution */
_Mkl_Api(int,viRngUniform,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const int  ))
_MKL_API(int,VIRNGUNIFORM,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *))
_mkl_api(int,virnguniform,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *))

/* UniformBits distribution */
_Mkl_Api(int,viRngUniformBits,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , unsigned int []))
_MKL_API(int,VIRNGUNIFORMBITS,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []))
_mkl_api(int,virnguniformbits,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []))

/* UniformBits32 distribution */
_Mkl_Api(int,viRngUniformBits32,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , unsigned int []))
_MKL_API(int,VIRNGUNIFORMBITS32,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []))
_mkl_api(int,virnguniformbits32,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned int []))

/* UniformBits64 distribution */
_Mkl_Api(int,viRngUniformBits64,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , unsigned MKL_INT64 []))
_MKL_API(int,VIRNGUNIFORMBITS64,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned MKL_INT64 []))
_mkl_api(int,virnguniformbits64,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, unsigned MKL_INT64 []))

/* Geometric distribution */
_Mkl_Api(int,viRngGeometric,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  ))
_MKL_API(int,VIRNGGEOMETRIC,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *))
_mkl_api(int,virnggeometric,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *))

/* Binomial distribution */
_Mkl_Api(int,viRngBinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const double  ))
_MKL_API(int,VIRNGBINOMIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const double *))
_mkl_api(int,virngbinomial,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const double *))

/* Hypergeometric distribution */
_Mkl_Api(int,viRngHypergeometric,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const int  , const int  , const int  ))
_MKL_API(int,VIRNGHYPERGEOMETRIC,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *, const int *))
_mkl_api(int,virnghypergeometric,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const int *, const int *, const int *))

/* Negbinomial distribution */
_Mkl_Api(int,viRngNegbinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  , const double  ))
_Mkl_Api(int,viRngNegBinomial,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  , const double  ))
_MKL_API(int,VIRNGNEGBINOMIAL,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *, const double *))
_mkl_api(int,virngnegbinomial,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *, const double *))

/* Poisson distribution */
_Mkl_Api(int,viRngPoisson,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double  ))
_MKL_API(int,VIRNGPOISSON,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *))
_mkl_api(int,virngpoisson,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double *))

/* PoissonV distribution */
_Mkl_Api(int,viRngPoissonV,(const MKL_INT  , VSLStreamStatePtr  , const MKL_INT  , int [], const double []))
_MKL_API(int,VIRNGPOISSONV,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double []))
_mkl_api(int,virngpoissonv,(const MKL_INT *, VSLStreamStatePtr *, const MKL_INT *, int [], const double []))


/*
//++
//  VSL SERVICE FUNCTION DECLARATIONS.
//--
*/
/* NewStream - stream creation/initialization */
_Mkl_Api(int,vslNewStream,(VSLStreamStatePtr* , const MKL_INT  , const MKL_UINT  ))
_mkl_api(int,vslnewstream,(VSLStreamStatePtr* , const MKL_INT *, const MKL_UINT *))
_MKL_API(int,VSLNEWSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const MKL_UINT *))

/* NewStreamEx - advanced stream creation/initialization */
_Mkl_Api(int,vslNewStreamEx,(VSLStreamStatePtr* , const MKL_INT  , const MKL_INT  , const unsigned int[]))
_mkl_api(int,vslnewstreamex,(VSLStreamStatePtr* , const MKL_INT *, const MKL_INT *, const unsigned int[]))
_MKL_API(int,VSLNEWSTREAMEX,(VSLStreamStatePtr* , const MKL_INT *, const MKL_INT *, const unsigned int[]))

_Mkl_Api(int,vsliNewAbstractStream,(VSLStreamStatePtr* , const MKL_INT  , const unsigned int[], const iUpdateFuncPtr))
_mkl_api(int,vslinewabstractstream,(VSLStreamStatePtr* , const MKL_INT *, const unsigned int[], const iUpdateFuncPtr))
_MKL_API(int,VSLINEWABSTRACTSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const unsigned int[], const iUpdateFuncPtr))

_Mkl_Api(int,vsldNewAbstractStream,(VSLStreamStatePtr* , const MKL_INT  , const double[], const double  , const double  , const dUpdateFuncPtr))
_mkl_api(int,vsldnewabstractstream,(VSLStreamStatePtr* , const MKL_INT *, const double[], const double *, const double *, const dUpdateFuncPtr))
_MKL_API(int,VSLDNEWABSTRACTSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const double[], const double *, const double *, const dUpdateFuncPtr))

_Mkl_Api(int,vslsNewAbstractStream,(VSLStreamStatePtr* , const MKL_INT  , const float[], const float  , const float  , const sUpdateFuncPtr))
_mkl_api(int,vslsnewabstractstream,(VSLStreamStatePtr* , const MKL_INT *, const float[], const float *, const float *, const sUpdateFuncPtr))
_MKL_API(int,VSLSNEWABSTRACTSTREAM,(VSLStreamStatePtr* , const MKL_INT *, const float[], const float *, const float *, const sUpdateFuncPtr))

/* DeleteStream - delete stream */
_Mkl_Api(int,vslDeleteStream,(VSLStreamStatePtr*))
_mkl_api(int,vsldeletestream,(VSLStreamStatePtr*))
_MKL_API(int,VSLDELETESTREAM,(VSLStreamStatePtr*))

/* CopyStream - copy all stream information */
_Mkl_Api(int,vslCopyStream,(VSLStreamStatePtr*, const VSLStreamStatePtr))
_mkl_api(int,vslcopystream,(VSLStreamStatePtr*, const VSLStreamStatePtr))
_MKL_API(int,VSLCOPYSTREAM,(VSLStreamStatePtr*, const VSLStreamStatePtr))

/* CopyStreamState - copy stream state only */
_Mkl_Api(int,vslCopyStreamState,(VSLStreamStatePtr  , const VSLStreamStatePtr  ))
_mkl_api(int,vslcopystreamstate,(VSLStreamStatePtr *, const VSLStreamStatePtr *))
_MKL_API(int,VSLCOPYSTREAMSTATE,(VSLStreamStatePtr *, const VSLStreamStatePtr *))

/* LeapfrogStream - leapfrog method */
_Mkl_Api(int,vslLeapfrogStream,(VSLStreamStatePtr  , const MKL_INT  , const MKL_INT  ))
_mkl_api(int,vslleapfrogstream,(VSLStreamStatePtr *, const MKL_INT *, const MKL_INT *))
_MKL_API(int,VSLLEAPFROGSTREAM,(VSLStreamStatePtr *, const MKL_INT *, const MKL_INT *))

/* SkipAheadStream - skip-ahead method */
_Mkl_Api(int,vslSkipAheadStream,(VSLStreamStatePtr  , const long long int  ))
_mkl_api(int,vslskipaheadstream,(VSLStreamStatePtr *, const long long int *))
_MKL_API(int,VSLSKIPAHEADSTREAM,(VSLStreamStatePtr *, const long long int *))

/* GetStreamStateBrng - get BRNG associated with given stream */
_Mkl_Api(int,vslGetStreamStateBrng,(const VSLStreamStatePtr  ))
_mkl_api(int,vslgetstreamstatebrng,(const VSLStreamStatePtr *))
_MKL_API(int,VSLGETSTREAMSTATEBRNG,(const VSLStreamStatePtr *))

/* GetNumRegBrngs - get number of registered BRNGs */
_Mkl_Api(int,vslGetNumRegBrngs,(void))
_mkl_api(int,vslgetnumregbrngs,(void))
_MKL_API(int,VSLGETNUMREGBRNGS,(void))

/* RegisterBrng - register new BRNG */
_Mkl_Api(int,vslRegisterBrng,(const VSLBRngProperties* ))
_mkl_api(int,vslregisterbrng,(const VSLBRngProperties* ))
_MKL_API(int,VSLREGISTERBRNG,(const VSLBRngProperties* ))

/* GetBrngProperties - get BRNG properties */
_Mkl_Api(int,vslGetBrngProperties,(const int  , VSLBRngProperties* ))
_mkl_api(int,vslgetbrngproperties,(const int *, VSLBRngProperties* ))
_MKL_API(int,VSLGETBRNGPROPERTIES,(const int *, VSLBRngProperties* ))

/* SaveStreamF - save random stream descriptive data to file */
_Mkl_Api(int,vslSaveStreamF,(const VSLStreamStatePtr  , const char*             ))
_mkl_api(int,vslsavestreamf,(const VSLStreamStatePtr *, const char* , const int ))
_MKL_API(int,VSLSAVESTREAMF,(const VSLStreamStatePtr *, const char* , const int ))

/* LoadStreamF - load random stream descriptive data from file */
_Mkl_Api(int,vslLoadStreamF,(VSLStreamStatePtr *, const char*             ))
_mkl_api(int,vslloadstreamf,(VSLStreamStatePtr *, const char* , const int ))
_MKL_API(int,VSLLOADSTREAMF,(VSLStreamStatePtr *, const char* , const int ))

/* SaveStreamM - save random stream descriptive data to memory */
_Mkl_Api(int,vslSaveStreamM,(const VSLStreamStatePtr  , char* ))
_mkl_api(int,vslsavestreamm,(const VSLStreamStatePtr *, char* ))
_MKL_API(int,VSLSAVESTREAMM,(const VSLStreamStatePtr *, char* ))

/* LoadStreamM - load random stream descriptive data from memory */
_Mkl_Api(int,vslLoadStreamM,(VSLStreamStatePtr *, const char* ))
_mkl_api(int,vslloadstreamm,(VSLStreamStatePtr *, const char* ))
_MKL_API(int,VSLLOADSTREAMM,(VSLStreamStatePtr *, const char* ))

/* GetStreamSize - get size of random stream descriptive data */
_Mkl_Api(int,vslGetStreamSize,(const VSLStreamStatePtr))
_mkl_api(int,vslgetstreamsize,(const VSLStreamStatePtr))
_MKL_API(int,VSLGETSTREAMSIZE,(const VSLStreamStatePtr))

/*
//++
//  VSL CONVOLUTION AND CORRELATION FUNCTION DECLARATIONS.
//--
*/

_Mkl_Api(int,vsldConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vsldconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLDCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vslsConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vslsconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLSCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vslzConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vslzconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLZCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vslcConvNewTask,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vslcconvnewtask,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLCCONVNEWTASK,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vsldCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vsldcorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLDCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vslsCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vslscorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLSCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vslzCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vslzcorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLZCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))

_Mkl_Api(int,vslcCorrNewTask,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_mkl_api(int,vslccorrnewtask,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))
_MKL_API(int,VSLCCORRNEWTASK,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT []))


_Mkl_Api(int,vsldConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vsldconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLDCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vslsConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vslsconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLSCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vslzConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vslzconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLZCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vslcConvNewTask1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vslcconvnewtask1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLCCONVNEWTASK1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vsldCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vsldcorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLDCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vslsCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vslscorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLSCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vslzCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vslzcorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLZCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))

_Mkl_Api(int,vslcCorrNewTask1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT ,  const MKL_INT  ))
_mkl_api(int,vslccorrnewtask1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLCCORRNEWTASK1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* ))


_Mkl_Api(int,vsldConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []))
_mkl_api(int,vsldconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []))
_MKL_API(int,VSLDCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []))

_Mkl_Api(int,vslsConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []))
_mkl_api(int,vslsconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []))
_MKL_API(int,VSLSCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []))

_Mkl_Api(int,vslzConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []))
_mkl_api(int,vslzconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []))
_MKL_API(int,VSLZCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []))

_Mkl_Api(int,vslcConvNewTaskX,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []))
_mkl_api(int,vslcconvnewtaskx,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []))
_MKL_API(int,VSLCCONVNEWTASKX,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []))

_Mkl_Api(int,vsldCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []))
_mkl_api(int,vsldcorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []))
_MKL_API(int,VSLDCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const double [], const MKL_INT []))

_Mkl_Api(int,vslsCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []))
_mkl_api(int,vslscorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []))
_MKL_API(int,VSLSCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const float [],  const MKL_INT []))

_Mkl_Api(int,vslzCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []))
_mkl_api(int,vslzcorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []))
_MKL_API(int,VSLZCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT []))

_Mkl_Api(int,vslcCorrNewTaskX,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []))
_mkl_api(int,vslccorrnewtaskx,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []))
_MKL_API(int,VSLCCORRNEWTASKX,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT [], const MKL_INT [], const MKL_INT [], const MKL_Complex8 [],  const MKL_INT []))


_Mkl_Api(int,vsldConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const double [], const MKL_INT  ))
_mkl_api(int,vsldconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ))
_MKL_API(int,VSLDCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ))

_Mkl_Api(int,vslsConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const float [],  const MKL_INT  ))
_mkl_api(int,vslsconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ))
_MKL_API(int,VSLSCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ))

_Mkl_Api(int,vslzConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex16 [], const MKL_INT  ))
_mkl_api(int,vslzconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ))
_MKL_API(int,VSLZCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ))

_Mkl_Api(int,vslcConvNewTaskX1D,(VSLConvTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  ))
_mkl_api(int,vslcconvnewtaskx1d,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ))
_MKL_API(int,VSLCCONVNEWTASKX1D,(VSLConvTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ))

_Mkl_Api(int,vsldCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const double [], const MKL_INT  ))
_mkl_api(int,vsldcorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ))
_MKL_API(int,VSLDCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const MKL_INT* ))

_Mkl_Api(int,vslsCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const float [],  const MKL_INT  ))
_mkl_api(int,vslscorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ))
_MKL_API(int,VSLSCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float [],  const MKL_INT* ))

_Mkl_Api(int,vslzCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex16 [], const MKL_INT  ))
_mkl_api(int,vslzcorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ))
_MKL_API(int,VSLZCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex16 [], const MKL_INT* ))

_Mkl_Api(int,vslcCorrNewTaskX1D,(VSLCorrTaskPtr* , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  ))
_mkl_api(int,vslccorrnewtaskx1d,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ))
_MKL_API(int,VSLCCORRNEWTASKX1D,(VSLCorrTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* ))


_Mkl_Api(int,vslConvDeleteTask,(VSLConvTaskPtr* ))
_mkl_api(int,vslconvdeletetask,(VSLConvTaskPtr* ))
_MKL_API(int,VSLCONVDeleteTask,(VSLConvTaskPtr* ))

_Mkl_Api(int,vslCorrDeleteTask,(VSLCorrTaskPtr* ))
_mkl_api(int,vslcorrdeletetask,(VSLCorrTaskPtr* ))
_MKL_API(int,VSLCORRDeleteTask,(VSLCorrTaskPtr* ))


_Mkl_Api(int,vslConvCopyTask,(VSLConvTaskPtr* , const VSLConvTaskPtr  ))
_mkl_api(int,vslconvcopytask,(VSLConvTaskPtr* , const VSLConvTaskPtr* ))
_MKL_API(int,VSLCONVCopyTask,(VSLConvTaskPtr* , const VSLConvTaskPtr* ))

_Mkl_Api(int,vslCorrCopyTask,(VSLCorrTaskPtr* , const VSLCorrTaskPtr  ))
_mkl_api(int,vslcorrcopytask,(VSLCorrTaskPtr* , const VSLCorrTaskPtr* ))
_MKL_API(int,VSLCORRCopyTask,(VSLCorrTaskPtr* , const VSLCorrTaskPtr* ))


_Mkl_Api(int,vslConvSetMode,(VSLConvTaskPtr  , const MKL_INT  ))
_mkl_api(int,vslconvsetmode,(VSLConvTaskPtr* , const MKL_INT* ))
_MKL_API(int,VSLCONVSETMODE,(VSLConvTaskPtr* , const MKL_INT* ))

_Mkl_Api(int,vslCorrSetMode,(VSLCorrTaskPtr  , const MKL_INT  ))
_mkl_api(int,vslcorrsetmode,(VSLCorrTaskPtr* , const MKL_INT* ))
_MKL_API(int,VSLCORRSETMODE,(VSLCorrTaskPtr* , const MKL_INT* ))


_Mkl_Api(int,vslConvSetInternalPrecision,(VSLConvTaskPtr  , const MKL_INT  ))
_mkl_api(int,vslconvsetinternalprecision,(VSLConvTaskPtr* , const MKL_INT* ))
_MKL_API(int,VSLCONVSETINTERNALPRECISION,(VSLConvTaskPtr* , const MKL_INT* ))

_Mkl_Api(int,vslCorrSetInternalPrecision,(VSLCorrTaskPtr  , const MKL_INT  ))
_mkl_api(int,vslcorrsetinternalprecision,(VSLCorrTaskPtr* , const MKL_INT* ))
_MKL_API(int,VSLCORRSETINTERNALPRECISION,(VSLCorrTaskPtr* , const MKL_INT* ))


_Mkl_Api(int,vslConvSetStart,(VSLConvTaskPtr  , const MKL_INT []))
_mkl_api(int,vslconvsetstart,(VSLConvTaskPtr* , const MKL_INT []))
_MKL_API(int,VSLCONVSETSTART,(VSLConvTaskPtr* , const MKL_INT []))

_Mkl_Api(int,vslCorrSetStart,(VSLCorrTaskPtr  , const MKL_INT []))
_mkl_api(int,vslcorrsetstart,(VSLCorrTaskPtr* , const MKL_INT []))
_MKL_API(int,VSLCORRSETSTART,(VSLCorrTaskPtr* , const MKL_INT []))


_Mkl_Api(int,vslConvSetDecimation,(VSLConvTaskPtr  , const MKL_INT []))
_mkl_api(int,vslconvsetdecimation,(VSLConvTaskPtr* , const MKL_INT []))
_MKL_API(int,VSLCONVSETDECIMATION,(VSLConvTaskPtr* , const MKL_INT []))

_Mkl_Api(int,vslCorrSetDecimation,(VSLCorrTaskPtr  , const MKL_INT []))
_mkl_api(int,vslcorrsetdecimation,(VSLCorrTaskPtr* , const MKL_INT []))
_MKL_API(int,VSLCORRSETDECIMATION,(VSLCorrTaskPtr* , const MKL_INT []))


_Mkl_Api(int,vsldConvExec,(VSLConvTaskPtr  , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []))
_mkl_api(int,vsldconvexec,(VSLConvTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []))
_MKL_API(int,VSLDCONVEXEC,(VSLConvTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []))

_Mkl_Api(int,vslsConvExec,(VSLConvTaskPtr  , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []))
_mkl_api(int,vslsconvexec,(VSLConvTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []))
_MKL_API(int,VSLSCONVEXEC,(VSLConvTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []))

_Mkl_Api(int,vslzConvExec,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_mkl_api(int,vslzconvexec,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_MKL_API(int,VSLZCONVEXEC,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))

_Mkl_Api(int,vslcConvExec,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_mkl_api(int,vslcconvexec,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_MKL_API(int,VSLCCONVEXEC,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))

_Mkl_Api(int,vsldCorrExec,(VSLCorrTaskPtr  , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []))
_mkl_api(int,vsldcorrexec,(VSLCorrTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []))
_MKL_API(int,VSLDCORREXEC,(VSLCorrTaskPtr* , const double [], const MKL_INT [], const double [], const MKL_INT [], double [], const MKL_INT []))

_Mkl_Api(int,vslsCorrExec,(VSLCorrTaskPtr  , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []))
_mkl_api(int,vslscorrexec,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []))
_MKL_API(int,VSLSCORREXEC,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], const float [],  const MKL_INT [], float [],  const MKL_INT []))

_Mkl_Api(int,vslzCorrExec,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_mkl_api(int,vslzcorrexec,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_MKL_API(int,VSLZCORREXEC,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))

_Mkl_Api(int,vslcCorrExec,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_mkl_api(int,vslccorrexec,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_MKL_API(int,VSLCCORREXEC,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))


_Mkl_Api(int,vsldConvExec1D,(VSLConvTaskPtr  , const double [], const MKL_INT  , const double [], const MKL_INT  , double [], const MKL_INT  ))
_mkl_api(int,vsldconvexec1d,(VSLConvTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ))
_MKL_API(int,VSLDCONVEXEC1D,(VSLConvTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ))

_Mkl_Api(int,vslsConvExec1D,(VSLConvTaskPtr  , const float [],  const MKL_INT  , const float [],  const MKL_INT  , float [],  const MKL_INT  ))
_mkl_api(int,vslsconvexec1d,(VSLConvTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))
_MKL_API(int,VSLSCONVEXEC1D,(VSLConvTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))

_Mkl_Api(int,vslzConvExec1D,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ))
_mkl_api(int,vslzconvexec1d,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))
_MKL_API(int,VSLZCONVEXEC1D,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))

_Mkl_Api(int,vslcConvExec1D,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ))
_mkl_api(int,vslcconvexec1d,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))
_MKL_API(int,VSLCCONVEXEC1D,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))

_Mkl_Api(int,vsldCorrExec1D,(VSLCorrTaskPtr  , const double [], const MKL_INT  , const double [], const MKL_INT  , double [], const MKL_INT  ))
_mkl_api(int,vsldcorrexec1d,(VSLCorrTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ))
_MKL_API(int,VSLDCORREXEC1D,(VSLCorrTaskPtr* , const double [], const MKL_INT* , const double [], const MKL_INT* , double [], const MKL_INT* ))

_Mkl_Api(int,vslsCorrExec1D,(VSLCorrTaskPtr  , const float [],  const MKL_INT  , const float [],  const MKL_INT  , float [],  const MKL_INT  ))
_mkl_api(int,vslscorrexec1d,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))
_MKL_API(int,VSLSCORREXEC1D,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))

_Mkl_Api(int,vslzCorrExec1D,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ))
_mkl_api(int,vslzcorrexec1d,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))
_MKL_API(int,VSLZCORREXEC1D,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))

_Mkl_Api(int,vslcCorrExec1D,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ))
_mkl_api(int,vslccorrexec1d,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))
_MKL_API(int,VSLCCORREXEC1D,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))


_Mkl_Api(int,vsldConvExecX,(VSLConvTaskPtr  , const double [], const MKL_INT [], double [], const MKL_INT []))
_mkl_api(int,vsldconvexecx,(VSLConvTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []))
_MKL_API(int,VSLDCONVEXECX,(VSLConvTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []))

_Mkl_Api(int,vslsConvExecX,(VSLConvTaskPtr  , const float [],  const MKL_INT [], float [],  const MKL_INT []))
_mkl_api(int,vslsconvexecx,(VSLConvTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []))
_MKL_API(int,VSLSCONVEXECX,(VSLConvTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []))

_Mkl_Api(int,vslzConvExecX,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_mkl_api(int,vslzconvexecx,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_MKL_API(int,VSLZCONVEXECX,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))

_Mkl_Api(int,vslcConvExecX,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_mkl_api(int,vslcconvexecx,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_MKL_API(int,VSLCCONVEXECX,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))

_Mkl_Api(int,vsldCorrExecX,(VSLCorrTaskPtr  , const double [], const MKL_INT [], double [], const MKL_INT []))
_mkl_api(int,vsldcorrexecx,(VSLCorrTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []))
_MKL_API(int,VSLDCORREXECX,(VSLCorrTaskPtr* , const double [], const MKL_INT [], double [], const MKL_INT []))

_Mkl_Api(int,vslsCorrExecX,(VSLCorrTaskPtr  , const float [],  const MKL_INT [], float [],  const MKL_INT []))
_mkl_api(int,vslscorrexecx,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []))
_MKL_API(int,VSLSCORREXECX,(VSLCorrTaskPtr* , const float [],  const MKL_INT [], float [],  const MKL_INT []))

_Mkl_Api(int,vslzCorrExecX,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_mkl_api(int,vslzcorrexecx,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))
_MKL_API(int,VSLZCORREXECX,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT [], MKL_Complex16 [], const MKL_INT []))

_Mkl_Api(int,vslcCorrExecX,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_mkl_api(int,vslccorrexecx,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))
_MKL_API(int,VSLCCORREXECX,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT [], MKL_Complex8 [],  const MKL_INT []))


_Mkl_Api(int,vsldConvExecX1D,(VSLConvTaskPtr  , const double [], const MKL_INT  , double [], const MKL_INT  ))
_mkl_api(int,vsldconvexecx1d,(VSLConvTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ))
_MKL_API(int,VSLDCONVEXECX1D,(VSLConvTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ))

_Mkl_Api(int,vslsConvExecX1D,(VSLConvTaskPtr  , const float [],  const MKL_INT  , float [],  const MKL_INT  ))
_mkl_api(int,vslsconvexecx1d,(VSLConvTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))
_MKL_API(int,VSLSCONVEXECX1D,(VSLConvTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))

_Mkl_Api(int,vslzConvExecX1D,(VSLConvTaskPtr  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ))
_mkl_api(int,vslzconvexecx1d,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))
_MKL_API(int,VSLZCONVEXECX1D,(VSLConvTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))

_Mkl_Api(int,vslcConvExecX1D,(VSLConvTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ))
_mkl_api(int,vslcconvexecx1d,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))
_MKL_API(int,VSLCCONVEXECX1D,(VSLConvTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))

_Mkl_Api(int,vsldCorrExecX1D,(VSLCorrTaskPtr  , const double [], const MKL_INT  , double [], const MKL_INT  ))
_mkl_api(int,vsldcorrexecx1d,(VSLCorrTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ))
_MKL_API(int,VSLDCORREXECX1D,(VSLCorrTaskPtr* , const double [], const MKL_INT* , double [], const MKL_INT* ))

_Mkl_Api(int,vslsCorrExecX1D,(VSLCorrTaskPtr  , const float [],  const MKL_INT  , float [],  const MKL_INT  ))
_mkl_api(int,vslscorrexecx1d,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))
_MKL_API(int,VSLSCORREXECX1D,(VSLCorrTaskPtr* , const float [],  const MKL_INT* , float [],  const MKL_INT* ))

_Mkl_Api(int,vslzCorrExecX1D,(VSLCorrTaskPtr  , const MKL_Complex16 [], const MKL_INT  , MKL_Complex16 [], const MKL_INT  ))
_mkl_api(int,vslzcorrexecx1d,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))
_MKL_API(int,VSLZCORREXECX1D,(VSLCorrTaskPtr* , const MKL_Complex16 [], const MKL_INT* , MKL_Complex16 [], const MKL_INT* ))

_Mkl_Api(int,vslcCorrExecX1D,(VSLCorrTaskPtr  , const MKL_Complex8 [],  const MKL_INT  , MKL_Complex8 [],  const MKL_INT  ))
_mkl_api(int,vslccorrexecx1d,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))
_MKL_API(int,VSLCCORREXECX1D,(VSLCorrTaskPtr* , const MKL_Complex8 [],  const MKL_INT* , MKL_Complex8 [],  const MKL_INT* ))


/*
//++
//  SUMMARARY STATTISTICS LIBRARY ROUTINES
//--
*/

/*
//  Task constructors
*/
_Mkl_Api(int,vsldSSNewTask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const double [], const MKL_INT []))
_mkl_api(int,vsldssnewtask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const double [], const MKL_INT []))
_MKL_API(int,VSLDSSNEWTASK,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const double [], const double [], const MKL_INT []))

_Mkl_Api(int,vslsSSNewTask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float  [], const float  [], const MKL_INT []))
_mkl_api(int,vslsssnewtask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float  [], const float  [], const MKL_INT []))
_MKL_API(int,VSLSSSNEWTASK,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* , const MKL_INT* , const float  [], const float  [], const MKL_INT []))


/*
// Task editors
*/

/*
// Editor to modify a task parameter
*/
_Mkl_Api(int,vsldSSEditTask,(VSLSSTaskPtr  , const MKL_INT  , const double* ))
_mkl_api(int,vsldssedittask,(VSLSSTaskPtr* , const MKL_INT* , const double* ))
_MKL_API(int,VSLDSSEDITTASK,(VSLSSTaskPtr* , const MKL_INT* , const double* ))

_Mkl_Api(int,vslsSSEditTask,(VSLSSTaskPtr  , const MKL_INT  , const float* ))
_mkl_api(int,vslsssedittask,(VSLSSTaskPtr* , const MKL_INT* , const float* ))
_MKL_API(int,VSLSSSEDITTASK,(VSLSSTaskPtr* , const MKL_INT* , const float* ))

_Mkl_Api(int,vsliSSEditTask,(VSLSSTaskPtr  , const MKL_INT  , const MKL_INT* ))
_mkl_api(int,vslissedittask,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ))
_MKL_API(int,VSLISSEDITTASK,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ))

/*
// Task specific editors
*/

/*
// Editors to modify moments related parameters
*/
_Mkl_Api(int,vsldSSEditMoments,(VSLSSTaskPtr  , double* , double* , double* , double* , double* , double* , double* ))
_mkl_api(int,vsldsseditmoments,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ))
_MKL_API(int,VSLDSSEDITMOMENTS,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ))

_Mkl_Api(int,vslsSSEditMoments,(VSLSSTaskPtr  , float* , float* , float* , float* , float* , float* , float* ))
_mkl_api(int,vslssseditmoments,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ))
_MKL_API(int,VSLSSSEDITMOMENTS,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ))


/*
// Editors to modify sums related parameters
*/
_Mkl_Api(int,vsldSSEditSums,(VSLSSTaskPtr  , double* , double* , double* , double* , double* , double* , double* ))
_mkl_api(int,vsldsseditsums,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ))
_MKL_API(int,VSLDSSEDITSUMS,(VSLSSTaskPtr* , double* , double* , double* , double* , double* , double* , double* ))

_Mkl_Api(int,vslsSSEditSums,(VSLSSTaskPtr  , float* , float* , float* , float* , float* , float* , float* ))
_mkl_api(int,vslssseditsums,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ))
_MKL_API(int,VSLSSSEDITSUMS,(VSLSSTaskPtr* , float* , float* , float* , float* , float* , float* , float* ))


/*
// Editors to modify variance-covariance/correlation matrix related parameters
*/
_Mkl_Api(int,vsldSSEditCovCor,(VSLSSTaskPtr  , double* , double* ,  const MKL_INT* , double* , const MKL_INT* ))
_mkl_api(int,vsldsseditcovcor,(VSLSSTaskPtr* , double* , double* ,  const MKL_INT* , double* , const MKL_INT* ))
_MKL_API(int,VSLDSSEDITCOVCOR,(VSLSSTaskPtr* , double* , double* ,  const MKL_INT* , double* , const MKL_INT* ))

_Mkl_Api(int,vslsSSEditCovCor,(VSLSSTaskPtr  , float* , float* , const MKL_INT* , float* , const MKL_INT* ))
_mkl_api(int,vslssseditcovcor,(VSLSSTaskPtr* , float* , float* , const MKL_INT* , float* , const MKL_INT* ))
_MKL_API(int,VSLSSSEDITCOVCOR,(VSLSSTaskPtr* , float* , float* , const MKL_INT* , float* , const MKL_INT* ))


/*
// Editors to modify cross-product matrix related parameters
*/
_Mkl_Api(int,vsldSSEditCP,(VSLSSTaskPtr  , double* , double* ,  double* , const MKL_INT* ))
_mkl_api(int,vsldsseditcp,(VSLSSTaskPtr* , double* , double* ,  double* , const MKL_INT* ))
_MKL_API(int,VSLDSSEDITCP,(VSLSSTaskPtr* , double* , double* ,  double* , const MKL_INT* ))

_Mkl_Api(int,vslsSSEditCP,(VSLSSTaskPtr  , float* , float* , float* , const MKL_INT* ))
_mkl_api(int,vslssseditcp,(VSLSSTaskPtr* , float* , float* , float* , const MKL_INT* ))
_MKL_API(int,VSLSSSEDITCP,(VSLSSTaskPtr* , float* , float* , float* , const MKL_INT* ))


/*
// Editors to modify partial variance-covariance matrix related parameters
*/
_Mkl_Api(int,vsldSSEditPartialCovCor,(VSLSSTaskPtr  , const MKL_INT [], const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* , const MKL_INT* ))
_mkl_api(int,vsldsseditpartialcovcor,(VSLSSTaskPtr* , const MKL_INT [], const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* , const MKL_INT* ))
_MKL_API(int,VSLDSSEDITPARTIALCOVCOR,(VSLSSTaskPtr* , const MKL_INT [], const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* , const MKL_INT* ))

_Mkl_Api(int,vslsSSEditPartialCovCor,(VSLSSTaskPtr  , const MKL_INT [], const float* , const MKL_INT* , const float* , const MKL_INT* , float* ,  const MKL_INT* , float* ,  const MKL_INT* ))
_mkl_api(int,vslssseditpartialcovcor,(VSLSSTaskPtr* , const MKL_INT [], const float* , const MKL_INT* , const float* , const MKL_INT* , float* ,  const MKL_INT* , float* ,  const MKL_INT* ))
_MKL_API(int,VSLSSSEDITPARTIALCOVCOR,(VSLSSTaskPtr* , const MKL_INT [], const float* , const MKL_INT* , const float* , const MKL_INT* , float* ,  const MKL_INT* , float* ,  const MKL_INT* ))


/*
// Editors to modify quantiles related parameters
*/
_Mkl_Api(int,vsldSSEditQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const double* , double* , double* , const MKL_INT* ))
_mkl_api(int,vsldsseditquantiles,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , double* , const MKL_INT* ))
_MKL_API(int,VSLDSSEDITQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , double* , const MKL_INT* ))

_Mkl_Api(int,vslsSSEditQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const float* , float* , float* , const MKL_INT* ))
_mkl_api(int,vslssseditquantiles,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , float* , const MKL_INT* ))
_MKL_API(int,VSLSSSEDITQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , float* , const MKL_INT* ))


/*
// Editors to modify stream data quantiles related parameters
*/
_Mkl_Api(int,vsldSSEditStreamQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const double* , double* , const MKL_INT* , const double* ))
_mkl_api(int,vsldsseditstreamquantiles,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , const MKL_INT* , const double* ))
_MKL_API(int,VSLDSSEDITSTREAMQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* , const MKL_INT* , const double* ))

_Mkl_Api(int,vslsSSEditStreamQuantiles,(VSLSSTaskPtr  , const MKL_INT* , const float* , float* , const MKL_INT* , const float* ))
_mkl_api(int,vslssseditstreamquantiles,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , const MKL_INT* , const float* ))
_MKL_API(int,VSLSSSEDITSTREAMQUANTILES,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* , const MKL_INT* , const float* ))

/*
// Editors to modify pooled/group variance-covariance matrix related parameters
*/
_Mkl_Api(int,vsldSSEditPooledCovariance,(VSLSSTaskPtr  , const MKL_INT* , double* , double* , const MKL_INT* , double* , double* ))
_mkl_api(int,vsldsseditpooledcovariance,(VSLSSTaskPtr* , const MKL_INT* , double* , double* , const MKL_INT* , double* , double* ))
_MKL_API(int,VSLDSSEDITPOOLEDCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , double* , double* , const MKL_INT* , double* , double* ))

_Mkl_Api(int,vslsSSEditPooledCovariance,(VSLSSTaskPtr  , const MKL_INT* , float* , float* , const MKL_INT* , float* , float* ))
_mkl_api(int,vslssseditpooledcovariance,(VSLSSTaskPtr* , const MKL_INT* , float* , float* , const MKL_INT* , float* , float* ))
_MKL_API(int,VSLSSSEDITPOOLEDCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , float* , float* , const MKL_INT* , float* , float* ))


/*
// Editors to modify robust variance-covariance matrix related parameters
*/
_Mkl_Api(int,vsldSSEditRobustCovariance,(VSLSSTaskPtr  , const MKL_INT* , const MKL_INT* ,  const double* , double* , double* ))
_mkl_api(int,vsldsseditrobustcovariance,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const double* , double* , double* ))
_MKL_API(int,VSLDSSEDITROBUSTCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const double* , double* , double* ))

_Mkl_Api(int,vslsSSEditRobustCovariance,(VSLSSTaskPtr  , const MKL_INT* , const MKL_INT* ,  const float* , float* , float* ))
_mkl_api(int,vslssseditrobustcovariance,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const float* , float* , float* ))
_MKL_API(int,VSLSSSEDITROBUSTCOVARIANCE,(VSLSSTaskPtr* , const MKL_INT* , const MKL_INT* ,  const float* , float* , float* ))


/*
// Editors to modify outliers detection parameters
*/
_Mkl_Api(int,vsldSSEditOutliersDetection,(VSLSSTaskPtr  , const MKL_INT* , const double* , double* ))
_mkl_api(int,vsldsseditoutliersdetection,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* ))
_MKL_API(int,VSLDSSEDITOUTLIERSDETECTION,(VSLSSTaskPtr* , const MKL_INT* , const double* , double* ))

_Mkl_Api(int,vslsSSEditOutliersDetection,(VSLSSTaskPtr  , const MKL_INT* , const float* , float* ))
_mkl_api(int,vslssseditoutliersdetection,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* ))
_MKL_API(int,VSLSSSEDITOUTLIERSDETECTION,(VSLSSTaskPtr* , const MKL_INT* , const float* , float* ))

/*
// Editors to modify missing values support parameters
*/
_Mkl_Api(int,vsldSSEditMissingValues,(VSLSSTaskPtr  , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* ))
_mkl_api(int,vsldsseditmissingvalues,(VSLSSTaskPtr* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* ))
_MKL_API(int,VSLDSSEDITMISSINGVALUES,(VSLSSTaskPtr* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , const double* , const MKL_INT* , double* , const MKL_INT* , double* ))

_Mkl_Api(int,vslsSSEditMissingValues,(VSLSSTaskPtr  , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , float* , const MKL_INT* , float* ))
_mkl_api(int,vslssseditmissingvalues,(VSLSSTaskPtr* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , float* , const MKL_INT* , float* ))
_MKL_API(int,VSLSSSEDITMISSINGVALUES,(VSLSSTaskPtr* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , const float* , const MKL_INT* , float* , const MKL_INT* , float* ))

/*
// Editors to modify matrixparametrization parameters
*/
_Mkl_Api(int,vsldSSEditCorParameterization,(VSLSSTaskPtr  , const double* , const MKL_INT* , double* , const MKL_INT* ))
_mkl_api(int,vsldsseditcorparameterization,(VSLSSTaskPtr* , const double* , const MKL_INT* , double* , const MKL_INT* ))
_MKL_API(int,VSLDSSEDITCORPARAMETERIZATION,(VSLSSTaskPtr* , const double* , const MKL_INT* , double* , const MKL_INT* ))

_Mkl_Api(int,vslsSSEditCorParameterization,(VSLSSTaskPtr  , const float* , const MKL_INT* , float* , const MKL_INT* ))
_mkl_api(int,vslssseditcorparameterization,(VSLSSTaskPtr* , const float* , const MKL_INT* , float* , const MKL_INT* ))
_MKL_API(int,VSLSSSEDITCORPARAMETERIZATION,(VSLSSTaskPtr* , const float* , const MKL_INT* , float* , const MKL_INT* ))


/*
// Compute routines
*/
_Mkl_Api(int,vsldSSCompute,(VSLSSTaskPtr  , const unsigned MKL_INT64  , const MKL_INT  ))
_mkl_api(int,vsldsscompute,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ))
_MKL_API(int,VSLDSSCOMPUTE,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ))

_Mkl_Api(int,vslsSSCompute,(VSLSSTaskPtr  , const unsigned MKL_INT64  , const MKL_INT  ))
_mkl_api(int,vslssscompute,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ))
_MKL_API(int,VSLSSSCOMPUTE,(VSLSSTaskPtr* , const unsigned MKL_INT64* , const MKL_INT* ))


/*
// Task destructor
*/
_Mkl_Api(int,vslSSDeleteTask,(VSLSSTaskPtr* ))
_mkl_api(int,vslssdeletetask,(VSLSSTaskPtr* ))
_MKL_API(int,VSLSSDELETETASK,(VSLSSTaskPtr* ))

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VSL_FUNCTIONS_H__ */
/*******************************************************************************
* Copyright (c) 2015-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef _MKL_DNN_TYPES_H
#define _MKL_DNN_TYPES_H


#if defined(__cplusplus_cli)
struct _uniPrimitive_s {};
struct _dnnLayout_s {};
#endif

typedef struct _uniPrimitive_s* dnnPrimitive_t;
typedef struct _dnnLayout_s* dnnLayout_t;
typedef void* dnnPrimitiveAttributes_t;

#define DNN_MAX_DIMENSION       32
#define DNN_QUERY_MAX_LENGTH    128

typedef enum {
    E_SUCCESS                   =  0,
    E_INCORRECT_INPUT_PARAMETER = -1,
    E_UNEXPECTED_NULL_POINTER   = -2,
    E_MEMORY_ERROR              = -3,
    E_UNSUPPORTED_DIMENSION     = -4,
    E_UNIMPLEMENTED             = -127
} dnnError_t;

typedef enum {
    /** GEMM base convolution (unimplemented) */
    dnnAlgorithmConvolutionGemm,
    /** Direct convolution */
    dnnAlgorithmConvolutionDirect,
    /** FFT based convolution (unimplemented) */
    dnnAlgorithmConvolutionFFT,
    /** Maximum pooling */
    dnnAlgorithmPoolingMax,
    /** Minimum pooling */
    dnnAlgorithmPoolingMin,
    /** Average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvgExcludePadding,
    /** Alias for average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvg = dnnAlgorithmPoolingAvgExcludePadding,
    /** Average pooling (padded values are taken into account) */
    dnnAlgorithmPoolingAvgIncludePadding
} dnnAlgorithm_t;

typedef enum {
    dnnResourceSrc            = 0,
    dnnResourceFrom           = 0,
    dnnResourceDst            = 1,
    dnnResourceTo             = 1,
    dnnResourceFilter         = 2,
    dnnResourceScaleShift     = 2,
    dnnResourceBias           = 3,
    dnnResourceMean           = 3,
    dnnResourceDiffSrc        = 4,
    dnnResourceDiffFilter     = 5,
    dnnResourceDiffScaleShift = 5,
    dnnResourceDiffBias       = 6,
    dnnResourceVariance       = 6,
    dnnResourceDiffDst        = 7,
    dnnResourceWorkspace      = 8,
    dnnResourceMultipleSrc    = 16,
    dnnResourceMultipleDst    = 24,
    dnnResourceNumber         = 32
} dnnResourceType_t;

typedef enum {
    dnnBorderZeros          = 0x0,
    dnnBorderZerosAsymm     = 0x100,
    dnnBorderExtrapolation  = 0x3
} dnnBorder_t;

typedef enum {
    dnnUseInputMeanVariance = 0x1U,
    dnnUseScaleShift        = 0x2U
} dnnBatchNormalizationFlag_t;

#endif
/*******************************************************************************
* Copyright (c) 2015-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef _MKL_DNN_TYPES_H
#define _MKL_DNN_TYPES_H


#if defined(__cplusplus_cli)
struct _uniPrimitive_s {};
struct _dnnLayout_s {};
#endif

typedef struct _uniPrimitive_s* dnnPrimitive_t;
typedef struct _dnnLayout_s* dnnLayout_t;
typedef void* dnnPrimitiveAttributes_t;

#define DNN_MAX_DIMENSION       32
#define DNN_QUERY_MAX_LENGTH    128

typedef enum {
    E_SUCCESS                   =  0,
    E_INCORRECT_INPUT_PARAMETER = -1,
    E_UNEXPECTED_NULL_POINTER   = -2,
    E_MEMORY_ERROR              = -3,
    E_UNSUPPORTED_DIMENSION     = -4,
    E_UNIMPLEMENTED             = -127
} dnnError_t;

typedef enum {
    /** GEMM base convolution (unimplemented) */
    dnnAlgorithmConvolutionGemm,
    /** Direct convolution */
    dnnAlgorithmConvolutionDirect,
    /** FFT based convolution (unimplemented) */
    dnnAlgorithmConvolutionFFT,
    /** Maximum pooling */
    dnnAlgorithmPoolingMax,
    /** Minimum pooling */
    dnnAlgorithmPoolingMin,
    /** Average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvgExcludePadding,
    /** Alias for average pooling (padded values are not taken into account) */
    dnnAlgorithmPoolingAvg = dnnAlgorithmPoolingAvgExcludePadding,
    /** Average pooling (padded values are taken into account) */
    dnnAlgorithmPoolingAvgIncludePadding
} dnnAlgorithm_t;

typedef enum {
    dnnResourceSrc            = 0,
    dnnResourceFrom           = 0,
    dnnResourceDst            = 1,
    dnnResourceTo             = 1,
    dnnResourceFilter         = 2,
    dnnResourceScaleShift     = 2,
    dnnResourceBias           = 3,
    dnnResourceMean           = 3,
    dnnResourceDiffSrc        = 4,
    dnnResourceDiffFilter     = 5,
    dnnResourceDiffScaleShift = 5,
    dnnResourceDiffBias       = 6,
    dnnResourceVariance       = 6,
    dnnResourceDiffDst        = 7,
    dnnResourceWorkspace      = 8,
    dnnResourceMultipleSrc    = 16,
    dnnResourceMultipleDst    = 24,
    dnnResourceNumber         = 32
} dnnResourceType_t;

typedef enum {
    dnnBorderZeros          = 0x0,
    dnnBorderZerosAsymm     = 0x100,
    dnnBorderExtrapolation  = 0x3
} dnnBorder_t;

typedef enum {
    dnnUseInputMeanVariance = 0x1U,
    dnnUseScaleShift        = 0x2U
} dnnBatchNormalizationFlag_t;

#endif
/*******************************************************************************
* Copyright (c) 2015-2017, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

#ifndef _MKL_DNN_H
#define _MKL_DNN_H



#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * F32 section: single precision
 ******************************************************************************/

dnnError_t dnnLayoutCreate_F32(
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t dnnLayoutCreateFromPrimitive_F32(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);

/** Returns the size of buffer required to serialize dnnLayout_t structure. */
size_t dnnLayoutSerializationBufferSize_F32();

/** Serializes given @p layout into buffer @p buf. User-provided buffer @p buf
 * should have enough space to store dnnLayout_t structure.
 * @sa dnnLayoutSerializationBufferSize_F32 */
dnnError_t dnnLayoutSerialize_F32(const dnnLayout_t layout, void *buf);

/** Creates new layout restored from previously serialized one. */
dnnError_t dnnLayoutDeserialize_F32(dnnLayout_t *pLayout, const void *buf);

size_t dnnLayoutGetMemorySize_F32(
        const dnnLayout_t layout);
int dnnLayoutCompare_F32(
        const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t dnnAllocateBuffer_F32(
        void **pPtr, dnnLayout_t layout);
dnnError_t dnnReleaseBuffer_F32(
        void *ptr);
dnnError_t dnnLayoutDelete_F32(
        dnnLayout_t layout);

dnnError_t dnnPrimitiveAttributesCreate_F32(
        dnnPrimitiveAttributes_t *attributes);
dnnError_t dnnPrimitiveAttributesDestroy_F32(
        dnnPrimitiveAttributes_t attributes);
dnnError_t dnnPrimitiveGetAttributes_F32(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);

dnnError_t dnnExecute_F32(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnExecuteAsync_F32(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnWaitFor_F32(
        dnnPrimitive_t primitive);
dnnError_t dnnDelete_F32(
        dnnPrimitive_t primitive);

dnnError_t dnnConversionCreate_F32(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
dnnError_t dnnConversionExecute_F32(
        dnnPrimitive_t conversion, void *from, void *to);

dnnError_t dnnSumCreate_F32(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, float *coefficients);
dnnError_t dnnConcatCreate_F32(
        dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
dnnError_t dnnSplitCreate_F32(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t dnnScaleCreate_F32(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float alpha);

dnnError_t dnnConvolutionCreateForward_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardData_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardFilter_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);

dnnError_t dnnGroupsConvolutionCreateForward_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateForwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardData_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardBias_F32(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);

dnnError_t dnnReLUCreateForward_F32(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float negativeSlope);
dnnError_t dnnReLUCreateBackward_F32(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope);

dnnError_t dnnLRNCreateForward_F32(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);
dnnError_t dnnLRNCreateBackward_F32(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);

dnnError_t dnnBatchNormalizationCreateForward_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);
dnnError_t dnnBatchNormalizationCreateBackwardData_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps);

dnnError_t dnnBatchNormalizationCreateForward_v2_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps,
        unsigned int flags);
dnnError_t dnnBatchNormalizationCreateBackward_v2_F32(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, float eps,
        unsigned int flags);

dnnError_t dnnPoolingCreateForward_F32(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnPoolingCreateBackward_F32(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);

dnnError_t dnnInnerProductCreateForward_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateForwardBias_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardData_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardFilter_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardBias_F32(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]);

/*******************************************************************************
 * F64 section: double precision
 ******************************************************************************/

dnnError_t dnnLayoutCreate_F64 (
        dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
dnnError_t dnnLayoutCreateFromPrimitive_F64(
        dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);

/** Returns the size of buffer required to serialize dnnLayout_t structure. */
size_t dnnLayoutSerializationBufferSize_F64();

/** Serializes given @p layout into buffer @p buf. User-provided buffer @p buf
 * should have enough space to store dnnLayout_t structure.
 * @sa dnnLayoutSerializationBufferSize_F64 */
dnnError_t dnnLayoutSerialize_F64(const dnnLayout_t layout, void *buf);

/** Creates new layout restored from previously serialized one. */
dnnError_t dnnLayoutDeserialize_F64(dnnLayout_t *pLayout, const void *buf);

size_t dnnLayoutGetMemorySize_F64(
        const dnnLayout_t layout);
int dnnLayoutCompare_F64(
        const dnnLayout_t l1, const dnnLayout_t l2);
dnnError_t dnnAllocateBuffer_F64(
        void **pPtr, dnnLayout_t layout);
dnnError_t dnnReleaseBuffer_F64(
        void *ptr);
dnnError_t dnnLayoutDelete_F64(
        dnnLayout_t layout);

dnnError_t dnnPrimitiveAttributesCreate_F64(
        dnnPrimitiveAttributes_t *attributes);
dnnError_t dnnPrimitiveAttributesDestroy_F64(
        dnnPrimitiveAttributes_t attributes);
dnnError_t dnnPrimitiveGetAttributes_F64(
        dnnPrimitive_t primitive,
        dnnPrimitiveAttributes_t *attributes);

dnnError_t dnnExecute_F64(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnExecuteAsync_F64(
        dnnPrimitive_t primitive, void *resources[]);
dnnError_t dnnWaitFor_F64(
        dnnPrimitive_t primitive);
dnnError_t dnnDelete_F64(
        dnnPrimitive_t primitive);

dnnError_t dnnConversionCreate_F64(
        dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
dnnError_t dnnConversionExecute_F64(
        dnnPrimitive_t conversion, void *from, void *to);

dnnError_t dnnSumCreate_F64(
        dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
        dnnLayout_t layout, double *coefficients);
dnnError_t dnnConcatCreate_F64(
         dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
dnnError_t dnnSplitCreate_F64(
        dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
        dnnLayout_t layout, size_t dstChannelSize[]);
dnnError_t dnnScaleCreate_F64(
        dnnPrimitive_t *pScale,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double alpha);

dnnError_t dnnConvolutionCreateForward_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateForwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardData_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardFilter_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnConvolutionCreateBackwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t dimension, const size_t dstSize[]);

dnnError_t dnnGroupsConvolutionCreateForward_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateForwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardData_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
        const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnGroupsConvolutionCreateBackwardBias_F64(
        dnnPrimitive_t* pConvolution,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t algorithm,
        size_t groups, size_t dimension, const size_t dstSize[]);

dnnError_t dnnReLUCreateForward_F64(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double negativeSlope);
dnnError_t dnnReLUCreateBackward_F64(
        dnnPrimitive_t* pRelu,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, double negativeSlope);

dnnError_t dnnLRNCreateForward_F64(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);
dnnError_t dnnLRNCreateBackward_F64(
        dnnPrimitive_t* pLrn,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);

dnnError_t dnnBatchNormalizationCreateForward_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);
dnnError_t dnnBatchNormalizationCreateBackwardData_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps);

dnnError_t dnnBatchNormalizationCreateForward_v2_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps,
        unsigned int flags);
dnnError_t dnnBatchNormalizationCreateBackward_v2_F64(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout, double eps,
        unsigned int flags);

dnnError_t dnnPoolingCreateForward_F64(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);
dnnError_t dnnPoolingCreateBackward_F64(
        dnnPrimitive_t* pPooling,
        dnnPrimitiveAttributes_t attributes,
        dnnAlgorithm_t op,
        const dnnLayout_t srcLayout,
        const size_t kernelSize[], const size_t kernelStride[],
        const int inputOffset[], const dnnBorder_t borderType);

dnnError_t dnnInnerProductCreateForward_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateForwardBias_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardData_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardFilter_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t srcSize[],
        size_t outputChannels);
dnnError_t dnnInnerProductCreateBackwardBias_F64(
        dnnPrimitive_t *pInnerProduct,
        dnnPrimitiveAttributes_t attributes,
        size_t dimensions,
        const size_t dstSize[]);

#ifdef __cplusplus
}
#endif

#endif
