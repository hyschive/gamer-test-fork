#ifndef __NUCLEAREOS_H__
#define __NUCLEAREOS_H__



#include "CUFLU.h"


#define NUC_TABLE_NVAR       16     // number of variables in the EoS table lookup
#define NUC_TABLE_NPTR        8     // number of table pointers to be sent to GPU


// auxiliary array indices
#define NUC_AUX_ESHIFT        0     // AuxArray_Flt: energy_shift
#define NUC_AUX_DENS2CGS      1     // AuxArray_Flt: convert density    to cgs
#define NUC_AUX_PRES2CGS      2     // AuxArray_Flt: convert pressure   to cgs
#define NUC_AUX_VSQR2CGS      3     // AuxArray_Flt: convert velocity^2 to cgs
#define NUC_AUX_PRES2CODE     4     // AuxArray_Flt: convert pressure   to code unit
#define NUC_AUX_VSQR2CODE     5     // AuxArray_Flt: convert velocity^2 to code unit
#define NUC_AUX_KELVIN2MEV    6     // AuxArray_Flt: convert kelvin     to MeV
#define NUC_AUX_MEV2KELVIN    7     // AuxArray_Flt: convert MeV        to kelvin

#define NUC_AUX_NRHO          0     // AuxArray_Int: nrho
#define NUC_AUX_NEPS          1     // AuxArray_Int: neps
#define NUC_AUX_NYE           2     // AuxArray_Int: nye
#define NUC_AUX_NMODE         3     // AuxArray_Int: nmode


// table indices
#define NUC_TAB_ALL           0     // alltables
#define NUC_TAB_ALL_MODE      1     // alltables_mode
#define NUC_TAB_RHO           2     // logrho
#define NUC_TAB_EPS           3     // logeps
#define NUC_TAB_YE            4     // yes
#define NUC_TAB_TEMP_MODE     5     // logtemp_mode
#define NUC_TAB_ENTR_MODE     6     // entr_mode
#define NUC_TAB_PRES_MODE     7     // logprss_mode


// EoS modes
#define NUC_MODE_ENGY         0     // energy mode
#define NUC_MODE_TEMP         1     // temperature mode
#define NUC_MODE_ENTR         2     // entropy mode
#define NUC_MODE_PRES         3     // pressure mode



#endif // __NUCLEAREOS_H__
