#ifndef __NUCLEAREOS_H__
#define __NUCLEAREOS_H__



#include "CUFLU.h"


#define NUC_TABLE_NVAR       16     // number of variables in the EoS table lookup
#define NUC_TABLE_NPTR       10     // number of table pointers to be sent to GPU


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
#define NUC_AUX_NTORE         1     // AuxArray_Int: ntemp/neps
#define NUC_AUX_NYE           2     // AuxArray_Int: nye
#define NUC_AUX_NRHO_MODE     3     // AuxArray_Int: nrho_mode
#define NUC_AUX_NMODE         4     // AuxArray_Int: nmode
#define NUC_AUX_NYE_MODE      5     // AuxArray_Int: nye_mode


// table indices
#define NUC_TAB_ALL           0     // alltables
#define NUC_TAB_ALL_MODE      1     // alltables_mode
#define NUC_TAB_RHO           2     // logrho
#define NUC_TAB_TORE          3     // logtemp/logenergy
#define NUC_TAB_YE            4     // yes
#define NUC_TAB_RHO_MODE      5     // logrho_mode 
#define NUC_TAB_ENTE_MODE     6     // logenergy_mode/logtemp_mode
#define NUC_TAB_ENTR_MODE     7     // entr_mode
#define NUC_TAB_PRES_MODE     8     // logprss_mode
#define NUC_TAB_YE_MODE       9     // yes_mode


// EoS table default mode
#define NUC_TABLE_MODE_TEMP   0     // temperature mode
#define NUC_TABLE_MODE_ENGY   1     // energy mode


// EoS modes
#define NUC_MODE_ENGY         0     // energy mode
#define NUC_MODE_TEMP         1     // temperature mode
#define NUC_MODE_ENTR         2     // entropy mode
#define NUC_MODE_PRES         3     // pressure mode



#endif // __NUCLEAREOS_H__
