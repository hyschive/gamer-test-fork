#include "NuclearEoS.h"
#ifdef __CUDACC__
#include "CUAPI.h"
#include "CUFLU_Shared_FluUtility.cu"
#endif

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

#include "NuclearEoS.cu"
__device__ static real EoS_DensEint2Pres_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                                  const double AuxArray_Flt[], const int AuxArray_Int[],
                                                  const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] );
__device__ static void EoS_General_Nuclear( const int Mode, real Out[], const real In[], const double AuxArray_Flt[],
                                            const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] );

#else

// global variables
int    g_nrho;
int    g_nye;
int    g_nrho_mode;
int    g_nmode;
int    g_nye_mode;
double g_energy_shift;

real *g_alltables;
real *g_alltables_mode;
real *g_logrho;
real *g_yes;
real *g_logrho_mode;
real *g_entr_mode;
real *g_logprss_mode;
real *g_yes_mode;

#if   ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
int    g_ntemp;
real   *g_logtemp;
real   *g_logeps_mode;
#elif ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
int    g_neps;
real   *g_logeps;
real   *g_logtemp_mode;
#endif


// prototypes
void nuc_eos_C_short( const real xrho, real *xtemp, const real xye,
                      real *xenr, real *xent, real *xprs,
                      real *xcs2, real *xmunu, const real energy_shift,
                      const int nrho, const int ntoreps, const int nye,
                      const int nrho_mode, const int nmode, const int nye_mode,
                      const real *alltables, const real *alltables_mode,
                      const real *logrho, const real *logtoreps, const real *yes, const real *logrho_mode,
                      const real *logepsort_mode, const real *entr_mode, const real *logprss_mode, const real *yes_mode,
                      const int keymode, int *keyerr, const real rfeps );
void nuc_eos_C_ReadTable( char *nuceos_table_name );
void CUAPI_PassNuclearEoSTable2GPU();

static real EoS_DensEint2Pres_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] );
static void EoS_General_Nuclear( const int Mode, real Out[], const real In[], const double AuxArray_Flt[],
                                 const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] );
#endif // #ifdef __CUDACC__ ... else ...




/********************************************************
1. Nuclear EoS (EOS_NUCLEAR)

2. This file is shared by both CPU and GPU

   GPU_EoS_Nuclear.cu -> CPU_EoS_Nuclear.cpp

3. Three steps are required to implement an EoS

   I.   Set EoS auxiliary arrays
   II.  Implement EoS conversion functions
   III. Set EoS initialization functions

4. All EoS conversion functions must be thread-safe and
   not use any global variable

5. When an EoS conversion function fails, it is recommended
   to return NAN in order to trigger auto-correction such as
   "OPT__1ST_FLUX_CORR" and "AUTO_REDUCE_DT"
********************************************************/



// =============================================
// I. Set EoS auxiliary arrays
// =============================================

//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_SetAuxArray_Nuclear
// Description :  Set the auxiliary arrays AuxArray_Flt/Int[]
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//                2. AuxArray_Flt/Int[] have the size of EOS_NAUX_MAX defined in Macro.h (default = 20)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
#ifndef __CUDACC__
void EoS_SetAuxArray_Nuclear( double AuxArray_Flt[], int AuxArray_Int[] )
{

   AuxArray_Flt[NUC_AUX_ESHIFT    ] = g_energy_shift;
   AuxArray_Flt[NUC_AUX_DENS2CGS  ] = UNIT_D;
   AuxArray_Flt[NUC_AUX_PRES2CGS  ] = UNIT_P;
   AuxArray_Flt[NUC_AUX_VSQR2CGS  ] = SQR( UNIT_V );
   AuxArray_Flt[NUC_AUX_PRES2CODE ] = 1.0 / UNIT_P;
   AuxArray_Flt[NUC_AUX_VSQR2CODE ] = 1.0 / SQR(UNIT_V);
   AuxArray_Flt[NUC_AUX_KELVIN2MEV] = Const_kB_eV*1.0e-6;
   AuxArray_Flt[NUC_AUX_MEV2KELVIN] = 1.0 / AuxArray_Flt[NUC_AUX_KELVIN2MEV];

   AuxArray_Int[NUC_AUX_NRHO      ] = g_nrho;
#if   ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   AuxArray_Int[NUC_AUX_NTORE     ] = g_ntemp;
#elif ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
   AuxArray_Int[NUC_AUX_NTORE     ] = g_neps;
#endif
   AuxArray_Int[NUC_AUX_NYE       ] = g_nye;
   AuxArray_Int[NUC_AUX_NRHO_MODE ] = g_nrho_mode;
   AuxArray_Int[NUC_AUX_NMODE     ] = g_nmode;
   AuxArray_Int[NUC_AUX_NYE_MODE  ] = g_nye_mode;

} // FUNCTION : EoS_SetAuxArray_Nuclear
#endif // #ifndef __CUDACC__



// =============================================
// II. Implement EoS conversion functions
//     (1) EoS_DensEint2Pres_*
//     (2) EoS_DensPres2Eint_*
//     (3) EoS_DensPres2CSqr_*
//     (4) EoS_DensEint2Temp_*
//     (5) EoS_DensTemp2Pres_*
//     (6) EoS_General_*
// =============================================

#ifdef GAMER_DEBUG
//-------------------------------------------------------------------------------------------------------
// Function    :  Nuc_Overflow
// Description :  Check whether the input floating-point value is finite
//
// Note        :  1. Definition of "finite" --> not NaN, Inf, -Inf
//                2. Overflow may occur during unit conversion for improper code units
//
// Parameter   :  x : Floating-point value to be checked
//
// Return      :  true  : infinite
//                false : finite
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE static
bool Nuc_Overflow( const real x )
{

   if ( x != x  ||  x < -__FLT_MAX__  ||  x > __FLT_MAX__ )    return true;
   else                                                        return false;

} // FUNCTION : Nuc_Overflow
#endif // #ifdef GAMER_DEBUG



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Pres_Nuclear
// Description :  Convert gas mass density and internal energy density to gas pressure
//
// Note        :  1. Internal energy density here is per unit volume instead of per unit mass
//                2. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//                3. Also return temperature and entropy if ExtraInOut[] != NULL
//
// Parameter   :  Dens_Code    : Gas mass density            (in code unit)
//                Eint_Code    : Gas internal energy density (in code unit)
//                Passive_Code : Passive scalars             (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//                ExtraInOut   : Array to store extra output variables
//                               --> [0] = gas temperature (in kelvin    )
//                                   [1] = entropy         (in kB/baryon )
//                               --> Optional and only used when it is not NULL
//
// Return      :  Gas pressure (in code unit), ExtraInOut[] (optional; see above)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Pres_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] )
{

// check
#  ifdef GAMER_DEBUG
#  if ( NCOMP_PASSIVE > 0 )
   if ( Passive_Code == NULL )   printf( "ERROR : Passive_Code == NULL in %s !!\n", __FUNCTION__ );
#  endif
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );

   if ( Hydro_CheckNegative(Dens_Code) )
      printf( "ERROR : invalid input density (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Dens_Code, __FILE__, __LINE__, __FUNCTION__ );

// still require Eint>0 for the nuclear EoS
   if ( Hydro_CheckNegative(Eint_Code) )
      printf( "ERROR : invalid input internal energy (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Eint_Code, __FILE__, __LINE__, __FUNCTION__ );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS  ];
   const real sEint2CGS   = AuxArray_Flt[NUC_AUX_VSQR2CGS  ];
   const real Pres2Code   = AuxArray_Flt[NUC_AUX_PRES2CODE ];
   const real MeV2Kelvin  = AuxArray_Flt[NUC_AUX_MEV2KELVIN];

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO     ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE    ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE      ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE    ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE ];

   int  Mode      = NUC_MODE_ENGY;
   real Dens_CGS  = Dens_Code * Dens2CGS;
   real sEint_CGS = ( Eint_Code / Dens_Code ) * sEint2CGS - EnergyShift;
   real Ye        = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
   real Temp_MeV  = NULL_REAL;
   real Entr      = NULL_REAL;
   real Pres_CGS  = NULL_REAL;
   real mu_nu     = NULL_REAL;
   real Useless   = NULL_REAL;

   int  Err       = NULL_INT;
#ifdef FLOAT8
   const real Tolerance = 1e-10;
#else
   const real Tolerance = 1e-6;
#endif


// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, __FUNCTION__ );

   //if ( Nuc_Overflow(sEint_CGS) )
   //   printf( "ERROR : EoS overflow (sEint_CGS %13.7e, Eint_Code %13.7e, Dens_Code %13.7e, sEint2CGS %13.7e) in %s() !!\n",
   //           sEint_CGS, Eint_Code, Dens_Code, sEint2CGS, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG


// invoke the nuclear EoS driver
   nuc_eos_C_short( Dens_CGS, &Temp_MeV, Ye, &sEint_CGS, &Entr, &Pres_CGS, &Useless, &Useless,
                    EnergyShift, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE], 
                    Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_ENTE_MODE], Table[NUC_TAB_ENTR_MODE],  Table[NUC_TAB_PRES_MODE],
                    Table[NUC_TAB_YE_MODE], Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )
   {
      Temp_MeV = NAN;
      Entr     = NAN;
      Pres_CGS = NAN;
      mu_nu    = NAN;
   }

   const real Temp_Kelv = Temp_MeV * MeV2Kelvin;
   const real Pres_Code = Pres_CGS * Pres2Code;


// final check
#  ifdef GAMER_DEBUG
   if ( Hydro_CheckNegative(Temp_Kelv) )
   {
      printf( "ERROR : invalid output temperature (%13.7e K) in %s() !!\n", Temp_Kelv, __FUNCTION__ );
      printf( "        Dens=%13.7e code units, Eint=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Eint_Code, Ye, Mode );
      printf( "        EoS error code: %d\n", Err );
   }

   if ( Hydro_CheckNegative(Entr) )
   {
      printf( "ERROR : invalid output entropy (%13.7e kB/baryon) in %s() !!\n", Entr, __FUNCTION__ );
      printf( "        Dens=%13.7e code units, Eint=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Eint_Code, Ye, Mode );
      printf( "        EoS error code: %d\n", Err );
   }

   if ( Hydro_CheckNegative(Pres_Code) )
   {
      printf( "ERROR : invalid output pressure (%13.7e code units) in %s() !!\n", Pres_Code, __FUNCTION__ );
      printf( "        Dens=%13.7e code units, Eint=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Eint_Code, Ye, Mode );
      printf( "        EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG


// store extra output variables if required
   if ( ExtraInOut != NULL )
   {
      ExtraInOut[0] = Temp_Kelv;
      ExtraInOut[1] = Entr;
      ExtraInOut[2] = mu_nu;
   }


   return Pres_Code;

} // FUNCTION : EoS_DensEint2Pres_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensPres2Eint_Nuclear
// Description :  Convert gas mass density and pressure to gas internal energy density
//
// Note        :  1. See EoS_DensEint2Pres_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density (in code unit)
//                Pres_Code    : Gas pressure     (in code unit)
//                Passive_Code : Passive scalars  (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//                ExtraInOut   : Array to store extra input and output variables if required
//                               --> Optional and only used when it is not NULL
//
// Return      :  Gas internal energy density (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensPres2Eint_Nuclear( const real Dens_Code, const real Pres_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] )
{

// check
#  ifdef GAMER_DEBUG
#  if ( NCOMP_PASSIVE > 0 )
   if ( Passive_Code == NULL )   printf( "ERROR : Passive_Code == NULL in %s !!\n", __FUNCTION__ );
#  endif
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );

   if ( Hydro_CheckNegative(Dens_Code) )
      printf( "ERROR : invalid input density (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Dens_Code, __FILE__, __LINE__, __FUNCTION__ );

   if ( Hydro_CheckNegative(Pres_Code) )
      printf( "ERROR : invalid input pressure (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Pres_Code, __FILE__, __LINE__, __FUNCTION__ );
#  endif // GAMER_DEBUG



   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT   ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS ];
   const real Pres2CGS    = AuxArray_Flt[NUC_AUX_PRES2CGS ];
   const real sEint2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE];

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO     ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE    ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE      ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE    ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE ];

   int  Mode      = NUC_MODE_PRES;
   real Dens_CGS  = Dens_Code * Dens2CGS;
   real Pres_CGS  = Pres_Code * Pres2CGS;
   real Ye        = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
   real sEint_CGS = NULL_REAL;
   real Useless   = NULL_REAL;
   int  Err       = NULL_INT;
#ifdef FLOAT8
   const real Tolerance = 1e-10;
#else
   const real Tolerance = 1e-6;
#endif


// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, __FUNCTION__ );

   if ( Nuc_Overflow(Pres_CGS) )
      printf( "ERROR : EoS overflow (Pres_CGS %13.7e, Pres_Code %13.7e, Pres2CGS %13.7e) in %s() !!\n",
              Pres_CGS, Pres_Code, Pres2CGS, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG


// invoke the nuclear EoS driver
   nuc_eos_C_short( Dens_CGS, &Useless, Ye, &sEint_CGS, &Useless, &Pres_CGS, &Useless, &Useless,
                    EnergyShift, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE], 
                    Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_ENTE_MODE], Table[NUC_TAB_ENTR_MODE],  Table[NUC_TAB_PRES_MODE],
                    Table[NUC_TAB_YE_MODE], Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )  sEint_CGS = NAN;

   const real Eint_Code = (  ( sEint_CGS + EnergyShift ) * sEint2Code  ) * Dens_Code;


// final check
#  ifdef GAMER_DEBUG
// still require Eint>0 for the nuclear EoS
   if ( Hydro_CheckNegative(Eint_Code) )
   {
      printf( "ERROR : invalid output internal energy density (%13.7e code units) in %s() !!\n", Eint_Code, __FUNCTION__ );
      printf( "        Dens=%13.7e code units, Pres=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Pres_Code, Ye, Mode );
      printf( "        EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG


// store extra output variables if required
   /*
   if ( ExtraInOut != NULL )
   {
      ExtraInOut[...] = ...;
   }
   */


   return Eint_Code;

} // FUNCTION : EoS_DensPres2Eint_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensPres2CSqr_Nuclear
// Description :  Convert gas mass density and pressure to sound speed squared
//
// Note        :  1. See EoS_DensEint2Pres_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density (in code unit)
//                Pres_Code    : Gas pressure     (in code unit)
//                Passive_Code : Passive scalars  (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//                ExtraInOut   : Array to store extra input and output variables if required
//                               --> Optional and only used when it is not NULL
//
// Return      :  Sound speed square (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensPres2CSqr_Nuclear( const real Dens_Code, const real Pres_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] )
{

// check
#  ifdef GAMER_DEBUG
#  if ( NCOMP_PASSIVE > 0 )
   if ( Passive_Code == NULL )   printf( "ERROR : Passive_Code == NULL in %s !!\n", __FUNCTION__ );
#  endif
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );

   if ( Hydro_CheckNegative(Dens_Code) )
      printf( "ERROR : invalid input density (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Dens_Code, __FILE__, __LINE__, __FUNCTION__ );

   if ( Hydro_CheckNegative(Pres_Code) )
      printf( "ERROR : invalid input pressure (%14.7e) at file <%s>, line <%d>, function <%s>\n",
              Pres_Code, __FILE__, __LINE__, __FUNCTION__ );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT   ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS ];
   const real Pres2CGS    = AuxArray_Flt[NUC_AUX_PRES2CGS ];
   const real CsSqr2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE];

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO     ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE    ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE      ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE    ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE ];

   int  Mode     = NUC_MODE_PRES;
   real Dens_CGS = Dens_Code * Dens2CGS;
   real Pres_CGS = Pres_Code * Pres2CGS;
   real Ye       = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;
   real Cs2_CGS  = NULL_REAL;
   real Useless  = NULL_REAL;
   int  Err      = NULL_INT;
#ifdef FLOAT8
   const real Tolerance = 1e-10;
#else
   const real Tolerance = 1e-6;
#endif


// check floating-point overflow and Ye
#  ifdef GAMER_DEBUG
   if ( Nuc_Overflow(Dens_CGS) )
      printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e) in %s() !!\n",
              Dens_CGS, Dens_Code, Dens2CGS, __FUNCTION__ );

   if ( Nuc_Overflow(Pres_CGS) )
      printf( "ERROR : EoS overflow (Pres_CGS %13.7e, Pres_Code %13.7e, Pres2CGS %13.7e) in %s() !!\n",
              Pres_CGS, Pres_Code, Pres2CGS, __FUNCTION__ );

   if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
      printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e) in %s() !!\n",
              Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], __FUNCTION__ );
#  endif // GAMER_DEBUG


// invoke the nuclear EoS driver
   nuc_eos_C_short( Dens_CGS, &Useless, Ye, &Useless, &Useless, &Pres_CGS, &Cs2_CGS, &Useless,
                    EnergyShift, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                    Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE], 
                    Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_ENTE_MODE], Table[NUC_TAB_ENTR_MODE],  Table[NUC_TAB_PRES_MODE],
                    Table[NUC_TAB_YE_MODE], Mode, &Err, Tolerance );

// trigger a *hard failure* if the EoS driver fails
   if ( Err )  Cs2_CGS = NAN;

   const real Cs2_Code = Cs2_CGS * CsSqr2Code;


// final check
#  ifdef GAMER_DEBUG
   if ( Hydro_CheckNegative(Cs2_Code) )
   {
      printf( "ERROR : invalid output sound speed squared (%13.7e) in %s() !!\n", Cs2_Code, __FUNCTION__ );
      printf( "        Dens=%13.7e code units, Pres=%13.7e code units, Ye=%13.7e, Mode %d\n", Dens_Code, Pres_Code, Ye, Mode );
      printf( "        EoS error code: %d\n", Err );
   }
#  endif // GAMER_DEBUG


// store extra output variables if required
   /*
   if ( ExtraInOut != NULL )
   {
      ExtraInOut[...] = ...;
   }
   */


   return Cs2_Code;

} // FUNCTION : EoS_DensPres2CSqr_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensEint2Temp_Nuclear
// Description :  Convert gas mass density and internal energy density to gas temperature
//
// Note        :  1. Internal energy density here is per unit volume instead of per unit mass
//                2. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//                3. Temperature is in kelvin
//                4. Invoke EoS_DensEint2Pres_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density            (in code unit)
//                Eint_Code    : Gas internal energy density (in code unit)
//                Passive_Code : Passive scalars             (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//                ExtraInOut   : Array to store extra input and output variables if required
//                               --> Optional and only used when it is not NULL
//
// Return      :  Gas temperature in kelvin
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensEint2Temp_Nuclear( const real Dens_Code, const real Eint_Code, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] )
{

   real Out[2];

   EoS_DensEint2Pres_Nuclear( Dens_Code, Eint_Code, Passive_Code, AuxArray_Flt, AuxArray_Int, Table, Out );

   return Out[0];

} // FUNCTION : EoS_DensEint2Temp_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_DensTemp2Pres_Nuclear
// Description :  Convert gas mass density and temperature to gas pressure
//
// Note        :  1. See EoS_SetAuxArray_Nuclear() for the values stored in AuxArray_Flt/Int[]
//                2. Temperature is in kelvin
//                3. Invoke EoS_General_Nuclear()
//
// Parameter   :  Dens_Code    : Gas mass density (in code unit)
//                Temp_Kelv    : Gas temperature  (in kelvin   )
//                Passive_Code : Passive scalars  (in code unit)
//                AuxArray_*   : Auxiliary arrays (see the Note above)
//                Table        : EoS tables
//                ExtraInOut   : Array to store extra input and output variables if required
//                               --> Optional and only used when it is not NULL
//
// Return      :  Gas pressure (in code unit)
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real EoS_DensTemp2Pres_Nuclear( const real Dens_Code, const real Temp_Kelv, const real Passive_Code[],
                                       const double AuxArray_Flt[], const int AuxArray_Int[],
                                       const real *const Table[EOS_NTABLE_MAX], real ExtraInOut[] )
{

   real In[3], Out[3], Pres_Code;

   In[0] = Dens_Code;
   In[1] = Temp_Kelv;
   In[2] = Passive_Code[ YE - NCOMP_FLUID ] / Dens_Code;

   EoS_General_Nuclear( NUC_MODE_TEMP, Out, In, AuxArray_Flt, AuxArray_Int, Table );

   Pres_Code = Out[2];

   return Pres_Code;

} // FUNCTION : EoS_DensTemp2Pres_Nuclear



//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_General_Nuclear
// Description :  General EoS converter: In[] -> Out[]
//
// Note        :  1. See EoS_DensEint2Pres_Nuclear()
//                2. In[] and Out[] must NOT overlap
//                3. Support the temperature and entropy modes:
//                   (1) Temperature mode:
//                         In [0] = mass density in code units
//                         In [1] = temperature in kelvin
//                         In [2] = Ye
//                         Out[0] = volume energy density in code units (with energy shift)
//                         Out[1] = entropy in kB/baryon
//                         Out[2] = pressure in code units
//                   (2) Entropy mode:
//                         In [0] = mass density in code units
//                         In [1] = entropy in kB/baryon
//                         In [2] = Ye
//                         Out[0] = volume energy density in code units (with energy shift)
//                   --> Both In[] and Out[] can be extended if necessary
//                4. No units conversion will be applied to entropy and Ye
//
// Parameter   :  Mode       : NUC_MODE_TEMP or NUC_MODE_ENTR
//                Out        : Output array     (see the Note above)
//                In         : Input array      (see the Note above)
//                AuxArray_* : Auxiliary arrays (see the Note above)
//                Table      : EoS tables
//
// Return      :  Out[]
//-------------------------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static void EoS_General_Nuclear( const int Mode, real Out[], const real In[], const double AuxArray_Flt[],
                                 const int AuxArray_Int[], const real *const Table[EOS_NTABLE_MAX] )
{

// general check
#  ifdef GAMER_DEBUG
   if ( Out          == NULL )   printf( "ERROR : Out == NULL in %s !!\n", __FUNCTION__ );
   if ( In           == NULL )   printf( "ERROR : In == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Flt == NULL )   printf( "ERROR : AuxArray_Flt == NULL in %s !!\n", __FUNCTION__ );
   if ( AuxArray_Int == NULL )   printf( "ERROR : AuxArray_Int == NULL in %s !!\n", __FUNCTION__ );
#  endif // GAMER_DEBUG


   const real EnergyShift = AuxArray_Flt[NUC_AUX_ESHIFT    ];
   const real Dens2CGS    = AuxArray_Flt[NUC_AUX_DENS2CGS  ];
   const real Kelvin2MeV  = AuxArray_Flt[NUC_AUX_KELVIN2MEV];
   const real sEint2Code  = AuxArray_Flt[NUC_AUX_VSQR2CODE ];
   const real Pres2Code   = AuxArray_Flt[NUC_AUX_PRES2CODE];

   const int  NRho        = AuxArray_Int[NUC_AUX_NRHO     ];
   const int  NTorE       = AuxArray_Int[NUC_AUX_NTORE    ];
   const int  NYe         = AuxArray_Int[NUC_AUX_NYE      ];
   const int  NRho_Mode   = AuxArray_Int[NUC_AUX_NRHO_MODE];
   const int  NMode       = AuxArray_Int[NUC_AUX_NMODE    ];
   const int  NYe_Mode    = AuxArray_Int[NUC_AUX_NYE_MODE ];


   switch ( Mode )
   {
//    temperature mode
      case NUC_MODE_TEMP :
      {
         const real Dens_Code = In[0];
         const real Temp_Kelv = In[1];
         const real Ye        = In[2];

//       check
#        ifdef GAMER_DEBUG
         if ( Hydro_CheckNegative(Dens_Code) )
            printf( "ERROR : invalid input density (%14.7e) at file <%s>, line <%d>, function <%s>\n",
                    Dens_Code, __FILE__, __LINE__, __FUNCTION__ );

         if ( Hydro_CheckNegative(Temp_Kelv) )
            printf( "ERROR : invalid input temperature (%14.7e) at file <%s>, line <%d>, function <%s>\n",
                    Temp_Kelv, __FILE__, __LINE__, __FUNCTION__ );

         if ( Hydro_CheckNegative(Ye) )
            printf( "ERROR : invalid input Ye (%14.7e) at file <%s>, line <%d>, function <%s>\n",
                    Ye, __FILE__, __LINE__, __FUNCTION__ );
#        endif // GAMER_DEBUG


         real Dens_CGS  = Dens_Code * Dens2CGS;
         real Temp_MeV  = Temp_Kelv * Kelvin2MeV;
         real sEint_CGS = NULL_REAL;
         real Entr      = NULL_REAL;
         real Pres_CGS  = NULL_REAL;
         real Useless   = NULL_REAL;
         int  Err       = NULL_INT;
#ifdef FLOAT8
   const real Tolerance = 1e-10;
#else
   const real Tolerance = 1e-6;
#endif

//       check floating-point overflow and Ye
#        ifdef GAMER_DEBUG
         if ( Nuc_Overflow(Dens_CGS) )
            printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e, Mode %d) in %s() !!\n",
                    Dens_CGS, Dens_Code, Dens2CGS, Mode, __FUNCTION__ );

         if ( Nuc_Overflow(Temp_MeV) )
            printf( "ERROR : EoS overflow (Temp_MeV %13.7e, Temp_Kelv %13.7e, Kelvin2MeV %13.7e, Mode %d) in %s() !!\n",
                    Temp_MeV, Temp_Kelv, Kelvin2MeV, Mode, __FUNCTION__ );

         if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
            printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e, Mode %d) in %s() !!\n",
                    Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], Mode, __FUNCTION__ );
#        endif // GAMER_DEBUG


//       invoke the nuclear EoS driver
         nuc_eos_C_short( Dens_CGS, &Temp_MeV, Ye, &sEint_CGS, &Entr, &Pres_CGS, &Useless, &Useless,
                          EnergyShift, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                          Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE], 
                          Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_ENTE_MODE], Table[NUC_TAB_ENTR_MODE],  Table[NUC_TAB_PRES_MODE],
                          Table[NUC_TAB_YE_MODE], Mode, &Err, Tolerance );

//       trigger a *hard failure* if the EoS driver fails
         if ( Err )
         {
            sEint_CGS = NAN;
            Entr      = NAN;
            Pres_CGS  = NAN;
         }


//       store the output results
         const real Eint_Code = (  ( sEint_CGS + EnergyShift ) * sEint2Code  ) * Dens_Code;
         const real Pres_Code = Pres_CGS * Pres2Code;
         Out[0] = Eint_Code;  // volume energy density in code units (with energy shift)
         Out[1] = Entr;       // entropy in kB/baryon
         Out[2] = Pres_Code;  // pressure in code units


//       final check
#        ifdef GAMER_DEBUG
//       still require Eint>0 for the nuclear EoS
         if ( Hydro_CheckNegative(Eint_Code) )
         {
            printf( "ERROR : invalid output internal energy density (%13.7e code units) in %s() !!\n", Eint_Code, __FUNCTION__ );
            printf( "        Dens=%13.7e code units, Temp=%13.7e K, Ye=%13.7e, Mode %d\n", Dens_Code, Temp_Kelv, Ye, Mode );
            printf( "        EoS error code: %d\n", Err );
         }

         if ( Hydro_CheckNegative(Entr) )
         {
            printf( "ERROR : invalid output entropy (%13.7e code units) in %s() !!\n", Entr, __FUNCTION__ );
            printf( "        Dens=%13.7e code units, Temp=%13.7e K, Ye=%13.7e, Mode %d\n", Dens_Code, Temp_Kelv, Ye, Mode );
            printf( "        EoS error code: %d\n", Err );
         }

         if ( Hydro_CheckNegative(Pres_Code) )
         {
            printf( "ERROR : invalid output pressure (%13.7e code units) in %s() !!\n", Pres_Code, __FUNCTION__ );
            printf( "        Dens=%13.7e code units, Temp=%13.7e K, Ye=%13.7e, Mode %d\n", Dens_Code, Temp_Kelv, Ye, Mode );
            printf( "        EoS error code: %d\n", Err );
         }
#        endif // GAMER_DEBUG
      } // case NUC_MODE_TEMP
      break;


//    entropy mode
      case NUC_MODE_ENTR :
      {
         const real Dens_Code = In[0];
               real Entr      = In[1];
         const real Ye        = In[2];

//       check
#        ifdef GAMER_DEBUG
         if ( Hydro_CheckNegative(Dens_Code) )
            printf( "ERROR : invalid input density (%14.7e) at file <%s>, line <%d>, function <%s>\n",
                    Dens_Code, __FILE__, __LINE__, __FUNCTION__ );

         if ( Hydro_CheckNegative(Entr) )
            printf( "ERROR : invalid input entropy (%14.7e) at file <%s>, line <%d>, function <%s>\n",
                    Entr, __FILE__, __LINE__, __FUNCTION__ );

         if ( Hydro_CheckNegative(Ye) )
            printf( "ERROR : invalid input Ye (%14.7e) at file <%s>, line <%d>, function <%s>\n",
                    Ye, __FILE__, __LINE__, __FUNCTION__ );
#        endif // GAMER_DEBUG


         real Dens_CGS  = Dens_Code * Dens2CGS;
         real sEint_CGS = NULL_REAL;
         real Useless   = NULL_REAL;
         int  Err       = NULL_INT;
#ifdef FLOAT8
   const real Tolerance = 1e-10;
#else
   const real Tolerance = 1e-6;
#endif

//       check floating-point overflow and Ye
#        ifdef GAMER_DEBUG
         if ( Nuc_Overflow(Dens_CGS) )
            printf( "ERROR : EoS overflow (Dens_CGS %13.7e, Dens_Code %13.7e, Dens2CGS %13.7e, Mode %d) in %s() !!\n",
                    Dens_CGS, Dens_Code, Dens2CGS, Mode, __FUNCTION__ );

         if ( Ye < (real)Table[NUC_TAB_YE][0]  ||  Ye > (real)Table[NUC_TAB_YE][NYe-1] )
            printf( "ERROR : invalid Ye = %13.7e (min = %13.7e, max = %13.7e, Mode %d) in %s() !!\n",
                    Ye, Table[NUC_TAB_YE][0], Table[NUC_TAB_YE][NYe-1], Mode, __FUNCTION__ );
#        endif // GAMER_DEBUG


//       invoke the nuclear EoS driver
         nuc_eos_C_short( Dens_CGS, &Useless, Ye, &sEint_CGS, &Entr, &Useless, &Useless, &Useless,
                          EnergyShift, NRho, NTorE, NYe, NRho_Mode, NMode, NYe_Mode,
                          Table[NUC_TAB_ALL], Table[NUC_TAB_ALL_MODE], Table[NUC_TAB_RHO], Table[NUC_TAB_TORE], Table[NUC_TAB_YE], 
                          Table[NUC_TAB_RHO_MODE], Table[NUC_TAB_ENTE_MODE], Table[NUC_TAB_ENTR_MODE],  Table[NUC_TAB_PRES_MODE],
                          Table[NUC_TAB_YE_MODE], Mode, &Err, Tolerance );

//       trigger a *hard failure* if the EoS driver fails
         if ( Err )  sEint_CGS = NAN;


//       store the output results
         const real Eint_Code = (  ( sEint_CGS + EnergyShift ) * sEint2Code  ) * Dens_Code;
         Out[0] = Eint_Code;  // volume energy density in code units (with energy shift)


//       final check
#        ifdef GAMER_DEBUG
//       still require Eint>0 for the nuclear EoS
         if ( Hydro_CheckNegative(Eint_Code) )
         {
            printf( "ERROR : invalid output internal energy density (%13.7e code units) in %s() !!\n", Eint_Code, __FUNCTION__ );
            printf( "        Dens=%13.7e code units, Entr=%13.7e kB/baryon, Ye=%13.7e, Mode %d\n", Dens_Code, Entr, Ye, Mode );
            printf( "        EoS error code: %d\n", Err );
         }
#        endif // GAMER_DEBUG
      } // case NUC_MODE_ENTR
      break;


      default :
      {
#        ifdef GAMER_DEBUG
         printf( "ERROR : unsupported mode (%d) at file <%s>, line <%d>, function <%s>\n",
                 Mode, __FILE__, __LINE__, __FUNCTION__ );
#        endif

//       trigger a *hard failure* here
         Out[0] = NAN;
      } // default

   } // switch ( Mode )

} // FUNCTION : EoS_General_Nuclear



// =============================================
// III. Set EoS initialization functions
// =============================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE EoS_DE2P_t EoS_DensEint2Pres_Ptr = EoS_DensEint2Pres_Nuclear;
FUNC_SPACE EoS_DP2E_t EoS_DensPres2Eint_Ptr = EoS_DensPres2Eint_Nuclear;
FUNC_SPACE EoS_DP2C_t EoS_DensPres2CSqr_Ptr = EoS_DensPres2CSqr_Nuclear;
FUNC_SPACE EoS_DE2T_t EoS_DensEint2Temp_Ptr = EoS_DensEint2Temp_Nuclear;
FUNC_SPACE EoS_DT2P_t EoS_DensTemp2Pres_Ptr = EoS_DensTemp2Pres_Nuclear;
FUNC_SPACE EoS_GENE_t EoS_General_Ptr       = EoS_General_Nuclear;

//-----------------------------------------------------------------------------------------
// Function    :  EoS_SetCPU/GPUFunc_Nuclear
// Description :  Return the function pointers of the CPU/GPU EoS routines
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//                2. Must obtain the CPU and GPU function pointers by **separate** routines
//                   since CPU and GPU functions are compiled completely separately in GAMER
//                   --> In other words, a unified routine like the following won't work
//
//                      EoS_SetFunc_Nuclear( CPU_FuncPtr, GPU_FuncPtr );
//
//                3. Call-by-reference
//
// Parameter   :  EoS_DensEint2Pres_CPU/GPUPtr : CPU/GPU function pointers to be set
//                EoS_DensPres2Eint_CPU/GPUPtr : ...
//                EoS_DensPres2CSqr_CPU/GPUPtr : ...
//                EoS_DensEint2Temp_CPU/GPUPtr : ...
//                EoS_DensTemp2Pres_CPU/GPUPtr : ...
//                EoS_General_CPU/GPUPtr       : ...
//
// Return      :  EoS_DensEint2Pres_CPU/GPUPtr, EoS_DensPres2Eint_CPU/GPUPtr,
//                EoS_DensPres2CSqr_CPU/GPUPtr, EoS_DensEint2Temp_CPU/GPUPtr,
//                EoS_DensTemp2Pres_CPU/GPUPtr, EoS_General_CPU/GPUPtr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void EoS_SetGPUFunc_Nuclear( EoS_DE2P_t &EoS_DensEint2Pres_GPUPtr,
                             EoS_DP2E_t &EoS_DensPres2Eint_GPUPtr,
                             EoS_DP2C_t &EoS_DensPres2CSqr_GPUPtr,
                             EoS_DE2T_t &EoS_DensEint2Temp_GPUPtr,
                             EoS_DT2P_t &EoS_DensTemp2Pres_GPUPtr,
                             EoS_GENE_t &EoS_General_GPUPtr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensEint2Pres_GPUPtr, EoS_DensEint2Pres_Ptr, sizeof(EoS_DE2P_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensPres2Eint_GPUPtr, EoS_DensPres2Eint_Ptr, sizeof(EoS_DP2E_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensPres2CSqr_GPUPtr, EoS_DensPres2CSqr_Ptr, sizeof(EoS_DP2C_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensEint2Temp_GPUPtr, EoS_DensEint2Temp_Ptr, sizeof(EoS_DE2T_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_DensTemp2Pres_GPUPtr, EoS_DensTemp2Pres_Ptr, sizeof(EoS_DT2P_t) )  );
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &EoS_General_GPUPtr,       EoS_General_Ptr,       sizeof(EoS_GENE_t) )  );
}

#else // #ifdef __CUDACC__

void EoS_SetCPUFunc_Nuclear( EoS_DE2P_t &EoS_DensEint2Pres_CPUPtr,
                             EoS_DP2E_t &EoS_DensPres2Eint_CPUPtr,
                             EoS_DP2C_t &EoS_DensPres2CSqr_CPUPtr,
                             EoS_DE2T_t &EoS_DensEint2Temp_CPUPtr,
                             EoS_DT2P_t &EoS_DensTemp2Pres_CPUPtr,
                             EoS_GENE_t &EoS_General_CPUPtr )
{
   EoS_DensEint2Pres_CPUPtr = EoS_DensEint2Pres_Ptr;
   EoS_DensPres2Eint_CPUPtr = EoS_DensPres2Eint_Ptr;
   EoS_DensPres2CSqr_CPUPtr = EoS_DensPres2CSqr_Ptr;
   EoS_DensEint2Temp_CPUPtr = EoS_DensEint2Temp_Ptr;
   EoS_DensTemp2Pres_CPUPtr = EoS_DensTemp2Pres_Ptr;
   EoS_General_CPUPtr       = EoS_General_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...



#ifndef __CUDACC__

// local function prototypes
void EoS_SetAuxArray_Nuclear( double [], int [] );
void EoS_SetCPUFunc_Nuclear( EoS_DE2P_t &, EoS_DP2E_t &, EoS_DP2C_t &, EoS_DE2T_t &, EoS_DT2P_t &, EoS_GENE_t & );
#ifdef GPU
void EoS_SetGPUFunc_Nuclear( EoS_DE2P_t &, EoS_DP2E_t &, EoS_DP2C_t &, EoS_DE2T_t &, EoS_DT2P_t &, EoS_GENE_t & );
#endif

//-----------------------------------------------------------------------------------------
// Function    :  EoS_Init_Nuclear
// Description :  Initialize EoS
//
// Note        :  1. Set auxiliary arrays by invoking EoS_SetAuxArray_*()
//                   --> It will be copied to GPU automatically in CUAPI_SetConstMemory()
//                2. Set the CPU/GPU EoS routines by invoking EoS_SetCPU/GPUFunc_*()
//                3. Invoked by EoS_Init()
//                   --> Enable it by linking to the function pointer "EoS_Init_Ptr"
//                4. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void EoS_Init_Nuclear()
{

// check
// default maximum table size must be large enough
   if ( EOS_NTABLE_MAX < NUC_TABLE_NPTR )
      Aux_Error( ERROR_INFO, "EOS_NTABLE_MAX (%d) < NUC_TABLE_NPTR (%d) for the nuclear EoS !!\n",
                 EOS_NTABLE_MAX, NUC_TABLE_NPTR );

// must enable units
   if ( ! OPT__UNIT )
      Aux_Error( ERROR_INFO, "must enable OPT__UNIT for EOS_NUCLEAR !!\n " );


   nuc_eos_C_ReadTable( NUC_TABLE );

   EoS_SetAuxArray_Nuclear( EoS_AuxArray_Flt, EoS_AuxArray_Int );
   EoS_SetCPUFunc_Nuclear( EoS_DensEint2Pres_CPUPtr, EoS_DensPres2Eint_CPUPtr,
                           EoS_DensPres2CSqr_CPUPtr, EoS_DensEint2Temp_CPUPtr,
                           EoS_DensTemp2Pres_CPUPtr, EoS_General_CPUPtr );
#  ifdef GPU
   EoS_SetGPUFunc_Nuclear( EoS_DensEint2Pres_GPUPtr, EoS_DensPres2Eint_GPUPtr,
                           EoS_DensPres2CSqr_GPUPtr, EoS_DensEint2Temp_GPUPtr,
                           EoS_DensTemp2Pres_GPUPtr, EoS_General_GPUPtr );
#  endif

#  ifdef GPU
   CUAPI_PassNuclearEoSTable2GPU();
#  endif

} // FUNCTION : EoS_Init_Nuclear

#endif // #ifndef __CUDACC__



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
