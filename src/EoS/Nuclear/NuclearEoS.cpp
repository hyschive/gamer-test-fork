#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

#include "cubinterp_some.cu"
#include "linterp_some.cu"
#include "findenergy.cu"
#include "findtemp.cu"
#include "findtemp2.cu"


#else

void nuc_eos_C_cubinterp_some( const real x, const real y, const real z,
                               real *output_vars, const real *alltables,
                               const int nx, const int ny, const int nz, const int nvars,
                               const real *xt, const real *yt, const real *zt );
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );
void findtemp( const real x, const real y, const real z,
               real *found_lt, const real *alltables_mode,
               const int nx, const int ny, const int nz, const int ntemp,
               const real *xt, const real *yt, const real *zt,
               const real *logtemp, const int keymode, int *keyerr );
void findtemp2( const real lr, const real lt0, const real ye, const real varin, real *ltout,
                const int nrho, const int ntemp, const int nye, const real *alltables, 
                const real *logrho, const real *logtemp, const real *yes,
                const int keymode, int *keyerrt, const real prec );
void findenergy( const real x, const real y, const real z,
                 real *found_leps, const real *alltables_mode,
                 const int nx, const int ny, const int nz, const int neps,
                 const real *xt, const real *yt, const real *zt,
                 const real *logeps, const int keymode, int *keyerr );

#endif // #ifdef __CUDACC__ ... else ...

//-----------------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_short
// Description :  Function to find thermodynamic varibles by searching
//                a pre-calculated nuclear equation of state table
// 
// Note        :  It will strictly return values in cgs or MeV
//                Four modes are supported
//                The defalut mode is temperature (1) mode
//                In case three other modes are available for finding temperature
//                energy      (0) mode
//                temperature (1) mode
//                entropy     (2) mode
//                pressure    (3) mode
//
// Parameter   :  xrho            : input density (rho (g/cm^3))
//                xtemp           : input (temperature mode)
//                                  or ouput temperature in MeV
//                xye             : electron fraction (Y_e)
//                xenr            : input specific internal energy (energy mode)
//                                  or output specific internal energy
//                xent            : input (entropy mode)
//                                  or output specific entropy (e)
//                xprs            : input (pressure mode)
//                                  or output pressure
//                xcs2            : output sound speed
//                xmunu           : output chemcial potential
//                energy_shift    : energy_shift
//                nrho            : size of density array in the Nuclear EoS table
//                ntoreps         : size of (temperature/energy) array in the Nuclear EoS table (temp/energy-based table)
//                nye             : size of Y_e array in the Nuclear EoS table
//                nmode           : size of log(eps)   (0)
//                                          entropy    (2)
//                                          log(P)     (3) array in the Nuclear EoS table
//                                                         for each mode
//                alltables       : Nuclear EoS table
//                alltables_mode  : Auxiliary log(T) arrays for energy mode
//                                                              entropy mode
//                                                              pressure mode
//                logrho          : log(rho) array in the table
//                logtoreps       : log(T) or log(eps) array for (T/eps) mode (temp/energy-based table)
//                yes             : Y_e      array in the table
//                logepsort_mode  : log(eps) or log(T) array for (eps/T) mode (temp/energy-based table)
//                entropy_mode    : entropy  array for entropy mode
//                logpress_mode   : log(P)   array for pressure mode
//                keymode         : which mode we will use
//                                  0 : energy mode      (coming in with eps)
//                                  1 : temperature mode (coming in with T)
//                                  2 : entropy mode     (coming in with entropy)
//                                  3 : pressure mode    (coming in with P)
//                keyerr          : output error
//                                  667 : fail in finding T/e (temp/energy-based table)
//                                  101 : Y_e too high
//                                  102 : Y_e too low
//                                  103 : temp too high 
//                                  104 : temp too low
//                                  105 : rho too high
//                                  106 : rho too low
//                                  107 : eps too high     (energy-based table)
//                                  108 : eps too low      (energy-based table)
//                                  109 : temp too high    (energy-based table)
//                                  110 : temp too low     (energy-based table)
//                                  111 : entropy too high (energy-based table)
//                                  112 : entropy too low  (energy-based table)
//                                  113 : log(P) too high  (energy-based table)
//                                  114 : log(P) too low   (energy-based table)
//                rfeps           : tolerence for interpolations
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
void nuc_eos_C_short( const real xrho, real *xtemp, const real xye,
                      real *xenr, real *xent, real *xprs,
                      real *xcs2, real *xmunu, const real energy_shift,
                      const int nrho, const int ntoreps, const int nye,
                      const int nrho_mode, const int nmode, const int nye_mode,
                      const real *alltables, const real *alltables_mode,
                      const real *logrho, const real *logtoreps, const real *yes, const real *logrho_mode,
                      const real *logepsort_mode, const real *entr_mode, const real *logprss_mode, const real *yes_mode,
                      const int keymode, int *keyerr, const real rfeps )
{

// check whether the input density and Ye are within the table
   const real lr = LOG10( xrho );
   *keyerr = 0;


   if ( lr > logrho[nrho-1] )  {  *keyerr = 105;  return;  }
   if ( lr < logrho[     0] )  {  *keyerr = 106;  return;  }

   if ( xye > yes  [nye -1] )  {  *keyerr = 101;  return;  }
   if ( xye < yes  [     0] )  {  *keyerr = 102;  return;  }
   

// find temperature (temp-based table)
#if ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )

   const int ntemp = ntoreps;
   const real *logtemp = logtoreps;
   real lt  = LOG10( *xtemp );
   real lt0 = LOG10( 63.0 ); //lt;

   switch ( keymode )
   {
      case NUC_MODE_ENGY :
      {
         const real leps = LOG10( MAX( (*xenr + energy_shift), 1.0 ) );
         const real *logeps_mode = logepsort_mode;

         findtemp( lr, leps, xye, &lt, alltables_mode, nrho_mode, nmode, nye_mode, ntemp,
                    logrho_mode, logeps_mode, yes_mode, logtemp, keymode, keyerr );

         if ( *keyerr != 0 ) 
         {
            findtemp2( lr, lt0, xye, leps, &lt, nrho, ntemp, nye, alltables,
                        logrho, logtemp, yes, keymode, keyerr, rfeps );
            if ( *keyerr != 0 ) return;
         }
      }
      break;

      case NUC_MODE_TEMP :
      {
         if ( lt > logtemp[ntemp-1] )        {  *keyerr = 103; return;  }
         if ( lt < logtemp[      0] )        {  *keyerr = 104; return;  }
      }
      break;

      case NUC_MODE_ENTR :
      {
         const real entr = *xent;

         findtemp( lr, entr, xye, &lt, alltables_mode, nrho_mode, nmode, nye_mode, ntemp,
                    logrho_mode, entr_mode, yes_mode, logtemp, keymode, keyerr );

         if ( *keyerr != 0 ) 
         {
            findtemp2( lr, lt0, xye, entr, &lt, nrho, ntemp, nye, alltables,
                        logrho, logtemp, yes, keymode, keyerr, rfeps );
            if ( *keyerr != 0 ) return;
         }
      }
      break;

      case NUC_MODE_PRES :
      {
         const real lprs = LOG10( *xprs );

         findtemp( lr, lprs, xye, &lt, alltables_mode, nrho_mode, nmode, nye_mode, ntemp,
                    logrho_mode, logprss_mode, yes_mode, logtemp, keymode, keyerr );

         if ( *keyerr != 0 ) 
         {
            findtemp2( lr, lt0, xye, lprs, &lt, nrho, ntemp, nye, alltables,
                        logrho, logtemp, yes, keymode, keyerr, rfeps );
            if ( *keyerr != 0 ) return;
         }
      }
      break;
   } // switch ( keymode )


   real res[5]; // result array

// linear interolation for other variables
   nuc_eos_C_linterp_some( lr, lt, xye, res, alltables,
                           nrho, ntemp, nye, 5, logrho, logtemp, yes );

// cubic interpolation for other variables
   //nuc_eos_C_cubinterp_some( lr, lt, xye, res, alltables,
   //                          nrho, ntemp, nye, 5, logrho, logtemp, yes );
   

// assign results
   if ( keymode != NUC_MODE_TEMP ) *xtemp = POW( (real)10.0, lt );
   if ( keymode != NUC_MODE_PRES ) *xprs  = POW( (real)10.0, res[0] );
   if ( keymode != NUC_MODE_ENGY ) *xenr  = POW( (real)10.0, res[1] ) - energy_shift;
   if ( keymode != NUC_MODE_ENTR ) *xent  = res[2];

   *xmunu = res[3];
   *xcs2  = res[4];


// find energy (energy-based table)
#elif ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )

   const int  neps    = ntoreps;
   const real *logeps = logtoreps;
   real       leps    = NULL_REAL;
   switch ( keymode )
   {
      case NUC_MODE_ENGY :
      {
         leps = LOG10( *xenr + energy_shift );

         if ( leps > logeps[neps-1] )        {  *keyerr = 107;  return;  }
         if ( leps < logeps[     0] )        {  *keyerr = 108;  return;  }
      }
      break;

      case NUC_MODE_TEMP :
      {
         const real lt = LOG10( *xtemp );
         const real *logtemp_mode = logepsort_mode;

         if ( lt > logtemp_mode[nmode-1] )   {  *keyerr = 109;  return;  }
         if ( lt < logtemp_mode[      0] )   {  *keyerr = 110;  return;  }

         findenergy( lr, lt, xye, &leps, alltables_mode, nrho, nmode, nye, neps,
                     logrho_mode, logtemp_mode, yes_mode, logeps, keymode, keyerr );

         if ( *keyerr != 0 )  return;
      }
      break;

      case NUC_MODE_ENTR :
      {
         const real entr = *xent;

         if ( entr > entr_mode[nmode-1] )    {  *keyerr = 111;  return;  }
         if ( entr < entr_mode[      0] )    {  *keyerr = 112;  return;  }

         findenergy( lr, entr, xye, &leps, alltables_mode, nrho, nmode, nye, neps,
                     logrho_mode, entr_mode, yes_mode, logeps, keymode, keyerr );

         if ( *keyerr != 0 )  return;
      }
      break;

      case NUC_MODE_PRES :
      {
         const real lprs = LOG10( *xprs );

         if ( lprs > logprss_mode[nmode-1] ) {  *keyerr = 113;  return;  }
         if ( lprs < logprss_mode[      0] ) {  *keyerr = 114;  return;  }

         findenergy( lr, lprs, xye, &leps, alltables_mode, nrho, nmode, nye, neps,
                     logrho_mode, logprss_mode, yes_mode, logeps, keymode, keyerr );

         if ( *keyerr != 0 )  return;
      }
      break;
   } // switch ( keymode )


   real res[5]; // result array

// linear interolation for other variables
   nuc_eos_C_linterp_some( lr, leps, xye, res, alltables,
                           nrho, neps, nye, 5, logrho, logeps, yes );

// cubic interpolation for other variables
   //nuc_eos_C_cubinterp_some( lr, leps, xye, res, alltables, nrho, neps, nye, 5,
   //                          logrho, logeps, yes );

// assign results
   if ( keymode != NUC_MODE_ENGY )  *xenr  = POW( (real)10.0, leps ) - energy_shift;
   if ( keymode != NUC_MODE_PRES )  *xprs  = POW( (real)10.0, res[0] );
   if ( keymode != NUC_MODE_TEMP )  *xtemp = POW( (real)10.0, res[1] );
   if ( keymode != NUC_MODE_ENTR )  *xent  = res[2];

   *xmunu = res[3];
   *xcs2  = res[4];

#endif // #elif Table_Mode == TABLE_MODE_ENGY


   return;

} // FUNCTION : nuc_eos_C_short


#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
