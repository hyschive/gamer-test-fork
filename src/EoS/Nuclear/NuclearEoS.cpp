#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__

#include "cubinterp_some.cu"
#include "linterp_some.cu"
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
void find_temp( const real x, const real y, const real z,
                real *found_lt, const real *alltables_mode,
                const int nx, const int ny, const int nz, const int ntemp,
                const real *xt, const real *yt, const real *zt,
                const real *logtemp, const int keymode, int *keyerr );
void find_temp2( const real lr, const real lt0, const real ye, const real varin, real *ltout,
                 const int nrho, const int ntemp, const int nye, const real *alltables, 
                 const real *logrho, const real *logtemp, const real *yes,
                 const int keymode, int *keyerrt, const real prec );

#endif // #ifdef __CUDACC__ ... else ...

//-----------------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_short
// Description :  Function to find thermodynamic varibles by searching
//                a pre-calculated nuclear equation of state table
// 
// Note        :  It will strictly return values in cgs or MeV
//                Four modes are supported
//                The defalut mode is temperature (0) mode
//                In case three other modes are available for finding temperature
//                energy      (0) mode
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
//                ntemp           : size of temperature array in the Nuclear EoS table
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
//                logtemp         : log(T)   array for temperature mode
//                yes             : Y_e      array in the table
//                logenergy_mode  : log(eps) array for energy mode
//                entropy_mode    : entropy  array for entropy mode
//                logpress_mode   : log(P)   array for pressure mode
//                keymode         : which mode we will use
//                                  0 : energy mode      (coming in with eps)
//                                  1 : temperature mode (coming in with T)
//                                  2 : entropy mode     (coming in with entropy)
//                                  3 : pressure mode    (coming in with P)
//                keyerr          : output error
//                                  667 : fail in finding temp (eps, e, P modes)
//                                  101 : Y_e too high
//                                  102 : Y_e too low
//                                  103 : temp too high (if keymode = 1) 
//                                  104 : temp too low  (if keymode = 1)
//                                  105 : rho too high
//                                  106 : rho too low
//                                  107 : eps too high     (if keymode = 0)
//                                  108 : eps too low      (if keymode = 0)
//                                  109 : entropy too high (if keymode = 2)
//                                  110 : entropy too low  (if keymode = 2)
//                                  111 : log(P) too high  (if keymode = 3)
//                                  112 : log(P) too low   (if keymode = 3)
//                rfeps           : tolerence for interpolations
//-----------------------------------------------------------------------------------------------
GPU_DEVICE
void nuc_eos_C_short( const real xrho, real *xtemp, const real xye,
                      real *xenr, real *xent, real *xprs,
                      real *xcs2, real *xmunu, const real energy_shift,
                      const int nrho, const int ntemp, const int nye, const int nmode,
                      const real *alltables, const real *alltables_mode,
                      const real *logrho, const real *logtemp, const real *yes,
                      const real *logeps_mode, const real *entr_mode, const real *logprss_mode,
                      const int keymode, int *keyerr, const real rfeps )
{

// check whether the input density and Ye are within the table
   const real lr = LOG10( xrho );
   *keyerr = 0;


   if ( lr > logrho[nrho-1] )  {  *keyerr = 105;  return;  }
   if ( lr < logrho[     0] )  {  *keyerr = 106;  return;  }

   if ( xye > yes  [nye -1] )  {  *keyerr = 101;  return;  }
   if ( xye < yes  [     0] )  {  *keyerr = 102;  return;  }
   

// find temperature
   real lt  = LOG10( *xtemp );
   real lt0 = LOG10( 63.0 ); //lt;

   switch ( keymode )
   {
      case NUC_MODE_ENGY :
      {
         const real leps = LOG10( MAX( (*xenr + energy_shift), 1.0 ) );
         
         if ( leps > logeps_mode[nmode-1] )        {  *keyerr = 107; return;  }
         if ( leps < logeps_mode[      0] )        {  *keyerr = 108; return;  }
         
         find_temp( lr, leps, xye, &lt, alltables_mode, nrho, nmode, nye, ntemp,
                    logrho, logeps_mode, yes, logtemp, keymode, keyerr );
         
         if ( *keyerr != 0 ) 
         {
            find_temp2( lr, lt0, xye, leps, &lt, nrho, ntemp, nye, alltables,
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
         
         if ( entr > entr_mode[nmode-1] )    {  *keyerr = 109; return;  }
         if ( entr < entr_mode[      0] )    {  *keyerr = 110; return;  }
          
         find_temp( lr, entr, xye, &lt, alltables_mode, nrho, nmode, nye, ntemp,
                    logrho, entr_mode, yes, logtemp, keymode, keyerr );
          
         if ( *keyerr != 0 ) 
         {
            find_temp2( lr, lt0, xye, entr, &lt, nrho, ntemp, nye, alltables,
                        logrho, logtemp, yes, keymode, keyerr, rfeps );
            if ( *keyerr != 0 ) return;
         }
      }
      break;
      
      case NUC_MODE_PRES :
      {
         const real lprs = LOG10( *xprs );
         
         if ( lprs > logprss_mode[nmode-1] ) {  *keyerr = 111; return;  }
         if ( lprs < logprss_mode[      0] ) {  *keyerr = 112; return;  }
         
         find_temp( lr, lprs, xye, &lt, alltables_mode, nrho, nmode, nye, ntemp,
                    logrho, logprss_mode, yes, logtemp, keymode, keyerr );
         
         if ( *keyerr != 0 ) 
         {
            find_temp2( lr, lt0, xye, lprs, &lt, nrho, ntemp, nye, alltables,
                        logrho, logtemp, yes, keymode, keyerr, rfeps );
            if ( *keyerr != 0 ) return;
         }
      }
      break;
   } // switch ( keymode )

 
   real res[5]; // result array

// linear interolation for other variables
   nuc_eos_C_linterp_some(lr, lt, xye, res, alltables,
                          nrho, ntemp, nye, 5, logrho, logtemp, yes);
   
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


   return;

} // FUNCTION : nuc_eos_C_short


#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
