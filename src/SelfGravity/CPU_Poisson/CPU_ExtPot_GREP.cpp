#include "CUPOT.h"
#ifdef __CUDACC__
#include "CUAPI.h"
#include "CUDA_ConstMemory.h"
#else
#include "GAMER.h"
#endif


#define LinearInterp( x, xa, xb, ya, yb )   (  ( ((x) - (xa)) * (yb) + ((xb) - (x)) * (ya) ) / ((xb) - (xa))  )


#ifndef __CUDACC__
extern Profile_t *DensAve [NLEVEL+1][2];
extern Profile_t *EngyAve [NLEVEL+1][2];
extern Profile_t *VrAve   [NLEVEL+1][2];
extern Profile_t *PresAve [NLEVEL+1][2];
extern Profile_t *Phi_eff [NLEVEL  ][2];

extern int    GREP_LvUpdate;
extern int    GREPSg     [NLEVEL];
extern double GREPSgTime [NLEVEL][2];
extern double GREP_Prof_Center   [3];
extern double GREP_Prof_MaxRadius;
extern double GREP_Prof_MinBinSize;

double *h_GREP_Lv_Data_New;
double *h_GREP_FaLv_Data_New;
double *h_GREP_FaLv_Data_Old;
double *h_GREP_Lv_Radius_New;
double *h_GREP_FaLv_Radius_New;
double *h_GREP_FaLv_Radius_Old;
int     h_GREP_Lv_NBin_New;
int     h_GREP_FaLv_NBin_New;
int     h_GREP_FaLv_NBin_Old;


extern void (*Poi_UserWorkBeforePoisson_Ptr)( const double Time, const int lv );
extern void Poi_UserWorkBeforePoisson_GREP( const double Time, const int lv );
#endif // #ifndef __CUDACC__



// =========================================================
// I. Initialize the GREP Profile_t objects and parameters
// =========================================================

#ifndef __CUDACC__
//-------------------------------------------------------------------------------------------------------
// Function    :  Init_GREP
// Description :  Initialize the GREP Profile_t objects and parameters
//
// Note        :  1. DensAve, EngyAve, VrAve, and PresAve
//                                        : Profile_t objects storing the spherical-averaged
//                                          density, energy, radial velocity, and pressure.
//                2. Phi_eff              : Profile_t objects storing the GR effective potential
//                3. GREPSg               : Sandglass of the current profile data [0/1]
//                4. GREPSgTime           : Physical time of profile
//                5. GREP_Prof_MinBinSize : Minimum bin size used to initialize the Profile_t object
//                6. GREP_Prof_MaxRadius  : Maximum radius   used to initialize the Profile_t object
//-------------------------------------------------------------------------------------------------------
void Init_GREP()
{

// (1) intialize the GREP profiles
   for (int Sg=0; Sg<2; Sg++)
   for (int lv=0; lv<=NLEVEL; lv++)
   {
      DensAve [lv][Sg] = new Profile_t();
      EngyAve [lv][Sg] = new Profile_t();
      VrAve   [lv][Sg] = new Profile_t();
      PresAve [lv][Sg] = new Profile_t();

      if ( lv < NLEVEL )
      Phi_eff [lv][Sg] = new Profile_t();
   }


// (2) initialize GREP Sg and SgTime
   for (int lv=0; lv<NLEVEL; lv++)
   {
//    GREPSg must be initialized to [0/1]. Otherwise, it will fail when determining Sg in Poi_Prepare_GREP()
      GREPSg[lv] = 0;
      for (int Sg=0; Sg<2; Sg++)   GREPSgTime[lv][Sg] = -__FLT_MAX__;
   }


// (3) initialize the GREP parameters
   switch ( GREP_CENTER_METHOD )
   {
      case 1:
         for (int i=0; i<3; i++)   GREP_Prof_Center[i] = amr->BoxCenter[i];
      break;

      default:
         Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "GREP_CENTER_METHOD", GREP_CENTER_METHOD );
   }

   GREP_Prof_MinBinSize = ( GREP_MINBINSIZE > 0.0 ) ? GREP_MINBINSIZE : amr->dh[MAX_LEVEL];

   GREP_Prof_MaxRadius  = ( GREP_MAXRADIUS > 0.0 )
                        ? GREP_MAXRADIUS
                        : SQRT( SQR( MAX( amr->BoxSize[0] - GREP_Prof_Center[0], GREP_Prof_Center[0] ) )
                        +       SQR( MAX( amr->BoxSize[1] - GREP_Prof_Center[1], GREP_Prof_Center[1] ) )
                        +       SQR( MAX( amr->BoxSize[2] - GREP_Prof_Center[2], GREP_Prof_Center[2] ) ) );

} // FUNCTION : Init_GREP



// =================================
// II. Set an auxiliary array
// =================================

//-------------------------------------------------------------------------------------------------------
// Function    :  SetExtPotAuxArray_GREP
// Description :  Set the auxiliary arrays ExtPot_AuxArray_Flt/Int[] used by ExtPot_GREP()
//
// Note        :  1. Invoked by Init_ExtPot_GREP()
//                2. AuxArray_Flt/Int[] have the size of EXT_POT_NAUX_MAX defined in Macro.h (default = 20)
//                3. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  AuxArray_Flt/Int : Floating-point/Integer arrays to be filled up
//
// Return      :  AuxArray_Flt/Int[]
//-------------------------------------------------------------------------------------------------------
void SetExtPotAuxArray_GREP( double AuxArray_Flt[], int AuxArray_Int[] )
{

   const int Lv   = GREP_LvUpdate;
   const int FaLv = ( Lv > 0 ) ? Lv - 1 : Lv;

   const int Sg_Lv   = GREPSg[Lv];
   const int Sg_FaLv = GREPSg[FaLv];

   AuxArray_Flt[0] = GREP_Prof_Center[0];                // x coordinate of the GREP profile center
   AuxArray_Flt[1] = GREP_Prof_Center[1];                // y coordinate of the GREP profile center
   AuxArray_Flt[2] = GREP_Prof_Center[2];                // z coordinate of the GREP profile center
   AuxArray_Flt[3] = GREPSgTime[ FaLv ][     Sg_FaLv ];  // new physical time of GREP on father level
   AuxArray_Flt[4] = GREPSgTime[ FaLv ][ 1 - Sg_FaLv ];  // old physical time of GREP on father level

} // FUNCTION : SetExtPotAuxArray_GREP
#endif // #ifndef __CUDACC__



// =================================
// III. Specify external potential
// =================================

//-----------------------------------------------------------------------------------------
// Function    :  ExtPot_GREP
// Description :  Calculate the external potential at the given coordinates and time
//
// Note        :  1. This function is shared by CPU and GPU
//                2. Auxiliary arrays UserArray_Flt/Int[] are set by SetExtPotAuxArray_GREP()
//                3. Currently it does not support the soften length
//
// Parameter   :  x/y/z             : Target spatial coordinates
//                Time              : Target physical time
//                UserArray_Flt/Int : User-provided floating-point/integer auxiliary arrays
//                Usage             : Different usages of external potential when computing total potential on level Lv
//                                    --> EXT_POT_USAGE_ADD     : add external potential on Lv
//                                        EXT_POT_USAGE_SUB     : subtract external potential for preparing self-gravity potential on Lv-1
//                                        EXT_POT_USAGE_SUB_TINT: like SUB but for temporal interpolation
//                                    --> This parameter is useless in most cases
//                PotTable          : 3D potential table used by EXT_POT_TABLE
//                GenePotTable      : Array of pointers for general potential tables
//
// Return      :  External potential at (x,y,z,Time)
//-----------------------------------------------------------------------------------------
GPU_DEVICE_NOINLINE
static real ExtPot_GREP( const double x, const double y, const double z, const double Time,
                         const double UserArray_Flt[], const int UserArray_Int[],
                         const ExtPotUsage_t Usage, const real PotTable[], void **GenePotTable )
{

   int     NBin;
   double  pot;
   double *effpot;
   double *radius;

   const real dx = (real)( x - UserArray_Flt[0] );
   const real dy = (real)( y - UserArray_Flt[1] );
   const real dz = (real)( z - UserArray_Flt[2] );
   const real r  = SQRT( SQR(dx) + SQR(dy) + SQR(dz) );


// use Usage to determine which Data and Radius profiles, and NBin are used
#ifdef __CUDACC__
   if ( Usage == EXT_POT_USAGE_ADD )
   {
      effpot = c_GREP_Lv_Data_New;
      radius = c_GREP_Lv_Radius_New;
      NBin   = c_GREP_Lv_NBin_New;
   }
#else // #ifdef __CUDACC__
   switch ( Usage )
   {
      case EXT_POT_USAGE_ADD:
         effpot = h_GREP_Lv_Data_New;
         radius = h_GREP_Lv_Radius_New;
         NBin   = h_GREP_Lv_NBin_New;
      break;

      case EXT_POT_USAGE_SUB:
      case EXT_POT_USAGE_SUB_TINT:
         if      (  Mis_CompareRealValue( Time, UserArray_Flt[3], NULL, false )  )
         {
            effpot = h_GREP_FaLv_Data_New;
            radius = h_GREP_FaLv_Radius_New;
            NBin   = h_GREP_FaLv_NBin_New;
         }

         else if (  Mis_CompareRealValue( Time, UserArray_Flt[4], NULL, false )  )
         {
            effpot = h_GREP_FaLv_Data_Old;
            radius = h_GREP_FaLv_Radius_Old;
            NBin   = h_GREP_FaLv_NBin_Old;
         }

         else
         {
            Aux_Error( ERROR_INFO, "No GREP Profile matches the specified time: %.15e !!\n", Time );
         }

      break;
   }
#endif // #ifdef __CUDACC__ ... else ...


// compute the potential
   if ( r < (real)radius[0] )
      pot = effpot[0];

   else if ( r < (real)radius[NBin-1] )
   {
      int Idx;
      int Min = 0;
      int Max = NBin-1;

      while (  ( Idx=(Min+Max)/2 ) != Min  )
      {
         if   ( (real)radius[Idx] > r )   Max = Idx;
         else                             Min = Idx;
      }

      const real rL      = (real)radius[Idx  ];
      const real rR      = (real)radius[Idx+1];
      const real effpotL = (real)effpot[Idx  ];
      const real effpotR = (real)effpot[Idx+1];

      pot = LinearInterp( r, rL, rR, effpotL, effpotR );
   }

   else
      pot = effpot[NBin-1];

   return pot;

} // FUNCTION : ExtPot_GREP



// =================================
// IV. Set initialization functions
// =================================

#ifdef __CUDACC__
#  define FUNC_SPACE __device__ static
#else
#  define FUNC_SPACE            static
#endif

FUNC_SPACE ExtPot_t ExtPot_Ptr = ExtPot_GREP;

//-----------------------------------------------------------------------------------------
// Function    :  SetCPU/GPUExtPot_GREP
// Description :  Return the function pointers of the CPU/GPU external potential routines
//
// Note        :  1. Invoked by Init_ExtPot_GREP()
//                2. Must obtain the CPU and GPU function pointers by **separate** routines
//                   since CPU and GPU functions are compiled completely separately in GAMER
//                   --> In other words, a unified routine like the following won't work
//
//                      SetExtPot_GREP( ExtPot_t &CPUExtPot_Ptr, ExtPot_t &GPUExtPot_Ptr )
//
// Parameter   :  CPU/GPUExtPot_Ptr (call-by-reference)
//
// Return      :  CPU/GPUExtPot_Ptr
//-----------------------------------------------------------------------------------------
#ifdef __CUDACC__
__host__
void SetGPUExtPot_GREP( ExtPot_t &GPUExtPot_Ptr )
{
   CUDA_CHECK_ERROR(  cudaMemcpyFromSymbol( &GPUExtPot_Ptr, ExtPot_Ptr, sizeof(ExtPot_t) )  );
}

#else // #ifdef __CUDACC__

void SetCPUExtPot_GREP( ExtPot_t &CPUExtPot_Ptr )
{
   CPUExtPot_Ptr = ExtPot_Ptr;
}

#endif // #ifdef __CUDACC__ ... else ...



#ifndef __CUDACC__

// local function prototypes
void Init_GREP();
void SetExtPotAuxArray_GREP( double [], int [] );
void SetCPUExtPot_GREP( ExtPot_t & );
#ifdef GPU
void SetGPUExtPot_GREP( ExtPot_t & );
#endif

//-----------------------------------------------------------------------------------------
// Function    :  Init_ExtPot_GREP
// Description :  Initialize external potential
//
// Note        :  1. Set an auxiliary array by invoking SetExtPotAuxArray_*()
//                   --> It will be copied to GPU automatically in CUAPI_SetConstMemory()
//                2. Set the CPU/GPU external potential major routines by invoking SetCPU/GPUExtPot_*()
//                3. Invoked by Init_ExtAccPot()
//                   --> Enable it by linking to the function pointer "Init_ExtPot_Ptr"
//                4. Add "#ifndef __CUDACC__" since this routine is only useful on CPU
//
// Parameter   :  None
//
// Return      :  None
//-----------------------------------------------------------------------------------------
void Init_ExtPot_GREP()
{

// set the function pointer for the built-in GREP
   Poi_UserWorkBeforePoisson_Ptr = Poi_UserWorkBeforePoisson_GREP;

   Init_GREP();
   SetExtPotAuxArray_GREP( ExtPot_AuxArray_Flt, ExtPot_AuxArray_Int );
   SetCPUExtPot_GREP( CPUExtPot_Ptr );
#  ifdef GPU
   SetGPUExtPot_GREP( GPUExtPot_Ptr );
#  endif

} // FUNCTION : Init_ExtPot_GREP



//-------------------------------------------------------------------------------------------------------
// Function    :  End_ExtPot_GREP
// Description :  Free memory previously allocated by Init_GREP()
//
// Parameter   :  None
//-------------------------------------------------------------------------------------------------------
void End_ExtPot_GREP()
{

   for (int Sg=0; Sg<2; Sg++)
   for (int lv=0; lv<=NLEVEL; lv++)
   {
      DensAve [lv][Sg]->FreeMemory();
      EngyAve [lv][Sg]->FreeMemory();
      VrAve   [lv][Sg]->FreeMemory();
      PresAve [lv][Sg]->FreeMemory();

      if ( lv < NLEVEL )
      Phi_eff [lv][Sg]->FreeMemory();
   }

} // FUNCTION : End_ExtPot_GREP

#endif // #ifndef __CUDACC__
