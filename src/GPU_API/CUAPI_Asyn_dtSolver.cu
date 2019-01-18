#include "CUAPI.h"
#include "CUFLU.h"
#ifdef GRAVITY
#include "CUPOT.h"
#endif

#ifdef GPU



#if   ( MODEL == HYDRO )
__global__
void CUFLU_dtSolver_HydroCFL(       real   g_dt_Array[],
                              const real   g_Flu_Array[][NCOMP_FLUID][ CUBE(PS1) ],
                              const double g_Corner_Array[][3],
                              const int lv,
                              const real Safety, const real Gamma, const real MinPres );
#ifdef GRAVITY
__global__
void CUPOT_dtSolver_HydroGravity(       real   g_dt_Array[],
                                  const real   g_Pot_Array[][ CUBE(GRA_NXT) ],
                                  const double g_Corner_Array[][3],
                                  const int lv,
                                  const real Safety, const bool P5_Gradient, const OptGravityType_t GravityType,
                                  const double ExtAcc_Time );
#endif
#elif ( MODEL == MHD )
#warning : WAIT MHD !!!

#elif ( MODEL == ELBDM )

#else
#error : ERROR : unsupported MODEL !!
#endif // MODEL

template <typename T> bool  Mis_CompareRealValue( const T Input1, const T Input2, const char *comment, const bool Verbose );


// device pointers
extern real *d_dt_Array_T;
extern real (*d_Flu_Array_T)[NCOMP_FLUID][ CUBE(PS1) ];
extern double (*d_Corner_Array_T)[3];
#ifdef GRAVITY
extern real (*d_Pot_Array_T)[ CUBE(GRA_NXT) ];
#endif

extern cudaStream_t *Stream;




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_Asyn_dtSolver
// Description :  Invoke various dt solvers
//
//                ***********************************************************
//                **                Asynchronous Function                  **
//                **                                                       **
//                **  will return before the execution in GPU is complete  **
//                ***********************************************************
//
// Note        :  1. Use streams for the asychronous memory copy between device and host
//                2. Prefix "d" : for pointers pointing to the "Device" memory space
//                   Prefix "h" : for pointers pointing to the "Host"   memory space
//
// Parameter   :  TSolver        : Target dt solver
//                                 --> DT_FLU_SOLVER : dt solver for fluid
//                                     DT_GRA_SOLVER : dt solver for gravity
//                h_dt_Array     : Host array to store the minimum dt in each target patch
//                h_Flu_Array    : Host array storing the prepared fluid     data of each target patch
//                h_Pot_Array    : Host array storing the prepared potential data of each target patch
//                h_Corner_Array : Array storing the physical corner coordinates of each patch
//                NPatchGroup    : Number of patch groups evaluated simultaneously by GPU
//                lv             : Target AMR level
//                Safety         : dt safety factor
//                Gamma          : Ratio of specific heats
//                MinPres        : Minimum allowed pressure
//                P5_Gradient    : Use 5-points stencil to evaluate the potential gradient
//                GravityType    : Types of gravity --> self-gravity, external gravity, both
//                ExtPot         : Add the external potential for ELBDM
//                TargetTime     : Target physical time
//                GPU_NStream    : Number of CUDA streams for the asynchronous memory copy
//
// Return      :  h_dt_Array
//-------------------------------------------------------------------------------------------------------
void CUAPI_Asyn_dtSolver( const Solver_t TSolver, real h_dt_Array[], const real h_Flu_Array[][NCOMP_FLUID][ CUBE(PS1) ],
                          const real h_Pot_Array[][ CUBE(GRA_NXT) ], const double h_Corner_Array[][3],
                          const int NPatchGroup, const int lv, const real Safety,
                          const real Gamma, const real MinPres, const bool P5_Gradient,
                          const OptGravityType_t GravityType, const bool ExtPot, const double TargetTime,
                          const int GPU_NStream )
{

// determine whether or not to prepare the corner array
   bool PrepareCorner = false;

#  ifdef GRAVITY
// ExtPot is not actually used in this function yet
   if (  TSolver == DT_GRA_SOLVER  && ( GravityType == GRAVITY_EXTERNAL || GravityType == GRAVITY_BOTH || ExtPot )  )
      PrepareCorner = true;
#  endif

#  if ( COORDINATE != CARTESIAN )
      PrepareCorner = true;
#  endif


// check
#  ifdef GAMER_DEBUG
   if ( TSolver != DT_FLU_SOLVER )
#  ifdef GRAVITY
   if ( TSolver != DT_GRA_SOLVER )
#  endif
      Aux_Error( ERROR_INFO, "TSolver != DT_FLU_SOLVER / DT_GRA_SOLVER !!\n" );

   if ( h_dt_Array == NULL )
      Aux_Error( ERROR_INFO, "h_dt_Array == NULL !!\n" );

   if ( TSolver == DT_FLU_SOLVER  &&  h_Flu_Array == NULL )
      Aux_Error( ERROR_INFO, "h_Flu_Array == NULL !!\n" );

#  ifdef GRAVITY
   if ( TSolver == DT_GRA_SOLVER )
   {
      if ( h_Pot_Array == NULL )
         Aux_Error( ERROR_INFO, "h_Pot_Array == NULL !!\n" );
   }
#  endif

   if ( PrepareCorner )
   {
      if ( h_Corner_Array   == NULL )  Aux_Error( ERROR_INFO, "h_Corner_Array == NULL !!\n" );
      if ( d_Corner_Array_T == NULL )  Aux_Error( ERROR_INFO, "d_Corner_Array_T == NULL !!\n" );
   }

#  if ( COORDINATE == CARTESIAN )
   if (  !Mis_CompareRealValue( amr->dh[lv][0], amr->dh[lv][1], NULL, false )  ||
         !Mis_CompareRealValue( amr->dh[lv][0], amr->dh[lv][2], NULL, false )    )
      Aux_Error( ERROR_INFO, "currently the Cartesian coordinates only work with cubic cells --> dh = (%20.14e, %20.14e, %20.14e) !!\n",
                 amr->dh[lv][0], amr->dh[lv][1], amr->dh[lv][2] );
#  else
   Aux_Error( ERROR_INFO, "non-Cartesian coordinates do not support %s() yet !!\n", __FUNCTION__ );
#  endif
#  endif // #ifdef GAMER_DEBUG


// set the block size
   const int NPatch = NPatchGroup*8;
   dim3 BlockDim_dtSolver( 1, 1, 1 );

   switch ( TSolver )
   {
      case DT_FLU_SOLVER:
         BlockDim_dtSolver.x = DT_FLU_BLOCK_SIZE;
      break;

#     ifdef GRAVITY
      case DT_GRA_SOLVER:
         BlockDim_dtSolver.x = DT_GRA_BLOCK_SIZE;
      break;
#     endif

      default :
         Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "TSolver", TSolver );
   }


// set the number of patches and the corresponding data size to be transferred into GPU in each stream
   int *NPatch_per_Stream = new int [GPU_NStream];
   int *UsedPatch         = new int [GPU_NStream];
   int *Data_MemSize      = new int [GPU_NStream];
   int *dt_MemSize        = new int [GPU_NStream];
   int *Corner_MemSize    = new int [GPU_NStream];


// number of patches in each stream
   UsedPatch[0] = 0;

   if ( GPU_NStream == 1 )    NPatch_per_Stream[0] = NPatch;
   else
   {
      for (int s=0; s<GPU_NStream-1; s++)
      {
         NPatch_per_Stream[s] = NPatch / GPU_NStream;
         UsedPatch[s+1] = UsedPatch[s] + NPatch_per_Stream[s];
      }

      NPatch_per_Stream[GPU_NStream-1] = NPatch - UsedPatch[GPU_NStream-1];
   }

// corresponding data size to be transferred into GPU in each stream
   for (int s=0; s<GPU_NStream; s++)
   {
      switch ( TSolver )
      {
         case DT_FLU_SOLVER:
            Data_MemSize  [s] = sizeof(real  )*NPatch_per_Stream[s]*CUBE(PS1)*NCOMP_FLUID;
         break;

#        ifdef GRAVITY
         case DT_GRA_SOLVER:
            Data_MemSize  [s] = sizeof(real  )*NPatch_per_Stream[s]*CUBE(GRA_NXT);
         break;
#        endif

         default :
            Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "TSolver", TSolver );
      }

      dt_MemSize    [s] = sizeof(real)*NPatch_per_Stream[s];

      if ( PrepareCorner )
      Corner_MemSize[s] = sizeof(double)*NPatch_per_Stream[s]*3;
   }


// a. copy data from host to device
//=========================================================================================
   for (int s=0; s<GPU_NStream; s++)
   {
      if ( NPatch_per_Stream[s] == 0 )    continue;

      switch ( TSolver )
      {
         case DT_FLU_SOLVER:
            CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Flu_Array_T + UsedPatch[s], h_Flu_Array + UsedPatch[s],
                               Data_MemSize[s], cudaMemcpyHostToDevice, Stream[s] )  );
         break;

#        ifdef GRAVITY
         case DT_GRA_SOLVER:
            CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Pot_Array_T + UsedPatch[s], h_Pot_Array + UsedPatch[s],
                               Data_MemSize[s], cudaMemcpyHostToDevice, Stream[s] )  );
         break;
#        endif

         default :
            Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "TSolver", TSolver );
      }

      if ( PrepareCorner )
         CUDA_CHECK_ERROR(  cudaMemcpyAsync( d_Corner_Array_T + UsedPatch[s], h_Corner_Array + UsedPatch[s],
                            Corner_MemSize[s], cudaMemcpyHostToDevice, Stream[s] )  );
   } // for (int s=0; s<GPU_NStream; s++)


// b. execute the kernel
//=========================================================================================
   for (int s=0; s<GPU_NStream; s++)
   {
      if ( NPatch_per_Stream[s] == 0 )    continue;

#     if   ( MODEL == HYDRO )
      switch ( TSolver )
      {
         case DT_FLU_SOLVER:
            CUFLU_dtSolver_HydroCFL <<< NPatch_per_Stream[s], BlockDim_dtSolver, 0, Stream[s] >>>
                                    ( d_dt_Array_T     + UsedPatch[s],
                                      d_Flu_Array_T    + UsedPatch[s],
                                      d_Corner_Array_T + UsedPatch[s],
                                      lv, Safety, Gamma, MinPres );
         break;

#        ifdef GRAVITY
         case DT_GRA_SOLVER:
            CUPOT_dtSolver_HydroGravity <<< NPatch_per_Stream[s], BlockDim_dtSolver, 0, Stream[s] >>>
                                        ( d_dt_Array_T     + UsedPatch[s],
                                          d_Pot_Array_T    + UsedPatch[s],
                                          d_Corner_Array_T + UsedPatch[s],
                                          lv, Safety, P5_Gradient, GravityType, TargetTime );
         break;
#        endif

         default :
            Aux_Error( ERROR_INFO, "incorrect parameter %s = %d !!\n", "TSolver", TSolver );
      }


#     elif ( MODEL == MHD )
#     warning :: WAIT MHD !!!

#     elif ( MODEL == ELBDM )

#     else
#        error : unsupported MODEL !!
#     endif // MODEL

      CUDA_CHECK_ERROR( cudaGetLastError() );
   } // for (int s=0; s<GPU_NStream; s++)


// c. copy data from device to host
//=========================================================================================
   for (int s=0; s<GPU_NStream; s++)
   {
      if ( NPatch_per_Stream[s] == 0 )    continue;

      CUDA_CHECK_ERROR(  cudaMemcpyAsync( h_dt_Array + UsedPatch[s], d_dt_Array_T + UsedPatch[s],
                         dt_MemSize[s], cudaMemcpyDeviceToHost, Stream[s] )  );
   } // for (int s=0; s<GPU_NStream; s++)


   delete [] NPatch_per_Stream;
   delete [] UsedPatch;
   delete [] Data_MemSize;
   delete [] dt_MemSize;
   delete [] Corner_MemSize;

} // FUNCTION : CUAPI_Asyn_dtSolver



#endif // #ifdef GPU
