#include "CUAPI.h"
#include "CUFLU.h"
#include "NuclearEoS.h"

#if ( defined GPU  &&  MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )


extern int g_nrho;
extern int g_neps;
extern int g_nye;
extern int g_nmode;

extern real *d_EoS_Table[EOS_NTABLE_MAX];




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_PassNuclearEoSTable2GPU
// Description :  Allocate GPU memory and transfer data to GPU for the nuclear EoS
//
// Note        :  1. Must be invoked BEFORE calling CUAPI_SetConstMemory() to correctly set the
//                   constant-memory pointer array c_EoS_Table[]
//
// Parameter   :  None
//
// Return      :  d_EoS_Table[]
//-------------------------------------------------------------------------------------------------------
void CUAPI_PassNuclearEoSTable2GPU()
{

// set the table size
   long EoS_TableSize[NUC_TABLE_NPTR];

   EoS_TableSize[NUC_TAB_ALL      ] = sizeof(real)*g_nrho*g_neps*g_nye*NUC_TABLE_NVAR;
   EoS_TableSize[NUC_TAB_ALL_MODE ] = sizeof(real)*g_nrho*g_nmode*g_nye*3;
   EoS_TableSize[NUC_TAB_RHO      ] = sizeof(real)*g_nrho;
   EoS_TableSize[NUC_TAB_EPS      ] = sizeof(real)*g_neps;
   EoS_TableSize[NUC_TAB_YE       ] = sizeof(real)*g_nye;
   EoS_TableSize[NUC_TAB_TEMP_MODE] = sizeof(real)*g_nmode;
   EoS_TableSize[NUC_TAB_ENTR_MODE] = sizeof(real)*g_nmode;
   EoS_TableSize[NUC_TAB_PRES_MODE] = sizeof(real)*g_nmode;

   if ( MPI_Rank == 0 )
   {
      long TotalSize = 0;
      for (int t=0; t<NUC_TABLE_NPTR; t++)   TotalSize += EoS_TableSize[t];

      Aux_Message( stdout, "NOTE : total memory requirement in GPU nuclear EoS table = %ld MB\n", TotalSize/(1<<20) );
   }


// allocate GPU memory and transfer tables to GPU
// --> unlike other CPU-GPU data transfer in the code, here we do not allocate page-locked host memory (i.e., cudaHostAlloc())
//     since these tables will be transferred just once
   for (int t=0; t<NUC_TABLE_NPTR; t++)
   {
      CUDA_CHECK_ERROR(  cudaMalloc( (void**) &d_EoS_Table[t], EoS_TableSize[t] )  );
      CUDA_CHECK_ERROR(  cudaMemcpy( d_EoS_Table[t], h_EoS_Table[t], EoS_TableSize[t], cudaMemcpyHostToDevice )  );
   }

} // FUNCTION : CUAPI_PassNuclearEoSTable2GPU



#endif // #if ( defined GPU  &&  MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
