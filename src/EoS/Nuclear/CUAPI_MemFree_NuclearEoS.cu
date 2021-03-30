#include "CUAPI.h"
#include "CUFLU.h"
#include "NuclearEoS.h"

#if ( defined GPU  &&  MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )


extern real *d_EoS_Table[EOS_NTABLE_MAX];




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_MemFree_NuclearEoS
// Description :  Free the GPU memory of the nuclear EoS
//
// Note        :  1. Invoked by EoS_End_Nuclear()
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void CUAPI_MemFree_NuclearEoS()
{

   for (int t=0; t<NUC_TABLE_NPTR; t++)
   {
      if ( d_EoS_Table[t] != NULL ) {  CUDA_CHECK_ERROR(  cudaFree( d_EoS_Table[t])  );  d_EoS_Table[t] = NULL;  }
   }

} // FUNCTION : CUAPI_MemFree_NuclearEoS



#endif // #if ( defined GPU  &&  MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
