#include "GAMER.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )


#ifdef GPU
void CUAPI_MemFree_NuclearEoS();
#endif


extern real *g_alltables;
extern real *g_alltables_mode;
extern real *g_logrho;
extern real *g_yes;
extern real *g_logrho_mode;
extern real *g_entr_mode;
extern real *g_logprss_mode;
extern real *g_yes_mode;

#if   ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
extern real   *g_logtemp;
extern real   *g_logeps_mode;
#elif ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
extern real   *g_logeps;
extern real   *g_logtemp_mode;
#endif




//-------------------------------------------------------------------------------------------------------
// Function    :  EoS_End_Nuclear
// Description :  Free the resources used by the nuclear EoS routines
//
// Note        :  1. Invoked by EoS_End()
//                   --> Linked to the function pointer "EoS_End_Ptr"
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void EoS_End_Nuclear()
{

// CPU memory
   free( g_alltables      );  g_alltables      = NULL;
   free( g_alltables_mode );  g_alltables_mode = NULL;
   free( g_logrho         );  g_logrho         = NULL;
   free( g_yes            );  g_yes            = NULL;
   free( g_entr_mode      );  g_entr_mode      = NULL;
   free( g_logprss_mode   );  g_logprss_mode   = NULL;
#if   ( NUC_TABLE_MODE == NUC_TABLE_MODE_TEMP )
   free( g_logtemp        );  g_logtemp        = NULL;
   free( g_logeps_mode    );  g_logeps_mode    = NULL;
#elif ( NUC_TABLE_MODE == NUC_TABLE_MODE_ENGY )
   free( g_logeps         );  g_logeps         = NULL;
   free( g_logtemp_mode   );  g_logtemp_mode   = NULL;
#endif



// GPU memory
#  ifdef GPU
   CUAPI_MemFree_NuclearEoS();
#  endif

} // FUNCTION : EoS_End_Nuclear



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
