#include "CUAPI.h"
#include "CUPOT.h"

#if ( defined GPU  &&  defined GRAVITY )


extern real   *h_ExtPotTable;
extern double *h_GREP_Lv_Data_New;
extern double *h_GREP_Lv_Radius_New;
extern int     h_GREP_Lv_NBin_New;

extern void CUAPI_SendExtPotTable2GPU( const real *h_Table );




//-------------------------------------------------------------------------------------------------------
// Function    :  CUAPI_SendExtPotGREP2GPU
// Description :  Send the GREP potential table to GPU using the interface for EXT_POT_TABLE
//
// Note        :  1. Invoked by Poi_UserWorkBeforePoisson_GREP()
//                2. EXT_POT_GREP_NAUX_MAX is defined in Macro.h (default = 4000)
//-------------------------------------------------------------------------------------------------------
void CUAPI_SendExtPotGREP2GPU()
{

   if ( h_GREP_Lv_NBin_New > EXT_POT_GREP_NAUX_MAX )
      Aux_Error( ERROR_INFO, "Too many bins in GREP profiles %d !!\n", h_GREP_Lv_NBin_New );

   for (int b=0; b<h_GREP_Lv_NBin_New; b++) {
      h_ExtPotTable[b]                         = (real) h_GREP_Lv_Data_New  [b];
      h_ExtPotTable[b + EXT_POT_GREP_NAUX_MAX] = (real) h_GREP_Lv_Radius_New[b];
   }

   CUAPI_SendExtPotTable2GPU( h_ExtPotTable );

} // FUNCTION : CUAPI_SendExtPotGREP2GPU



#endif // #if ( defined GPU  &&  defined GRAVITY )
