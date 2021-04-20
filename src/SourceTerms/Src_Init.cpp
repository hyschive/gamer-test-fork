#include "GAMER.h"



// prototypes of built-in source terms
#if ( MODEL == HYDRO )
void Src_Init_Deleptonization();
void Src_Init_LightBulb();
#endif

// this function pointer can be set by a test problem initializer for a user-specified source term
void (*Src_Init_User_Ptr)() = NULL;




//-------------------------------------------------------------------------------------------------------
// Function    :  Src_Init
// Description :  Initialize the source terms
//
// Note        :  1. Invoked by Init_GAMER()
//                2. Set SrcTerms
//                3. Set "Src_Init_User_Ptr" in a test problem initializer for a user-specified source term
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Src_Init()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// check if at least one source term is activated
   if (
#       if ( MODEL == HYDRO )
        SrcTerms.Deleptonization  ||
        SrcTerms.Neutrino_Scheme  ||
#       endif
        SrcTerms.User
      )
      SrcTerms.Any = true;
   else
      SrcTerms.Any = false;


// set auxiliary parameters
   for (int d=0; d<3; d++)    SrcTerms.BoxCenter[d] = amr->BoxCenter[d];

   SrcTerms.Unit_L = UNIT_L;
   SrcTerms.Unit_M = UNIT_M;
   SrcTerms.Unit_T = UNIT_T;
   SrcTerms.Unit_V = UNIT_V;
   SrcTerms.Unit_D = UNIT_D;
   SrcTerms.Unit_E = UNIT_E;
   SrcTerms.Unit_P = UNIT_P;
#  ifdef MHD
   SrcTerms.Unit_B = UNIT_B;
#  endif


// initialize all function pointers as NULL
#  if ( MODEL == HYDRO )
   SrcTerms.Dlep_FuncPtr              = NULL;
   SrcTerms.Dlep_AuxArrayDevPtr_Flt   = NULL;
   SrcTerms.Dlep_AuxArrayDevPtr_Int   = NULL;
   SrcTerms.Dlep_Profile_DataDevPtr   = NULL;
   SrcTerms.Dlep_Profile_RadiusDevPtr = NULL;
   SrcTerms.LigB_FuncPtr              = NULL;
   SrcTerms.LigB_AuxArrayDevPtr_Flt   = NULL;
   SrcTerms.LigB_AuxArrayDevPtr_Int   = NULL;
#  endif
   SrcTerms.User_FuncPtr              = NULL;
   SrcTerms.User_AuxArrayDevPtr_Flt   = NULL;
   SrcTerms.User_AuxArrayDevPtr_Int   = NULL;


// initialize all source terms
// (1) deleptonization and lightbulb
#  if ( MODEL == HYDRO )
   if ( SrcTerms.Deleptonization )
   {
      Src_Init_Deleptonization();

//    check if the source-term function is set properly
      if ( SrcTerms.Dlep_FuncPtr == NULL )   Aux_Error( ERROR_INFO, "SrcTerms.Dlep_FuncPtr == NULL !!\n" );
   }
   if ( SrcTerms.Neutrino_Scheme == LIGHTBULB )
   {
      Src_Init_LightBulb();

//    check if the source-term function is set properly
      if ( SrcTerms.LigB_FuncPtr == NULL )   Aux_Error( ERROR_INFO, "SrcTerms.LigB_FuncPtr == NULL !!\n" );
   }
#  endif

// (2) user-specified source term
   if ( SrcTerms.User )
   {
      if ( Src_Init_User_Ptr == NULL )       Aux_Error( ERROR_INFO, "Src_Init_User_Ptr == NULL !!\n" );

      Src_Init_User_Ptr();

//    check if the source-term function is set properly
      if ( SrcTerms.User_FuncPtr == NULL )   Aux_Error( ERROR_INFO, "SrcTerms.User_FuncPtr == NULL !!\n" );
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Src_Init
