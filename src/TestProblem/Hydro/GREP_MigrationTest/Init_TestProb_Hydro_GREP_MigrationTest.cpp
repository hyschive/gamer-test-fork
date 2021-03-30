#include "GAMER.h"
#include "TestProb.h"


// problem-specific global variables
// ===============================================================================================
// Parameters for a toroidal B field
#ifdef MHD
static double  Bfield_Ab;                       // magnetic field strength   [1e15]
static double  Bfield_np;                       // dependence on the density [0.0]
#endif

// Parameters for initial condition
static double *NeutronStar_Prof = NULL;         // radial progenitor model
static int     NeutronStar_NBin;                // number of radial bins in the progenitor model
static char    NeutronStar_ICFile[MAX_STRING];  // Filename of initial condition
// ===============================================================================================


static void LoadICTable();
static void Record_CentralDens();




//-------------------------------------------------------------------------------------------------------
// Function    :  Validate
// Description :  Validate the compilation flags and runtime parameters for this test problem
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Validate()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ...\n", TESTPROB_ID );


#  if ( MODEL != HYDRO )
   Aux_Error( ERROR_INFO, "MODEL != HYDRO !!\n" );
#  endif

#  ifndef GRAVITY
   Aux_Error( ERROR_INFO, "GRAVITY must be enabled !!\n" );
#  endif

#  ifdef GRAVITY
   if ( OPT__EXT_POT != EXT_POT_GREP )
      Aux_Error( ERROR_INFO, "must set OPT__EXT_POT = EXT_POT_GREP !!\n" );
#  endif

#  if ( EOS != EOS_GAMMA )
   Aux_Error( ERROR_INFO, "EOS != EOS_GAMMA !!\n" );
#  endif

   if ( !OPT__UNIT )
      Aux_Error( ERROR_INFO, "OPT__UNIT must be enabled !!\n" );

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Validating test problem %d ... done\n", TESTPROB_ID );

} // FUNCTION : Validate



#if ( MODEL == HYDRO )
//-------------------------------------------------------------------------------------------------------
// Function    :  SetParameter
// Description :  Load and set the problem-specific runtime parameters
//
// Note        :  1. Filename is set to "Input__TestProb" by default
//                2. Major tasks in this function:
//                   (1) load the problem-specific runtime parameters
//                   (2) set the problem-specific derived parameters
//                   (3) reset other general-purpose parameters if necessary
//                   (4) make a note of the problem-specific parameters
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void SetParameter()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ...\n" );


// (1) load the problem-specific runtime parameters
   const char FileName[] = "Input__TestProb";
   ReadPara_t *ReadPara  = new ReadPara_t;

// (1-1) add parameters in the following format:
// --> note that VARIABLE, DEFAULT, MIN, and MAX must have the same data type
// --> some handy constants (e.g., Useless_bool, Eps_double, NoMin_int, ...) are defined in "include/ReadPara.h"
// ********************************************************************************************************************************
// ReadPara->Add( "KEY_IN_THE_FILE",   &VARIABLE,              DEFAULT,       MIN,              MAX               );
// ********************************************************************************************************************************
   ReadPara->Add( "NeutronStar_ICFile",   NeutronStar_ICFile,    Useless_str,   Useless_str,      Useless_str       );
#  ifdef MHD
   ReadPara->Add( "Bfield_Ab",           &Bfield_Ab,             1.0e15,        0.0,              NoMax_double      );
   ReadPara->Add( "Bfield_np",           &Bfield_np,             0.0,           NoMin_double,     NoMax_double      );
#  endif

   ReadPara->Read( FileName );

   delete ReadPara;

// (1-2) set the default values

// (1-3) check the runtime parameters


// (2) set the problem-specific derived parameters


// (3) reset other general-purpose parameters
//     --> a helper macro PRINT_WARNING is defined in TestProb.h
   const long   End_Step_Default = __INT_MAX__;
   const double End_T_Default    = __FLT_MAX__;

   if ( END_STEP < 0 ) {
      END_STEP = End_Step_Default;
      PRINT_WARNING( "END_STEP", END_STEP, FORMAT_LONG );
   }

   if ( END_T < 0.0 ) {
      END_T = End_T_Default;
      PRINT_WARNING( "END_T", END_T, FORMAT_REAL );
   }


// (4) make a note
   if ( MPI_Rank == 0 )
   {
      Aux_Message( stdout, "=============================================================================\n" );
      Aux_Message( stdout, "  test problem ID           = %d\n",      TESTPROB_ID );
      Aux_Message( stdout, "  NeutronStar_ICFile        = %s\n",      NeutronStar_ICFile );
#     ifdef MHD
      Aux_Message( stdout, "  Bfield_Ab                 = %13.7e\n",  Bfield_Ab );
      Aux_Message( stdout, "  Bfield_np                 = %13.7e\n",  Bfield_np );
#     endif
      Aux_Message( stdout, "=============================================================================\n" );
   }


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "   Setting runtime parameters ... done\n" );

} // FUNCTION : SetParameter



//-------------------------------------------------------------------------------------------------------
// Function    :  SetGridIC
// Description :  Set the problem-specific initial condition on grids
//
// Note        :  1. This function may also be used to estimate the numerical errors when OPT__OUTPUT_USER is enabled
//                   --> In this case, it should provide the analytical solution at the given "Time"
//                2. This function will be invoked by multiple OpenMP threads when OPENMP is enabled
//                   --> Please ensure that everything here is thread-safe
//                3. Even when DUAL_ENERGY is adopted for HYDRO, one does NOT need to set the dual-energy variable here
//                   --> It will be calculated automatically
//
// Parameter   :  fluid    : Fluid field to be initialized
//                x/y/z    : Physical coordinates
//                Time     : Physical time
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  fluid
//-------------------------------------------------------------------------------------------------------
void SetGridIC( real fluid[], const double x, const double y, const double z, const double Time,
                const int lv, double AuxArray[] )
{

   const double  BoxCenter[3] = { 0.5*amr->BoxSize[0], 0.5*amr->BoxSize[1], 0.5*amr->BoxSize[2] };
   const double *Table_R      = NeutronStar_Prof + 0*NeutronStar_NBin;
   const double *Table_Velr   = NeutronStar_Prof + 1*NeutronStar_NBin;
   const double *Table_Dens   = NeutronStar_Prof + 2*NeutronStar_NBin;
   const double *Table_Pres   = NeutronStar_Prof + 3*NeutronStar_NBin;

   double Dens, Velr, Pres, Momx, Momy, Momz, Eint, Etot;

   const double x0 = x - BoxCenter[0];
   const double y0 = y - BoxCenter[1];
   const double z0 = z - BoxCenter[2];
   const double r  = SQRT( SQR( x0 ) + SQR( y0 ) + SQR( z0 ) );

   Dens = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Dens, r);
   Velr = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Velr, r);
   Pres = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Pres, r);

   Momx = Dens*Velr*x0/r;
   Momy = Dens*Velr*y0/r;
   Momz = Dens*Velr*z0/r;

   Eint = EoS_DensPres2Eint_CPUPtr( Dens, Pres, NULL, EoS_AuxArray_Flt,
                                    EoS_AuxArray_Int, h_EoS_Table, NULL ); // assuming EoS requires no passive scalars
   Etot = Hydro_ConEint2Etot( Dens, Momx, Momy, Momz, Eint, 0.0 );         // do NOT include magnetic energy here


   fluid[DENS] = Dens;
   fluid[MOMX] = Momx;
   fluid[MOMY] = Momy;
   fluid[MOMZ] = Momz;
   fluid[ENGY] = Etot;

} // FUNCTION : SetGridIC


#ifdef MHD
//-------------------------------------------------------------------------------------------------------
// Function    :  SetBFieldIC
// Description :  Set the problem-specific initial condition of magnetic field
//
// Note        :  1. This function will be invoked by multiple OpenMP threads when OPENMP is enabled
//                   (unless OPT__INIT_GRID_WITH_OMP is disabled)
//                   --> Please ensure that everything here is thread-safe
//                2. Generate poloidal B field from vector potential in a form similar
//                   to that defined in Liu+ 2008, Phys. Rev. D78, 024012
//                     A_phi = Ab * \bar\omega^2 * (1 - rho / rho_max)^np * (P / P_max)
//                   where
//                     \omega^2 = (x - x_center)^2 + y^2
//
//                     A_x = -(y / \bar\omega^2) * A_phi
//                     A_y =  (x / \bar\omega^2) * A_phi
//                     A_z =  0
//
// Parameter   :  magnetic : Array to store the output magnetic field
//                x/y/z    : Target physical coordinates
//                Time     : Target physical time
//                lv       : Target refinement level
//                AuxArray : Auxiliary array
//
// Return      :  magnetic
//-------------------------------------------------------------------------------------------------------
void SetBFieldIC( real magnetic[], const double x, const double y, const double z, const double Time,
                  const int lv, double AuxArray[] )
{

   const double  BoxCenter[3] = { 0.5*amr->BoxSize[0], 0.5*amr->BoxSize[1], 0.5*amr->BoxSize[2] };
   const double *Table_R      = NeutronStar_Prof + 0*NeutronStar_NBin;
   const double *Table_Dens   = NeutronStar_Prof + 2*NeutronStar_NBin;
   const double *Table_Pres   = NeutronStar_Prof + 3*NeutronStar_NBin;

   const double x0 = x - BoxCenter[0];
   const double y0 = y - BoxCenter[1];
   const double z0 = z - BoxCenter[2];

   // approximate the central density and pressure by the data at the first row
   const double dens_c = Table_Dens[0];
   const double pres_c = Table_Pres[0];
   const double Ab     = Bfield_Ab / UNIT_B;

   // Use finite difference to compute the B field
   double delta = amr->dh[MAX_LEVEL];
   double r,    dens,    pres;
   double r_xp, dens_xp, pres_xp;
   double r_yp, dens_yp, pres_yp;
   double r_zp, dens_zp, pres_zp;

   r       = SQRT( SQR( y0 ) + SQR( z0 ) + SQR( x0         ) );
   r_xp    = SQRT( SQR( y0 ) + SQR( z0 ) + SQR( x0 + delta ) );
   r_yp    = SQRT( SQR( z0 ) + SQR( x0 ) + SQR( y0 + delta ) );
   r_zp    = SQRT( SQR( x0 ) + SQR( y0 ) + SQR( z0 + delta ) );

   dens    = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Dens, r   );
   dens_xp = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Dens, r_xp);
   dens_yp = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Dens, r_yp);
   dens_zp = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Dens, r_zp);

   pres    = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Pres, r   );
   pres_xp = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Pres, r_xp);
   pres_yp = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Pres, r_yp);
   pres_zp = Mis_InterpolateFromTable(NeutronStar_NBin, Table_R, Table_Pres, r_zp);

   double dAy_dx = ( ( x0 + delta )*POW( 1.0 - dens_xp/dens_c, Bfield_np )*( pres_xp / pres_c )   \
                   - ( x0         )*POW( 1.0 - dens   /dens_c, Bfield_np )*( pres    / pres_c ) ) \
                 / delta;

   double dAx_dy = ( -( y0 + delta )*POW( 1.0 - dens_yp/dens_c, Bfield_np )*( pres_yp / pres_c )   \
                   - -( y0         )*POW( 1.0 - dens   /dens_c, Bfield_np )*( pres    / pres_c ) ) \
                 / delta;

   double dAphi_dz = ( POW( 1.0 - dens_zp/dens_c, Bfield_np )*( pres_zp / pres_c )   \
                     - POW( 1.0 - dens   /dens_c, Bfield_np )*( pres    / pres_c ) ) \
                   / delta;


   magnetic[MAGX] = -x0 * Ab * dAphi_dz;
   magnetic[MAGY] = -y0 * Ab * dAphi_dz;
   magnetic[MAGZ] =       Ab * ( dAy_dx - dAx_dy );

} // FUNCTION : SetBFieldIC
#endif // #ifdef MHD
#endif // #if ( MODEL == HYDRO )



//-------------------------------------------------------------------------------------------------------
// Function    :  LoadICTable
// Description :  Load data from the file assigned to 'NeutronStar_ICFile' in Input__TestProb
//-------------------------------------------------------------------------------------------------------
void LoadICTable()
{

   const bool RowMajor_No  = false;              // load data into the column-major order
   const bool AllocMem_Yes = true;               // allocate memort for NeutronStar_Prof
   const int  NCol         = 4;                  // total number of columns to load
   const int  TargetCols[NCol] = { 0, 1, 2, 3 }; // target columns: {radius, vr, density, pressure}

   double *Table_R, *Table_Dens, *Table_Pres, *Table_Velr;

   NeutronStar_NBin = Aux_LoadTable( NeutronStar_Prof, NeutronStar_ICFile, NCol, TargetCols, RowMajor_No, AllocMem_Yes );

   Table_R    = NeutronStar_Prof + 0*NeutronStar_NBin;
   Table_Velr = NeutronStar_Prof + 1*NeutronStar_NBin;
   Table_Dens = NeutronStar_Prof + 2*NeutronStar_NBin;
   Table_Pres = NeutronStar_Prof + 3*NeutronStar_NBin;

   // convert to code units (assuming progentior model is in cgs)
   for (int b=0; b<NeutronStar_NBin; b++)
   {
      Table_R[b]    /= UNIT_L;
      Table_Velr[b] /= UNIT_V;
      Table_Dens[b] /= UNIT_D;
      Table_Pres[b] /= UNIT_P;
   }

} // FUNCTION : LoadICTable()



//-------------------------------------------------------------------------------------------------------
// Function    :  End_GREP_MigrationTest
// Description :  Free memory before terminating the program
//
// Note        :  1. Linked to the function pointer "End_User_Ptr" to replace "End_User()"
//
// Parameter   :  None
//-------------------------------------------------------------------------------------------------------
void End_GREP_MigrationTest()
{

   delete [] NeutronStar_Prof;
   NeutronStar_Prof = NULL;

} // FUNCTION : End_GREP_MigrationTest



//-------------------------------------------------------------------------------------------------------
// Function    :  Record_MigrationTest
// Description :  Interface for calling multiple record functions
//-------------------------------------------------------------------------------------------------------
void Record_MigrationTest()
{

// the maximum density around the box center
   Record_CentralDens();

} // FUNCTION : Record_MigrationTest()



//-------------------------------------------------------------------------------------------------------
// Function    :  Record_CentralDens
// Description :  Record the maximum density around the box center
//-------------------------------------------------------------------------------------------------------
void Record_CentralDens()
{

   const char   filename_central_dens[] = "Record__CentralDens";
   const double BoxCenter[3]            = { 0.5*amr->BoxSize[0], 0.5*amr->BoxSize[1], 0.5*amr->BoxSize[2] };

// allocate memory for per-thread arrays
#  ifdef OPENMP
   const int NT = OMP_NTHREAD;   // number of OpenMP threads
#  else
   const int NT = 1;
#  endif

   double DataCoord[4] = { -__DBL_MAX__ }, **OMP_DataCoord=NULL;
   Aux_AllocateArray2D( OMP_DataCoord, NT, 4 );


#  pragma omp parallel
   {
#     ifdef OPENMP
      const int TID = omp_get_thread_num();
#     else
      const int TID = 0;
#     endif

//    initialize arrays
      OMP_DataCoord[TID][0] = -__DBL_MAX__;
      for (int b=1; b<4; b++)   OMP_DataCoord[TID][b] = 0.0;

      for (int lv=0; lv<NLEVEL; lv++)
      {
         const double dh = amr->dh[lv];

#        pragma omp for schedule( runtime )
         for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
         {
            if ( amr->patch[0][lv][PID]->son != -1 )  continue;

            for (int k=0; k<PS1; k++)  {  const double z = amr->patch[0][lv][PID]->EdgeL[2] + (k+0.5)*dh;
            for (int j=0; j<PS1; j++)  {  const double y = amr->patch[0][lv][PID]->EdgeL[1] + (j+0.5)*dh;
            for (int i=0; i<PS1; i++)  {  const double x = amr->patch[0][lv][PID]->EdgeL[0] + (i+0.5)*dh;

               const double dx = x - BoxCenter[0];
               const double dy = y - BoxCenter[1];
               const double dz = z - BoxCenter[2];
               const double r2 = SQR(dx) + SQR(dy) + SQR(dz);

               const double dens = amr->patch[ amr->FluSg[lv] ][lv][PID]->fluid[DENS][k][j][i];

               if ( dens > OMP_DataCoord[TID][0] )
               {
                  OMP_DataCoord[TID][0] = dens;
                  OMP_DataCoord[TID][1] = x;
                  OMP_DataCoord[TID][2] = y;
                  OMP_DataCoord[TID][3] = z;
               }

            }}} // i,j,k
         } // for (int PID=0; PID<amr->NPatchComma[lv][1]; PID++)
      } // for (int lv=0; lv<NLEVEL; lv++)
   } // OpenMP parallel region


// find the maximum over all OpenMP threads
   for (int TID=0; TID<NT; TID++)
   {
      if ( OMP_DataCoord[TID][0] > DataCoord[0] )
         for (int b=0; b<4; b++)   DataCoord[b] = OMP_DataCoord[TID][b];
   }

// free per-thread arrays
   Aux_DeallocateArray2D( OMP_DataCoord );


// collect data from all ranks
# ifndef SERIAL
   {
      double DataCoord_All[4 * MPI_NRank];

      MPI_Allgather( DataCoord, 4, MPI_DOUBLE, DataCoord_All, 4, MPI_DOUBLE, MPI_COMM_WORLD );

      for (int i=0; i<MPI_NRank; i++)
      {
         if ( DataCoord_All[4 * i] > DataCoord[0] )
            for (int b=0; b<4; b++)   DataCoord[b] = DataCoord_All[4 * i + b];
      }
   }
# endif // ifndef SERIAL


// output to file
   if ( MPI_Rank == 0 )
   {

      static bool FirstTime = true;

//    output file header
      if ( FirstTime )
      {
         if ( Aux_CheckFileExist(filename_central_dens) )
         {
             Aux_Message( stderr, "WARNING : file \"%s\" already exists !!\n", filename_central_dens );
         }

         else
         {
             FILE *file_max_dens = fopen( filename_central_dens, "w" );
             fprintf( file_max_dens, "#%14s %12s %16s %16s %16s %16s\n",
                                     "Time", "Step", "Dens", "PosX", "PosY", "PosZ" );
             fclose( file_max_dens );
         }

         FirstTime = false;
      }

      FILE *file_max_dens = fopen( filename_central_dens, "a" );
      fprintf( file_max_dens, "%15.7e %12ld %16.7e %16.7e %16.7e %16.7e\n",
               Time[0]*UNIT_T, Step, DataCoord[0]*UNIT_D, DataCoord[1]*UNIT_L, DataCoord[2]*UNIT_L, DataCoord[3]*UNIT_L );
      fclose( file_max_dens );

   } // if ( MPI_Rank == 0 )

} // FUNCTION : Record_CentralDens()



//-------------------------------------------------------------------------------------------------------
// Function    :  Init_TestProb_Hydro_GREP_MigrationTest
// Description :  Test problem initializer
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
void Init_TestProb_Hydro_GREP_MigrationTest()
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


// validate the compilation flags and runtime parameters
   Validate();


#  if ( MODEL == HYDRO )
// set the problem-specific runtime parameters
   SetParameter();

// Load IC Table
   if ( OPT__INIT != INIT_BY_RESTART )   LoadICTable();

// procedure to enable a problem-specific function:
// 1. define a user-specified function (example functions are given below)
// 2. declare its function prototype on the top of this file
// 3. set the corresponding function pointer below to the new problem-specific function
// 4. enable the corresponding runtime option in "Input__Parameter"
//    --> for instance, enable OPT__OUTPUT_USER for Output_User_Ptr
   Init_Function_User_Ptr         = SetGridIC;
#  ifdef MHD
   Init_Function_BField_User_Ptr  = SetBFieldIC;
#  endif
   Flag_User_Ptr                  = NULL;
   Aux_Record_User_Ptr            = Record_MigrationTest;
   End_User_Ptr                   = End_GREP_MigrationTest;
#  endif // #if ( MODEL == HYDRO )


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : Init_TestProb_Hydro_GREP_MigrationTest
