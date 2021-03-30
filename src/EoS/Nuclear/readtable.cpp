#include "NuclearEoS.h"
#include "GAMER.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )


# ifdef SUPPORT_HDF5

#  define H5_USE_16_API    1
#  include "hdf5.h"

#  ifdef FLOAT8
#    define H5T_GAMER_REAL H5T_NATIVE_DOUBLE
#  else
#    define H5T_GAMER_REAL H5T_NATIVE_FLOAT
#  endif

# else

#  error : ERROR : must enable SUPPORT_HDF5 for EOS_NUCLEAR !!

# endif // #ifdef SUPPORT_HDF5 ... else ...



extern int    g_nrho;
extern int    g_neps;
extern int    g_nye;
extern int    g_nmode;
extern double g_energy_shift;

extern real *g_alltables;
extern real *g_alltables_mode;
extern real *g_logrho;
extern real *g_logeps;
extern real *g_yes;
extern real *g_logtemp_mode;
extern real *g_entr_mode;
extern real *g_logprss_mode;



// catch HDF5 errors
#define HDF5_ERROR( fn_call )                                           \
{                                                                       \
   const int _error_code = fn_call;                                     \
   if ( _error_code < 0 )                                               \
   {                                                                    \
      Aux_Error( ERROR_INFO, "HDF5 call '%s' returned error code %d",   \
                 #fn_call, _error_code );                               \
   }                                                                    \
}



//-------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_ReadTable
// Description :  Load the EoS table from the disk
//
// Note        :  1. Invoked by EoS_Init_Nuclear()
//
// Parameter   :  nuceos_table_name : Filename
//
// Return      :  EoS tables
//-------------------------------------------------------------------------------------
void nuc_eos_C_ReadTable( char *nuceos_table_name )
{

   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ...\n", __FUNCTION__ );


   if ( MPI_Rank == 0 )
      Aux_Message( stdout, "   Reading nuclear EoS table: %s\n", nuceos_table_name );

// check file existence
   if ( !Aux_CheckFileExist(nuceos_table_name) )
      Aux_Error( ERROR_INFO, "file \"%s\" does not exist !!\n", nuceos_table_name );


// use these two macros to easily read in a lot of variables in the same way
// --> the first reads in one variable of a given type completely
#  define READ_EOS_HDF5( NAME, VAR, TYPE, MEM )                                        \
   {                                                                                   \
      hid_t dataset;                                                                   \
      HDF5_ERROR(  dataset = H5Dopen( file, NAME )  );                                 \
      HDF5_ERROR(  H5Dread( dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR )  );        \
      HDF5_ERROR(  H5Dclose( dataset )  );                                             \
   }

#  define READ_EOSTABLE_HDF5( NAME, OFF )                                              \
   {                                                                                   \
      hsize_t offset[2] = { OFF, 0 };                                                  \
      H5Sselect_hyperslab( mem3, H5S_SELECT_SET, offset, NULL, var3, NULL );           \
      READ_EOS_HDF5( NAME, alltables_temp, H5T_GAMER_REAL, mem3 );                     \
   }

#  define READ_EOSTABLE_MODE_HDF5( NAME,OFF )                                          \
   {                                                                                   \
      hsize_t offset[2] = { OFF,0 };                                                   \
      H5Sselect_hyperslab( mem3_mode, H5S_SELECT_SET, offset, NULL, var3_mode, NULL ); \
      READ_EOS_HDF5( NAME, alltables_mode_temp, H5T_GAMER_REAL, mem3_mode );           \
   }



// open file
   hid_t file;
   HDF5_ERROR(  file = H5Fopen( nuceos_table_name, H5F_ACC_RDONLY, H5P_DEFAULT )  );


// read size of tables
   READ_EOS_HDF5( "pointsrho",    &g_nrho,  H5T_NATIVE_INT, H5S_ALL );
   READ_EOS_HDF5( "pointsenergy", &g_neps,  H5T_NATIVE_INT, H5S_ALL );
   READ_EOS_HDF5( "pointsye",     &g_nye,   H5T_NATIVE_INT, H5S_ALL );
   READ_EOS_HDF5( "points_mode",  &g_nmode, H5T_NATIVE_INT, H5S_ALL );


// allocate memory for tables
   real *alltables_temp      = NULL;
   real *alltables_mode_temp = NULL;

   if (  ! ( alltables_temp      = (real*)malloc(g_nrho*g_neps*g_nye*NUC_TABLE_NVAR*sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( alltables_mode_temp = (real*)malloc(g_nrho*g_nmode*g_nye*3            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_alltables         = (real*)malloc(g_nrho*g_neps*g_nye*NUC_TABLE_NVAR*sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n");

   if (  ! ( g_alltables_mode    = (real*)malloc(g_nrho*g_nmode*g_nye*3            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n");

   if (  ! ( g_logrho            = (real*)malloc(g_nrho                            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_logeps            = (real*)malloc(g_neps                            *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n" );

   if (  ! ( g_yes               = (real*)malloc(g_nye                             *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n");

   if (  ! ( g_logtemp_mode      = (real*)malloc(g_nmode                           *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n");

   if (  ! ( g_entr_mode         = (real*)malloc(g_nmode                           *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n");

   if (  ! ( g_logprss_mode      = (real*)malloc(g_nmode                           *sizeof(real)) )  )
      Aux_Error( ERROR_INFO, "cannot allocate memory for EOS table !!\n");


// prepare HDF5 to read hyperslabs into alltables_temp[]
   hsize_t table_dims[2]      = { NUC_TABLE_NVAR, g_nrho*g_neps*g_nye };
   hsize_t var3[2]            = { 1, g_nrho*g_neps*g_nye };
   hid_t   mem3               = H5Screate_simple( 2, table_dims, NULL );

   hsize_t table_dims_mode[2] = { 3, g_nrho*g_nmode*g_nye };
   hsize_t var3_mode[2]       = { 1, g_nrho*g_nmode*g_nye };
   hid_t   mem3_mode          = H5Screate_simple( 2, table_dims_mode, NULL );


// read alltables_temp[]
   READ_EOSTABLE_HDF5( "logpress",  0 );
   READ_EOSTABLE_HDF5( "logtemp",   1 );
   READ_EOSTABLE_HDF5( "entropy",   2 );
   READ_EOSTABLE_HDF5( "munu",      3 );
   READ_EOSTABLE_HDF5( "cs2",       4 );

// chemical potentials
   READ_EOSTABLE_HDF5( "muhat",     5 );
   READ_EOSTABLE_HDF5( "mu_e",      6 );
   READ_EOSTABLE_HDF5( "mu_p",      7 );
   READ_EOSTABLE_HDF5( "mu_n",      8 );

// compositions
   READ_EOSTABLE_HDF5( "Xa",        9 );
   READ_EOSTABLE_HDF5( "Xh",       10 );
   READ_EOSTABLE_HDF5( "Xn",       11 );
   READ_EOSTABLE_HDF5( "Xp",       12 );

// average nucleus
   READ_EOSTABLE_HDF5( "Abar",     13 );
   READ_EOSTABLE_HDF5( "Zbar",     14 );

// Gamma
   READ_EOSTABLE_HDF5( "gamma",    15 );

// energy for temp, entr modes
   READ_EOSTABLE_MODE_HDF5( "logenergy_temp", 0 );
   READ_EOSTABLE_MODE_HDF5( "logenergy_entr", 1 );
   READ_EOSTABLE_MODE_HDF5( "logenergy_prss", 2 );

// read additional tables and variables
   READ_EOS_HDF5( "logrho",        g_logrho,        H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "logenergy",     g_logeps,        H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "ye",            g_yes,           H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "logtemp_mode",  g_logtemp_mode,  H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "entr_mode",     g_entr_mode,     H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "logprss_mode",  g_logprss_mode,  H5T_GAMER_REAL,    H5S_ALL );
   READ_EOS_HDF5( "energy_shift", &g_energy_shift,  H5T_NATIVE_DOUBLE, H5S_ALL );

   HDF5_ERROR(  H5Sclose( mem3      )  );
   HDF5_ERROR(  H5Sclose( mem3_mode )  );
   HDF5_ERROR(  H5Fclose( file      )  );


// change ordering of g_alltables[] so that the table kind is the fastest changing index
   for (int iv=0; iv<NUC_TABLE_NVAR; iv++)
   for (int k=0; k<g_nye;  k++)
   for (int j=0; j<g_neps; j++)
   for (int i=0; i<g_nrho; i++)
   {
      const long indold = i + g_nrho*( j + g_neps*(k + g_nye*iv) );
      const long indnew = iv + NUC_TABLE_NVAR*( i + g_nrho*(j + g_neps*k) );

      g_alltables[indnew] = alltables_temp[indold];
   }

   for (int iv=0; iv<3; iv++)
   for (int k=0; k<g_nye;   k++)
   for (int j=0; j<g_nmode; j++)
   for (int i=0; i<g_nrho;  i++)
   {
      const long indold = i + g_nrho*( j + g_nmode*(k + g_nye*iv) );
      const long indnew = iv + 3*( i + g_nrho*(j + g_nmode*k) );

      g_alltables_mode[indnew] = alltables_mode_temp[indold];
   }


// free memory of temporary arrays
   free( alltables_temp      );
   free( alltables_mode_temp );


// set the EoS table pointers
   h_EoS_Table[NUC_TAB_ALL      ] = g_alltables;
   h_EoS_Table[NUC_TAB_ALL_MODE ] = g_alltables_mode;
   h_EoS_Table[NUC_TAB_RHO      ] = g_logrho;
   h_EoS_Table[NUC_TAB_EPS      ] = g_logeps;
   h_EoS_Table[NUC_TAB_YE       ] = g_yes;
   h_EoS_Table[NUC_TAB_TEMP_MODE] = g_logtemp_mode;
   h_EoS_Table[NUC_TAB_ENTR_MODE] = g_entr_mode;
   h_EoS_Table[NUC_TAB_PRES_MODE] = g_logprss_mode;


   if ( MPI_Rank == 0 )    Aux_Message( stdout, "%s ... done\n", __FUNCTION__ );

} // FUNCTION : nuc_eos_C_ReadTable



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
