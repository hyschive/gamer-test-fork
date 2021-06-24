#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



GPU_DEVICE static
void find_temp_bdry( const real x, const real y, const real z, 
                     real *found_lt, const real *alltables_mode,
                     const int nx, const int ny, const int nz, const int ntemp, 
                     const real *xt, const real *yt, const real *zt, 
                     const real *logtemp, const int keymode, int *keyerr );
#ifdef __CUDACC__
GPU_DEVICE static
void find_temp( const real x, const real y, const real z, 
                real *found_lt, const real *alltables_mode,
                const int nx, const int ny, const int nz, const int ntemp, 
                const real *xt, const real *yt, const real *zt, 
                const real *logtemp, const int keymode, int *keyerr );
#endif




//-------------------------------------------------------------------------------------
// Function    :  find_temp
// Description :  Finding energy from specific internal energy (0)
//                                    entropy                  (2)
//                                    pressure                 (3) mode
//                Using 3D Catmull-Rom cubic interpolation formula
//                to search the corresponding temperature given (rho, (eps, e, P), Y_e)
//
// Note        :  1. Use 3D Catmull-Rom cubic interpolation formula
//                   to search the corresponding energy given (rho, (eps, e, P), Y_e)
//                2. Invoked by nuc_eos_C_short()
//
// Parameter   :  x              : input vector of first  variable (rho)
//                y              : input vector of second variable (eps, e, P)
//                z              : input vector of third  variable (Y_e)
//                found_lt       : output log(temp) of interpolated function values
//                alltables_mode : 3d array of tabulated logtemp
//                nx             : x-dimension of table
//                ny             : y-dimension of table
//                nz             : z-dimension of table
//                ntemp          : size of temperature array in the Nuclear Eos table
//                xt             : vector of x-coordinates of table
//                yt             : vector of y-coordinates of table
//                zt             : vector of z-coordinates of table
//                logtemp        : log(T) array in the table
//                keymode        : which mode we will use
//                                 0: energy mode      (coming in with eps)
//                                 2: entropy mode     (coming in with entropy)
//                                 3: pressure mode    (coming in with P)
//                keyerr         : output error
//-------------------------------------------------------------------------------------
GPU_DEVICE
void find_temp( const real x, const real y, const real z, 
                real *found_lt, const real *alltables_mode,
                const int nx, const int ny, const int nz, const int ntemp, 
                const real *xt, const real *yt, const real *zt, 
                const real *logtemp, const int keymode, int *keyerr )
{

   const real *pv = NULL;

   real   dx, dy, dz, dxi, dyi, dzi;
   real   delx, dely, delz;
   real   u[4], v[4], w[4];
   real   r[4], q[4];
   real   vox;
   int    ix, iy, iz;
   int    nxy;

   nxy = nx*ny;

// determine spacing parameters of equidistant (!!!) table
#if 1
   dx = ( xt[nx-1] - xt[0] ) / (real)(nx-1);
   dy = ( yt[ny-1] - yt[0] ) / (real)(ny-1);
   dz = ( zt[nz-1] - zt[0] ) / (real)(nz-1);
   
   dxi = (real)1.0 / dx;
   dyi = (real)1.0 / dy;
   dzi = (real)1.0 / dz;
#endif

#if 0
   dx = drho;
   dy = dtemp;
   dz = dye;
   dxi = drhoi;
   dyi = dtempi;
   dzi = dyei;
#endif


   // determine location in table
   ix = (int)( (x - xt[0] + (real)1.0e-10)*dxi );
   iy = (int)( (y - yt[0] + (real)1.0e-10)*dyi );
   iz = (int)( (z - zt[0] + (real)1.0e-10)*dzi );
   
    
   // linear interpolation at boundaries
   if ( ix == 0 || iy == 0 || iz == 0 ||
        ix == nx-2 || iy == ny-2 || iz == nz-2 )
   {
      find_temp_bdry( x, y, z, found_lt, alltables_mode,
                      nx, ny, nz, ntemp, xt, yt, zt, logtemp, keymode, keyerr );
      return;
   }


// differences
   delx = ( x - xt[ix] )*dxi;
   dely = ( y - yt[iy] )*dyi;
   delz = ( z - zt[iz] )*dzi;


// factors for Catmull-Rom interpolation
   const real delx2 =  SQR( delx );
   const real dely2 =  SQR( dely );
   const real delz2 =  SQR( delz );
   const real delx3 = CUBE( delx );
   const real dely3 = CUBE( dely );
   const real delz3 = CUBE( delz );

   u[0] = (real)-0.5*delx3 +           delx2 - (real)0.5*delx;
   u[1] = (real) 1.5*delx3 - (real)2.5*delx2 + (real)1.0;
   u[2] = (real)-1.5*delx3 + (real)2.0*delx2 + (real)0.5*delx;
   u[3] = (real) 0.5*delx3 - (real)0.5*delx2;

   v[0] = (real)-0.5*dely3 +           dely2 - (real)0.5*dely;
   v[1] = (real) 1.5*dely3 - (real)2.5*dely2 + (real)1.0;
   v[2] = (real)-1.5*dely3 + (real)2.0*dely2 + (real)0.5*dely;
   v[3] = (real) 0.5*dely3 - (real)0.5*dely2;

   w[0] = (real)-0.5*delz3 +           delz2 - (real)0.5*delz;
   w[1] = (real) 1.5*delz3 - (real)2.5*delz2 + (real)1.0;
   w[2] = (real)-1.5*delz3 + (real)2.0*delz2 + (real)0.5*delz;
   w[3] = (real) 0.5*delz3 - (real)0.5*delz2;
   
   
   int iv;
   if      ( keymode == NUC_MODE_ENGY ) iv = 0; // energy table for the energy mode
   else if ( keymode == NUC_MODE_ENTR ) iv = 1; // energy table for the entropy mode
   else if ( keymode == NUC_MODE_PRES ) iv = 2; // energy table for the pressure mode
   
   vox = (real)0.0;
    
   pv = alltables_mode + iv + 3*( (ix-1) + (iy-1)*nx + (iz-1)*nxy );

   for (int k=0; k<4; k++)
   {
      q[k] = (real)0.0;

      for (int j=0; j<4; j++)
      {
         r[j] = (real)0.0;

         for (int i=0; i<4; i++)
         {
            r[j] += u[i]* *pv;
            pv   += 3;
         }

         q[k] += v[j]*r[j];
         pv   += 3*nx - 4*3;
      }

      vox += w[k]*q[k];
      pv  += nxy*3 - 4*3*nx;
   } // for (int k=0; k<4; k++)

   *found_lt = vox;


// linear interpolation when cubic interpolations failed
   if ( vox != vox )
   {
      if ( keymode == NUC_MODE_PRES ) { *keyerr = 683; return; }
      find_temp_bdry( x, y, z, found_lt, alltables_mode,
                      nx, ny, nz, ntemp, xt, yt, zt, logtemp, keymode, keyerr );
      return;
   }


  return;

} // FUNCTION : find_temp



//-------------------------------------------------------------------------------------
// Function    :  find_temp_bdry
// Description :  Finding energy from specific internal energy (0)
//                                    entropy                  (2)
//                                    pressure                 (3) mode
//                Using linear interpolation at boundaries of table
//                to search the corresponding energy given
//                (rho, (eps, e, P), Y_e)
//
// Parameter   :  x              : input vector of first  variable (rho)
//                y              : input vector of second variable (eps, e, P)
//                z              : input vector of third  variable (Y_e)
//                found_t        : output log(T) of interpolated function values
//                alltables_mode : 3d array of tabulated logenergy
//                nx             : x-dimension of table
//                ny             : y-dimension of table
//                nz             : z-dimension of table
//                ntemp          : size of temperature array in the Nuclear Eos table
//                xt             : vector of x-coordinates of table
//                yt             : vector of y-coordinates of table
//                zt             : vector of z-coordinates of table
//                logtemp        : log(temp) array in the table
//                keymode        : which mode we will use
//                                 0: energy mode      (coming in with eps)
//                                 2: entropy mode     (coming in with entropy)
//                                 3: pressure mode    (coming in with P)
//                keyerr         : output error
//-------------------------------------------------------------------------------------
GPU_DEVICE
void find_temp_bdry( const real x, const real y, const real z, 
                     real *found_lt, const real *alltables_mode,
                     const int nx, const int ny, const int nz, const int ntemp, 
                     const real *xt, const real *yt, const real *zt, 
                     const real *logtemp, const int keymode, int *keyerr )
{

// helper variables
  real fh[8], delx, dely, delz, a[8];
  real dx, dy, dz, dxi, dyi, dzi, dxyi, dxzi, dyzi, dxyzi;
  int  ix, iy, iz;


// determine spacing parameters of equidistant (!!!) table
#  if 1
   dx  = ( xt[nx-1] - xt[0] ) / (real)(nx-1);
   dy  = ( yt[ny-1] - yt[0] ) / (real)(ny-1);
   dz  = ( zt[nz-1] - zt[0] ) / (real)(nz-1);
   
   dxi = (real)1.0 / dx;
   dyi = (real)1.0 / dy;
   dzi = (real)1.0 / dz;
#endif

#if 0
   dx = drho;
   dy = dtemp;
   dz = dye;

   dxi = drhoi;
   dyi = dtempi;
   dzi = dyei;
#endif
  
   dxyi  = dxi*dyi;
   dxzi  = dxi*dzi;
   dyzi  = dyi*dzi;
   dxyzi = dxi*dyi*dzi;


// determine location in table
   ix = 1 + (int)( (x - xt[0] - (real)1.0e-10)*dxi );
   iy = 1 + (int)( (y - yt[0] - (real)1.0e-10)*dyi );
   iz = 1 + (int)( (z - zt[0] - (real)1.0e-10)*dzi );
   
   ix = MAX( 1, MIN( ix, nx-1 ) );
   iy = MAX( 1, MIN( iy, ny-1 ) );
   iz = MAX( 1, MIN( iz, nz-1 ) );
   

// set up aux vars for interpolation   
   delx = xt[ix] - x;
   dely = yt[iy] - y;
   delz = zt[iz] - z;
   
   int idx[8];
   idx[0] = 3*(  (ix  ) + nx*( (iy  ) + ny*(iz  ) )  );
   idx[1] = 3*(  (ix-1) + nx*( (iy  ) + ny*(iz  ) )  );
   idx[2] = 3*(  (ix  ) + nx*( (iy-1) + ny*(iz  ) )  );
   idx[3] = 3*(  (ix  ) + nx*( (iy  ) + ny*(iz-1) )  );
   idx[4] = 3*(  (ix-1) + nx*( (iy-1) + ny*(iz  ) )  );
   idx[5] = 3*(  (ix-1) + nx*( (iy  ) + ny*(iz-1) )  );
   idx[6] = 3*(  (ix  ) + nx*( (iy-1) + ny*(iz-1) )  );
   idx[7] = 3*(  (ix-1) + nx*( (iy-1) + ny*(iz-1) )  );
   
   int iv;
   if      ( keymode == NUC_MODE_ENGY )  iv = 0; // energy table for the temperature mode
   else if ( keymode == NUC_MODE_ENTR )  iv = 1; // energy table for the entropy mode
   else if ( keymode == NUC_MODE_PRES )  iv = 2; // energy table for the pressure mode
   
   
// set up aux vars for interpolation assuming array ordering (iv, ix, iy, iz)
   fh[0] = alltables_mode[ iv + idx[0] ];
   fh[1] = alltables_mode[ iv + idx[1] ];
   fh[2] = alltables_mode[ iv + idx[2] ];
   fh[3] = alltables_mode[ iv + idx[3] ];
   fh[4] = alltables_mode[ iv + idx[4] ];
   fh[5] = alltables_mode[ iv + idx[5] ];
   fh[6] = alltables_mode[ iv + idx[6] ];
   fh[7] = alltables_mode[ iv + idx[7] ];


// set up coeffs of interpolation polynomical and evaluate function values
   a[0] = fh[0];
   a[1] = dxi  *( fh[1] - fh[0] );
   a[2] = dyi  *( fh[2] - fh[0] );
   a[3] = dzi  *( fh[3] - fh[0] );
   a[4] = dxyi *( fh[4] - fh[1] - fh[2] + fh[0] );
   a[5] = dxzi *( fh[5] - fh[1] - fh[3] + fh[0] );
   a[6] = dyzi *( fh[6] - fh[2] - fh[3] + fh[0] );
   a[7] = dxyzi*( fh[7] - fh[0] + fh[1] + fh[2] +
                  fh[3] - fh[4] - fh[5] - fh[6] );

   *found_lt = a[0]
             + a[1]*delx
             + a[2]*dely
             + a[3]*delz
             + a[4]*delx*dely
             + a[5]*delx*delz
             + a[6]*dely*delz
             + a[7]*delx*dely*delz;
   
   if ( *found_lt != *found_lt || 
        ! ( *found_lt>logtemp[0] && *found_lt<logtemp[ntemp-1] )  )
      *keyerr = 665;


  return;

} // FUNCTION : find_temp_bdry



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
