#include "NuclearEoS.h"

#if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )



#ifdef __CUDACC__
GPU_DEVICE static
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt );
#endif




//-------------------------------------------------------------------------------------
// Function    :  nuc_eos_C_linterp_some
// Description :  Find thermodynamic variables using linear interpolation by searching
//                the tabulated nuclear EoS
//
// Note        :  1. Invoked by nuc_eos_C_short() and nuc_eos_C_cubinterp_some()
//
// Parameter   :  x           : Input vector of first  variable (rho)
//                y           : Input vector of second variable (eps)
//                z           : Input vector of third  variable (Y_e)
//                output_vars : Output variables of interpolated function values
//                alltables   : 3D array of tabulated variables
//                nx          : X-dimension of table
//                ny          : Y-dimension of table
//                nz          : Z-dimension of table
//                nvars       : Number of variables we will find
//                xt          : Vector of x-coordinates of table
//                yt          : Vector of y-coordinates of table
//                zt          : Vector of z-coordinates of table
//
// Return      :  output_vars
//-------------------------------------------------------------------------------------
GPU_DEVICE
void nuc_eos_C_linterp_some( const real x, const real y, const real z,
                             real *output_vars, const real *alltables,
                             const int nx, const int ny, const int nz, const int nvars,
                             const real *xt, const real *yt, const real *zt )
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
#  endif

#  if 0
   dx  = drho;
   dy  = deps;
   dz  = dye;

   dxi = drhoi;
   dyi = depsi;
   dzi = dyei;
#  endif

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

   idx[0] = NUC_TABLE_NVAR*(  (ix  ) + nx*( (iy  ) + ny*(iz  ) )  );
   idx[1] = NUC_TABLE_NVAR*(  (ix-1) + nx*( (iy  ) + ny*(iz  ) )  );
   idx[2] = NUC_TABLE_NVAR*(  (ix  ) + nx*( (iy-1) + ny*(iz  ) )  );
   idx[3] = NUC_TABLE_NVAR*(  (ix  ) + nx*( (iy  ) + ny*(iz-1) )  );
   idx[4] = NUC_TABLE_NVAR*(  (ix-1) + nx*( (iy-1) + ny*(iz  ) )  );
   idx[5] = NUC_TABLE_NVAR*(  (ix-1) + nx*( (iy  ) + ny*(iz-1) )  );
   idx[6] = NUC_TABLE_NVAR*(  (ix  ) + nx*( (iy-1) + ny*(iz-1) )  );
   idx[7] = NUC_TABLE_NVAR*(  (ix-1) + nx*( (iy-1) + ny*(iz-1) )  );


   for (int iv=0; iv<nvars; iv++)
   {
//    set up aux vars for interpolation assuming array ordering (iv, ix, iy, iz)
      fh[0] = alltables[ iv + idx[0] ];
      fh[1] = alltables[ iv + idx[1] ];
      fh[2] = alltables[ iv + idx[2] ];
      fh[3] = alltables[ iv + idx[3] ];
      fh[4] = alltables[ iv + idx[4] ];
      fh[5] = alltables[ iv + idx[5] ];
      fh[6] = alltables[ iv + idx[6] ];
      fh[7] = alltables[ iv + idx[7] ];

//    set up coeffs of interpolation polynomical and evaluate function values
      a[0] = fh[0];
      a[1] = dxi  *( fh[1] - fh[0] );
      a[2] = dyi  *( fh[2] - fh[0] );
      a[3] = dzi  *( fh[3] - fh[0] );
      a[4] = dxyi *( fh[4] - fh[1] - fh[2] + fh[0] );
      a[5] = dxzi *( fh[5] - fh[1] - fh[3] + fh[0] );
      a[6] = dyzi *( fh[6] - fh[2] - fh[3] + fh[0] );
      a[7] = dxyzi*( fh[7] - fh[0] + fh[1] + fh[2] +
                     fh[3] - fh[4] - fh[5] - fh[6] );

      output_vars[iv] = a[0]
                      + a[1]*delx
                      + a[2]*dely
                      + a[3]*delz
                      + a[4]*delx*dely
                      + a[5]*delx*delz
                      + a[6]*dely*delz
                      + a[7]*delx*dely*delz;
   } // for (int iv=0; iv<nvars; iv++)


   return;

} // FUNCTION : nuc_eos_C_linterp_some



#endif // #if ( MODEL == HYDRO  &&  EOS == EOS_NUCLEAR )
