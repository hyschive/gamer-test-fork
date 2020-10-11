The procedures of this LSS simulation includes:

(a) Using CAMB/axionCAMB to produce transfer function (the input file planck_2018.ini/planck_2018_axion.ini for CAMB/axionCAMB is attached for reference)

(b) Run the script "./set_velocities_zero.py F(T)" to produce the CAMB(axionCAMB) transfer function: planck_2018_transfer_out_no_vel.dat(planck_2018_axion_transfer_out_no_vel.dat) needed by MUSIC.

(c) Modifying the cosmology paramters consistent with CAMB/axionCAMB .ini files, as well as the input file parameters in ics_example.conf (transfer = camb_file; transfer_file = planck_2018_axion_transfer_out_no_vel.dat/planck_2018_axion_transfer_out_no_vel.dat; format = generic; filename = planck_2018_tf_no_vel.hdf5/planck_2018_axion_tf_no_vel.hdf5). Then run the MUSIC: "./MUSIC ics_example.conf".

(d) Run the script "./make_umic_from_hdf5.py S(D) hdf5_filename" to produce UM_IC file for single/double precision GAMER.

(e) Run the GAMER.
