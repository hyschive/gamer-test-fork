#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Plot the slice of density profile at the center (w/o grid)
#    and generate the movies
#
#  Last Updated: 2019/11/01
#  He-Feng Hsieh


import matplotlib
matplotlib.use("Agg")

import os
import yt
from glob import glob
import matplotlib.pyplot as plt


# setting
Plot_Density      = True  # plot the slice of density at the center
Plot_Density_grid = True  # plot the slice of density at the center with grid
Gene_animation    = True  # generate animation


# obtain all the HDF5 output
fn_in = glob("Data_" + "[0-9]" * 6)
fn_in.sort()

fn_out_fmt      = "{}_Slice_x_density.png"
fn_out_fmt_grid = "{}_Slice_x_density_grid.png"


# load data
for fn in fn_in:
    ds = yt.load(fn)

    # retrieve the simulation time of lv = 0
    time = ds.parameters["Time"][0]

    if Plot_Density:
        slc = yt.SlicePlot(ds, "x", "density", center = "c")
        slc.annotate_title("Time = {:.2f} ms".format(time))
        slc.save(fn_out_fmt.format(fn))

    if Plot_Density_grid:
        try:
            slc.annotate_grids()
        except:
            slc = yt.SlicePlot(ds, "x", "density", center = "c")
            slc.annotate_title("Time = {:.2f} ms".format(time))
            slc.annotate_grids()

        slc.save(fn_out_fmt_grid.format(fn))

    # free memory
    if Plot_Density or Plot_Density_grid:
        del slc

    del ds


# generate animation
if Gene_animation:
    if Plot_Density:
        fn_in  = r"Data_%6d_Slice_x_density.png"
        fn_out = "Slice_density.mp4"
        cmd = 'ffmpeg -r 25 -i {} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec h264 -pix_fmt yuv420p {}'.format(fn_in, fn_out)

        os.system(cmd)


    if Plot_Density_grid:
        fn_in  = r"Data_%6d_Slice_x_density_grid.png"
        fn_out = "Slice_density_grid.mp4"
        cmd = 'ffmpeg -r 25 -i {} -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec h264 -pix_fmt yuv420p {}'.format(fn_in, fn_out)

        os.system(cmd)

