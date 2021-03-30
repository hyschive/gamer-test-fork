#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Plot the evolution of central density
#
#  Last Updated: 2020/08/20
#  He-Feng Hsieh

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt


# load data
time, rhoc = np.genfromtxt("Record__CentralDens", usecols = [0, 2], unpack = 1)

# plot the evolution of central density
fig, ax = plt.subplots()

ax.plot(time * 1e3, rhoc / 1e14, c = "k")
ax.set_xlabel("Time (ms)")
ax.set_ylabel(r"Central density ($10^{14}$ g cm$^{-3}$)")

fig.tight_layout()
plt.savefig("Rhoc.png")
