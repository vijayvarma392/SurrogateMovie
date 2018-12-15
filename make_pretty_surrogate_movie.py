#######################################################################################
##
##      Filename: make_pretty_video.py
##
##      Author: Vijay Varma
##
##      Created: 08-01-2018
##
##      Last Modified:
##
##      Description: Make video demonstrating how surrogates work.
##
#######################################################################################

import numpy as np
import matplotlib.pyplot as P
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set_style("ticks")
from palettable.wesanderson import Darjeeling2_5        # pip install palettable
from palettable.wesanderson import FantasticFox2_5
from palettable.wesanderson import GrandBudapest5_5
from palettable.wesanderson import GrandBudapest1_4
from palettable.wesanderson import GrandBudapest5_5 as tmp_wave_colors
import matplotlib.image as mgimg
from matplotlib import animation

import os, sys, string, subprocess
from scipy.optimize import curve_fit
from glob import glob

import LALPy    # from surrogate_modelin repo

# plot settings
my_dpi = 300.
marker_size=8
marker_size_square=5
label_fontsize = 8
title_fontsize = 16
ticks_fontsize = 6
line_width = 1.2
legend_size = 14
label_pad = -5
tick_pad = -2
colors = Darjeeling2_5.mpl_colors       # append all colors
colors += FantasticFox2_5.mpl_colors
colors += GrandBudapest5_5.mpl_colors
colors += GrandBudapest1_4.mpl_colors

#for i in range(len(colors)):
#    P.plot(range(10), np.ones(10)*i, color=colors[i], lw=3)
#P.savefig('test.png')
#P.close()
#print len(colors)
#exit(-1)

color_wave = colors[3]
color_amp = colors[1]
color_fit = colors[18]
color_marker = colors[7]
color_ht = colors[9]
color_marker_nd = colors[15]
color_marker_test_nd = colors[17]
color_test = colors[17]

color_wave_list = tmp_wave_colors.mpl_colors
color_wave_list = [color_wave_list[0], color_wave_list[4], color_wave_list[2], color_wave_list[1]]

# some fixed settings
ZLIM = 0.25     # zlim for plot
# number of steps in the increment plots
NSTEPS = 120

# ---------------------------------------------------------------------------------------
def get_wave(q):
    """ Returns waves of constant length from peak.
    """

    approximant = 'SEOBNRv4'
    chi1 = [0,0,0]
    chi2 = [0,0,0]
    deltaTOverM = 0.1
    omega0 = 2e-2

    t, h = LALPy.generate_LAL_waveform(approximant, q, chi1, chi2, deltaTOverM, omega0)

    Amp = np.abs(h)
    peakIdx = np.argmax(Amp)

    t -= t[peakIdx]

    tmin = -500
    if min(t) > tmin:
        raise Exception('Data not long enough, decrease omega0.')
    keepIdx = t - tmin > -1e-3      # simple hack to ensure t_vec is always nearly the same
    t = t[keepIdx]
    h = h[keepIdx]

    tmax = 100
    keepIdx = t - tmax < 1e-3
    t = t[keepIdx]
    h = h[keepIdx]

    return t, h

# ---------------------------------------------------------------------------------------
def setup_plot():

    fig = P.figure(figsize=(1000./my_dpi, 750./my_dpi))
    ax = P.axes(projection='3d')

    ax.invert_xaxis()       # so that q axis looks prettier
    view_theta = 23
    view_phi = -27
    ax.view_init(view_theta, view_phi)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    #ax.set_axis_off()

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_yticks([-500,0])
    ax.zaxis.set_major_locator(MaxNLocator(5))

    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize, pad=tick_pad)

    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad
    ax.zaxis.labelpad = label_pad

    return ax, fig


# ---------------------------------------------------------------------------------------
def save_plot(fname):
    P.subplots_adjust(left=-0.1, right=0.99, bottom=0.06, top=1.05, wspace=0.2, hspace=0.2)
    P.savefig(fname, dpi=my_dpi)


def get_z_order_waves(q_vec, i):
    q_temp = np.sort(q_vec)
    zorder = np.where(q_vec[i] == q_temp)[0][0]
    return zorder

# ---------------------------------------------------------------------------------------
def make_greedy_plots(Nsteps=NSTEPS):
    """ Makes Nsteps plots of the real part of h22, adding more waveform in each step.
        Also add one waveform at a time.
    """

    ax, fig = setup_plot()
    for i in range(len(q_vec)):
        for k in range(Nsteps):
            dataLen = len(t_vec)
            startIdx = k*dataLen/Nsteps
            endIdx = (k+1)*dataLen/Nsteps
            xVals = np.ones(dataLen)*q_vec[i]
            yVals = t_vec
            zVals = h_vec[i]
            zorder_waves = get_z_order_waves(q_vec, i)

            # plot real parts in increments
            ax.plot3D(xVals[startIdx:endIdx], yVals[startIdx:endIdx], \
                zs=np.real(zVals[startIdx:endIdx]), color=color_wave_list[i], \
                lw=line_width, zorder=zorder_waves)

            ax.set_xlim(0, 10)
            ax.set_ylim(min(t_vec), max(t_vec))
            ax.set_zlim(-ZLIM, ZLIM)
            ax.set_xlabel('$q$', fontsize=label_fontsize)
            ax.set_ylabel('$t$ $(M)$', fontsize=label_fontsize)

            # Make sure to save such that files end up in the same alphabetical order as you
            # want in the video. Here we just add a fake prefix.
            save_plot('%s/aa_qidx%d_wave_%.4d.png'%(plotdir, i, k))
    P.close()

# ---------------------------------------------------------------------------------------
def make_wave_plots(Nsteps=NSTEPS, plotAmp=0):
    """ Makes Nsteps plots of the real part or amplitude of h22,
        adding more waveform in each step.
        If plotAmp = 0, plot only real part in increments, but plot all waveforms simultaneously.
        If plotAmp = 1, plot real part and amplitude. Amplitude in increments.
        If plotAmp = 2, plot amplitude, then roll along t and add empirical nodes.
    """

    if plotAmp != 2:
        ax, fig = setup_plot()

    for k in range(Nsteps):
        if plotAmp == 2:            # separate plot for each step
            ax, fig = setup_plot()

        for i in range(len(q_vec)):
            dataLen = len(t_vec)
            startIdx = k*dataLen/Nsteps
            endIdx = (k+1)*dataLen/Nsteps
            xVals = np.ones(dataLen)*q_vec[i]
            yVals = t_vec
            zVals = h_vec[i]
            zorder_waves = get_z_order_waves(q_vec, i)

            if plotAmp==0:  # plot real parts in increments
                ax.plot3D(xVals[startIdx:endIdx], yVals[startIdx:endIdx], \
                    zs=np.real(zVals[startIdx:endIdx]), color=color_wave_list[i], \
                    lw=line_width, zorder=zorder_waves)
            elif plotAmp==1: # plot fixed real part, but amplitude in increments
                if k == 0:
                    ax.plot3D(xVals, yVals, zs=np.real(zVals), color=color_wave_list[i], \
                        lw=line_width, zorder=zorder_waves)
                ax.plot3D(xVals[startIdx:endIdx], yVals[startIdx:endIdx], \
                    zs=np.abs(zVals[startIdx:endIdx]), color=color_amp, \
                    lw=line_width, zorder=zorder_waves)
            elif plotAmp==2: # plot amplitude, roll along t with height, plot empirical time nodes
                ax.plot3D(xVals, yVals, zs=np.abs(zVals), color=color_amp, lw=line_width)      # plot amp
                tmp_dataLen = 1000
                tmp_x = q_vec[i]
                tmp_y = t_vec[startIdx]
                tmp_z = np.abs(h_vec[i][startIdx])
                tmp_xVals = np.ones(tmp_dataLen)*tmp_x
                tmp_yVals = np.ones(tmp_dataLen)*tmp_y
                tmp_zVals = np.linspace(0, tmp_z, tmp_dataLen)
                ax.plot3D(tmp_xVals, tmp_yVals, zs=tmp_zVals, color=color_ht, lw=line_width)   # roll along t with height
                if tmp_y < 75:      # plot the marker for the roll
                    ax.scatter(tmp_x, tmp_y, tmp_z, c=color_marker, marker='s', s=marker_size_square)

                # plot empirical nodes after rolling past them
                for emp_time in empirical_node_times:
                    if tmp_y >= emp_time:  # If we rolled past this emp node, plot it always
                        emp_idx = np.argmin(np.abs(t_vec - emp_time))
                        this_x = q_vec[i]
                        this_y = t_vec[emp_idx]
                        this_z = np.abs(h_vec[i][emp_idx])
                        # plot marker for empirical nodes
                        ax.scatter(this_x, this_y, this_z, c=color_marker_nd, \
                            marker='o', s=marker_size)

        ax.set_xlim(0, 10)
        ax.set_ylim(min(t_vec), max(t_vec))
        ax.set_zlim(-ZLIM, ZLIM)
        ax.set_xlabel('$q$', fontsize=label_fontsize)
        ax.set_ylabel('$t$ $(M)$', fontsize=label_fontsize)


        # Make sure to save such that files end up in the same alphabetical order as you
        # want in the video. Here we just add a fake prefix.
        if plotAmp==0:
            save_plot('%s/aa_wave_%.4d.png'%(plotdir, k))
        elif plotAmp==1:
            save_plot('%s/ab_waveamp_%.4d.png'%(plotdir, k))
        elif plotAmp==2:
            ax.set_zlim(0, ZLIM)
            save_plot('%s/ba_amp_%.4d.png'%(plotdir, k))
            P.close()

    P.close()


# ---------------------------------------------------------------------------------------
def func(x, a, b, c):
    return a*x**2 + b*x + c

# ---------------------------------------------------------------------------------------
def make_fit_plots(Nsteps=NSTEPS, plotTest=0):
    """ Plot fits accross params space at empirical nodes
        If plotTest = 0, plot fits in increments.
        If plotTest = 1, plot fits, then plot empirical nodes at a
            test case, then plot test amplitude in increments.
    """
    ax, fig = setup_plot()
    dataLen = 1000

    for i in range(len(q_vec)):
        # plot amplitudes
        ax.plot3D(np.ones(len(t_vec))*q_vec[i], t_vec, zs=np.abs(h_vec[i]), color=color_amp, lw=line_width)
        for emp_time in empirical_node_times:
            emp_idx = np.argmin(np.abs(t_vec - emp_time))
            this_x = q_vec[i]
            this_y = t_vec[emp_idx]
            this_z = np.abs(h_vec[i][emp_idx])
            # plot markers for empirical nodes
            ax.scatter(this_x, this_y, this_z, c=color_marker_nd, marker='o', s=marker_size)

    # fit each node
    xVals_vec = []
    yVals_vec = []
    zVals_vec = []
    for emp_time in empirical_node_times:
        emp_idx = np.argmin(np.abs(t_vec - emp_time))
        data_zVals = []
        for i in range(len(q_vec)):
            data_zVals.append(np.abs(h_vec[i][emp_idx]))
        xVals = np.linspace(min(q_vec), max(q_vec), dataLen)
        yVals = np.ones(dataLen)*emp_time
        popt, pcov = curve_fit(func, q_vec, data_zVals)
        zVals = func(xVals, *popt)
        xVals_vec.append(xVals)
        yVals_vec.append(yVals)
        zVals_vec.append(zVals)

    if plotTest == 0:       # plot fits in increments
        for k in range(Nsteps):
            startIdx = k*dataLen/Nsteps
            endIdx = (k+1)*dataLen/Nsteps
            for j in range(len(empirical_node_times)):
                 ax.plot3D(xVals_vec[j][startIdx:endIdx], yVals_vec[j][startIdx:endIdx], \
                     zs=zVals_vec[j][startIdx:endIdx], color=color_fit, lw=line_width)

            ax.set_xlim(0, 10)
            ax.set_ylim(min(t_vec), max(t_vec))
            ax.set_xlabel('$q$', fontsize=label_fontsize)
            ax.set_ylabel('$t$ $(M)$', fontsize=label_fontsize)
            ax.set_zlim(0, ZLIM)
            save_plot('%s/ca_amp_fits_%.4d.png'%(plotdir, k))
    elif plotTest == 1:     # plot entire fit, and then plot test case in increments
        for j in range(len(empirical_node_times)):
            # plot entire fit
            ax.plot3D(xVals_vec[j], yVals_vec[j], zs=zVals_vec[j], color=color_fit, lw=line_width)

        ax.set_xlim(0, 10)
        ax.set_ylim(min(t_vec), max(t_vec))
        ax.set_xlabel('$q$', fontsize=label_fontsize)
        ax.set_ylabel('$t$ $(M)$', fontsize=label_fontsize)
        ax.set_zlim(0, ZLIM)

        # plot nodes for the test case
        for emp_time in empirical_node_times:
            emp_idx = np.argmin(np.abs(t_vec - emp_time))
            this_x = q_test
            this_y = t_vec[emp_idx]
            this_z = np.abs(h_test[emp_idx])
            # plot nodes for the test case
            ax.scatter(this_x, this_y, this_z, c=color_marker_test_nd, marker='x', s=marker_size)

        # plot test case in increments
        for k in range(Nsteps):
            startIdx = k*len(t_vec)/Nsteps
            endIdx = (k+1)*len(t_vec)/Nsteps
            # plot test case in increments
            ax.plot3D(np.ones(len(t_vec))[startIdx:endIdx]*q_test, t_vec[startIdx:endIdx], \
                zs=np.abs(h_test)[startIdx:endIdx], color=color_test, lw=line_width)
            save_plot('%s/da_test_amp_%.4d.png'%(plotdir, k))

    P.close()


# ---------------------------------------------------------------------------------------
def make_video(pattern, plotdir, moviedir, movienametag):
    """ Combine particular png files to make a video
    """
    images_list = glob('%s/%s'%(plotdir, pattern))
    images_list.sort()
    # save all required files into tmp_moviedir, with simple filenames: %.4d.png
    tmp_moviedir = '%s/tmp_movie_%s'%(plotdir, movienametag)
    os.system('mkdir -p %s'%tmp_moviedir)
    for i in range(len(images_list)):
        fname = images_list[i].split('%s/'%plotdir)[-1].split('.png')[0]
        os.system('cp %s/%s.png %s/%.4d.png'%(plotdir, fname, tmp_moviedir, i))

    os.system('avconv -i %s'%tmp_moviedir +'/%04d.png ' \
        +' -y -c:v libx264 -pix_fmt yuv420p %s/%s.mp4'%(moviedir, movienametag))


###########################################   main   ########################################
plotdir = 'Movie'
os.system('mkdir -p %s'%plotdir)

q_test = 5.5
t_vec, h_test = get_wave(q_test)

q_vec = np.array([7, 1, 10, 3.5])       # This order will be used to add the greedy basis
h_vec = []
# assuming here that t_vec is same for all q
for q in q_vec:
    t_vec, h = get_wave(q)
    if len(h) != len(h_test):
        raise Exception("Lengths don't match")
    h_vec.append(h)

empirical_node_times = [min(t_vec), -200, -50, 0]
if len(empirical_node_times) != len(q_vec):
    raise Exception('Number of empirical nodes should be same as number of basis.')

#NOTE: We want to make four videos, as follows:
# We will add prefixes to png filenames to ensure they have the right alphabetical order.
# 1. Video of real parts and amplitudes in increments.
#       Images saved as aa_qidx%d_wave_%.4d.png and ab_waveamp_%.4d.png respectively.
#       Video saved as waves.mp4.
# 2. Video of rolling along t and adding empirical nodes.
#       Images saved as ba_amp_%.4d.png.
#       Video saved as ei.mp4.
# 3. Video of empirical node fits in increments.
#       Images saved as ca_amp_fits_%.4d.png.
#       Video saved as fits.mp4
# 4. Video of test case amplitude in increments.
#       Images saved as da_test_amp_%.4d.png
#       Video saved as eval.mp4

make_greedy_plots()
print 'Made waves'

make_wave_plots(plotAmp=1)
print 'Made amplitude'

make_wave_plots(plotAmp=2)
print 'Made empirical nodes'

make_fit_plots(plotTest=0)
print 'Made fits'

make_fit_plots(plotTest=1)
print 'Made eval\n'


moviedir = '%s/surrogates_demo'%plotdir
os.system('mkdir -p %s'%moviedir)

# make videos, these different types are expained above
make_video('a?_*.png', plotdir, moviedir, 'waves')
make_video('b?_*.png', plotdir, moviedir, 'ei')
make_video('c?_*.png', plotdir, moviedir, 'fits')
make_video('d?_*.png', plotdir, moviedir, 'eval')

# also make full movie
make_video('*.png', plotdir, moviedir, 'full_movie')

print '\nMade videos'
