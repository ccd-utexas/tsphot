#!/usr/bin/env python

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

from astropy.io import fits
import matplotlib.cm as cm
import glob

global iopen,fphot

fphot = 'phot_coords'
f=open(fphot,'w')
f.write('')
f.close()

iopen = 0

fits_files = glob.glob('A????.????.fits')
fimage = fits_files[0]

list = fits.open(fimage)
imdata = list[0].data
imdata = np.array(imdata)
#imdata = np.sqrt(imdata)
imdata = np.log(imdata)
rimdata = imdata.ravel()

vmax=max(rimdata)
vmin=min(rimdata)

shape = np.shape(imdata)
xdim = shape[0]
ydim = shape[1]
# print vmin, vmax, xdim, ydim

x = range(xdim)
y = range(ydim)
xv, yv = np.meshgrid(x,y)

vmax1 = vmin + 0.3*(vmax-vmin)
vmin1 = vmin + 0.05*(vmax-vmin)
dv = (vmax1-vmin1)/10.
levels = np.arange(vmin1,vmax1,dv)
#print vmin,vmax
#exit()

fig = plt.figure(figsize=(8,8))
#ax = fig.add_subplot(111)
ax = fig.add_axes([0,0,1,1])
ax.contourf(xv,yv,imdata,levels=levels,cmap = cm.gray, vmin=vmin1,vmax=vmax1)
ax.set_aspect('equal')
ax.tick_params(\
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off',      # ticks along the bottom edge are off
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelleft='off', # labels along the bottom edge are off
    labelbottom='off') # labels along the bottom edge are off

#plt.tight_layout()
#ax.contourf(imdata,aspect='equal',vmin=vmin,vmax=vmax)


def onclick(event):
    global f,iopen
    # print event.button, event.x, event.y, event.xdata, event.ydata,event.key
    inv = ax.transData.inverted()
    vec = inv.transform((event.x,event.y))
    str = '{0:5.3f} {1:5.3f}\n'.format(vec[0],vec[1])
    print '(x,y) =',vec
    if iopen == 1:
        f.write(str)

def onpress(event):
    global f,iopen,fphot
    if event.key=='q':
        print '\nClosing plot...'
        plt.close()
        if iopen == 1:
            f.close()
            print '\nClosing',fphot,'\n'
    if event.key=='w':
        if iopen == 0:
            print '\nOpening',fphot,'for writing\n'
            f=open(fphot,'a')
            iopen = 1
        else:
            print '\nClosing',fphot,'for writing\n'
            f.close()
            iopen = 0

cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', onpress)

plt.show()

