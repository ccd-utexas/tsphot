#!/usr/bin/env python 
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import numpy as np
import scipy 
from scipy.signal import lombscargle
import matplotlib.gridspec as gridspec


def list_powerset2(lst):
    return reduce(lambda result, x: result + [subset + [x] for subset in result], lst, [[]])

# Make a plot of light curve scatter versus aperture size
def applot(aplist,sigvec,apmin):
    apfile = 'aperture.pdf'
    fig=figure(1)
    ax1 = fig.add_subplot()
    plot(aplist,sigvec,'-o')
    xlabel('Aperture size')
    ylabel('Scatter')
    tstring = 'Optimal Aperture is {0}'.format(apmin)
    #fig.suptitle(tstring, fontsize=13)
    title(tstring, fontsize=16)
    #ax1.set_title(tstring)
    savefig(apfile,transparent=True,bbox_inches='tight')
    close()
    print 'Aperture optimization stored in',apfile
    
# Make a plot of the optimal light curve file
def lcplot(time,target,comp,sky):
    ratio = target/comp
    ratio_norm = ratio/np.mean(ratio) - 1.
    #scipy.convolve(y, ones(N)/N)
    nfilt=5
    ratio_norm_smooth = scipy.convolve(ratio_norm, np.ones(nfilt)/nfilt)
    time_smooth = scipy.convolve(time, np.ones(nfilt)/nfilt)
    target_norm = target/np.mean(target) - 1.
    comp_norm = comp/np.mean(comp) - 1.
    #sky_norm = sky/np.mean(sky) - 1.
    sky_norm = sky/np.amin(sky)
    ndata = len(time)
    dt = time[1]-time[0]
    fmax = 1./(2.*dt)
    #fmax = 0.01
    #df = 1./(5.*(time[-1]-time[0]))
    df = 1./(2.*(time[-1]-time[0]))
    fmin = df
    fmin = 0.01*df
    df = (fmax-fmin)/1000.
    farray = np.arange(fmin,fmax,df)
    omarray = 2.* np.pi * farray
    pow = lombscargle(time,ratio_norm,omarray)
    pow = pow * 4/ndata
    amp = np.sqrt(pow)

    fig=figure(1,figsize=(14, 14),frameon=False)
    gs1 = gridspec.GridSpec(6, 1,hspace=0.0)
    ax1 = fig.add_subplot(gs1[0])
    ax1.get_xaxis().set_ticklabels([])
    plot(time,ratio_norm)
    ylabel(r'$\delta I/I$',size='x-large')
    leg=ax1.legend(['target'],'best',fancybox=True,shadow=False,handlelength=0.0)
    leg.draw_frame(False)

    ax2 = fig.add_subplot(gs1[1],sharex=ax1)
    plot(time_smooth[0:-nfilt],ratio_norm_smooth[:-nfilt])
    ylabel(r'$\delta I/I$',size='x-large')
    leg=ax2.legend(['target smoothed'],'best',fancybox=True,shadow=False,handlelength=0.0)
    leg.draw_frame(False)

    ax3 = fig.add_subplot(gs1[2],sharex=ax1)
    plot(time,comp_norm)
    ylabel(r'$\delta I/I$',size='x-large')
    leg=ax3.legend(['comparison'],'best',fancybox=True,shadow=False,handlelength=0.0)
    leg.draw_frame(False)

    ax4 = fig.add_subplot(gs1[3])
    plot(time,sky_norm)
    ylabel(r'$I/I_{\rm min}$',size='x-large')
    xlabel(r'${\rm time \, (sec)}$',size='x-large')
    leg=ax4.legend(['sky'],'best',fancybox=True,shadow=False,handlelength=0.0)
    leg.draw_frame(False)

    gs2 = gridspec.GridSpec(4, 1,hspace=0.1)
    ax5 = fig.add_subplot(gs2[3])

    freqs = farray * 1.e+6
    plot(freqs,amp)
    xlabel(r'Frequency ($\mu$Hz)',size='large')
    ylabel('Amplitude',size='large')
    leg=ax5.legend(['FT'],'best',fancybox=True,shadow=False,handlelength=0.0)
    leg.draw_frame(False)

    filebase = 'lc.pdf'
    savefig(filebase,transparent=True,bbox_inches='tight')
    close()
    print 'Optimal light curve plot stored in',filebase,'\n'

# ysig is normalized so that it represents the point-to-point 
# scatter sigma_i, assuming uncorrelated, random noise
def scatter(lcvec):
    ndata = len(lcvec)
    ivec = np.arange(0,ndata-1)
    ivecp = ivec + 1
    dy    = lcvec[ivecp] - lcvec[ivec]
    ysig  = np.sqrt(np.dot(dy,dy)/(2.*(ndata-1.)))
    return ysig

if __name__ == '__main__':

    lcfile = 'lightcurve.app'

    # Get number of stars
    f = open(lcfile,'r')
    line = f.readline()
    s = line.split()
    nstars = int(s[-1])
    f.close()

    print '\nThe Number of stars is',nstars

    cols=range(0,3*nstars+1)
    cols=range(0,3*nstars+1)
    print cols
    # var = np.loadtxt(lcfile,usecols=cols)
    var = np.loadtxt(lcfile)
    print var
    exit()
    
    jdarr  = var[:,0]
    aparr  = var[:,1]
    fluxes = var[:,2:2+nstars]
    sky    = var[:,2+nstars:2+2*nstars]
    pos    = var[:,2+2*nstars:2+4*nstars]
    fwhm   = var[:,2+4*nstars:2+5*nstars]

    # Get the list of unique aperture sizes
    apset = set(aparr)
    aplist = list(apset)
    aplist = np.array(aplist)

    print '\nUsing the following apertures:'
    print aplist

    # Cyle through all possible combinations of comparison stars and choose the one 
    # that minimizes the point-to-point scatter ysig

    starlist =  np.arange(1,nstars)
    pset = list_powerset2(starlist)
    del pset[0]

    # Now that we've got the list of comparison stars, let's find the optimal aperture

    sigvec = []

    # create arrays to store lightcurves for different apertures
    # Use median aperture for this
    apmed = float(int(np.median(aplist)))
    mask = np.where(aparr == apmed)
    stars = fluxes[mask,:][0]
    target = stars[:,0]
    ntarg = len(target)
    nap   = len(aplist)
    ncom  = len(pset)
    nlcs = nap*ncom

    targs   = np.zeros((ntarg,nap))
    comps   = np.zeros((ntarg,nap,ncom))
    skyvals = np.zeros((ntarg,nap))
    ysigarr = np.ones((nap,ncom))
    ysigarr = 1000000.*ysigarr
    # counter for apertures
    iap = 0
    for app in aplist:

        mask = np.where(aparr == app)

        jd =  jdarr[mask]
        ap =  aparr[mask]
        stars = fluxes[mask,:][0]
        skys  = sky[mask,:][0]
        xpos  = pos[mask,:][0]
        fw =   fwhm[mask]

        ta = stars[:,0]

        # store target and sky lightcurves
        targs[:,iap]=ta
        skyvals[:,iap]=skys[:,0]

        # Loop over all possible comparison stars
        # counter for combination of comparison stars
        nc = 0
        for term in pset:
            compstar = 0.*ta
            for ii in term:
                compstar = compstar + stars[:,ii]
            
            # divide by comparison, normalize, and shift to zero
            ratio = ta/compstar
            ratio = ratio/np.mean(ratio) - 1.0
            ysig = scatter(ratio)

            #sigvec.append(ysig)
            ysigarr[iap,nc] = ysig
            # print iap, nc, term, ysig

            # store comparison lightcurve
            comps[:,iap,nc]=compstar
            nc = nc + 1

        iap = iap + 1

    # print ysigarr

    # Find optimal aperture and its index
    sigmin = np.amin(ysigarr)
    isigmin = np.argmin(ysigarr)
    iarg = np.unravel_index(isigmin,(nap,ncom))
    iapmin, ncmin = iarg
    print nap, ncom
    #print ysigarr[iarg], iapmin, ncmin

    apmin = aplist[iapmin]
    report  = '\nThe optimal aperture is {0} pixels, the point-to-point scatter is {1:0.5f},'.format(apmin,sigmin)
    print report
    print 'and the optimal comparison star combination is',pset[ncmin],'\n'

    # Make plot of scatter versus aperture size
    sigvec = ysigarr[:,ncmin]
    applot(aplist,sigvec,apmin)

    time = 86400.*(jd-jd[0])
    target = targs[:,iapmin]
    compstar = comps[:,iapmin,ncmin]
    sky0 = skyvals[:,iapmin]

    # Make online plot of lightcurves, sky, and the FT
    lcplot(time,target,compstar,sky0)
