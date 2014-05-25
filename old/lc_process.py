#!/usr/bin/env python 
import matplotlib
matplotlib.use('Agg')
import pylab as P
from matplotlib.pyplot import *
from matplotlib.transforms import offset_copy
import numpy as np
from scipy import interpolate
from scipy.signal import lombscargle
from scipy.optimize import leastsq
import glob
import os 
from matplotlib.backends.backend_pdf import PdfPages


# Make a plot of the optimal light curve file
def lcplot(prfile):
    vars = np.loadtxt(prfile)
    time = vars[:,0]
    target = vars[:,1]
    comp = vars[:,2]
    sky = vars[:,-1]
    ratio = target/comp
    ratio_norm = ratio/np.mean(ratio) - 1.
    target_norm = target/np.mean(target) - 1.
    comp_norm = comp/np.mean(comp) - 1.
    sky_norm = sky/np.mean(sky) - 1.
    ndata = len(time)
    dt = time[1]-time[0]
    fmax = 1./(2.*dt)
    fmax = 0.01
    df = 1./(5.*(time[-1]-time[0]))
    fmin = df
    df = (fmax-fmin)/1000.
    farray = np.arange(fmin,fmax,df)
    omarray = 2.* np.pi * farray
    pow = lombscargle(time,ratio_norm,omarray)
    pow = pow * 4/ndata
    amp = np.sqrt(pow)

    fig=figure(1,figsize=(16, 12),frameon=False)
    ax1 = fig.add_subplot(411)
    plot(time,ratio_norm)
    ylabel(r'$\delta I/I$')

    ax2 = fig.add_subplot(412,sharex=ax1)
    plot(time,comp_norm)
    ylabel(r'$\delta I/I$')

    ax3 = fig.add_subplot(413,sharex=ax2)
    plot(time,sky_norm)
    ylabel(r'$\delta I/I$')
    xlabel(r'${\rm time \, (sec)}$',size='x-large')

    subplots_adjust(hspace = .001)

    ax = fig.add_subplot(414)
    freqs = farray * 1.e+6
    plot(freqs,amp)
    xlabel(r'Frequency ($\mu$Hz)')
    ylabel('Amplitude')

    filebase = 'lc.pdf'
    #print fileout
    savefig(filebase,transparent=True,bbox_inches='tight')
    close()
    exit()

# ysig is normalized so that it represents the point-to-point 
# scatter sigma_i, assuming uncorrelated, random noise
def scatter(lcvec):
    ndata = len(lcvec)
    ivec = np.arange(0,ndata-1)
    ivecp = ivec + 1
    dy    = lcvec[ivecp] - lcvec[ivec]
    dy = dy
    ysig  = np.sqrt(np.dot(dy,dy)/(2.*(ndata-1.)))
    return ysig

if __name__ == '__main__':

    prfile = 'optimal_5.0.dat'
    lcplot(prfile)

    lcfile = 'lightcurve.app'

    # Get number of stars
    f = open(lcfile,'r')
    line = f.readline()
    s = line.split()
    nstars = int(s[-1])
    f.close()

    #print 'nstars=',nstars

    cols=range(0,3*nstars+1)
    cols=range(0,3*nstars+1)
    var = np.loadtxt(lcfile,usecols=cols)

    jdarr  = var[:,0]
    aparr  = var[:,1]
    fluxes = var[:,2:2+nstars]
    sky    = var[:,2+nstars:2+2*nstars]
    pos    = var[:,2+2*nstars:2+4*nstars]
    fwhm   = var[:,2+4*nstars:2+5*nstars]

    apset = set(aparr)
    aplist = list(apset)
    print aplist

    apmin = min(aparr)
    apmax = max(aparr)
    print '\nUsing apertures from',int(apmin),'to',int(apmax),'pixels...\n'
    apvals = np.arange(apmin,apmax+0.5)

    appfile='app_fits2.dat'
    f=open(appfile,'w')

    # Make a list of all permutations of including/not including comparison stars
    # This algorithm is hopefully not transparent, but it works!
    ns = nstars - 1
    combos = []
    for i in range(1,2**ns):
        num = np.base_repr(i)
        stf = '{0:0' + '{0}b'.format(ns) + '}'
        li = stf.format(i)
        perms = list(li)
        plist = map(float,perms)
        combos.append(plist)

    combos.reverse() 
    combos = np.array(combos)
    # 
    for app in aplist:

        mask = np.where(aparr == app)

        jd =  jdarr[mask]
        ap =  aparr[mask]
        stars = fluxes[mask,:][0]
        skys  = sky[mask,:][0]
        xpos  = pos[mask,:][0]
        fw =   fwhm[mask]

        ta = stars[:,0]
        tanorm = ta/np.mean(ta)

        # Generate all possible combinations of comparison stars
        enum = 0
        for com in combos:
            enum = enum + 1
            co = 0.*ta
            for ic in range(ns):
                co = co + com[ic] * stars[:,ic+1]

            ysig_ta = scatter(tanorm)

            comean = np.mean(co)
            print ' comean =', comean
            if comean <= 0.:
                ysig_co = 1.e+99
                ysig_ra = 1.e+99
                break

            conorm = co/comean
            if np.any(conorm <= 0.):
                print '*** Problem ***'
                ysig_co = 1.e+99
                ysig_ra = 1.e+99
                break

            tadivco = tanorm/conorm

            ysig_co = scatter(conorm)
            ysig_ra = scatter(tadivco)

            if np.isnan(ysig_ra)  :
                print '*** Error ***'
                print ysig_ra
                exit()

            t1string = 'Aperture = {0} pixels, '.format(int(app))
            t4string = 'ysig_ta = {0:9.5f}, ysig_co = {1:9.5f}, ysig_ra = {2:9.5f}'.format(ysig_ta,ysig_co,ysig_ra)

            dstring0 = '{0:3d} {1:12.6f} {2:12.6f} {3:12.6f} '.format(int(app),ysig_ta,ysig_co,ysig_ra)
            dstring1 = '   {0}'.format(enum)
            for ic in range(ns):
                dstring1 = dstring1 + '   {0}'.format(int(com[ic]))
            dstring = dstring0 + dstring1 + '\n'
            tstring = t1string + t4string + dstring1
            print tstring
            f.write(dstring)

    f.close()

    var = np.loadtxt(appfile)

    apps0=var[:,0]
    yscat_ta0=var[:,1]
    yscat_co0=var[:,2]
    yscat_ra0=var[:,3]
    coms     =var[:,4]
    nvar = len(var[0,:])
    #print 'nvar=',nvar
    ncomps = nstars - 1
    ucoms = list(set(coms))
    ncoms = len(ucoms)
    #print ncoms,ucoms,nvar
    

    lab = 'C = '
    pp = PdfPages('app_fits.pdf')
    for com in ucoms:
        mask = np.where((apps0 > 0.) & (coms == com)) 
        apps  =  apps0[mask]
        yscat_ta = yscat_ta0[mask]
        yscat_co = yscat_co0[mask]
        yscat_ra = yscat_ra0[mask]
        mask2 = np.where((apps0 < 0.) & (coms == com)) 
        ind0 = mask[0][0]
        lcom = var[ind0,5:5+nstars-1]
        #tout = 'comp = {0}*comp1'.format(lcom[0])
        tout = 'C = {0}*C_1'.format(lcom[0])
        for i in range(1,ncomps):
            #tout = tout + ' + {0}*comp{1}'.format(lcom[i],i+1)
            tout = tout + ' + {0}*C_{1}'.format(lcom[i],i+1)
            #tout = tout + '$'
        tout = r'$' + tout + '$'
        sigmin = yscat_ra.min()
        tout = tout + '\n' + r'$\sigma_{\rm min}=' + r'{0}$'.format(sigmin)

        yscat_psf = yscat_ra0[mask2][0] + 0.*apps
        y2 = 5.*min(yscat_ra)

        fig=figure(1,frameon=False)
        ax = fig.add_subplot(111)
        plot(apps,yscat_ra)
        plot(apps,yscat_ta)
        plot(apps,yscat_co)
        plot(apps,yscat_psf,'--')
        ylim(0.0,y2)
        xlabel('Aperture (pixels)')
        ylabel('point-to-point scatter')
        leglabel2 = ['target/comparison', 'target', 'comparison','PSF fit']
        leg=ax.legend(leglabel2,'best',fancybox=True,shadow=False)
        leg.draw_frame(False)
        t1='Light Curve Scatter as a Function of Aperture Size'
        t2 = t1 + tout
        ax.set_title(t1,size='small')
        ax.text(0.03, 0.11, tout, transform=ax.transAxes, fontsize=14,
        verticalalignment='top',horizontalalignment='left',size='x-small')

        #fileoutapp = 'app_fits_{0}.pdf'.format(com)
        #savefig(fileoutapp,transparent=True,bbox_inches='tight')
        pp.savefig(transparent=True,bbox_inches='tight')
        cla()

    pp.close()
    imin = yscat_ra0.argmin()
    com0 = coms[imin]
    app = apps0[imin]
    sout  = '\nOptimal light curve has an aperture size of {0} pixels, an rms of {1}, \nand a composite comparison star C = '.format(app,yscat_ra0[imin])
    sout3 = '# Optimal light curve has an aperture size of {0} pixels, an rms of {1}, \n# and a composite comparison star comp = '.format(app,yscat_ra0[imin])
    lcom = var[imin][5:5+nstars-1]
    tout = '{0}*C1'.format(lcom[0])
    for i in range(1,ncomps):
        tout = tout + ' + {0}*C{1}'.format(lcom[i],i+1)
    sout2 = sout + tout + '\n'
    sout4 = sout3 + tout + '\n'
    print sout2

    mask = np.where(aparr == app)

    jd =  jdarr[mask]
    stars = fluxes[mask,:][0]
    skys  = sky[mask,:][0]

    time = (jd-jd[0])*86400.
    ndata = len(time)

    fig=figure(1,frameon=False)
    ax = fig.add_subplot(111)
    for n in range(nstars):
        if n==0:
            legvec = ['target']
        else:
            lab = 'comp {0}'.format(n)
            legvec.append(lab)
        st = stars[:,n]
        stnorm = st/np.median(st) - 0.15*n
        plot(time,stnorm)

    sk = skys[:,0]
    sknorm = sk/np.median(sk) - 0.15*nstars
    plot(time,sknorm)
    legvec.append('sky')
    leg=ax.legend(legvec,'best',fancybox=True,shadow=False)
    leg.draw_frame(False)
    y1,y2 = ylim()
    ylim(0,y2)
    xlabel('time (sec)')
    ylabel(r'$I(t)/\langle I \rangle$')
    fileoutapp = 'lcnorm.pdf'
    savefig(fileoutapp,transparent=True,bbox_inches='tight')
    close()

    # This writes out the optimal profile
    prfile='optimal_{0}.dat'.format(app)
    f=open(prfile,'w')
    f.write(sout4)
    for i in range(ndata):
        dstring0 = '{0:15.5f} '.format(time[i]) 
        dstring1 = ' '
        for n in range(nstars):
            dstring1 = dstring1 + '{0:13.5f} '.format(stars[i,n])
        dstring2 = '{0:15.5f} \n'.format(skys[i,0])
        dstring = dstring0 + dstring1 + dstring2
        f.write(dstring)
        #print i,dstring,

    f.close()

    # Everything below this point merely creates a .wq file. You can comment 
    # it out and just create it by hand if this doesn't work

    # Get list of all FITS images for run
    # fits_files = glob.glob('A????.????.fits')
    fits_files = glob.glob('GD244-????.fits')
    fimage = fits_files[0]
    com0 = 'whiff ' + fimage + ' -o mcd1'
    retvalue = os.system(com0)

    header = glob.glob('mcd1*head')[0]
    # This is the first image
    #run = fimage[1:5]
    run = header[0:-5]
    argos_out = run + '.wq'

    com1 = 'wqedit ' + header + ' -k Runname -v ' + run
    retvalue = os.system(com1)
    com2 = 'wqedit {0} -k Aperture -v {1} -c "Radius in pixels"'.format(header,app)
    retvalue = os.system(com2)
    com3 = 'cat ' + header + ' ' + prfile + ' > ' + argos_out
    retvalue = os.system(com3)
