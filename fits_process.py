#!/usr/bin/env python
# STH: hacks for online analysis prefixed with '# STH:'

import numpy as np
from astropy.io import fits
from astropy.time import Time
import glob
import os
import photutils
import scipy.optimize as sco

# Gaussian functional form assumed for PSF fits
def psf((xx,yy),s0,s1,x0,y0,w):
    elow = -50.
    arg = - w * ( (xx-x0)**2 + (yy-y0)**2 )
    arg[arg <= elow] = elow
    intensity = s0 + s1 * np.exp( arg )
    fwhm = 2. * np.sqrt(np.log(2.)/np.abs(w))
    # Turn 2D intensity array into 1D array
    return intensity.ravel()

# Total integrated flux and fwhm for the assumed Gaussian PSF
def psf_flux(s0,s1,x0,y0,w):
    flux = np.pi * s1/np.abs(w)
    fwhm = 2. * np.sqrt(np.log(2.)/np.abs(w))
    return fwhm, flux

def center(xx):
    global imdata
    # The aperture size for finding the location of the stars is arbitrarily set here
    app = 7.
    # STH: make sure imdata is 2d, but only here
    if imdata.ndim == 3:
        imdata2d = imdata[0]
        flux = -photutils.aperture_circular(imdata2d, xx[0], xx[1], app, method='exact',subpixels=10)
    else:
        flux = -photutils.aperture_circular(imdata, xx[0], xx[1], app, method='exact',subpixels=10)
    return flux

# The derivative of the Gaussian PSF with respect to x and y in pixels. eps = 0.2 pixels
def der_center(xx):
    eps = 0.2
    # The aperture size for finding the location of the stars is arbitrarily set here
    app = 7.
    der = np.zeros_like(xx)
    fxp1 =  center([ xx[0] + eps, xx[1] ])
    fxm1 =  center([ xx[0] - eps, xx[1] ])
    fyp1 =  center([ xx[0], xx[1] + eps ])
    fym1 =  center([ xx[0], xx[1] - eps ])

    der[0] = (fxp1-fxm1)/(2.*eps)
    der[1] = (fyp1-fym1)/(2.*eps)
    return der


# The main function for doing aperture photometry on individual FITS files
# app = -1 means the results are for a PSF fit instead of aperture photometry
def aperture(image,hdr,dnorm):
    global iap, pguess_old, nstars, svec
    rann1 = 18.
    dann = 2.
    rann2 = rann1 + dann
    app_min = 1.
    app_max = 19.
    dapp = 1.
    app_sizes = np.arange(app_min,app_max,dapp)

    # target= hdr['target']      # this header info exists for Argos .fits files
    # comp  = hdr['compstar']    # this header info exists for Argos .fits files
    # ts = target.split(',')
    # cs = comp.split(',')
    # xvec = map(float, [ ts[0], cs[0] ])
    # yvec = map(float, [ ts[1], cs[1] ])

    # If first time through, read in "guesses" for locations of stars
    if iap == 0:
        var = np.loadtxt('phot_coords')
        xvec = var[:,0]
        yvec = var[:,1]
        nstars = len(xvec)
        #print app_sizes,'\n'
    else:
        xvec = svec[:,0]
        yvec = svec[:,1]

    #print nstars, xvec, yvec
    # Find locations of stars
    dxx0 = 10.
    for i in range(nstars):
        xx0 = [xvec[i], yvec[i]]
        xbounds = (xx0[0]-dxx0,xx0[0]+dxx0)
        ybounds = (xx0[1]-dxx0,xx0[1]+dxx0)
        #res = sco.minimize(center, xx0, method='BFGS', jac=der_center)
        #res = sco.fmin_tnc(center, xx0, bounds=(xbounds,ybounds))
        #res = sco.minimize(center, xx0, method='tnc', bounds=(xbounds,ybounds))
        res = sco.minimize(center, xx0, method='L-BFGS-B', bounds=(xbounds,ybounds),jac=der_center)
        xx0=res.x
        xvec[i] = xx0[0]
        yvec[i] = xx0[1]

    #print nstars, xvec, yvec
    #exit()

    # Calculate sky around stars
    # STH: make sure image is 2d, but only here
    # STH: TODO: RESUME HERE:
    # ValueError: trailing dimension of 'apertures' must match the length of xc, yc
    if image.ndim == 3:
        image2d = image[0]
        sky  = photutils.annulus_circular(image2d, xvec, yvec, rann1, rann2, method='exact',subpixels=10)
    else:
        sky  = photutils.annulus_circular(image, xvec, yvec, rann1, rann2, method='exact',subpixels=10)

    # Do psf fits to stars. Results are stored in arrays fwhm, pflux, psky, psf_x, and psf_y
    nx=10
    ny=nx
    psky  = np.zeros(nstars)
    psf_x = np.zeros(nstars)
    psf_y = np.zeros(nstars)
    fwhm  = np.zeros(nstars)
    pflux = np.zeros(nstars)
    if iap == 0:
        pguess_old = np.zeros((nstars,5))
    for i in range(nstars):
        xc = int(xvec[i])
        yc = int(yvec[i])
        x=range(xc-nx,xc+nx+1)
        y=range(yc-ny,yc+ny+1)
        x, y = np.meshgrid(x,y)
        img = image[y,x]
        # Make 2D array into 1D array so it can be fed to curve_fit()
        imgr = img.ravel()
        if iap == 0:
            pvals = [ dnorm, 2000., .3, nstars]
            pguess=np.array([ dnorm, 2000., xvec[i], yvec[i], .3])
        else:
            pguess = pguess_old[i]
        pguess[2]=xvec[i]
        pguess[3]=yvec[i]
        popt, pcov = sco.curve_fit(psf, (x, y), imgr, p0=pguess)
        pguess_old[i,:] = popt
        dlevel = popt[0]
        fwhm[i], pflux[i] = psf_flux(*popt)
        psky[i]  = np.pi * (2.*fwhm[i])**2 * dlevel
        psf_x[i] = popt[2]
        psf_y[i] = popt[3]

    # Make stacked array of star positions from aperture photometry
    svec = np.dstack((xvec,yvec))[0]
    #print svec

    # Make stacked array of star positions from PSF fitting
    pvec = np.dstack((psf_x,psf_y))[0]

    iap = iap + 1

    starr = []
    apvec = []
    app=-1.0

    starr.append([pflux,psky,fwhm])
    apvec.append(app)

    # Get time of observation from the header
    date = hdr['DATE-OBS']
    utc  = hdr['UTC']
    times = date + "  " + utc
    t = Time(times, format='iso', scale='utc')
    # Calculate Julian Date of observation
    jd  = t.jd
    for app in app_sizes:
        flux = photutils.aperture_circular(image, xvec, yvec, app, method='exact',subpixels=10)
        skyc = sky*app**2/(rann2**2 - rann1**2)
        fluxc = flux  - skyc
        starr.append([fluxc,skyc,fwhm])
        apvec.append(app)
    starr = np.array(starr)
    apvec = np.array(apvec)
    #print starr
    return jd,svec,pvec,apvec,starr

def combine(pattern,fileout):
    files = glob.glob(pattern)
    #print 'files= ',files
    n=0
    print "  Combining frames: "
    for file in files:
        pa,fi = os.path.split(file)
        print fi,
        if ( (n+1)/4 )*4 == n+1:
            print " "
        list = fits.open(file)
        imdata = list[0].data
        hdr    = list[0].header
        if n==0:
            exptime0 = hdr['exptime']
        exptime = hdr['exptime']
        if exptime != exptime0:
            print "\nError: Exposure time mismatch in",file,"\n"
            exit()
        if file == files[0]:
            comb = imdata
        else:
            comb = comb + imdata
        list.close()
        n=n+1

    print ""
    comb = comb/float(n)
    out = fits.open(files[0])
    hdr = out[0].header
    hdr.rename_keyword('NTP:GPS','NTP-GPS')
    combdata = out[0].data
    combdata = comb
    out.writeto(fileout)
    out.close()
    return exptime, files, comb


if __name__ == '__main__':

    global imdata, iap, nstars

    # # STH: don't do calibration frames.
    # # Make master Dark file
    # darks_pattern = 'd17apr5/*.fits'
    # print '\nComputing master darks for',darks_pattern
    # dexptime, dfiles, master_dark = combine(darks_pattern,'Dark.fits')
    # dnorm = np.mean(master_dark)
    # print 'Avg dark pixel = ',dnorm,' counts'

    # # Make master Flat file
    # flats_pattern = 'f17apr10/*.fits'
    # print '\nComputing master flats for',flats_pattern
    # fexptime, ffiles, master_flat = combine(flats_pattern,'Flat.fits')

    # if dexptime != fexptime:
    #     print '\nWarning: Exposure times for darks and flats do not match.\n' 
    #     #print '\nWarning: Exposure times for darks and flats do not match. You must either provide bias files '
    #     #print 'or dark and flat files with matching exposure times.\n'
    #     #exit()
    # master_flat_c = master_flat - master_dark
    # fnorm = np.mean(master_flat)
    # print 'Avg flat pixel = ',fnorm,' counts\n'
    # master_flat_n = master_flat_c/fnorm
    # arg_probs = np.argwhere(master_flat_n <= 0.0)
    # master_flat_n[master_flat_n <= 0.0] = 1.0

    # # STH: TODO: call spe file here and use astropy to convert to nddata
    # Get list of all FITS images for run
    # fits_files = glob.glob('A????.????.fits')
    fits_files = glob.glob("*.fits")
    # This is the first image
    fimage = fits_files[0]

    # # STH: Don't do corrections
    # print "Dark correcting and flat-fielding files...\n"
    list = fits.open(fimage)
    hdr = list[0].header
    # # STH: Don't do corrections
    object= 'object' # hdr['object']
    run= 'run' # hdr['run']
    fout=open('lightcurve_old.app','w')
    efout=open('lightcurve.app','w')
    nstars = 3
    dform0='#   Aperture reductions for run {0} on target {1}. Total number of stars is {2}\n'.format(run,object,nstars)
    dform='#    time (JD)      App (pix)   Target Counts    Comparison Counts     Sky Counts     Target Position    Comp Position    FWHM        Fits File\n'
    fout.write(dform0)
    efout.write(dform0)
    fout.write(dform)

    # Format header for the general case of nstars stars
    eform0='#    time (JD)      App (pix)   Target Counts'
    'Comparison Counts     Sky Counts     Target Position    Comp Position    FWHM        Fits File\n'
    for i in range(1,nstars):
        eform0 = eform0 + '      Comp {0} Counts'.format(i)
    eform0 = eform0 + '       Target Sky'
    for i in range(1,nstars):
        eform0 = eform0 + '         Comp {0} Sky'.format(i)
    eform0 = eform0 + '     Target Position'
    for i in range(1,nstars):
        eform0 = eform0 + '  Comp {0} Position'.format(i)
    eform0 = eform0 + '  Target FWHM'
    for i in range(1,nstars):
        eform0 = eform0 + ' Comp {0} FWHM'.format(i)
    eform0 = eform0 + '   FITS File\n'
    #print eform0
    efout.write(eform0)

    #print 'Calculating apertures:'

    iap = 0
    icount = 1
    fcount = ''
    print 'Processing files:'
    for file in fits_files:
        fcount = fcount + '  ' + file
        if np.remainder(icount,5) == 0:
            print fcount
            fcount = ''
        else:
            if file == fits_files[-1]:
                print fcount
        icount = icount + 1
        # # STH: Don't do corrections
        # # Create modified file name for "corrected" FITS files
        # s=file.split('.')
        # filec = s[0] + 'c.' + s[1] + '.' + s[2]
        # open FITS file
        list = fits.open(file)
        imdata = list[0].data
        # # STH: Don't do calibrations for online analysis hack
        # # Dark and Flat correct the data
        # imdata = imdata - master_dark
        # imdata = imdata/master_flat_n
        hdr = list[0].header
        # # STH: No corrections
        # # Replace offending ":" character wrt the FITS standard with the accepted "-" character
        # hdr.rename_keyword('NTP:GPS','NTP-GPS')
        # # Write out corrected FITS file
        # list.writeto(filec)
        list.close()

        # Call aperture photometry routine. Get times, positions, and fluxes
        # var2 contains the list [fluxc,skyc,fwhm])
        # jd,svec,pvec,apvec,starr
        # fluxc, skyc, and fwhm are all lists of length nstars
        # STH: hack since no corrections
        dnorm = np.mean(imdata)
        jd, svec, pvec, apvec, var2 = aperture(imdata,hdr,dnorm)
        ndim = len(apvec)

        # STH: Don't use fixed-width format. Use csv.
        # loop over apertures
        for i in range(0,ndim):
            if apvec[i] >= 0.0:
                dform = '{0:18.8f}  {1:7.2f} {2:17.8f}  {3:17.8f}  {4:17.8f}    {5:6.2f} {6:6.2f} {7:10.2f} {8:6.2f} {9:8.3f}    {10}\n'.format(jd,apvec[i],var2[i,0,0],var2[i,0,1],var2[i,1,0], svec[0,0], svec[0,1], svec[1,0], svec[1,1], var2[2,2,0], file)     
                eform = '{0:18.8f}  {1:7.2f} '.format(jd,apvec[i])
                for j in range(0,nstars):
                    eform = eform + '{0:17.8f}  '.format(var2[i,0,j])
                for j in range(0,nstars):
                    eform = eform + '{0:17.8f}  '.format(var2[i,1,j])
                for j in range(0,nstars):
                    eform = eform + '{0:8.2f} {1:7.2f} '.format(svec[j,0],svec[j,1])
                for j in range(0,nstars):
                    eform = eform + '{0:8.3f}    '.format(var2[i,2,j])
                eform = eform + file + '\n'
            else:
                eform = '{0:18.8f}  {1:7.2f} '.format(jd,apvec[i])
                for j in range(0,nstars):
                    eform = eform + '{0:17.8f}  '.format(var2[i,0,j])
                for j in range(0,nstars):
                    eform = eform + '{0:17.8f}  '.format(var2[i,1,j])
                for j in range(0,nstars):
                    eform = eform + '{0:8.2f} {1:7.2f} '.format(pvec[j,0],pvec[j,1])
                for j in range(0,nstars):
                    eform = eform + '{0:8.3f}    '.format(var2[i,2,j])
                eform = eform + file + '\n'
            fout.write(dform)
            efout.write(eform)
            #print eform,


    fout.close()
