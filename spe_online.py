#!/usr/bin/env python
"""
Read .spe file and do aperture photometry.
"""

from astropy.io import fits
from astropy.time import Time
import glob
import os
import photutils
import scipy.optimize as sco
import numpy as np
import argparse
import read_spe

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
def aperture(image,hdr):
    global iap, pguess_old, nstars, svec
    dnorm = 500.
    rann1 = 18.
    dann = 2.
    rann2 = rann1 + dann
    app_min = 1.
    app_max = 19.
    dapp = 1.
    app_sizes = np.arange(app_min,app_max,dapp)

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


    # Calculate sky around stars
    sky  = photutils.annulus_circular(image, xvec, yvec, rann1, rann2, method='exact',subpixels=10)

    # Do psf fits to stars. Results are stored in arrays fwhm, pflux, psky, psf_x, and psf_y
    fwhm  = np.zeros(nstars)

    # Make stacked array of star positions from aperture photometry
    svec = np.dstack((xvec,yvec))[0]
    #print svec

    # Make stacked array of star positions from PSF fitting
    # pvec = np.dstack((psf_x,psf_y))[0]
    pvec = svec

    iap = iap + 1

    starr = []
    apvec = []
    app=-1.0

    # Get time of observation from the header
    #date = hdr['DATE-OBS']   # for Argos files
    #utc  = hdr['UTC']         # for Argos files
    date = hdr['UTC-DATE']    # for Pro-EM files
    utc  = hdr['UTC-BEG']     # for Pro-EM files
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

# Not called.
# def combine(pattern,fileout):
#     files = glob.glob(pattern)
#     print 'files= ',files
#     n=0
#     print "  Combining frames: "
#     for file in files:
#         pa,fi = os.path.split(file)
#         print fi,
#         if ( (n+1)/4 )*4 == n+1:
#             print " "
#         list = fits.open(file)
#         imdata = list[0].data
#         hdr    = list[0].header
#         if n==0:
#             exptime0 = hdr['exptime']
#         exptime = hdr['exptime']
#         if exptime != exptime0:
#             print "\nError: Exposure time mismatch in",file,"\n"
#             exit()
#         if file == files[0]:
#             comb = imdata
#         else:
#             comb = comb + imdata
#         list.close()
#         n=n+1

#     print ""
#     comb = comb/float(n)
#     out = fits.open(files[0])
#     hdr = out[0].header
#     #hdr.rename_keyword('NTP:GPS','NTP-GPS')  # Argos
#     combdata = out[0].data
#     combdata = comb
#     out.writeto(fileout)
#     out.close()
#     return exptime, files, comb

def head_write(ffile,object,nstars):
    dform0='#   Aperture reductions for target {0}. Total number of stars is {1}\n'.format(object,nstars)
    ffile.write(dform0)

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
    ffile.write(eform0)

def app_write(efout,ndim,nstars,jd,apvec,svec,pvec,var2):
    for i in range(0,ndim):
        if apvec[i] >= 0.0:
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
        efout.write(eform)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read .spe file and do aperture photometry.")
    parser.add_argument("--fpath",
                        required=True,
                        help=("Input .spe file."))
    parser.add_argument("--verbose", "-v",
                        action='store_true',
                        help=("Print 'INFO:' messages to stdout."))
    args = parser.parse_args()
    if args.verbose:
        print "INFO: Arguments:"
        for arg in args.__dict__:
            print ' ', arg, args.__dict__[arg]

    # TODO: use classes to retain state information.
    global imdata, iap, nstars

    # # Get list of all FITS images for run
    # run_pattern = 'GD244-????.fits'
    # #fits_files = glob.glob('A????.????.fits')
    # fits_files = glob.glob(run_pattern)
    # # This is the first image
    # fimage = fits_files[0]

    # #print "Dark correcting and flat-fielding files...\n"
    # list = fits.open(fimage)
    # hdr = list[0].header
    # object= hdr['object']
    # #run= hdr['run']

    # TODO: use .csv and .txt. ".app" has special meaning on Mac OS.
    efout=open('lightcurve.app','w')

    #print 'Calculating apertures:'

    iap = 0
    icount = 1
    fcount = ''
    
    print 'Processing files:'
    # for file in fits_files:
    # TODO: Fix namespace error so that don't need to call twice.
    # TODO: Only reduce new frames, not all again.
    spe = read_spe.File(args.fpath)
    num_frames = spe.get_num_frames()
    for idx in xrange(num_frames):
        # fcount = fcount + '  ' + file
        # Note: Frame index is Python indexed and begins at 0.
        # frame_tracking_number from LightField begins at 1.
        fcount = fcount + '  ' + ("frame_idx_{num}".format(num=idx))
        if np.remainder(icount,5) == 0:
            print fcount
            fcount = ''
        else:
            # if file == fits_files[-1]:
            # If last frame.
            if idx == num_frames - 1:
                print fcount
        icount = icount + 1

        # # open FITS file
        # list = fits.open(file)
        # imdata = list[0].data
        # Read SPE file
        (imdata, metadata) = spe.get_frame(idx)

        # hdr = list[0].header

        # Call aperture photometry routine. Get times, positions, and fluxes
        # var2 contains the list [fluxc,skyc,fwhm])
        # jd,svec,pvec,apvec,starr
        # fluxc, skyc, and fwhm are all lists of length nstars
        # jd, svec, pvec, apvec, var2 = aperture(imdata,hdr,dnorm)
        # jd, svec, pvec, apvec, var2 = aperture(imdata,hdr)
        # TODO: Give aperture a hdr dict with keys 'UTC-DATE', 'UTC-BEG'.
        print metadata
        exit()
        (jd, svec, pvec, apvec, var2) = aperture(image=imdata, hdr=)
        ndim = len(apvec)

        # First time through write header
        if icount == 2:
            fname_base = os.path.basename(args.fpath)
            # head_write(efout,object,nstars)
            # TODO: object is a reserved word. Don't use.
            head_write(efout, object=fname_base ,nstars)

        # Write out results for all apertures
        app_write(efout,ndim,nstars,jd,apvec,svec,pvec,var2)

    spe.close()
